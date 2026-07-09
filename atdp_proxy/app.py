from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from atdp_proxy.config import Settings
from atdp_proxy.datasets import export_areal_minimal_jsonl, export_step_jsonl
from atdp_proxy.evolution import compute_evolution_signals
from atdp_proxy.ingress import BoundaryEventIn, boundary_to_event
from atdp_proxy.learning import DatasetFilters, build_steps
from atdp_proxy.providers import ModelProvider, OpenAICompatibleProvider
from atdp_proxy.recorder import ATDPRecorder
from atdp_proxy.schemas import ATDPEvent, ATDPEventCreate, RewardUpdateIn, SessionEndIn, TrajectoryResponse
from atdp_proxy.storage import ATDPStorage
from atdp_proxy.telemetry import ATDPTelemetry


def _headers(request: Request) -> dict[str, str]:
    return {key.lower(): value for key, value in request.headers.items()}


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _stream_capture_body(payload: dict[str, Any], chunks: list[str]) -> dict[str, Any]:
    raw_text = "".join(chunks)
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    finish_reason = None
    for line in raw_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = parsed.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or choice.get("message") or {}
            if isinstance(delta, dict):
                if isinstance(delta.get("content"), str):
                    content_parts.append(delta["content"])
                if isinstance(delta.get("tool_calls"), list):
                    tool_calls.extend(delta["tool_calls"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    content = "".join(content_parts)
    return {
        "id": None,
        "object": "chat.completion.stream.capture",
        "model": payload.get("model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "tool_calls": tool_calls,
        "stream_capture": {"raw_chunks": raw_text, "chunk_count": len(chunks)},
    }


def build_provider(settings: Settings) -> ModelProvider:
    if settings.provider == "openai-compatible":
        return OpenAICompatibleProvider(settings)
    raise RuntimeError(f"Unsupported provider: {settings.provider}")


def create_app(
    settings: Settings | None = None,
    provider: ModelProvider | None = None,
    storage: ATDPStorage | None = None,
    telemetry: ATDPTelemetry | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    storage = storage or ATDPStorage(settings)
    storage.initialize()
    telemetry = telemetry or ATDPTelemetry(settings)
    provider = provider or build_provider(settings)
    recorder = ATDPRecorder(storage, settings)

    app = FastAPI(
        title="ATDP Data Proxy",
        version="0.1.0",
        description="Provider-agnostic proxy for ATDP trajectories and OpenTelemetry-compatible spans.",
    )
    app.state.settings = settings
    app.state.storage = storage
    app.state.telemetry = telemetry
    app.state.provider = provider
    app.state.recorder = recorder

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "provider": settings.provider,
            "redaction_mode": settings.redaction_mode,
            "otel_enabled": telemetry.enabled,
            "otel_setup_error": telemetry.setup_error,
        }

    async def chat_completions(payload: dict[str, Any], request: Request) -> JSONResponse:
        inbound_headers = _headers(request)
        session_id = recorder.resolve_session_id(payload, inbound_headers)
        base_attributes = {
            "atdp.session_id": session_id,
            "atdp.provider": settings.provider,
            "llm.request.type": "chat",
            "llm.request.model": payload.get("model"),
        }
        with telemetry.start_span("atdp.llm.chat", base_attributes, inbound_headers) as span:
            request_event = recorder.record_llm_request(
                session_id=session_id,
                payload=payload,
                headers=inbound_headers,
                trace_id=span.trace_id,
                span_id=span.span_id,
            )
            span.set_attribute("atdp.request_event_id", request_event.id)

            trace_headers: dict[str, str] = {}
            telemetry.inject_headers(trace_headers)
            if payload.get("stream") is True:
                start = time.perf_counter()
                chunks: list[str] = []

                async def stream_and_record():
                    try:
                        async for chunk in provider.stream_chat_completions(payload, inbound_headers, trace_headers):
                            chunks.append(chunk.decode("utf-8", errors="replace"))
                            yield chunk
                        latency = time.perf_counter() - start
                        response_body = _stream_capture_body(payload, chunks)
                        response_event = recorder.record_llm_response(
                            session_id=session_id,
                            request_event=request_event,
                            response_body=response_body,
                            status_code=200,
                            latency_seconds=latency,
                            trace_id=span.trace_id,
                            span_id=span.span_id,
                        )
                        span.set_attribute("atdp.response_event_id", response_event.id)
                        span.set_attribute("atdp.latency_seconds", latency)
                    except Exception as exc:
                        latency = time.perf_counter() - start
                        span.record_exception(exc)
                        recorder.record_llm_exception(
                            session_id=session_id,
                            request_event=request_event,
                            exc=exc,
                            latency_seconds=latency,
                            trace_id=span.trace_id,
                            span_id=span.span_id,
                        )
                        yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode("utf-8")

                return StreamingResponse(
                    stream_and_record(),
                    media_type="text/event-stream",
                    headers={
                        "X-ATDP-Session-ID": session_id,
                        "X-ATDP-Request-Event-ID": request_event.id,
                    },
                )

            start = time.perf_counter()
            try:
                provider_response = await provider.chat_completions(payload, inbound_headers, trace_headers)
            except Exception as exc:
                latency = time.perf_counter() - start
                span.record_exception(exc)
                error_event = recorder.record_llm_exception(
                    session_id=session_id,
                    request_event=request_event,
                    exc=exc,
                    latency_seconds=latency,
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                )
                raise HTTPException(
                    status_code=502,
                    detail={
                        "message": "Upstream provider request failed.",
                        "error": str(exc),
                        "atdp_event_id": error_event.id,
                    },
                ) from exc

            latency = time.perf_counter() - start
            response_event = recorder.record_llm_response(
                session_id=session_id,
                request_event=request_event,
                response_body=provider_response.body,
                status_code=provider_response.status_code,
                latency_seconds=latency,
                trace_id=span.trace_id,
                span_id=span.span_id,
            )
            span.set_attribute("atdp.response_event_id", response_event.id)
            span.set_attribute("http.response.status_code", provider_response.status_code)
            span.set_attribute("atdp.latency_seconds", latency)
            response_headers = {
                "X-ATDP-Session-ID": session_id,
                "X-ATDP-Request-Event-ID": request_event.id,
                "X-ATDP-Response-Event-ID": response_event.id,
            }
            response_headers.update(provider_response.headers)
            return JSONResponse(
                status_code=provider_response.status_code,
                content=provider_response.body,
                headers=response_headers,
            )

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(payload: dict[str, Any], request: Request) -> JSONResponse:
        return await chat_completions(payload, request)

    @app.post("/chat/completions")
    async def root_chat_completions(payload: dict[str, Any], request: Request) -> JSONResponse:
        return await chat_completions(payload, request)

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        inbound_headers = _headers(request)
        trace_headers: dict[str, str] = {}
        telemetry.inject_headers(trace_headers)
        provider_response = await provider.models(inbound_headers, trace_headers)
        return JSONResponse(status_code=provider_response.status_code, content=provider_response.body)

    @app.post("/v1/atdp/events", response_model=ATDPEvent)
    async def record_event(event: ATDPEventCreate, request: Request) -> ATDPEvent:
        inbound_headers = _headers(request)
        session_id = event.session_id or recorder.resolve_session_id({}, inbound_headers)
        event.session_id = session_id
        with telemetry.start_span(
            f"atdp.event.{event.type}",
            {"atdp.session_id": session_id, "atdp.event.type": event.type},
            inbound_headers,
        ) as span:
            event.trace_id = event.trace_id or span.trace_id
            event.span_id = event.span_id or span.span_id
            return recorder.record(event)

    @app.post("/v1/atdp/boundary-events", response_model=ATDPEvent)
    async def record_boundary_event(body: BoundaryEventIn, request: Request) -> ATDPEvent:
        inbound_headers = _headers(request)
        session_id = body.session_id or recorder.resolve_session_id({}, inbound_headers)
        event = boundary_to_event(body, session_id)
        with telemetry.start_span(
            f"atdp.boundary.{body.boundary.value}",
            {
                "atdp.session_id": session_id,
                "atdp.boundary": body.boundary.value,
                "atdp.phase": body.phase.value,
            },
            inbound_headers,
        ) as span:
            event.trace_id = event.trace_id or span.trace_id
            event.span_id = event.span_id or span.span_id
            recorded = recorder.record(event)
            if body.boundary.value == "final_outcome":
                storage.mark_session_ended(session_id, recorded.metadata)
            return recorded

    @app.post("/v1/atdp/sessions/{session_id}/end", response_model=ATDPEvent)
    async def end_session(session_id: str, body: SessionEndIn | None = None) -> ATDPEvent:
        body = body or SessionEndIn()
        return recorder.record_session_end(session_id=session_id, metadata=body.metadata, reward=body.reward)

    @app.get("/v1/atdp/sessions")
    async def list_sessions() -> dict[str, Any]:
        sessions = storage.list_sessions()
        return {"sessions": [session.model_dump(mode="json") for session in sessions]}

    @app.get("/v1/atdp/sessions/{session_id}/trajectory", response_model=TrajectoryResponse)
    async def trajectory(session_id: str) -> TrajectoryResponse:
        events = storage.events_for_session(session_id)
        return TrajectoryResponse(session_id=session_id, event_count=len(events), events=events)

    @app.get("/v1/atdp/events/{event_id}", response_model=ATDPEvent)
    async def get_event(event_id: str) -> ATDPEvent:
        event = storage.get_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="ATDP event not found")
        return event

    @app.post("/v1/atdp/events/{event_id}/reward", response_model=ATDPEvent)
    async def update_reward(event_id: str, body: RewardUpdateIn) -> ATDPEvent:
        reward = recorder.reward_from_update(body)
        event = storage.append_reward_update(event_id, reward)
        if event is None:
            raise HTTPException(status_code=404, detail="ATDP event not found")
        recorder.record(
            ATDPEventCreate(
                session_id=event.session_id,
                type="reward.updated",
                parent_event_id=event.id,
                observation={"event_id": event.id},
                outcome={"reward": reward.model_dump(mode="json")},
                metadata={"source": reward.source},
                governance=event.governance,
            )
        )
        return event

    @app.get("/v1/atdp/export/jsonl")
    async def export_jsonl(session_id: str | None = None) -> PlainTextResponse:
        return PlainTextResponse(storage.export_jsonl(session_id), media_type="application/jsonl")

    def dataset_filters(
        session_id: str | None,
        tenant_id: str | None,
        training_eligible: bool | None,
        include_ineligible: bool,
        event_type: str | None,
        from_value: str | None,
        to_value: str | None,
    ) -> DatasetFilters:
        return DatasetFilters(
            session_id=session_id,
            tenant_id=tenant_id,
            training_eligible=training_eligible,
            include_ineligible=include_ineligible,
            event_type=event_type,
            from_timestamp=_parse_datetime(from_value),
            to_timestamp=_parse_datetime(to_value),
        )

    def events_for_export(filters: DatasetFilters) -> list[ATDPEvent]:
        if filters.session_id:
            return storage.events_for_session(filters.session_id)
        events: list[ATDPEvent] = []
        for session in storage.list_sessions():
            events.extend(storage.events_for_session(session.session_id))
        return events

    @app.get("/v1/atdp/datasets/steps")
    async def dataset_steps(
        session_id: str | None = None,
        tenant_id: str | None = None,
        training_eligible: bool | None = True,
        include_ineligible: bool = False,
        event_type: str | None = None,
        from_: str | None = Query(default=None, alias="from"),
        to: str | None = None,
    ) -> PlainTextResponse:
        filters = dataset_filters(session_id, tenant_id, training_eligible, include_ineligible, event_type, from_, to)
        steps = build_steps(events_for_export(filters), filters)
        return PlainTextResponse(export_step_jsonl(steps), media_type="application/jsonl")

    @app.get("/v1/atdp/datasets/export")
    async def dataset_export(
        format: str = "atdp_step_jsonl",
        session_id: str | None = None,
        tenant_id: str | None = None,
        training_eligible: bool | None = True,
        include_ineligible: bool = False,
        event_type: str | None = None,
        from_: str | None = Query(default=None, alias="from"),
        to: str | None = None,
    ) -> PlainTextResponse:
        filters = dataset_filters(session_id, tenant_id, training_eligible, include_ineligible, event_type, from_, to)
        steps = build_steps(events_for_export(filters), filters)
        if format == "atdp_step_jsonl":
            body = export_step_jsonl(steps)
        elif format == "areal_minimal":
            body = export_areal_minimal_jsonl(steps)
        else:
            raise HTTPException(status_code=400, detail="Unsupported dataset export format")
        return PlainTextResponse(body, media_type="application/jsonl")

    @app.get("/v1/atdp/evolution/signals")
    async def evolution_signals(
        session_id: str | None = None,
        tenant_id: str | None = None,
        from_: str | None = Query(default=None, alias="from"),
        to: str | None = None,
    ) -> dict[str, Any]:
        filters = dataset_filters(session_id, tenant_id, None, True, None, from_, to)
        events = events_for_export(filters)
        steps = build_steps(events, filters)
        return compute_evolution_signals(events, steps).model_dump(mode="json")

    return app

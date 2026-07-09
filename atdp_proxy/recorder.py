from __future__ import annotations

import hashlib
import time
from typing import Any
from uuid import uuid4

from atdp_proxy.config import Settings
from atdp_proxy.governance.redaction import redact_mapping, redact_value
from atdp_proxy.schemas import (
    ATDPEvent,
    ATDPEventCreate,
    EventType,
    GovernanceMetadata,
    ReplayMetadata,
    ReplayMode,
    RewardSignal,
    RewardUpdateIn,
)
from atdp_proxy.storage import ATDPStorage


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_headers(headers: dict[str, str]) -> dict[str, str]:
    blocked = {"authorization", "cookie", "set-cookie", "x-api-key"}
    return {key: value for key, value in headers.items() if key.lower() not in blocked}


def _decode_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    blocked = {"messages", "tools", "functions"}
    return {key: value for key, value in payload.items() if key not in blocked}


def _system_prompt_fingerprint(payload: dict[str, Any]) -> str | None:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None
    system_parts = [str(msg.get("content", "")) for msg in messages if isinstance(msg, dict) and msg.get("role") == "system"]
    if not system_parts:
        return None
    return _sha256("\n".join(system_parts))


def _tool_calls(response_body: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    choices = response_body.get("choices")
    if not isinstance(choices, list):
        return calls
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("tool_calls"), list):
            calls.extend(message["tool_calls"])
    return calls


def _response_ids(response_body: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider_response_id": response_body.get("id"),
        "provider_object": response_body.get("object"),
        "created": response_body.get("created"),
    }


def _token_data(response_body: dict[str, Any]) -> dict[str, Any]:
    logprobs: list[float] = []
    tokens: list[str] = []
    choices = response_body.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_logprobs = choice.get("logprobs")
            if not isinstance(choice_logprobs, dict):
                continue
            content = choice_logprobs.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("token"), str):
                    tokens.append(item["token"])
                if isinstance(item.get("logprob"), (int, float)):
                    logprobs.append(float(item["logprob"]))
    if tokens or logprobs:
        return {"token_data_status": "available", "tokens": tokens or None, "logprobs": logprobs or None}
    return {"token_data_status": "unavailable"}


class ATDPRecorder:
    def __init__(self, storage: ATDPStorage, settings: Settings):
        self.storage = storage
        self.settings = settings

    def resolve_session_id(self, payload: dict[str, Any] | None, headers: dict[str, str]) -> str:
        header_value = headers.get("x-atdp-session-id") or headers.get("x-session-id")
        if header_value:
            return header_value
        if payload:
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                for key in ("atdp_session_id", "session_id", "thread_id", "conversation_id"):
                    if metadata.get(key):
                        return str(metadata[key])
        return str(uuid4())

    def governance_from_payload(self, payload: dict[str, Any] | None) -> GovernanceMetadata:
        metadata = payload.get("metadata") if payload else None
        governance = metadata.get("governance") if isinstance(metadata, dict) else None
        if isinstance(governance, dict):
            data = {"tenant_id": self.settings.default_tenant_id, **governance}
            return GovernanceMetadata.model_validate(data)
        tenant_id = self.settings.default_tenant_id
        if isinstance(metadata, dict) and metadata.get("tenant_id"):
            tenant_id = str(metadata["tenant_id"])
        return GovernanceMetadata(tenant_id=tenant_id)

    def record(self, event: ATDPEventCreate) -> ATDPEvent:
        return self.storage.append_event(self._redact_event(event))

    def _redact_event(self, event: ATDPEventCreate) -> ATDPEventCreate:
        if self.settings.redaction_mode == "off":
            return event
        data = redact_mapping(event.model_dump(mode="json"), mode=self.settings.redaction_mode)
        governance = data.get("governance")
        if isinstance(governance, dict) and governance.get("redaction_status") == "unredacted":
            governance["redaction_status"] = "redacted"
        return ATDPEventCreate.model_validate(data)

    def record_session_start(self, session_id: str, metadata: dict[str, Any] | None = None) -> ATDPEvent:
        return self.record(
            ATDPEventCreate(
                session_id=session_id,
                type=EventType.SESSION_START.value,
                observation={"session_id": session_id},
                metadata=metadata or {},
                governance=GovernanceMetadata(tenant_id=self.settings.default_tenant_id),
                replay=ReplayMetadata(mode=ReplayMode.DETERMINISTIC, reason="Session lifecycle marker."),
            )
        )

    def record_session_end(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        reward: RewardSignal | None = None,
    ) -> ATDPEvent:
        self.storage.mark_session_ended(session_id, metadata)
        return self.record(
            ATDPEventCreate(
                session_id=session_id,
                type=EventType.SESSION_END.value,
                outcome={"ended": True},
                reward=reward,
                metadata=metadata or {},
                governance=GovernanceMetadata(tenant_id=self.settings.default_tenant_id),
                replay=ReplayMetadata(mode=ReplayMode.DETERMINISTIC, reason="Session lifecycle marker."),
            )
        )

    def record_llm_request(
        self,
        session_id: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        trace_id: str | None,
        span_id: str | None,
    ) -> ATDPEvent:
        tools = payload.get("tools") or payload.get("functions") or []
        return self.record(
            ATDPEventCreate(
                session_id=session_id,
                type=EventType.LLM_REQUEST.value,
                trace_id=trace_id,
                span_id=span_id,
                observation={
                    "messages": payload.get("messages", []),
                    "headers": _safe_headers(headers),
                },
                decision_context={
                    "model": payload.get("model"),
                    "parameters": _decode_parameters(payload),
                    "prompt_template_fingerprint": payload.get("metadata", {}).get("prompt_template_fingerprint")
                    if isinstance(payload.get("metadata"), dict)
                    else None,
                    "system_prompt_fingerprint": _system_prompt_fingerprint(payload),
                    "exposed_tools": tools,
                },
                action={
                    "operation": "chat.completions",
                    "provider": self.settings.provider,
                    "upstream_base_url": self.settings.upstream_base_url,
                    "raw_provider_request": payload,
                },
                metadata={
                    "stream": bool(payload.get("stream")),
                    "tool_count": len(tools) if isinstance(tools, list) else 0,
                },
                governance=self.governance_from_payload(payload),
                replay=ReplayMetadata(mode=ReplayMode.APPROXIMATE, reason="LLM sampling is approximate unless all provider state is pinned."),
            )
        )

    def record_llm_response(
        self,
        session_id: str,
        request_event: ATDPEvent,
        response_body: dict[str, Any],
        status_code: int,
        latency_seconds: float,
        trace_id: str | None,
        span_id: str | None,
    ) -> ATDPEvent:
        usage = response_body.get("usage") if isinstance(response_body, dict) else None
        calls = _tool_calls(response_body)
        response_ids = _response_ids(response_body) if isinstance(response_body, dict) else {}
        token_data = _token_data(response_body) if isinstance(response_body, dict) else {"token_data_status": "unavailable"}
        return self.record(
            ATDPEventCreate(
                session_id=session_id,
                type=EventType.LLM_RESPONSE.value if status_code < 400 else EventType.LLM_ERROR.value,
                parent_event_id=request_event.id,
                trace_id=trace_id,
                span_id=span_id,
                observation={"request_event_id": request_event.id},
                action={"operation": "provider.response", "provider": self.settings.provider},
                outcome={
                    "status_code": status_code,
                    "response": response_body,
                    "raw_provider_response": response_body,
                    "usage": usage,
                    "tool_calls": calls,
                    "token_data": token_data,
                },
                metadata={
                    "latency_seconds": latency_seconds,
                    "model": response_body.get("model") if isinstance(response_body, dict) else None,
                    "provider": self.settings.provider,
                    **response_ids,
                    "token_data": token_data,
                    "finish_reasons": [
                        choice.get("finish_reason")
                        for choice in response_body.get("choices", [])
                        if isinstance(choice, dict)
                    ]
                    if isinstance(response_body.get("choices"), list)
                    else [],
                },
                governance=request_event.governance,
                replay=ReplayMetadata(mode=ReplayMode.APPROXIMATE, reason="Provider response can be used as an approximate replay fixture."),
            )
        )

    def record_llm_exception(
        self,
        session_id: str,
        request_event: ATDPEvent | None,
        exc: BaseException,
        latency_seconds: float,
        trace_id: str | None,
        span_id: str | None,
    ) -> ATDPEvent:
        return self.record(
            ATDPEventCreate(
                session_id=session_id,
                type=EventType.LLM_ERROR.value,
                parent_event_id=request_event.id if request_event else None,
                trace_id=trace_id,
                span_id=span_id,
                observation={"request_event_id": request_event.id if request_event else None},
                outcome={
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                },
                metadata={"latency_seconds": latency_seconds},
                governance=request_event.governance if request_event else GovernanceMetadata(tenant_id=self.settings.default_tenant_id),
                replay=ReplayMetadata(mode=ReplayMode.NON_REPLAYABLE, reason="Request failed before a provider response was available."),
            )
        )

    def reward_from_update(self, update: RewardUpdateIn) -> RewardSignal:
        return RewardSignal(
            value=update.value,
            label=update.label,
            critique=redact_value(update.critique, mode=self.settings.redaction_mode),
            source=update.source,
            evaluator_version=update.evaluator_version,
            training_eligible=update.training_eligible,
            metadata=redact_mapping(update.metadata, mode=self.settings.redaction_mode),
        )

    @staticmethod
    def monotonic() -> float:
        return time.perf_counter()

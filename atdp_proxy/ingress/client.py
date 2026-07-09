from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import httpx


class ATDPClient:
    def __init__(self, base_url: str = "http://localhost:8080/v1", session_id: str | None = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.timeout = timeout

    def _headers(self, session_id: str | None = None) -> dict[str, str]:
        resolved = session_id or self.session_id
        return {"X-ATDP-Session-ID": resolved} if resolved else {}

    def capture_boundary(
        self,
        *,
        boundary: str,
        phase: str = "event",
        name: str,
        version: str | None = None,
        input: Any = None,
        output: Any = None,
        error: Any = None,
        session_id: str | None = None,
        parent_event_id: str | None = None,
        correlation_id: str | None = None,
        server_name: str | None = None,
        server_version: str | None = None,
        input_schema_version: str | None = None,
        side_effects_happened: bool | None = None,
        idempotency_key: str | None = None,
        resources: list[dict[str, Any]] | None = None,
        prompts: list[dict[str, Any]] | None = None,
        replay_of_session_id: str | None = None,
        replay_of_event_id: str | None = None,
        replay_passed: bool | None = None,
        decision_context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        governance: dict[str, Any] | None = None,
        replay: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "boundary": boundary,
            "phase": phase,
            "name": name,
            "version": version,
            "parent_event_id": parent_event_id,
            "input": input,
            "output": output,
            "error": error,
            "correlation_id": correlation_id,
            "server_name": server_name,
            "server_version": server_version,
            "input_schema_version": input_schema_version,
            "side_effects_happened": side_effects_happened,
            "idempotency_key": idempotency_key,
            "resources": resources or [],
            "prompts": prompts or [],
            "replay_of_session_id": replay_of_session_id,
            "replay_of_event_id": replay_of_event_id,
            "replay_passed": replay_passed,
            "decision_context": decision_context or {},
            "metadata": metadata or {},
            "governance": governance or {},
            "replay": replay or {},
        }
        resolved_session_id = session_id or self.session_id
        if resolved_session_id:
            payload["session_id"] = resolved_session_id
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/atdp/boundary-events",
                headers=self._headers(session_id),
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    @contextmanager
    def capture_tool(
        self,
        name: str,
        *,
        input: Any = None,
        version: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        start = self.capture_boundary(
            boundary="tool",
            phase="start",
            name=name,
            version=version,
            input=input,
            session_id=session_id,
            metadata=metadata,
        )
        try:
            yield start
        except Exception as exc:
            self.capture_boundary(
                boundary="tool",
                phase="end",
                name=name,
                version=version,
                input=input,
                error={"type": exc.__class__.__name__, "message": str(exc)},
                session_id=session_id,
                parent_event_id=start["id"],
                metadata=metadata,
            )
            raise

    def capture_retrieval(self, name: str, *, query: Any = None, results: Any = None, **kwargs: Any) -> dict[str, Any]:
        return self.capture_boundary(boundary="retrieval", phase="end", name=name, input=query, output=results, **kwargs)

    def capture_mcp_tool_call(self, name: str, *, arguments: Any = None, **kwargs: Any) -> dict[str, Any]:
        return self.capture_boundary(boundary="mcp", phase="start", name=name, input=arguments, **kwargs)

    def capture_mcp_tool_result(self, name: str, *, arguments: Any = None, result: Any = None, **kwargs: Any) -> dict[str, Any]:
        return self.capture_boundary(boundary="mcp", phase="end", name=name, input=arguments, output=result, **kwargs)

    def capture_memory_read(self, name: str, *, query: Any = None, results: Any = None, **kwargs: Any) -> dict[str, Any]:
        return self.capture_boundary(boundary="memory", phase="start", name=name, input=query, output=results, **kwargs)

    def capture_memory_write(self, name: str, *, input: Any = None, output: Any = None, **kwargs: Any) -> dict[str, Any]:
        return self.capture_boundary(boundary="memory", phase="end", name=name, input=input, output=output, **kwargs)

    def capture_file_action(self, name: str, *, input: Any = None, output: Any = None, **kwargs: Any) -> dict[str, Any]:
        return self.capture_boundary(boundary="file", phase="event", name=name, input=input, output=output, **kwargs)

    def capture_browser_action(self, name: str, *, input: Any = None, output: Any = None, **kwargs: Any) -> dict[str, Any]:
        return self.capture_boundary(boundary="browser", phase="event", name=name, input=input, output=output, **kwargs)

    def capture_human_feedback(self, name: str, *, input: Any = None, output: Any = None, **kwargs: Any) -> dict[str, Any]:
        return self.capture_boundary(boundary="human_feedback", phase="event", name=name, input=input, output=output, **kwargs)

    def capture_guardrail_decision(self, name: str, *, input: Any = None, output: Any = None, **kwargs: Any) -> dict[str, Any]:
        return self.capture_boundary(boundary="guardrail", phase="event", name=name, input=input, output=output, **kwargs)

    def capture_replay_result(self, name: str = "replay_result", *, output: Any = None, **kwargs: Any) -> dict[str, Any]:
        return self.capture_boundary(boundary="replay", phase="event", name=name, output=output, **kwargs)

    def capture_final_outcome(self, name: str = "final_outcome", *, output: Any = None, **kwargs: Any) -> dict[str, Any]:
        return self.capture_boundary(boundary="final_outcome", phase="event", name=name, output=output, **kwargs)

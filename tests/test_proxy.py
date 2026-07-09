from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from atdp_proxy.app import create_app
from atdp_proxy.config import Settings
from atdp_proxy.providers.base import ProviderResponse


class StaticProvider:
    name = "static"

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def chat_completions(
        self,
        payload: dict[str, Any],
        inbound_headers: dict[str, str],
        trace_headers: dict[str, str],
    ) -> ProviderResponse:
        self.requests.append(payload)
        return ProviderResponse(
            status_code=200,
            body={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "test response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    async def stream_chat_completions(
        self,
        payload: dict[str, Any],
        inbound_headers: dict[str, str],
        trace_headers: dict[str, str],
    ):
        self.requests.append(payload)
        yield b'data: {"choices":[{"delta":{"role":"assistant","content":"stream "},"index":0}]}\n\n'
        yield b'data: {"choices":[{"delta":{"content":"response"},"finish_reason":"stop","index":0}]}\n\n'
        yield b"data: [DONE]\n\n"

    async def models(self, inbound_headers: dict[str, str], trace_headers: dict[str, str]) -> ProviderResponse:
        return ProviderResponse(status_code=200, body={"object": "list", "data": [{"id": "test-model"}]})


class ErrorProvider(StaticProvider):
    async def chat_completions(
        self,
        payload: dict[str, Any],
        inbound_headers: dict[str, str],
        trace_headers: dict[str, str],
    ) -> ProviderResponse:
        raise RuntimeError("provider unavailable")


def make_client(tmp_path):
    settings = Settings(
        provider="openai-compatible",
        upstream_base_url="http://provider.invalid/v1",
        data_dir=str(tmp_path),
        otel_enabled=False,
    )
    provider = StaticProvider()
    app = create_app(settings=settings, provider=provider)
    return TestClient(app), provider


def test_chat_completion_records_llm_events(tmp_path):
    client, provider = make_client(tmp_path)

    response = client.post(
        "/v1/chat/completions",
        headers={"X-ATDP-Session-ID": "session-1", "Authorization": "Bearer secret"},
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
            "metadata": {
                "governance": {
                    "tenant_id": "tenant-1",
                    "redaction_status": "redacted",
                    "training_eligible": False,
                }
            },
        },
    )

    assert response.status_code == 200
    assert response.headers["x-atdp-session-id"] == "session-1"
    assert provider.requests[0]["model"] == "test-model"

    trajectory = client.get("/v1/atdp/sessions/session-1/trajectory").json()
    assert trajectory["event_count"] == 2
    assert [event["type"] for event in trajectory["events"]] == ["llm.request", "llm.response"]
    assert trajectory["events"][0]["observation"]["headers"].get("authorization") is None
    assert trajectory["events"][0]["schema_version"] == "atdp.proxy.v0.1"
    assert trajectory["events"][0]["trajectory_id"] == "session-1"
    assert trajectory["events"][0]["event_hash"]
    assert trajectory["events"][1]["previous_event_hash"] == trajectory["events"][0]["event_hash"]
    assert trajectory["events"][0]["action"]["raw_provider_request"]["messages"][0]["content"] == "hello"
    assert trajectory["events"][1]["outcome"]["usage"]["total_tokens"] == 5
    assert trajectory["events"][1]["outcome"]["raw_provider_response"]["id"] == "chatcmpl-test"
    assert trajectory["events"][1]["governance"]["tenant_id"] == "tenant-1"
    assert trajectory["events"][1]["governance"]["redaction_status"] == "redacted"
    assert trajectory["events"][1]["governance"]["training_eligible"] is False

    default_steps = client.get("/v1/atdp/datasets/steps?session_id=session-1").text.strip()
    assert default_steps == ""

    included_steps = client.get(
        "/v1/atdp/datasets/steps?session_id=session-1&include_ineligible=true"
    ).text.strip().splitlines()
    assert len(included_steps) == 1
    step = json.loads(included_steps[0])
    assert step["source_event_ids"] == [trajectory["events"][0]["id"], trajectory["events"][1]["id"]]
    assert step["metadata"]["token_data_status"] == "unavailable"


def test_explicit_tool_event_and_late_reward(tmp_path):
    client, _provider = make_client(tmp_path)

    tool_response = client.post(
        "/v1/atdp/events",
        json={
            "session_id": "session-2",
            "type": "tool.result",
            "decision_context": {"tool_name": "calculator", "tool_version": "1.0"},
            "action": {"name": "calculator", "arguments": {"expr": "2+2"}},
            "outcome": {"result": 4},
            "replay": {"mode": "deterministic", "reason": "Pure function."},
            "governance": {"tenant_id": "tenant-a", "training_eligible": True},
        },
    )

    assert tool_response.status_code == 200
    event_id = tool_response.json()["id"]

    reward_response = client.post(
        f"/v1/atdp/events/{event_id}/reward",
        json={"value": 1.0, "label": "accepted", "source": "pytest"},
    )

    assert reward_response.status_code == 200
    rewarded = reward_response.json()
    assert rewarded["reward"] is None
    assert rewarded["latest_reward"]["value"] == 1.0
    assert rewarded["latest_reward"]["source"] == "pytest"
    assert rewarded["reward_updates"][0]["update_hash"]

    trajectory = client.get("/v1/atdp/sessions/session-2/trajectory").json()
    assert [event["type"] for event in trajectory["events"]] == ["tool.result", "reward.updated"]
    assert trajectory["events"][0]["reward"] is None
    assert trajectory["events"][0]["latest_reward"]["label"] == "accepted"
    assert trajectory["events"][0]["event_hash"]

    session_export = client.get("/v1/atdp/export/jsonl?session_id=session-2").text.strip().splitlines()
    exported_types = [json.loads(line)["record_type"] for line in session_export]
    assert exported_types == ["event", "reward.update", "event"]

    step_export = client.get("/v1/atdp/datasets/export?format=atdp_step_jsonl&session_id=session-2").text
    step = json.loads(step_export.strip())
    assert step["reward"]["label"] == "accepted"
    assert step["action"]["arguments"]["expr"] == "2+2"

    areal_export = client.get("/v1/atdp/datasets/export?format=areal_minimal&session_id=session-2").text
    areal = json.loads(areal_export.strip())
    assert areal["reward"]["label"] == "accepted"
    assert areal["metadata"]["token_data_status"] == "unavailable"


def test_explicit_event_can_use_header_session_id(tmp_path):
    client, _provider = make_client(tmp_path)

    response = client.post(
        "/v1/atdp/events",
        headers={"X-ATDP-Session-ID": "session-from-header"},
        json={"type": "memory.read", "outcome": {"items": []}},
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "session-from-header"


def test_boundary_events_cover_capture_surfaces_and_evolution_signals(tmp_path):
    client, _provider = make_client(tmp_path)

    tool_start = client.post(
        "/v1/atdp/boundary-events",
        headers={"X-ATDP-Session-ID": "session-boundary"},
        json={
            "boundary": "tool",
            "phase": "start",
            "name": "calculator",
            "version": "1.0",
            "input": {"expr": "2+2"},
            "governance": {"tenant_id": "tenant-boundary", "training_eligible": True},
            "replay": {"mode": "deterministic", "reason": "Pure function."},
        },
    )
    assert tool_start.status_code == 200
    tool_end = client.post(
        "/v1/atdp/boundary-events",
        json={
            "session_id": "session-boundary",
            "boundary": "tool",
            "phase": "end",
            "name": "calculator",
            "version": "1.0",
            "parent_event_id": tool_start.json()["id"],
            "input": {"expr": "2+2"},
            "output": {"result": 4},
            "governance": {"tenant_id": "tenant-boundary", "training_eligible": True},
            "replay": {"mode": "deterministic", "reason": "Pure function."},
        },
    )
    assert tool_end.status_code == 200

    for payload in [
        {"boundary": "retrieval", "phase": "end", "name": "docs", "input": {"query": "atdp"}, "output": {"documents": []}},
        {"boundary": "memory", "phase": "start", "name": "short_term", "output": {"items": []}},
        {"boundary": "memory", "phase": "end", "name": "short_term", "input": {"item": "lesson"}},
        {"boundary": "file", "phase": "event", "name": "read", "input": {"path": "README.md"}},
        {"boundary": "browser", "phase": "event", "name": "click", "input": {"selector": "#ok"}},
        {"boundary": "human_feedback", "phase": "event", "name": "approval", "output": {"approved": True}},
        {"boundary": "final_outcome", "phase": "event", "name": "task_complete", "output": {"success": True}},
    ]:
        response = client.post(
            "/v1/atdp/boundary-events",
            headers={"X-ATDP-Session-ID": "session-boundary"},
            json=payload,
        )
        assert response.status_code == 200

    trajectory = client.get("/v1/atdp/sessions/session-boundary/trajectory").json()
    event_types = [event["type"] for event in trajectory["events"]]
    assert "tool.call" in event_types
    assert "tool.result" in event_types
    assert "retrieval.result" in event_types
    assert "memory.read" in event_types
    assert "memory.write" in event_types
    assert "file.action" in event_types
    assert "browser.action" in event_types
    assert "human.feedback" in event_types
    assert "session.end" in event_types

    signals = client.get("/v1/atdp/evolution/signals?session_id=session-boundary").json()
    assert signals["event_count"] == len(trajectory["events"])
    assert signals["step_count"] >= 1
    assert "needs_reward" in signals["recommendations"]

    step_lines = client.get("/v1/atdp/datasets/steps?session_id=session-boundary").text.strip().splitlines()
    paired_tool_step = next(
        json.loads(line)
        for line in step_lines
        if json.loads(line)["metadata"]["event_types"] == ["tool.call", "tool.result"]
    )
    assert paired_tool_step["outcome"]["output"]["result"] == 4
    assert paired_tool_step["source_event_ids"] == [tool_start.json()["id"], tool_end.json()["id"]]


def test_mcp_guardrail_and_replay_boundaries_are_learning_steps(tmp_path):
    client, _provider = make_client(tmp_path)

    mcp_call = client.post(
        "/v1/atdp/boundary-events",
        headers={"X-ATDP-Session-ID": "session-mcp"},
        json={
            "boundary": "mcp",
            "phase": "start",
            "name": "filesystem.read_file",
            "server_name": "filesystem",
            "server_version": "2026.7.0",
            "input_schema_version": "schema-v1",
            "correlation_id": "mcp-1",
            "input": {"path": "README.md"},
            "resources": [{"uri": "file://README.md"}],
            "replay": {"mode": "deterministic", "reason": "Input and result are captured."},
        },
    )
    assert mcp_call.status_code == 200

    mcp_result = client.post(
        "/v1/atdp/boundary-events",
        json={
            "session_id": "session-mcp",
            "boundary": "mcp",
            "phase": "end",
            "name": "filesystem.read_file",
            "parent_event_id": mcp_call.json()["id"],
            "server_name": "filesystem",
            "correlation_id": "mcp-1",
            "input": {"path": "README.md"},
            "output": {"content": "# ATDP Data Proxy"},
            "side_effects_happened": False,
            "replay": {"mode": "deterministic", "reason": "Input and result are captured."},
        },
    )
    assert mcp_result.status_code == 200

    guardrail = client.post(
        "/v1/atdp/boundary-events",
        headers={"X-ATDP-Session-ID": "session-mcp"},
        json={
            "boundary": "guardrail",
            "name": "pii_filter",
            "input": {"text": "hello"},
            "output": {"allowed": True},
        },
    )
    assert guardrail.status_code == 200

    replay = client.post(
        "/v1/atdp/boundary-events",
        headers={"X-ATDP-Session-ID": "session-mcp"},
        json={
            "boundary": "replay",
            "name": "lnl_replay",
            "replay_of_session_id": "failed-session",
            "replay_of_event_id": "failed-event",
            "replay_passed": True,
            "output": {"contains_expected_answer": True},
        },
    )
    assert replay.status_code == 200

    trajectory = client.get("/v1/atdp/sessions/session-mcp/trajectory").json()
    event_types = [event["type"] for event in trajectory["events"]]
    assert event_types == ["mcp.tool.call", "mcp.tool.result", "guardrail.decision", "replay.result"]
    assert trajectory["events"][0]["decision_context"]["server_name"] == "filesystem"
    assert trajectory["events"][1]["metadata"]["correlation_id"] == "mcp-1"
    assert trajectory["events"][3]["outcome"]["replay_passed"] is True

    step_lines = client.get("/v1/atdp/datasets/steps?session_id=session-mcp").text.strip().splitlines()
    paired_mcp_step = json.loads(step_lines[0])
    assert paired_mcp_step["metadata"]["event_types"] == ["mcp.tool.call", "mcp.tool.result"]
    assert paired_mcp_step["source_event_ids"] == [mcp_call.json()["id"], mcp_result.json()["id"]]


def test_lnl_feedback_generates_memory_write_replay_recommendation(tmp_path):
    client, _provider = make_client(tmp_path)

    chat = client.post(
        "/v1/chat/completions",
        headers={"X-ATDP-Session-ID": "session-lnl"},
        json={"model": "test-model", "messages": [{"role": "user", "content": "What is LNL?"}]},
    )
    assert chat.status_code == 200
    response_event_id = chat.headers["x-atdp-response-event-id"]

    reward_response = client.post(
        f"/v1/atdp/events/{response_event_id}/reward",
        json={
            "value": -1.0,
            "label": "wrong_answer",
            "source": "human",
            "critique": "Model did not know the private company glossary entry.",
        },
    )
    assert reward_response.status_code == 200

    feedback = client.post(
        "/v1/atdp/boundary-events",
        headers={"X-ATDP-Session-ID": "session-lnl"},
        json={
            "boundary": "human_feedback",
            "name": "user_correction",
            "parent_event_id": response_event_id,
            "output": {"text": "Wrong. LNL means Lorem Norway Limited, a company."},
        },
    )
    assert feedback.status_code == 200

    memory = client.post(
        "/v1/atdp/boundary-events",
        headers={"X-ATDP-Session-ID": "session-lnl"},
        json={
            "boundary": "memory",
            "phase": "end",
            "name": "company_glossary",
            "parent_event_id": feedback.json()["id"],
            "input": {"key": "LNL", "value": "Lorem Norway Limited, a company"},
            "output": {"approved": True},
            "replay": {"mode": "deterministic", "reason": "Memory key/value and approval are captured."},
        },
    )
    assert memory.status_code == 200

    replay = client.post(
        "/v1/atdp/boundary-events",
        headers={"X-ATDP-Session-ID": "session-lnl"},
        json={
            "boundary": "replay",
            "name": "lnl_replay",
            "replay_of_session_id": "session-lnl",
            "replay_of_event_id": response_event_id,
            "replay_passed": True,
            "output": {"answer_contains": "Lorem Norway Limited"},
            "replay": {"mode": "deterministic", "reason": "Replay evaluator result is captured."},
        },
    )
    assert replay.status_code == 200

    signals = client.get("/v1/atdp/evolution/signals?session_id=session-lnl").json()
    assert signals["memory_miss_patterns"] == {"LNL=Lorem Norway Limited, a company": 1}
    assert "missing_domain_knowledge" in signals["recommendations"]
    assert "memory_write" in signals["recommendations"]
    assert "replay_needed" in signals["recommendations"]


def test_dataset_step_ids_are_stable_and_reward_eligibility_is_honored(tmp_path):
    client, _provider = make_client(tmp_path)

    event_response = client.post(
        "/v1/atdp/events",
        json={
            "session_id": "session-stable-export",
            "type": "tool.result",
            "action": {"name": "lookup"},
            "outcome": {"result": "private"},
        },
    )
    event_id = event_response.json()["id"]
    reward = client.post(
        f"/v1/atdp/events/{event_id}/reward",
        json={"value": 1.0, "label": "debug_only", "source": "pytest", "training_eligible": False},
    )
    assert reward.status_code == 200

    first_export = client.get("/v1/atdp/datasets/steps?session_id=session-stable-export").text.strip()
    second_export = client.get("/v1/atdp/datasets/steps?session_id=session-stable-export").text.strip()
    assert first_export == second_export
    default_step = json.loads(first_export)
    assert default_step["reward"] is None
    assert default_step["reward_updates"] == []

    included = client.get(
        "/v1/atdp/datasets/steps?session_id=session-stable-export&include_ineligible=true"
    ).text.strip()
    included_step = json.loads(included)
    assert included_step["step_id"] == default_step["step_id"]
    assert included_step["reward"]["label"] == "debug_only"


def test_redaction_covers_bearer_tokens_cookies_and_secret_urls(tmp_path):
    client, _provider = make_client(tmp_path)

    response = client.post(
        "/v1/atdp/events",
        json={
            "session_id": "session-redaction",
            "type": "file.action",
            "observation": {
                "text": (
                    "Bearer abcdefghijklmnopqrstuvwxyz "
                    "Cookie: sessionid=super-secret; "
                    "https://example.test/path?api_key=super-secret&x=1"
                )
            },
        },
    )
    assert response.status_code == 200
    text = response.json()["observation"]["text"]
    assert "abcdefghijklmnopqrstuvwxyz" not in text
    assert "sessionid=super-secret" not in text
    assert "api_key=super-secret" not in text
    assert "[REDACTED_SECRET]" in text


def test_models_proxy(tmp_path):
    client, _provider = make_client(tmp_path)

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "test-model"


def test_streaming_is_buffered_and_recorded(tmp_path):
    client, _provider = make_client(tmp_path)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-ATDP-Session-ID": "session-stream"},
        json={"model": "test-model", "messages": [], "stream": True},
    ) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "stream " in body
    assert "response" in body
    trajectory = client.get("/v1/atdp/sessions/session-stream/trajectory").json()
    assert [event["type"] for event in trajectory["events"]] == ["llm.request", "llm.response"]
    assert trajectory["events"][1]["outcome"]["response"]["choices"][0]["message"]["content"] == "stream response"


def test_upstream_error_records_non_replayable_llm_error(tmp_path):
    settings = Settings(
        provider="openai-compatible",
        upstream_base_url="http://provider.invalid/v1",
        data_dir=str(tmp_path),
        otel_enabled=False,
    )
    client = TestClient(create_app(settings=settings, provider=ErrorProvider()))

    response = client.post(
        "/v1/chat/completions",
        headers={"X-ATDP-Session-ID": "session-error"},
        json={"model": "test-model", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 502
    trajectory = client.get("/v1/atdp/sessions/session-error/trajectory").json()
    assert [event["type"] for event in trajectory["events"]] == ["llm.request", "llm.error"]
    assert trajectory["events"][1]["replay"]["mode"] == "non_replayable"


def test_basic_redaction_runs_before_storage_and_export(tmp_path):
    client, _provider = make_client(tmp_path)

    response = client.post(
        "/v1/chat/completions",
        headers={"X-ATDP-Session-ID": "session-redact"},
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "email me at user@example.com api_key=secret123"}],
        },
    )

    assert response.status_code == 200
    trajectory = client.get("/v1/atdp/sessions/session-redact/trajectory").json()
    stored_content = trajectory["events"][0]["observation"]["messages"][0]["content"]
    assert "[REDACTED_EMAIL]" in stored_content
    assert "secret123" not in stored_content

    export = client.get("/v1/atdp/datasets/steps?session_id=session-redact").text
    assert "user@example.com" not in export
    assert "secret123" not in export

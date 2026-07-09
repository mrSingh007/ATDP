from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from atdp_proxy.schemas import ATDPEventCreate, GovernanceMetadata, ReplayMetadata, ReplayMode


class Boundary(str, Enum):
    MCP = "mcp"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    MEMORY = "memory"
    FILE = "file"
    BROWSER = "browser"
    HUMAN_FEEDBACK = "human_feedback"
    GUARDRAIL = "guardrail"
    REPLAY = "replay"
    FINAL_OUTCOME = "final_outcome"


class BoundaryPhase(str, Enum):
    START = "start"
    END = "end"
    EVENT = "event"


class BoundaryEventIn(BaseModel):
    session_id: str | None = None
    boundary: Boundary
    phase: BoundaryPhase = BoundaryPhase.EVENT
    name: str
    version: str | None = None
    parent_event_id: str | None = None
    input: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    output: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    error: dict[str, Any] | str | None = None
    correlation_id: str | None = None
    server_name: str | None = None
    server_version: str | None = None
    input_schema_version: str | None = None
    side_effects_happened: bool | None = None
    idempotency_key: str | None = None
    resources: list[dict[str, Any]] = Field(default_factory=list)
    prompts: list[dict[str, Any]] = Field(default_factory=list)
    replay_of_session_id: str | None = None
    replay_of_event_id: str | None = None
    replay_passed: bool | None = None
    decision_context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    governance: GovernanceMetadata = Field(default_factory=GovernanceMetadata)
    replay: ReplayMetadata = Field(default_factory=ReplayMetadata)


def _event_type(boundary: Boundary, phase: BoundaryPhase) -> str:
    if boundary == Boundary.MCP:
        return "mcp.tool.call" if phase == BoundaryPhase.START else "mcp.tool.result"
    if boundary == Boundary.TOOL:
        return "tool.call" if phase == BoundaryPhase.START else "tool.result"
    if boundary == Boundary.RETRIEVAL:
        return "retrieval.query" if phase == BoundaryPhase.START else "retrieval.result"
    if boundary == Boundary.MEMORY:
        if phase == BoundaryPhase.START:
            return "memory.read"
        if phase == BoundaryPhase.END:
            return "memory.write"
        return "memory.read"
    if boundary == Boundary.FILE:
        return "file.action"
    if boundary == Boundary.BROWSER:
        return "browser.action"
    if boundary == Boundary.HUMAN_FEEDBACK:
        return "human.feedback"
    if boundary == Boundary.GUARDRAIL:
        return "guardrail.decision"
    if boundary == Boundary.REPLAY:
        return "replay.result"
    if boundary == Boundary.FINAL_OUTCOME:
        return "session.end"
    raise ValueError(f"Unsupported boundary: {boundary}")


def boundary_to_event(body: BoundaryEventIn, session_id: str) -> ATDPEventCreate:
    replay = body.replay
    if (
        replay.mode == ReplayMode.UNKNOWN
        and body.side_effects_happened
        and body.boundary in {Boundary.MCP, Boundary.TOOL, Boundary.FILE, Boundary.BROWSER}
    ):
        replay = ReplayMetadata(
            mode=ReplayMode.NON_REPLAYABLE,
            reason="Boundary event had side effects and no replay fixture was supplied.",
            snapshot_refs=body.replay.snapshot_refs,
            side_effects=body.replay.side_effects,
        )
    elif replay.mode == ReplayMode.UNKNOWN and body.boundary in {Boundary.MCP, Boundary.TOOL, Boundary.RETRIEVAL, Boundary.MEMORY}:
        replay = ReplayMetadata(
            mode=ReplayMode.APPROXIMATE,
            reason="Boundary event captured with input/output; exact replay depends on external state snapshots.",
            snapshot_refs=body.replay.snapshot_refs,
            side_effects=body.replay.side_effects,
        )
    boundary_context = {
        "boundary": body.boundary.value,
        "phase": body.phase.value,
        "name": body.name,
        "version": body.version,
        "correlation_id": body.correlation_id,
        "server_name": body.server_name,
        "server_version": body.server_version,
        "input_schema_version": body.input_schema_version,
        "side_effects_happened": body.side_effects_happened,
        "idempotency_key": body.idempotency_key,
        "resources": body.resources,
        "prompts": body.prompts,
        "replay_of_session_id": body.replay_of_session_id,
        "replay_of_event_id": body.replay_of_event_id,
        "replay_passed": body.replay_passed,
    }
    decision_context = {
        **body.decision_context,
        **boundary_context,
    }
    action = {
        "boundary": body.boundary.value,
        "phase": body.phase.value,
        "name": body.name,
        "version": body.version,
        "input": body.input,
        "correlation_id": body.correlation_id,
        "server_name": body.server_name,
        "idempotency_key": body.idempotency_key,
    }
    outcome: dict[str, Any] = {
        "output": body.output,
        "error": body.error,
        "success": body.error is None,
        "replay_passed": body.replay_passed,
    }
    metadata = {
        **body.metadata,
        **{key: value for key, value in boundary_context.items() if value not in (None, [], {})},
    }
    return ATDPEventCreate(
        session_id=session_id,
        type=_event_type(body.boundary, body.phase),
        parent_event_id=body.parent_event_id,
        observation={"input": body.input} if body.input is not None else {},
        decision_context=decision_context,
        action=action,
        outcome=outcome,
        metadata=metadata,
        governance=body.governance,
        replay=replay,
    )

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

ATDP_SCHEMA_VERSION = "atdp.proxy.v0.1"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


class EventType(str, Enum):
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_ERROR = "llm.error"
    MCP_TOOL_CALL = "mcp.tool.call"
    MCP_TOOL_RESULT = "mcp.tool.result"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    RETRIEVAL_QUERY = "retrieval.query"
    RETRIEVAL_RESULT = "retrieval.result"
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    FILE_ACTION = "file.action"
    BROWSER_ACTION = "browser.action"
    HUMAN_FEEDBACK = "human.feedback"
    GUARDRAIL_DECISION = "guardrail.decision"
    REPLAY_RESULT = "replay.result"
    REWARD_UPDATED = "reward.updated"
    SESSION_START = "session.start"
    SESSION_END = "session.end"


class ReplayMode(str, Enum):
    DETERMINISTIC = "deterministic"
    APPROXIMATE = "approximate"
    NON_REPLAYABLE = "non_replayable"
    UNKNOWN = "unknown"


class GovernanceMetadata(BaseModel):
    tenant_id: str = "default"
    redaction_status: str = "unredacted"
    data_classification: str = "internal"
    retention_policy: str | None = None
    consent_basis: str | None = None
    human_review_status: str = "unreviewed"
    training_eligible: bool = True
    visibility: str = "restricted"
    policy_tags: list[str] = Field(default_factory=list)


class ReplayMetadata(BaseModel):
    mode: ReplayMode = ReplayMode.UNKNOWN
    reason: str | None = None
    snapshot_refs: dict[str, Any] = Field(default_factory=dict)
    side_effects: list[str] = Field(default_factory=list)


class RewardSignal(BaseModel):
    value: float | None = None
    label: str | None = None
    critique: str | None = None
    source: str = "unknown"
    evaluator_version: str | None = None
    training_eligible: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class RewardUpdateRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    event_id: str
    session_id: str
    reward: RewardSignal
    created_at: datetime = Field(default_factory=utc_now)
    update_hash: str | None = None


class ATDPEventBase(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str
    trajectory_id: str | None = None
    schema_version: str = ATDP_SCHEMA_VERSION
    type: str
    timestamp: datetime = Field(default_factory=utc_now)
    parent_event_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    observation: dict[str, Any] = Field(default_factory=dict)
    hidden_state: dict[str, Any] = Field(default_factory=dict)
    decision_context: dict[str, Any] = Field(default_factory=dict)
    action: dict[str, Any] = Field(default_factory=dict)
    outcome: dict[str, Any] = Field(default_factory=dict)
    reward: RewardSignal | None = None
    latest_reward: RewardSignal | None = None
    reward_updates: list[RewardUpdateRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    governance: GovernanceMetadata = Field(default_factory=GovernanceMetadata)
    replay: ReplayMetadata = Field(default_factory=ReplayMetadata)
    previous_event_hash: str | None = None
    event_hash: str | None = None


class ATDPEventCreate(ATDPEventBase):
    session_id: str | None = None
    id: str | None = None
    step: int | None = Field(default=None, ge=0)


class ATDPEvent(ATDPEventBase):
    id: str = Field(default_factory=new_id)
    step: int = Field(ge=0)


class RewardUpdateIn(BaseModel):
    value: float | None = None
    label: str | None = None
    critique: str | None = None
    source: str = "unknown"
    evaluator_version: str | None = None
    training_eligible: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionEndIn(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)
    reward: RewardSignal | None = None


class TrajectoryResponse(BaseModel):
    session_id: str
    event_count: int
    events: list[ATDPEvent]


class SessionSummary(BaseModel):
    session_id: str
    created_at: datetime
    ended_at: datetime | None = None
    event_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

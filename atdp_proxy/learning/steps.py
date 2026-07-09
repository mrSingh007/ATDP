from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from atdp_proxy.schemas import ATDPEvent, GovernanceMetadata, ReplayMetadata, RewardSignal, RewardUpdateRecord, new_id


class DatasetFilters(BaseModel):
    session_id: str | None = None
    tenant_id: str | None = None
    training_eligible: bool | None = True
    include_ineligible: bool = False
    event_type: str | None = None
    from_timestamp: datetime | None = None
    to_timestamp: datetime | None = None


class ATDPStep(BaseModel):
    step_id: str = Field(default_factory=new_id)
    session_id: str
    trajectory_id: str
    step_index: int
    source_event_ids: list[str]
    observation: dict[str, Any] = Field(default_factory=dict)
    hidden_state: dict[str, Any] = Field(default_factory=dict)
    decision_context: dict[str, Any] = Field(default_factory=dict)
    action: dict[str, Any] = Field(default_factory=dict)
    outcome: dict[str, Any] = Field(default_factory=dict)
    reward: RewardSignal | None = None
    reward_updates: list[RewardUpdateRecord] = Field(default_factory=list)
    governance: GovernanceMetadata = Field(default_factory=GovernanceMetadata)
    replay: ReplayMetadata = Field(default_factory=ReplayMetadata)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tokens: list[int] | None = None
    logprobs: list[float] | None = None
    model_version: str | None = None
    policy_version: str | None = None


def _stable_step_id(source_event_ids: list[str]) -> str:
    canonical = json.dumps(source_event_ids, sort_keys=True, separators=(",", ":"))
    return f"step_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:32]}"


def _passes_filters(event: ATDPEvent, filters: DatasetFilters) -> bool:
    if filters.session_id and event.session_id != filters.session_id:
        return False
    if filters.tenant_id and event.governance.tenant_id != filters.tenant_id:
        return False
    if filters.event_type and event.type != filters.event_type:
        return False
    if filters.from_timestamp and event.timestamp < filters.from_timestamp:
        return False
    if filters.to_timestamp and event.timestamp > filters.to_timestamp:
        return False
    if not filters.include_ineligible:
        if filters.training_eligible is not None and event.governance.training_eligible is not filters.training_eligible:
            return False
    return True


def _reward_allowed(reward: RewardSignal, filters: DatasetFilters) -> bool:
    if filters.include_ineligible or filters.training_eligible is None:
        return True
    if reward.training_eligible is None:
        return True
    return reward.training_eligible is filters.training_eligible


def _latest_reward(events: list[ATDPEvent], filters: DatasetFilters) -> RewardSignal | None:
    for event in reversed(events):
        if event.latest_reward is not None and _reward_allowed(event.latest_reward, filters):
            return event.latest_reward
        if event.reward is not None and _reward_allowed(event.reward, filters):
            return event.reward
    return None


def _reward_updates(events: list[ATDPEvent], filters: DatasetFilters) -> list[RewardUpdateRecord]:
    updates: list[RewardUpdateRecord] = []
    for event in events:
        updates.extend(update for update in event.reward_updates if _reward_allowed(update.reward, filters))
    return updates


def _token_metadata(*events: ATDPEvent) -> dict[str, Any]:
    for event in events:
        token_data = event.metadata.get("token_data") or event.outcome.get("token_data")
        if isinstance(token_data, dict):
            return token_data
    return {"token_data_status": "unavailable"}


def _llm_step(request: ATDPEvent, response: ATDPEvent, step_index: int, filters: DatasetFilters) -> ATDPStep:
    source_event_ids = [request.id, response.id]
    reward = _latest_reward([request, response], filters)
    updates = _reward_updates([request, response], filters)
    metadata = {
        **request.metadata,
        **response.metadata,
        **_token_metadata(request, response),
        "event_types": [request.type, response.type],
    }
    return ATDPStep(
        step_id=_stable_step_id(source_event_ids),
        session_id=request.session_id,
        trajectory_id=request.trajectory_id or request.session_id,
        step_index=step_index,
        source_event_ids=source_event_ids,
        observation=request.observation,
        hidden_state={**request.hidden_state, **response.hidden_state},
        decision_context=request.decision_context,
        action=request.action,
        outcome=response.outcome,
        reward=reward,
        reward_updates=updates,
        governance=response.governance,
        replay=response.replay,
        metadata=metadata,
        tokens=metadata.get("token_ids"),
        logprobs=metadata.get("logprobs"),
        model_version=metadata.get("model_version") or metadata.get("model"),
        policy_version=metadata.get("policy_version"),
    )


def _paired_step(start: ATDPEvent, end: ATDPEvent, step_index: int, filters: DatasetFilters) -> ATDPStep:
    source_event_ids = [start.id, end.id]
    reward = _latest_reward([start, end], filters)
    updates = _reward_updates([start, end], filters)
    metadata = {
        **start.metadata,
        **end.metadata,
        **_token_metadata(start, end),
        "event_types": [start.type, end.type],
    }
    return ATDPStep(
        step_id=_stable_step_id(source_event_ids),
        session_id=start.session_id,
        trajectory_id=start.trajectory_id or start.session_id,
        step_index=step_index,
        source_event_ids=source_event_ids,
        observation={**start.observation, "result_observation": end.observation},
        hidden_state={**start.hidden_state, **end.hidden_state},
        decision_context={**start.decision_context, **end.decision_context},
        action=start.action,
        outcome=end.outcome,
        reward=reward,
        reward_updates=updates,
        governance=end.governance,
        replay=end.replay,
        metadata=metadata,
        tokens=metadata.get("token_ids"),
        logprobs=metadata.get("logprobs"),
        model_version=metadata.get("model_version") or metadata.get("model"),
        policy_version=metadata.get("policy_version"),
    )


def _single_event_step(event: ATDPEvent, step_index: int, filters: DatasetFilters) -> ATDPStep:
    source_event_ids = [event.id]
    metadata = {**event.metadata, **_token_metadata(event), "event_types": [event.type]}
    return ATDPStep(
        step_id=_stable_step_id(source_event_ids),
        session_id=event.session_id,
        trajectory_id=event.trajectory_id or event.session_id,
        step_index=step_index,
        source_event_ids=source_event_ids,
        observation=event.observation,
        hidden_state=event.hidden_state,
        decision_context=event.decision_context,
        action=event.action,
        outcome=event.outcome,
        reward=_latest_reward([event], filters),
        reward_updates=_reward_updates([event], filters),
        governance=event.governance,
        replay=event.replay,
        metadata=metadata,
        tokens=metadata.get("token_ids"),
        logprobs=metadata.get("logprobs"),
        model_version=metadata.get("model_version") or metadata.get("model"),
        policy_version=metadata.get("policy_version"),
    )


def build_steps(events: list[ATDPEvent], filters: DatasetFilters | None = None) -> list[ATDPStep]:
    filters = filters or DatasetFilters(include_ineligible=True, training_eligible=None)
    events = [event for event in events if _passes_filters(event, filters)]
    by_id = {event.id: event for event in events}
    consumed: set[str] = set()
    steps: list[ATDPStep] = []

    for event in events:
        if event.id in consumed or event.type != "llm.request":
            continue
        response = next(
            (
                candidate
                for candidate in events
                if candidate.parent_event_id == event.id and candidate.type in {"llm.response", "llm.error"}
            ),
            None,
        )
        if response is None:
            continue
        steps.append(_llm_step(event, response, len(steps) + 1, filters))
        consumed.update({event.id, response.id})

    pair_types = {
        "mcp.tool.call": "mcp.tool.result",
        "tool.call": "tool.result",
        "retrieval.query": "retrieval.result",
    }
    for event in events:
        if event.id in consumed or event.type not in pair_types:
            continue
        end = next(
            (
                candidate
                for candidate in events
                if candidate.parent_event_id == event.id and candidate.type == pair_types[event.type]
            ),
            None,
        )
        if end is None:
            continue
        steps.append(_paired_step(event, end, len(steps) + 1, filters))
        consumed.update({event.id, end.id})

    for event in events:
        if event.id in consumed or event.type in {"reward.updated", "session.start"}:
            continue
        if event.parent_event_id and event.parent_event_id in by_id and by_id[event.parent_event_id].id in consumed:
            continue
        steps.append(_single_event_step(event, len(steps) + 1, filters))
        consumed.add(event.id)

    return steps

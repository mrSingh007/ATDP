from __future__ import annotations

from collections import Counter
import re
from typing import Any

from pydantic import BaseModel, Field

from atdp_proxy.learning.steps import ATDPStep
from atdp_proxy.schemas import ATDPEvent


class EvolutionSignals(BaseModel):
    event_count: int
    step_count: int
    rewarded_steps: int
    missing_reward_count: int
    reward_coverage: float
    error_count: int
    tool_failure_clusters: dict[str, int] = Field(default_factory=dict)
    retrieval_failure_clusters: dict[str, int] = Field(default_factory=dict)
    memory_miss_patterns: dict[str, int] = Field(default_factory=dict)
    ineligible_ratio: float
    replayability: dict[str, int] = Field(default_factory=dict)
    replay_gap_count: int = 0
    candidate_training_slices: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


CORRECTION_RE = re.compile(
    r"(?i)\b(?:wrong|incorrect|correction|actually)\b.*?\b([A-Z][A-Z0-9_-]{1,20})\s+(?:means|is|=)\s+([^.;\n]+)"
)


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_string_values(item))
        return values
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_string_values(item))
        return values
    return []


def _human_corrections(events: list[ATDPEvent]) -> Counter[str]:
    corrections: Counter[str] = Counter()
    for event in events:
        if event.type != "human.feedback":
            continue
        text = "\n".join(_string_values(event.observation) + _string_values(event.outcome))
        for match in CORRECTION_RE.finditer(text):
            key = match.group(1).strip()
            value = match.group(2).strip()
            corrections[f"{key}={value}"] += 1
    return corrections


def compute_evolution_signals(events: list[ATDPEvent], steps: list[ATDPStep]) -> EvolutionSignals:
    rewarded = sum(1 for step in steps if step.reward is not None)
    missing = max(len(steps) - rewarded, 0)
    error_count = sum(1 for event in events if event.type.endswith(".error") or event.outcome.get("error"))
    ineligible = sum(1 for event in events if not event.governance.training_eligible)
    replayability = Counter(event.replay.mode.value for event in events)
    tool_failures: Counter[str] = Counter()
    retrieval_failures: Counter[str] = Counter()
    for event in events:
        if event.type == "tool.result" and event.outcome.get("error"):
            tool_failures[str(event.decision_context.get("name") or event.action.get("name") or "unknown")] += 1
        if event.type == "retrieval.result" and (event.outcome.get("error") or event.outcome.get("output") in (None, [], {})):
            retrieval_failures[str(event.decision_context.get("name") or event.action.get("name") or "unknown")] += 1

    memory_misses = _human_corrections(events)
    replay_gap_count = replayability.get("unknown", 0) + replayability.get("non_replayable", 0)
    candidate_training_slices = [
        {
            "step_id": step.step_id,
            "session_id": step.session_id,
            "source_event_ids": step.source_event_ids,
            "reward_label": step.reward.label if step.reward else None,
        }
        for step in steps
        if step.reward is not None and step.governance.training_eligible
    ]

    recommendations: list[str] = []
    if missing:
        recommendations.append("needs_reward")
    if ineligible:
        recommendations.append("needs_redaction_review")
    if tool_failures:
        recommendations.append("tool_schema_review")
    if retrieval_failures:
        recommendations.append("retrieval_review")
    if memory_misses:
        recommendations.extend(["missing_domain_knowledge", "memory_write", "replay_needed"])
    if replay_gap_count:
        recommendations.append("replay_gap")
    if candidate_training_slices:
        recommendations.append("candidate_training_slice")
    if not recommendations:
        recommendations.append("no_op")

    return EvolutionSignals(
        event_count=len(events),
        step_count=len(steps),
        rewarded_steps=rewarded,
        missing_reward_count=missing,
        reward_coverage=(rewarded / len(steps)) if steps else 0.0,
        error_count=error_count,
        tool_failure_clusters=dict(tool_failures),
        retrieval_failure_clusters=dict(retrieval_failures),
        memory_miss_patterns=dict(memory_misses),
        ineligible_ratio=(ineligible / len(events)) if events else 0.0,
        replayability=dict(replayability),
        replay_gap_count=replay_gap_count,
        candidate_training_slices=candidate_training_slices,
        recommendations=recommendations,
    )

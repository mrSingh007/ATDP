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
STRUCTURED_CORRECTION_RE = re.compile(
    r"(?i)^\s*([A-Z][A-Z0-9_-]{1,20})\s+(?:means|is|=)\s+(.+?)\s*$"
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


def _latest_reward_value(event: ATDPEvent) -> float | None:
    reward = event.latest_reward or event.reward
    return reward.value if reward else None


def _is_negative_parent_response(event: ATDPEvent | None) -> bool:
    if event is None or event.type != "llm.response":
        return False
    value = _latest_reward_value(event)
    return value is not None and value < 0


def _structured_correction(output: Any) -> tuple[str, str] | None:
    if not isinstance(output, dict):
        return None
    if output.get("feedback_type") != "knowledge_correction":
        return None
    if output.get("suggested_intervention") not in (None, "memory_write", "memory_write_proposal"):
        return None
    correction = output.get("correction")
    if not isinstance(correction, str):
        return None
    match = STRUCTURED_CORRECTION_RE.match(correction)
    if match is None:
        return None
    key = match.group(1).strip()
    value = match.group(2).strip().rstrip(".;")
    if not key or not value:
        return None
    return key, value


def _text_corrections(value: Any) -> list[tuple[str, str]]:
    corrections: list[tuple[str, str]] = []
    text = "\n".join(_string_values(value))
    for match in CORRECTION_RE.finditer(text):
        corrections.append((match.group(1).strip(), match.group(2).strip().rstrip(".;")))
    return corrections


def _human_corrections(events: list[ATDPEvent]) -> Counter[str]:
    corrections: Counter[str] = Counter()
    events_by_id = {event.id: event for event in events}
    for event in events:
        if event.type != "human.feedback":
            continue
        parent = events_by_id.get(event.parent_event_id or "")
        if event.parent_event_id and not _is_negative_parent_response(parent):
            continue

        structured = _structured_correction(event.outcome.get("output"))
        correction_pairs = [structured] if structured else _text_corrections([event.observation, event.outcome])
        for key, value in correction_pairs:
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
        recommendations.extend(
            ["missing_domain_knowledge", "memory_write", "memory_write_proposal", "replay_needed"]
        )
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

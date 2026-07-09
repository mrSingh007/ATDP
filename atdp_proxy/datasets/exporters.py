from __future__ import annotations

import json
from typing import Any

from atdp_proxy.learning.steps import ATDPStep


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records) + (
        "\n" if records else ""
    )


def export_step_jsonl(steps: list[ATDPStep]) -> str:
    return _jsonl([step.model_dump(mode="json") for step in steps])


def export_areal_minimal_jsonl(steps: list[ATDPStep]) -> str:
    records: list[dict[str, Any]] = []
    for step in steps:
        messages = step.observation.get("messages") or step.observation.get("input") or []
        output_message_list = None
        response = step.outcome.get("response") if isinstance(step.outcome, dict) else None
        if isinstance(response, dict):
            choices = response.get("choices")
            if isinstance(choices, list):
                output_message_list = [choice.get("message") for choice in choices if isinstance(choice, dict)]
        if output_message_list is None:
            output_message_list = [{"role": "assistant", "content": step.outcome.get("output") or step.outcome}]
        records.append(
            {
                "interaction_id": step.step_id,
                "parent_id": step.source_event_ids[0] if step.source_event_ids else None,
                "messages": messages,
                "output_message_list": output_message_list,
                "reward": step.reward.model_dump(mode="json") if step.reward else None,
                "token_ids": step.tokens,
                "logprobs": step.logprobs,
                "metadata": {
                    **step.metadata,
                    "session_id": step.session_id,
                    "trajectory_id": step.trajectory_id,
                    "source_event_ids": step.source_event_ids,
                    "token_data_status": step.metadata.get("token_data_status", "available")
                    if step.tokens is not None or step.logprobs is not None
                    else "unavailable",
                },
            }
        )
    return _jsonl(records)

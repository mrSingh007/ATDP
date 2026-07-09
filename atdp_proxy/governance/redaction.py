from __future__ import annotations

import re
from typing import Any

SECRET_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "password",
    "secret",
    "token",
    "x-api-key",
}

STRUCTURAL_STRING_KEYS = {
    "timestamp",
    "created_at",
    "ended_at",
    "event_hash",
    "previous_event_hash",
    "update_hash",
    "schema_version",
    "trace_id",
    "span_id",
    "id",
    "event_id",
    "session_id",
    "trajectory_id",
    "parent_event_id",
}

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d .()\-]{7,}\d)(?!\d)")
BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{10,}")
COOKIE_RE = re.compile(r"(?i)\b(cookie|set-cookie)\s*:\s*[^;\n\r]+(?:;[^,\n\r]*)?")
SECRET_URL_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|token)=)[^&#\s]+"
)
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)\s*[:=]\s*['\"]?[^'\"\s,;]+"
)


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return key.lower() in SECRET_KEYS or normalized in SECRET_KEYS


def redact_string(value: str) -> str:
    value = EMAIL_RE.sub("[REDACTED_EMAIL]", value)
    value = PHONE_RE.sub("[REDACTED_PHONE]", value)
    value = BEARER_TOKEN_RE.sub("Bearer [REDACTED_SECRET]", value)
    value = COOKIE_RE.sub(lambda match: f"{match.group(1)}: [REDACTED_SECRET]", value)
    value = SECRET_URL_RE.sub(lambda match: f"{match.group(1)}[REDACTED_SECRET]", value)
    value = SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=[REDACTED_SECRET]", value)
    return value


def redact_value(value: Any, mode: str = "basic", parent_key: str | None = None) -> Any:
    if mode == "off":
        return value
    if parent_key and parent_key in STRUCTURAL_STRING_KEYS:
        return value
    if parent_key and _is_secret_key(parent_key):
        return "[REDACTED_SECRET]"
    if isinstance(value, str):
        return redact_string(value)
    if isinstance(value, list):
        return [redact_value(item, mode=mode) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item, mode=mode) for item in value)
    if isinstance(value, dict):
        return redact_mapping(value, mode=mode)
    return value


def redact_mapping(value: dict[str, Any], mode: str = "basic") -> dict[str, Any]:
    if mode == "off":
        return value
    return {key: redact_value(item, mode=mode, parent_key=key) for key, item in value.items()}

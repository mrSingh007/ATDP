from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    provider: str = "openai-compatible"
    upstream_base_url: str = "http://localhost:11434/v1"
    upstream_api_key: str | None = None
    forward_authorization: bool = False
    data_dir: str = "./data"
    sqlite_path: str | None = None
    jsonl_path: str | None = None
    request_timeout_seconds: float = 120.0
    default_tenant_id: str = "default"
    default_environment: str = "dev"
    redaction_mode: str = "basic"
    otel_enabled: bool = False
    otel_service_name: str = "atdp-data-proxy"
    otel_exporter_otlp_endpoint: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = os.getenv("ATDP_DATA_DIR", "./data")
        return cls(
            provider=os.getenv("ATDP_PROVIDER", "openai-compatible"),
            upstream_base_url=os.getenv("ATDP_UPSTREAM_BASE_URL", "http://localhost:11434/v1"),
            upstream_api_key=os.getenv("ATDP_UPSTREAM_API_KEY") or None,
            forward_authorization=_env_bool("ATDP_FORWARD_AUTHORIZATION", False),
            data_dir=data_dir,
            sqlite_path=os.getenv("ATDP_SQLITE_PATH") or None,
            jsonl_path=os.getenv("ATDP_JSONL_PATH") or None,
            request_timeout_seconds=float(os.getenv("ATDP_REQUEST_TIMEOUT_SECONDS", "120")),
            default_tenant_id=os.getenv("ATDP_DEFAULT_TENANT_ID", "default"),
            default_environment=os.getenv("ATDP_ENVIRONMENT", "dev"),
            redaction_mode=os.getenv("ATDP_REDACTION_MODE", "basic"),
            otel_enabled=_env_bool("ATDP_OTEL_ENABLED", False),
            otel_service_name=os.getenv("ATDP_OTEL_SERVICE_NAME", "atdp-data-proxy"),
            otel_exporter_otlp_endpoint=os.getenv("ATDP_OTEL_EXPORTER_OTLP_ENDPOINT") or None,
        )

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def resolved_sqlite_path(self) -> Path:
        if self.sqlite_path:
            return Path(self.sqlite_path)
        return self.data_path / "atdp.sqlite3"

    @property
    def resolved_jsonl_path(self) -> Path:
        if self.jsonl_path:
            return Path(self.jsonl_path)
        return self.data_path / "atdp_events.jsonl"

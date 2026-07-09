from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol


@dataclass(slots=True)
class ProviderResponse:
    status_code: int
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)


class ModelProvider(Protocol):
    name: str

    async def chat_completions(
        self,
        payload: dict[str, Any],
        inbound_headers: dict[str, str],
        trace_headers: dict[str, str],
    ) -> ProviderResponse:
        ...

    def stream_chat_completions(
        self,
        payload: dict[str, Any],
        inbound_headers: dict[str, str],
        trace_headers: dict[str, str],
    ) -> AsyncIterator[bytes]:
        ...

    async def models(self, inbound_headers: dict[str, str], trace_headers: dict[str, str]) -> ProviderResponse:
        ...

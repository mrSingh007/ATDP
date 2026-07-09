from __future__ import annotations

from typing import Any, AsyncIterator

import httpx

from atdp_proxy.config import Settings
from atdp_proxy.providers.base import ProviderResponse


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.upstream_base_url.rstrip("/")
        self.timeout = settings.request_timeout_seconds

    def _url(self, path: str) -> str:
        path = path if path.startswith("/") else f"/{path}"
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}{path}"
        return f"{self.base_url}/v1{path}"

    def _headers(self, inbound_headers: dict[str, str], trace_headers: dict[str, str]) -> dict[str, str]:
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        if self.settings.upstream_api_key:
            headers["authorization"] = f"Bearer {self.settings.upstream_api_key}"
        elif self.settings.forward_authorization and inbound_headers.get("authorization"):
            headers["authorization"] = inbound_headers["authorization"]
        headers.update(trace_headers)
        return headers

    async def chat_completions(
        self,
        payload: dict[str, Any],
        inbound_headers: dict[str, str],
        trace_headers: dict[str, str],
    ) -> ProviderResponse:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self._url("/chat/completions"),
                json=payload,
                headers=self._headers(inbound_headers, trace_headers),
            )
        return ProviderResponse(
            status_code=response.status_code,
            body=self._response_body(response),
            headers=self._response_headers(response),
        )

    async def stream_chat_completions(
        self,
        payload: dict[str, Any],
        inbound_headers: dict[str, str],
        trace_headers: dict[str, str],
    ) -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                self._url("/chat/completions"),
                json=payload,
                headers=self._headers(inbound_headers, trace_headers),
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk

    async def models(self, inbound_headers: dict[str, str], trace_headers: dict[str, str]) -> ProviderResponse:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                self._url("/models"),
                headers=self._headers(inbound_headers, trace_headers),
            )
        return ProviderResponse(
            status_code=response.status_code,
            body=self._response_body(response),
            headers=self._response_headers(response),
        )

    @staticmethod
    def _response_body(response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text}
        if isinstance(body, dict):
            return body
        return {"data": body}

    @staticmethod
    def _response_headers(response: httpx.Response) -> dict[str, str]:
        return {
            key: value
            for key, value in response.headers.items()
            if key.lower() in {"x-request-id", "openai-processing-ms"}
        }

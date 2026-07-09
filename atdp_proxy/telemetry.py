from __future__ import annotations

import json
import logging
from contextlib import AbstractContextManager
from typing import Any

from atdp_proxy.config import Settings

logger = logging.getLogger(__name__)


def _attribute_value(value: Any) -> str | bool | int | float:
    if isinstance(value, (str, bool, int, float)):
        return value
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


class NoopSpan(AbstractContextManager):
    trace_id: str | None = None
    span_id: str | None = None

    def __enter__(self) -> "NoopSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def record_exception(self, exc: BaseException) -> None:
        return None


class ManagedSpan(AbstractContextManager):
    def __init__(self, cm: AbstractContextManager, attributes: dict[str, Any]):
        self.cm = cm
        self.attributes = attributes
        self.span = None
        self.trace_id: str | None = None
        self.span_id: str | None = None

    def __enter__(self) -> "ManagedSpan":
        self.span = self.cm.__enter__()
        for key, value in self.attributes.items():
            self.set_attribute(key, value)
        context = self.span.get_span_context()
        if context and context.is_valid:
            self.trace_id = f"{context.trace_id:032x}"
            self.span_id = f"{context.span_id:016x}"
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None and self.span is not None:
            self.record_exception(exc)
        return bool(self.cm.__exit__(exc_type, exc, tb))

    def set_attribute(self, key: str, value: Any) -> None:
        if value is not None and self.span is not None:
            self.span.set_attribute(key, _attribute_value(value))

    def record_exception(self, exc: BaseException) -> None:
        if self.span is not None:
            self.span.record_exception(exc)


class ATDPTelemetry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = False
        self.setup_error: str | None = None
        self._trace = None
        self._propagate = None
        self._tracer = None

        if not settings.otel_enabled:
            return

        try:
            from opentelemetry import propagate, trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            resource = Resource.create(
                {
                    "service.name": settings.otel_service_name,
                    "deployment.environment": settings.default_environment,
                }
            )
            provider = TracerProvider(resource=resource)

            if settings.otel_exporter_otlp_endpoint:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))

            trace.set_tracer_provider(provider)
            self._trace = trace
            self._propagate = propagate
            self._tracer = trace.get_tracer(settings.otel_service_name)
            self.enabled = True
        except Exception as exc:
            self.setup_error = str(exc)
            logger.warning("OpenTelemetry setup failed; telemetry is disabled: %s", exc)
            self.enabled = False

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        inbound_headers: dict[str, str] | None = None,
    ) -> NoopSpan | ManagedSpan:
        if not self.enabled or self._tracer is None:
            return NoopSpan()

        context = None
        if inbound_headers and self._propagate is not None:
            context = self._propagate.extract(inbound_headers)
        cm = self._tracer.start_as_current_span(name, context=context)
        return ManagedSpan(cm, attributes or {})

    def inject_headers(self, headers: dict[str, str]) -> None:
        if self.enabled and self._propagate is not None:
            self._propagate.inject(headers)

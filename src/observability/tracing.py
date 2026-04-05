"""OpenTelemetry tracing setup + span helpers for agent nodes.

Enable with OTEL_ENABLED=true in .env.  Exporter is configurable:
  - OTEL_EXPORTER=console  (default, prints spans to stdout)
  - OTEL_EXPORTER=otlp     (sends to OTEL_ENDPOINT, default localhost:4317)

When disabled, all helpers are no-ops so the rest of the code is unaffected.
"""

from __future__ import annotations

import functools
import os
import uuid
from contextlib import contextmanager
from typing import Any, Generator

_ENABLED = os.getenv("OTEL_ENABLED", "").strip().lower() in ("1", "true", "yes")

_tracer: Any = None
_trace_id_var: str = ""


def _init_tracer() -> Any:
    """Lazily initialize the OTel tracer (only when OTEL_ENABLED)."""
    global _tracer
    if _tracer is not None:
        return _tracer

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        resource = Resource.create({"service.name": "bigdata-qna"})
        provider = TracerProvider(resource=resource)

        exporter_type = os.getenv("OTEL_EXPORTER", "console").strip().lower()
        if exporter_type == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                endpoint = os.getenv("OTEL_ENDPOINT", "http://localhost:4317")
                exporter = OTLPSpanExporter(endpoint=endpoint)
            except ImportError:
                exporter = ConsoleSpanExporter()
        else:
            exporter = ConsoleSpanExporter()

        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("bigdata_qna")
    except ImportError:
        _tracer = None
    return _tracer


def get_trace_id() -> str:
    """Return the current pipeline trace id (lightweight, always available)."""
    global _trace_id_var
    if not _trace_id_var:
        _trace_id_var = uuid.uuid4().hex[:16]
    return _trace_id_var


def new_trace_id() -> str:
    """Start a fresh trace id for a new pipeline invocation."""
    global _trace_id_var
    _trace_id_var = uuid.uuid4().hex[:16]
    return _trace_id_var


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Generator[Any, None, None]:
    """Context manager that creates an OTel span when enabled, otherwise a no-op."""
    if not _ENABLED:
        yield None
        return
    tracer = _init_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as s:
        if attributes:
            for k, v in attributes.items():
                s.set_attribute(k, _safe_attr(v))
        yield s


def traced(name: str | None = None):
    """Decorator that wraps a function in an OTel span."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            span_name = name or fn.__qualname__
            with span(span_name):
                return fn(*args, **kwargs)
        return wrapper
    return decorator


def _safe_attr(v: Any) -> Any:
    """Coerce value to an OTel-safe attribute type."""
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)[:500]

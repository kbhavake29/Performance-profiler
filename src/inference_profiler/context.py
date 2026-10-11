"""Request context tracking and W3C trace-context propagation helpers."""

from __future__ import annotations

import contextvars
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

if TYPE_CHECKING:
    from inference_profiler.instrumentation import RequestHandle

_PROPAGATOR = TraceContextTextMapPropagator()

_current_request: contextvars.ContextVar[RequestHandle | None] = contextvars.ContextVar(
    "inference_profiler_current_request", default=None
)


def current_request() -> RequestHandle | None:
    """Return the request scope active in this thread/task, if any.

    Lets deeply nested code add phases without threading the request object
    through every call::

        req = current_request()
        if req is not None:
            with req.phase("prefill"):
                ...
    """
    return _current_request.get()


def _set_current_request(handle: RequestHandle | None) -> contextvars.Token[RequestHandle | None]:
    return _current_request.set(handle)


def _reset_current_request(token: contextvars.Token[RequestHandle | None]) -> None:
    _current_request.reset(token)


def extract_context(carrier: Mapping[str, str]) -> otel_context.Context:
    """Extract a parent context from W3C ``traceparent``/``tracestate`` headers."""
    return _PROPAGATOR.extract(carrier=carrier)


def inject_context(
    carrier: MutableMapping[str, str], ctx: otel_context.Context | None = None
) -> MutableMapping[str, str]:
    """Write W3C trace headers for ``ctx`` (default: current context) into ``carrier``."""
    _PROPAGATOR.inject(carrier, context=ctx)
    return carrier


def traceparent_for(span: trace.Span) -> str | None:
    """Return the ``traceparent`` header value for ``span``, or ``None`` if it is invalid."""
    carrier: dict[str, str] = {}
    inject_context(carrier, trace.set_span_in_context(span))
    return carrier.get("traceparent")

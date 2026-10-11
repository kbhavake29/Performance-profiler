"""Core instrumentation: the :class:`Profiler`, request scopes and phase scopes.

Design rules enforced here:

* Instrumentation never swallows the application's exceptions; it records and re-raises.
* Internal instrumentation failures are logged and counted, not raised, unless
  ``strict=True``.
* Metric attributes are bounded (model, runtime version, status, phase, cache
  operation). Request IDs only ever appear on spans.
* Prompt text and generated text are never accepted by this API.
"""

from __future__ import annotations

import asyncio
import atexit
import functools
import inspect
import logging
import random
import re
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, MutableMapping, Sequence
from types import TracebackType
from typing import Any, ParamSpec, TypeVar, cast

from opentelemetry import context as otel_context
from opentelemetry import metrics as otel_metrics
from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricExporter,
    MetricExportResult,
    MetricReader,
    MetricsData,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.util.types import AttributeValue

from inference_profiler import metrics as m
from inference_profiler.config import ProfilerConfig
from inference_profiler.context import (
    _reset_current_request,
    _set_current_request,
    current_request,
    extract_context,
    inject_context,
    traceparent_for,
)
from inference_profiler.memory import MemorySampler
from inference_profiler.version import __version__

_log = logging.getLogger("inference_profiler")

P = ParamSpec("P")
R = TypeVar("R")

KNOWN_PHASES = frozenset({"queue", "prefill", "generation"})
_PHASE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_INSTRUMENTATION_SCOPE = "inference_profiler"


class RequestCancelled(Exception):
    """Raise inside a request scope to mark it cancelled rather than failed."""


# ============================================================ exporter wrappers


class _ExportStats:
    """Late-bound sink for export health metrics (instruments exist after providers)."""

    def __init__(self) -> None:
        self.instruments: m.Instruments | None = None
        self.failures = 0
        self.dropped = 0
        self._lock = threading.Lock()

    def record(self, signal: str, duration_s: float, ok: bool, items: int) -> None:
        if not ok:
            with self._lock:
                self.failures += 1
                self.dropped += items
        inst = self.instruments
        if inst is None:
            return
        attrs = {m.ATTR_SIGNAL: signal}
        inst.export_duration.record(duration_s, attrs)
        if not ok:
            inst.export_errors.add(1, attrs)
            inst.telemetry_dropped.add(items, {**attrs, m.ATTR_REASON: "export_failed"})


class InstrumentedSpanExporter(SpanExporter):
    """Wraps a span exporter to measure export latency and count failures.

    Exceptions from the delegate are converted to ``FAILURE`` so a broken
    collector can never propagate into the application.
    """

    def __init__(self, delegate: SpanExporter, stats: _ExportStats) -> None:
        self._delegate = delegate
        self._stats = stats

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        start = time.perf_counter()
        try:
            result = self._delegate.export(spans)
        except Exception:
            _log.warning("span export raised; dropping %d spans", len(spans), exc_info=True)
            result = SpanExportResult.FAILURE
        self._stats.record(
            "traces", time.perf_counter() - start, result is SpanExportResult.SUCCESS, len(spans)
        )
        return result

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._delegate.force_flush(timeout_millis)


class InstrumentedMetricExporter(MetricExporter):
    """Wraps a metric exporter, preserving its temporality/aggregation preferences."""

    def __init__(self, delegate: MetricExporter, stats: _ExportStats) -> None:
        super().__init__(
            preferred_temporality=delegate._preferred_temporality,
            preferred_aggregation=delegate._preferred_aggregation,
        )
        self._delegate = delegate
        self._stats = stats

    def export(
        self, metrics_data: MetricsData, timeout_millis: float = 10_000, **kwargs: Any
    ) -> MetricExportResult:
        start = time.perf_counter()
        try:
            result = self._delegate.export(metrics_data, timeout_millis=timeout_millis, **kwargs)
        except Exception:
            _log.warning("metric export raised; dropping batch", exc_info=True)
            result = MetricExportResult.FAILURE
        self._stats.record(
            "metrics", time.perf_counter() - start, result is MetricExportResult.SUCCESS, 1
        )
        return result

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        return self._delegate.force_flush(timeout_millis)

    def shutdown(self, timeout_millis: float = 30_000, **kwargs: Any) -> None:
        self._delegate.shutdown(timeout_millis=timeout_millis, **kwargs)


# ==================================================================== profiler


class Profiler:
    """Entry point of the SDK. Create one per process and share it.

    ``span_exporter`` / ``metric_reader`` are injection points for tests and
    custom pipelines. When a ``span_exporter`` is injected it is driven by a
    synchronous ``SimpleSpanProcessor`` unless ``batch_spans=True``.
    """

    def __init__(
        self,
        config: ProfilerConfig | None = None,
        *,
        span_exporter: SpanExporter | None = None,
        metric_reader: MetricReader | None = None,
        batch_spans: bool | None = None,
    ) -> None:
        self.config = config if config is not None else ProfilerConfig.from_env()
        self._stats = _ExportStats()
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False
        self._limiter = m.CardinalityLimiter(self.config.max_attribute_values)
        self._memory_sampler: MemorySampler | None = None
        self._instrumentation_error_logged = False
        self._tracer_provider: TracerProvider | None = None
        self._meter_provider: MeterProvider | None = None
        self.instruments: m.Instruments | None = None
        self._tracer: trace.Tracer = trace.NoOpTracer()

        if not self.config.enabled:
            _log.info("inference profiler disabled; requests will not be instrumented")
            return

        resource = Resource.create(
            {
                "service.name": self.config.service_name,
                "deployment.environment.name": self.config.deployment_environment,
                m.ATTR_RUNTIME_VERSION: self.config.runtime_version,
                "telemetry.distro.name": "inference-profiler",
                "telemetry.distro.version": __version__,
                **dict(self.config.resource_attributes),
            }
        )

        # ---- metrics first: the export-health instruments live on this provider.
        readers: list[MetricReader] = []
        if metric_reader is not None:
            readers.append(metric_reader)
        elif self.config.exporter != "none":
            readers.append(
                PeriodicExportingMetricReader(
                    InstrumentedMetricExporter(self._build_metric_exporter(), self._stats),
                    export_interval_millis=self.config.export_interval_ms,
                    export_timeout_millis=self.config.export_timeout_ms,
                )
            )
        self._meter_provider = MeterProvider(
            resource=resource, metric_readers=readers, shutdown_on_exit=False
        )
        meter = self._meter_provider.get_meter(_INSTRUMENTATION_SCOPE, __version__)
        self.instruments = m.Instruments.create(meter)
        self._stats.instruments = self.instruments
        meter.create_observable_gauge(
            m.PROFILER_ENABLED,
            callbacks=[lambda _opts: [otel_metrics.Observation(1)]],
            unit="",
            description="1 when the inference profiler SDK is enabled in this process.",
        )

        # ---- traces
        self._tracer_provider = TracerProvider(
            resource=resource,
            sampler=ParentBased(TraceIdRatioBased(self.config.sample_rate)),
            shutdown_on_exit=False,
        )
        processor = self._build_span_processor(span_exporter, batch_spans)
        if processor is not None:
            self._tracer_provider.add_span_processor(processor)
        self._tracer = self._tracer_provider.get_tracer(_INSTRUMENTATION_SCOPE, __version__)

        if self.config.set_global_providers:
            trace.set_tracer_provider(self._tracer_provider)
            otel_metrics.set_meter_provider(self._meter_provider)

        if self.config.memory_interval_ms > 0:
            meter.create_observable_gauge(
                m.PROCESS_MEMORY_USAGE,
                callbacks=[self._observe_rss],
                unit="By",
                description="Resident set size of the instrumented process (real measurement).",
            )
            self._memory_sampler = MemorySampler(self.config.memory_interval_ms)
            self._memory_sampler.start()

        atexit.register(self.shutdown)

    # ---------------------------------------------------------------- builders
    def _build_metric_exporter(self) -> MetricExporter:
        if self.config.exporter == "console":
            return ConsoleMetricExporter()
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        return OTLPMetricExporter(
            endpoint=self.config.metrics_endpoint,
            timeout=self.config.export_timeout_ms / 1000.0,
        )

    def _build_span_processor(
        self, injected: SpanExporter | None, batch: bool | None
    ) -> SpanProcessor | None:
        if injected is not None:
            exporter: SpanExporter = InstrumentedSpanExporter(injected, self._stats)
            if batch:
                return self._batch(exporter)
            return SimpleSpanProcessor(exporter)
        if self.config.exporter == "none":
            return None
        if self.config.exporter == "console":
            return SimpleSpanProcessor(ConsoleSpanExporter())
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        otlp = OTLPSpanExporter(
            endpoint=self.config.traces_endpoint,
            timeout=self.config.export_timeout_ms / 1000.0,
        )
        return self._batch(InstrumentedSpanExporter(otlp, self._stats))

    def _batch(self, exporter: SpanExporter) -> BatchSpanProcessor:
        return BatchSpanProcessor(
            exporter,
            max_queue_size=self.config.max_span_queue_size,
            max_export_batch_size=min(512, self.config.max_span_queue_size),
            schedule_delay_millis=min(self.config.export_interval_ms, 5_000),
            export_timeout_millis=self.config.export_timeout_ms,
        )

    def _observe_rss(self, _options: object) -> list[otel_metrics.Observation]:
        sampler = self._memory_sampler
        if sampler is None or sampler.samples == 0:
            return []
        return [otel_metrics.Observation(sampler.latest_bytes)]

    # ---------------------------------------------------------------- state
    @property
    def enabled(self) -> bool:
        return self.config.enabled and not self._is_shutdown

    @property
    def tracer(self) -> trace.Tracer:
        return self._tracer

    @property
    def export_failures(self) -> int:
        return self._stats.failures

    @property
    def dropped_items(self) -> int:
        return self._stats.dropped

    @property
    def memory_sampler(self) -> MemorySampler | None:
        return self._memory_sampler

    def status(self) -> dict[str, Any]:
        """Snapshot of SDK health, used by ``inference-profiler doctor``."""
        return {
            "enabled": self.enabled,
            "service_name": self.config.service_name,
            "otlp_endpoint": self.config.otlp_endpoint,
            "exporter": self.config.exporter,
            "sample_rate": self.config.sample_rate,
            "export_failures": self._stats.failures,
            "dropped_items": self._stats.dropped,
            "overflowed_attributes": sorted(self._limiter.overflowed),
            "shutdown": self._is_shutdown,
        }

    # ---------------------------------------------------------------- errors
    def _instrumentation_error(self, where: str, exc: BaseException) -> None:
        if self.config.strict:
            raise exc
        if not self._instrumentation_error_logged:
            _log.warning(
                "inference profiler error in %s (further errors logged at DEBUG)",
                where,
                exc_info=exc,
            )
            self._instrumentation_error_logged = True
        else:
            _log.debug("inference profiler error in %s", where, exc_info=exc)
        if self.instruments is not None:
            try:
                self.instruments.instrumentation_errors.add(1, {"where": where})
            except Exception:  # pragma: no cover - last line of defence
                pass

    def _attrs(self, **attrs: str) -> dict[str, str]:
        return self._limiter.attributes(attrs)

    # ---------------------------------------------------------------- requests
    def inference_request(
        self,
        *,
        model: str,
        request_id: str | None = None,
        prompt_tokens: int | None = None,
        runtime_version: str | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
        parent: Mapping[str, str] | otel_context.Context | None = None,
        kind: SpanKind = SpanKind.SERVER,
    ) -> RequestHandle:
        """Create a request scope. Use it as a (async) context manager or call ``start``/``end``.

        ``parent`` may be incoming HTTP/gRPC headers carrying W3C ``traceparent``
        or an OpenTelemetry ``Context``; otherwise the current context is used.
        """
        if not self.enabled:
            return RequestHandle()
        try:
            return InferenceRequest(
                self,
                model=model,
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                runtime_version=runtime_version or self.config.runtime_version,
                attributes=attributes,
                parent=parent,
                kind=kind,
            )
        except Exception as exc:
            self._instrumentation_error("inference_request", exc)
            return RequestHandle()

    start_request = inference_request

    def trace_request(
        self, *, model: str, runtime_version: str | None = None
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator wrapping a (sync or async) handler in a request scope.

        Inside the handler, ``current_request()`` returns the scope.
        """

        def decorator(fn: Callable[P, R]) -> Callable[P, R]:
            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                    async with self.inference_request(model=model, runtime_version=runtime_version):
                        return await cast(Callable[P, Awaitable[Any]], fn)(*args, **kwargs)

                return cast(Callable[P, R], async_wrapper)

            @functools.wraps(fn)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                with self.inference_request(model=model, runtime_version=runtime_version):
                    return fn(*args, **kwargs)

            return wrapper

        return decorator

    def phase(self, name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator recording a phase on the *current* request (no-op if there is none)."""

        def decorator(fn: Callable[P, R]) -> Callable[P, R]:
            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                    req = current_request()
                    if req is None:
                        return await cast(Callable[P, Awaitable[Any]], fn)(*args, **kwargs)
                    with req.phase(name):
                        return await cast(Callable[P, Awaitable[Any]], fn)(*args, **kwargs)

                return cast(Callable[P, R], async_wrapper)

            @functools.wraps(fn)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                req = current_request()
                if req is None:
                    return fn(*args, **kwargs)
                with req.phase(name):
                    return fn(*args, **kwargs)

            return wrapper

        return decorator

    # ---------------------------------------------------------------- gauges
    def set_cache_utilization(self, ratio: float, *, model: str = "unknown") -> None:
        """Report cache utilisation (0..1) measured by the integration."""
        if self.instruments is None:
            return
        try:
            self.instruments.cache_utilization.set(
                max(0.0, min(1.0, ratio)), self._attrs(**{m.ATTR_MODEL: model})
            )
        except Exception as exc:
            self._instrumentation_error("set_cache_utilization", exc)

    def set_simulated_memory(self, used_bytes: int, pressure: float, *, model: str) -> None:
        """Report SIMULATED memory figures. Labelled ``inference.simulated=true``."""
        if self.instruments is None:
            return
        attrs = self._attrs(**{m.ATTR_MODEL: model, m.ATTR_SIMULATED: "true"})
        self.instruments.simulated_memory_usage.set(used_bytes, attrs)
        self.instruments.simulated_memory_pressure.set(pressure, attrs)

    # ---------------------------------------------------------------- lifecycle
    def force_flush(self, timeout_ms: int | None = None) -> bool:
        if not self.config.enabled or self._is_shutdown:
            return True
        timeout = timeout_ms if timeout_ms is not None else self.config.export_timeout_ms
        ok = True
        if self._tracer_provider is not None:
            ok = self._tracer_provider.force_flush(timeout) and ok
        if self._meter_provider is not None:
            try:
                ok = self._meter_provider.force_flush(timeout) and ok
            except Exception as exc:
                self._instrumentation_error("force_flush", exc)
                ok = False
        return ok

    def shutdown(self, timeout_ms: int | None = None) -> bool:
        """Flush and stop exporters. Idempotent; bounded by ``shutdown_timeout_ms``.

        Returns ``False`` if the exporters did not finish within the timeout
        (the remaining work continues on a daemon thread and is abandoned at exit).
        """
        with self._shutdown_lock:
            if self._is_shutdown or not self.config.enabled:
                self._is_shutdown = True
                return True
            self._is_shutdown = True
        try:
            atexit.unregister(self.shutdown)
        except Exception:  # pragma: no cover
            pass
        if self._memory_sampler is not None:
            self._memory_sampler.stop()
        timeout = timeout_ms if timeout_ms is not None else self.config.shutdown_timeout_ms

        def _do() -> None:
            if self._tracer_provider is not None:
                self._tracer_provider.shutdown()
            if self._meter_provider is not None:
                self._meter_provider.shutdown(timeout_millis=max(timeout, 1))

        worker = threading.Thread(target=_do, name="inference-profiler-shutdown", daemon=True)
        worker.start()
        worker.join(timeout / 1000.0 if timeout > 0 else None)
        if worker.is_alive():
            _log.warning(
                "inference profiler shutdown exceeded %d ms; abandoning pending telemetry", timeout
            )
            return False
        return True

    def __enter__(self) -> Profiler:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


# ============================================================ request handles


class PhaseScope:
    """No-op phase scope; :class:`_Phase` is the recording implementation."""

    hit: bool | None = None

    def __enter__(self) -> PhaseScope:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    async def __aenter__(self) -> PhaseScope:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc, tb)

    def phase(self, name: str, **attributes: AttributeValue) -> PhaseScope:
        return PhaseScope()

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        return None


class RequestHandle:
    """Request scope API. This base class is the no-op used when profiling is
    disabled or failed to initialise, so call sites never need ``if`` checks."""

    status: str = m.STATUS_OK

    def __enter__(self) -> RequestHandle:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.end()

    async def __aenter__(self) -> RequestHandle:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc, tb)

    @property
    def is_recording(self) -> bool:
        return False

    @property
    def trace_id(self) -> str | None:
        return None

    @property
    def traceparent(self) -> str | None:
        return None

    def start(self) -> RequestHandle:
        return self

    def end(self) -> None:
        return None

    def phase(self, name: str, **attributes: AttributeValue) -> PhaseScope:
        return PhaseScope()

    def queue(self) -> PhaseScope:
        return self.phase("queue")

    def prefill(self) -> PhaseScope:
        return self.phase("prefill")

    def generation(self) -> PhaseScope:
        return self.phase("generation")

    def mark_first_token(self) -> None:
        return None

    def record_token(self) -> None:
        return None

    def record_generation(
        self,
        generated_tokens: int,
        *,
        time_to_first_token_ms: float | None = None,
        inter_token_latencies_ms: Sequence[float] | None = None,
        generation_duration_ms: float | None = None,
    ) -> None:
        return None

    def cache_lookup(self, cache: str = "kv") -> PhaseScope:
        return PhaseScope()

    def cache_insert(self, cache: str = "kv", entries: int = 1) -> PhaseScope:
        return PhaseScope()

    def cache_evict(self, cache: str = "kv", entries: int = 1) -> PhaseScope:
        return PhaseScope()

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        return None

    def cancel(self, reason: str | None = None) -> None:
        return None

    def fail(self, error: BaseException | str) -> None:
        return None

    def inject(self, carrier: MutableMapping[str, str]) -> MutableMapping[str, str]:
        return carrier


class _Phase(PhaseScope):
    """A timed child span of a request (or of another phase)."""

    def __init__(
        self,
        request: InferenceRequest,
        name: str,
        parent_span: trace.Span,
        attributes: Mapping[str, AttributeValue],
        *,
        span_name: str | None = None,
        cache_operation: str | None = None,
        entries: int = 1,
    ) -> None:
        self._req = request
        self.name = name
        self._parent_span = parent_span
        self._attributes = attributes
        self._span_name = span_name or f"inference.{name}"
        self._cache_operation = cache_operation
        self._entries = entries
        self._span: trace.Span = trace.INVALID_SPAN
        self._start = 0.0
        self._token: object | None = None
        self.hit = None
        self.duration_s: float | None = None

    def __enter__(self) -> _Phase:
        prof = self._req.profiler
        try:
            ctx = trace.set_span_in_context(self._parent_span)
            self._span = prof.tracer.start_span(
                self._span_name,
                context=ctx,
                kind=SpanKind.INTERNAL,
                attributes={m.ATTR_PHASE: self.name, **self._attributes},
            )
            self._token = otel_context.attach(trace.set_span_in_context(self._span))
            if self.name == "queue":
                self._req._enter_queue()
        except Exception as exc:
            prof._instrumentation_error("phase.enter", exc)
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        duration = time.perf_counter() - self._start
        self.duration_s = duration
        prof = self._req.profiler
        try:
            if self._token is not None:
                otel_context.detach(self._token)  # type: ignore[arg-type]
                self._token = None
            if exc is not None:
                if _is_cancellation(exc):
                    self._span.set_attribute(m.ATTR_STATUS, m.STATUS_CANCELLED)
                else:
                    self._span.record_exception(exc)
                    self._span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            if self._cache_operation is not None:
                self._finish_cache(duration)
            self._req._phase_finished(self.name, duration)
            self._span.end()
        except Exception as err:
            prof._instrumentation_error("phase.exit", err)

    def _finish_cache(self, duration: float) -> None:
        inst = self._req.profiler.instruments
        if inst is None:
            return
        attrs = self._req._metric_attrs()
        op = self._cache_operation
        inst.cache_operation_duration.record(
            duration, {**attrs, m.ATTR_CACHE_OPERATION: cast(str, op)}
        )
        if op == "lookup":
            if self.hit is None:
                return
            self._span.set_attribute("inference.cache.hit", self.hit)
            (inst.cache_hits if self.hit else inst.cache_misses).add(1, attrs)
            self._req._cache_counts["hits" if self.hit else "misses"] += 1
        elif op == "insert":
            inst.cache_insertions.add(self._entries, attrs)
            self._req._cache_counts["insertions"] += self._entries
        elif op == "evict":
            inst.cache_evictions.add(self._entries, attrs)
            self._req._cache_counts["evictions"] += self._entries

    def phase(self, name: str, **attributes: AttributeValue) -> PhaseScope:
        if not _PHASE_NAME.match(name):
            self._req.profiler._instrumentation_error(
                "phase", ValueError(f"invalid phase name {name!r}")
            )
            return PhaseScope()
        return _Phase(self._req, name, self._span, attributes)

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        self._span.set_attribute(key, value)


class InferenceRequest(RequestHandle):
    """Recording request scope. Create via :meth:`Profiler.inference_request`."""

    def __init__(
        self,
        profiler: Profiler,
        *,
        model: str,
        request_id: str | None,
        prompt_tokens: int | None,
        runtime_version: str,
        attributes: Mapping[str, AttributeValue] | None,
        parent: Mapping[str, str] | otel_context.Context | None,
        kind: SpanKind,
    ) -> None:
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if prompt_tokens is not None and prompt_tokens < 0:
            raise ValueError("prompt_tokens must be >= 0")
        self.profiler = profiler
        self.model = model
        self.request_id = request_id or uuid.uuid4().hex
        self.prompt_tokens = prompt_tokens
        self.runtime_version = runtime_version
        self._extra_attributes = dict(attributes or {})
        self._parent = parent
        self._kind = kind
        self._span: trace.Span = trace.INVALID_SPAN
        self._ctx_token: object | None = None
        self._req_token: Any = None
        self._lock = threading.Lock()
        self._started = False
        self._ended = False
        self._start_perf = 0.0
        self._start_epoch_ns = 0
        self._queued = False
        self._queue_s = 0.0
        self._generation_s: float | None = None
        self._first_token_perf: float | None = None
        self._last_token_perf: float | None = None
        self._token_count = 0
        self._token_spans_emitted = 0
        self._token_spans = False
        self._itl_samples = 0
        #: Inter-token gaps buffered during decoding; recorded once in end().
        self._itl_buffer: list[float] = []
        self._ttft_recorded = False
        self._cache_counts = {"hits": 0, "misses": 0, "insertions": 0, "evictions": 0}
        self._explicit_status: str | None = None
        self._error_type: str | None = None
        self.status = m.STATUS_OK
        self.duration_s: float | None = None
        self.generated_tokens: int | None = None
        self._metric_attrs_cache: dict[str, str] | None = None

    # ---------------------------------------------------------------- basics
    def _metric_attrs(self) -> dict[str, str]:
        if self._metric_attrs_cache is None:
            self._metric_attrs_cache = self.profiler._attrs(
                **{m.ATTR_MODEL: self.model, m.ATTR_RUNTIME_VERSION: self.runtime_version}
            )
        return self._metric_attrs_cache

    @property
    def is_recording(self) -> bool:
        return self._span.is_recording()

    @property
    def span(self) -> trace.Span:
        return self._span

    @property
    def trace_id(self) -> str | None:
        ctx = self._span.get_span_context()
        return format(ctx.trace_id, "032x") if ctx.is_valid else None

    @property
    def traceparent(self) -> str | None:
        return traceparent_for(self._span)

    def inject(self, carrier: MutableMapping[str, str]) -> MutableMapping[str, str]:
        """Write this request's ``traceparent`` into outgoing headers."""
        return inject_context(carrier, trace.set_span_in_context(self._span))

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> InferenceRequest:
        if self._started:
            return self
        self._started = True
        prof = self.profiler
        try:
            parent = self._parent
            if parent is None:
                parent_ctx = None
            elif isinstance(parent, otel_context.Context):
                parent_ctx = parent
            else:
                parent_ctx = extract_context(parent)
            self._start_epoch_ns = time.time_ns()
            attrs: dict[str, AttributeValue] = {
                m.ATTR_MODEL: self.model,
                m.ATTR_RUNTIME_VERSION: self.runtime_version,
                "inference.request.id": self.request_id,
                **self._extra_attributes,
            }
            if self.prompt_tokens is not None:
                attrs["gen_ai.usage.input_tokens"] = self.prompt_tokens
            self._span = prof.tracer.start_span(
                "inference.request",
                context=parent_ctx,
                kind=self._kind,
                attributes=attrs,
                start_time=self._start_epoch_ns,
            )
            self._ctx_token = otel_context.attach(trace.set_span_in_context(self._span))
            self._req_token = _set_current_request(self)
            cfg = prof.config
            self._token_spans = (
                cfg.token_spans
                and self._span.is_recording()
                and random.random() < cfg.token_span_sample_rate
            )
            inst = prof.instruments
            if inst is not None:
                attrs_m = self._metric_attrs()
                inst.requests_in_flight.add(1, attrs_m)
                inst.requests_active.add(1, attrs_m)
                if self.prompt_tokens:
                    inst.tokens.add(self.prompt_tokens, {**attrs_m, m.ATTR_TOKEN_TYPE: "input"})
        except Exception as exc:
            prof._instrumentation_error("request.start", exc)
        self._start_perf = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is not None:
            if _is_cancellation(exc):
                self.cancel(type(exc).__name__)
            else:
                self.fail(exc)
        self.end()

    def end(self) -> None:
        with self._lock:
            if self._ended or not self._started:
                return
            self._ended = True
        duration = time.perf_counter() - self._start_perf
        self.duration_s = duration
        prof = self.profiler
        try:
            if self._req_token is not None:
                try:
                    _reset_current_request(self._req_token)
                except ValueError:  # ended from a different context (e.g. another thread)
                    pass
            if self._ctx_token is not None:
                try:
                    otel_context.detach(self._ctx_token)  # type: ignore[arg-type]
                except Exception:
                    pass
            if self._queued:  # queue phase never closed
                self._leave_queue()
            self.status = self._explicit_status or m.STATUS_OK
            span = self._span
            span.set_attribute(m.ATTR_STATUS, self.status)
            span.set_attribute("inference.queue.duration_ms", round(self._queue_s * 1000.0, 3))
            if self.status == m.STATUS_ERROR:
                span.set_attribute("error.type", self._error_type or "error")
            inst = prof.instruments
            if inst is not None:
                attrs = self._metric_attrs()
                ctx = trace.set_span_in_context(span)
                inst.request_duration.record(
                    duration, {**attrs, m.ATTR_STATUS: self.status}, context=ctx
                )
                inst.requests.add(1, {**attrs, m.ATTR_STATUS: self.status})
                for gap in self._itl_buffer:
                    inst.inter_token_latency.record(gap, attrs)
                self._itl_buffer.clear()
                inst.requests_in_flight.add(-1, attrs)
                inst.requests_active.add(-1, attrs)
            span.end()
        except Exception as exc:
            prof._instrumentation_error("request.end", exc)

    # ---------------------------------------------------------------- status
    def cancel(self, reason: str | None = None) -> None:
        self._explicit_status = m.STATUS_CANCELLED
        if reason:
            self._span.set_attribute("inference.cancel.reason", reason[:128])

    def fail(self, error: BaseException | str) -> None:
        self._explicit_status = m.STATUS_ERROR
        if isinstance(error, BaseException):
            self._error_type = type(error).__name__
            self._span.record_exception(error)
            self._span.set_status(Status(StatusCode.ERROR, self._error_type))
        else:
            self._error_type = str(error)[:128]
            self._span.set_status(Status(StatusCode.ERROR, self._error_type))

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        self._span.set_attribute(key, value)

    # ---------------------------------------------------------------- phases
    def phase(self, name: str, **attributes: AttributeValue) -> PhaseScope:
        if not _PHASE_NAME.match(name):
            self.profiler._instrumentation_error(
                "phase", ValueError(f"invalid phase name {name!r}")
            )
            return PhaseScope()
        return _Phase(self, name, self._span, attributes)

    def _enter_queue(self) -> None:
        inst = self.profiler.instruments
        self._queued = True
        if inst is not None:
            attrs = self._metric_attrs()
            inst.queue_depth.add(1, attrs)
            inst.requests_active.add(-1, attrs)

    def _leave_queue(self) -> None:
        inst = self.profiler.instruments
        self._queued = False
        if inst is not None:
            attrs = self._metric_attrs()
            inst.queue_depth.add(-1, attrs)
            inst.requests_active.add(1, attrs)

    def _phase_finished(self, name: str, duration: float) -> None:
        inst = self.profiler.instruments
        if inst is None:
            return
        attrs = self._metric_attrs()
        ctx = trace.set_span_in_context(self._span)
        if name == "queue":
            if self._queued:
                self._leave_queue()
            self._queue_s += duration
            inst.queue_duration.record(duration, attrs, context=ctx)
        elif name == "prefill":
            inst.prefill_duration.record(duration, attrs, context=ctx)
        elif name == "generation":
            self._generation_s = (self._generation_s or 0.0) + duration
            inst.generation_duration.record(duration, attrs, context=ctx)
        elif name not in ("cache_lookup", "cache_insert", "cache_evict"):
            inst.phase_duration.record(
                duration,
                {**attrs, m.ATTR_PHASE: self.profiler._limiter.limit(m.ATTR_PHASE, name)},
                context=ctx,
            )

    def cache_lookup(self, cache: str = "kv") -> PhaseScope:
        """Time a cache lookup. Set ``scope.hit = True/False`` inside the block."""
        return _Phase(
            self,
            "cache_lookup",
            self._span,
            {"inference.cache.name": cache},
            span_name="inference.cache_lookup",
            cache_operation="lookup",
        )

    def cache_insert(self, cache: str = "kv", entries: int = 1) -> PhaseScope:
        return _Phase(
            self,
            "cache_insert",
            self._span,
            {"inference.cache.name": cache, "inference.cache.entries": entries},
            span_name="inference.cache_insert",
            cache_operation="insert",
            entries=entries,
        )

    def cache_evict(self, cache: str = "kv", entries: int = 1) -> PhaseScope:
        return _Phase(
            self,
            "cache_evict",
            self._span,
            {"inference.cache.name": cache, "inference.cache.entries": entries},
            span_name="inference.cache_evict",
            cache_operation="evict",
            entries=entries,
        )

    # ---------------------------------------------------------------- tokens
    def mark_first_token(self) -> None:
        """Call when the first output token is available. TTFT is measured from request start."""
        if self._first_token_perf is not None:
            return
        now = time.perf_counter()
        self._first_token_perf = now
        self._last_token_perf = now
        self._token_count = max(self._token_count, 1)
        self._record_ttft(now - self._start_perf)

    def record_token(self) -> None:
        """Call once per generated token for exact inter-token latency.

        The gap is appended to a per-request buffer (no OpenTelemetry call in the
        decode loop) and recorded into the histogram when the request ends.
        """
        if self._first_token_perf is None:
            self.mark_first_token()
            return
        now = time.perf_counter()
        prev = cast(float, self._last_token_perf)
        self._last_token_perf = now
        self._token_count += 1
        itl = now - prev
        self._itl_buffer.append(itl)
        self._itl_samples += 1
        if self._token_spans and self._token_spans_emitted < (
            self.profiler.config.max_token_spans_per_request
        ):
            self._token_spans_emitted += 1
            end_ns = time.time_ns()
            span = self.profiler.tracer.start_span(
                "inference.token",
                context=trace.set_span_in_context(self._span),
                start_time=end_ns - int(itl * 1e9),
                attributes={"inference.token.index": self._token_count - 1},
            )
            span.end(end_time=end_ns)

    def _record_ttft(self, ttft_s: float) -> None:
        if self._ttft_recorded:
            return
        self._ttft_recorded = True
        start_ns = self._start_epoch_ns
        end_ns = start_ns + int(ttft_s * 1e9)
        span = self.profiler.tracer.start_span(
            "inference.time_to_first_token",
            context=trace.set_span_in_context(self._span),
            start_time=start_ns,
            attributes={m.ATTR_PHASE: "time_to_first_token"},
        )
        span.end(end_time=end_ns)
        self._span.set_attribute("inference.time_to_first_token_ms", round(ttft_s * 1000.0, 3))
        inst = self.profiler.instruments
        if inst is not None:
            inst.time_to_first_token.record(
                ttft_s, self._metric_attrs(), context=trace.set_span_in_context(self._span)
            )

    def record_generation(
        self,
        generated_tokens: int,
        *,
        time_to_first_token_ms: float | None = None,
        inter_token_latencies_ms: Sequence[float] | None = None,
        generation_duration_ms: float | None = None,
    ) -> None:
        """Record generation results.

        ``time_to_first_token_ms`` is used only if :meth:`mark_first_token` was not
        called. ``inter_token_latencies_ms`` is used only if :meth:`record_token`
        was not called. ``generation_duration_ms`` overrides the measured
        ``generation`` phase duration for throughput.
        """
        prof = self.profiler
        try:
            if generated_tokens < 0:
                raise ValueError("generated_tokens must be >= 0")
            self.generated_tokens = generated_tokens
            span = self._span
            span.set_attribute("gen_ai.usage.output_tokens", generated_tokens)
            if time_to_first_token_ms is not None and not self._ttft_recorded:
                if time_to_first_token_ms < 0:
                    raise ValueError("time_to_first_token_ms must be >= 0")
                self._record_ttft(time_to_first_token_ms / 1000.0)
            inst = prof.instruments
            if inst is None:
                return
            attrs = self._metric_attrs()
            if generated_tokens:
                inst.tokens.add(generated_tokens, {**attrs, m.ATTR_TOKEN_TYPE: "output"})
            if inter_token_latencies_ms and self._itl_samples == 0:
                for value in inter_token_latencies_ms:
                    inst.inter_token_latency.record(value / 1000.0, attrs)
                self._itl_samples += len(inter_token_latencies_ms)
            gen_s = (
                generation_duration_ms / 1000.0
                if generation_duration_ms is not None
                else self._generation_s
            )
            if gen_s and gen_s > 0 and generated_tokens > 0:
                tps = generated_tokens / gen_s
                span.set_attribute("inference.generation.tokens_per_second", round(tps, 3))
                inst.generation_throughput.record(tps, attrs)
        except Exception as exc:
            prof._instrumentation_error("record_generation", exc)


def _is_cancellation(exc: BaseException) -> bool:
    return isinstance(
        exc, (asyncio.CancelledError, RequestCancelled, GeneratorExit, KeyboardInterrupt)
    )

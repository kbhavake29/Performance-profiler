"""Metric names, units, bucket layouts and the instrument set used by the profiler.

Names follow OpenTelemetry conventions (dot-separated, unit carried separately).
The collector's Prometheus exporter turns ``inference.request.duration`` with
unit ``s`` into ``inference_request_duration_seconds_*``; counters gain a
``_total`` suffix. See ``docs/metrics.md`` for the full mapping.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from opentelemetry.metrics import Counter, Histogram, Meter, UpDownCounter
from opentelemetry.metrics import _Gauge as Gauge

# ----------------------------------------------------------------------- names
REQUEST_DURATION: Final = "inference.request.duration"
QUEUE_DURATION: Final = "inference.queue.duration"
PREFILL_DURATION: Final = "inference.prefill.duration"
TIME_TO_FIRST_TOKEN: Final = "inference.time_to_first_token"
INTER_TOKEN_LATENCY: Final = "inference.inter_token.latency"
GENERATION_DURATION: Final = "inference.generation.duration"
PHASE_DURATION: Final = "inference.phase.duration"
GENERATION_THROUGHPUT: Final = "inference.generation.throughput"
REQUESTS: Final = "inference.requests"
TOKENS: Final = "inference.tokens"
REQUESTS_IN_FLIGHT: Final = "inference.requests.in_flight"
REQUESTS_ACTIVE: Final = "inference.requests.active"
QUEUE_DEPTH: Final = "inference.queue.depth"
CACHE_HITS: Final = "inference.cache.hits"
CACHE_MISSES: Final = "inference.cache.misses"
CACHE_INSERTIONS: Final = "inference.cache.insertions"
CACHE_EVICTIONS: Final = "inference.cache.evictions"
CACHE_OPERATION_DURATION: Final = "inference.cache.operation.duration"
CACHE_UTILIZATION: Final = "inference.cache.utilization"
PROCESS_MEMORY_USAGE: Final = "process.memory.usage"
SIMULATED_MEMORY_USAGE: Final = "inference.simulated.memory.usage"
SIMULATED_MEMORY_PRESSURE: Final = "inference.simulated.memory.pressure"
EXPORT_ERRORS: Final = "inference_profiler.export.errors"
EXPORT_DURATION: Final = "inference_profiler.export.duration"
TELEMETRY_DROPPED: Final = "inference_profiler.telemetry.dropped"
PROFILER_ENABLED: Final = "inference_profiler.enabled"
INSTRUMENTATION_ERRORS: Final = "inference_profiler.instrumentation.errors"

# ------------------------------------------------------------------ attributes
ATTR_MODEL: Final = "gen_ai.request.model"
ATTR_RUNTIME_VERSION: Final = "inference.runtime.version"
ATTR_STATUS: Final = "inference.status"
ATTR_PHASE: Final = "inference.phase"
ATTR_TOKEN_TYPE: Final = "gen_ai.token.type"
ATTR_CACHE_OPERATION: Final = "inference.cache.operation"
ATTR_SIGNAL: Final = "inference_profiler.signal"
ATTR_REASON: Final = "inference_profiler.reason"
ATTR_SIMULATED: Final = "inference.simulated"

STATUS_OK: Final = "ok"
STATUS_ERROR: Final = "error"
STATUS_CANCELLED: Final = "cancelled"

# Seconds. Covers sub-millisecond cache ops through minute-long generations.
LATENCY_BUCKETS_S: Final = (
    0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1,
    0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 30.0, 60.0,
)  # fmt: skip
INTER_TOKEN_BUCKETS_S: Final = (
    0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0,
)  # fmt: skip
THROUGHPUT_BUCKETS: Final = (
    1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0,
)  # fmt: skip

OTHER_VALUE: Final = "_other"


class CardinalityLimiter:
    """Caps the number of distinct values per attribute key.

    The first ``max_values`` distinct values seen for a key pass through;
    later values are replaced by ``_other``. Lookups of already-admitted values
    take no lock; only admitting a new value does.
    """

    def __init__(self, max_values: int) -> None:
        self._max = max_values
        self._seen: dict[str, set[str]] = {}
        self._lock = threading.Lock()
        self.overflowed: set[str] = set()

    def limit(self, key: str, value: str) -> str:
        seen = self._seen.get(key)
        if seen is not None and value in seen:
            return value
        with self._lock:
            seen = self._seen.setdefault(key, set())
            if value in seen:
                return value
            if len(seen) >= self._max:
                self.overflowed.add(key)
                return OTHER_VALUE
            seen.add(value)
            return value

    def attributes(self, attrs: Mapping[str, str]) -> dict[str, str]:
        return {k: self.limit(k, v) for k, v in attrs.items()}


@dataclass(slots=True)
class Instruments:
    """All synchronous instruments, created once per profiler."""

    request_duration: Histogram
    queue_duration: Histogram
    prefill_duration: Histogram
    time_to_first_token: Histogram
    inter_token_latency: Histogram
    generation_duration: Histogram
    phase_duration: Histogram
    generation_throughput: Histogram
    requests: Counter
    tokens: Counter
    requests_in_flight: UpDownCounter
    requests_active: UpDownCounter
    queue_depth: UpDownCounter
    cache_hits: Counter
    cache_misses: Counter
    cache_insertions: Counter
    cache_evictions: Counter
    cache_operation_duration: Histogram
    cache_utilization: Gauge
    simulated_memory_usage: Gauge
    simulated_memory_pressure: Gauge
    export_errors: Counter
    export_duration: Histogram
    telemetry_dropped: Counter
    instrumentation_errors: Counter

    @classmethod
    def create(cls, meter: Meter) -> Instruments:
        def hist(name: str, unit: str, desc: str, buckets: tuple[float, ...]) -> Histogram:
            return meter.create_histogram(
                name, unit=unit, description=desc, explicit_bucket_boundaries_advisory=buckets
            )

        lat = LATENCY_BUCKETS_S
        return cls(
            request_duration=hist(
                REQUEST_DURATION,
                "s",
                "End-to-end inference request latency, including queueing.",
                lat,
            ),
            queue_duration=hist(
                QUEUE_DURATION, "s", "Time a request waited before execution started.", lat
            ),
            prefill_duration=hist(
                PREFILL_DURATION, "s", "Prompt processing (prefill) duration.", lat
            ),
            time_to_first_token=hist(
                TIME_TO_FIRST_TOKEN,
                "s",
                "Time from request arrival to the first generated token.",
                lat,
            ),
            inter_token_latency=hist(
                INTER_TOKEN_LATENCY,
                "s",
                "Delay between consecutive generated tokens.",
                INTER_TOKEN_BUCKETS_S,
            ),
            generation_duration=hist(
                GENERATION_DURATION, "s", "Token generation (decode) duration.", lat
            ),
            phase_duration=hist(PHASE_DURATION, "s", "Duration of custom request phases.", lat),
            generation_throughput=hist(
                GENERATION_THROUGHPUT,
                "{token}/s",
                "Per-request decode throughput.",
                THROUGHPUT_BUCKETS,
            ),
            requests=meter.create_counter(
                REQUESTS, unit="{request}", description="Finished requests by status."
            ),
            tokens=meter.create_counter(
                TOKENS, unit="{token}", description="Prompt and generated tokens processed."
            ),
            requests_in_flight=meter.create_up_down_counter(
                REQUESTS_IN_FLIGHT,
                unit="{request}",
                description="Admitted requests that have not finished (queued + executing).",
            ),
            requests_active=meter.create_up_down_counter(
                REQUESTS_ACTIVE,
                unit="{request}",
                description="Requests currently executing (excludes queued).",
            ),
            queue_depth=meter.create_up_down_counter(
                QUEUE_DEPTH,
                unit="{request}",
                description="Requests currently waiting in the queue phase.",
            ),
            cache_hits=meter.create_counter(
                CACHE_HITS, unit="{lookup}", description="Cache lookups that found an entry."
            ),
            cache_misses=meter.create_counter(
                CACHE_MISSES, unit="{lookup}", description="Cache lookups that missed."
            ),
            cache_insertions=meter.create_counter(
                CACHE_INSERTIONS, unit="{entry}", description="Entries inserted into the cache."
            ),
            cache_evictions=meter.create_counter(
                CACHE_EVICTIONS, unit="{entry}", description="Entries evicted from the cache."
            ),
            cache_operation_duration=hist(
                CACHE_OPERATION_DURATION, "s", "Cache lookup/insert/evict duration.", lat
            ),
            cache_utilization=meter.create_gauge(
                CACHE_UTILIZATION,
                unit="1",
                description="Fraction of cache capacity in use, as reported by the integration.",
            ),
            simulated_memory_usage=meter.create_gauge(
                SIMULATED_MEMORY_USAGE,
                unit="By",
                description="SIMULATED KV-cache memory from the workload model. Not a real measurement.",
            ),
            simulated_memory_pressure=meter.create_gauge(
                SIMULATED_MEMORY_PRESSURE,
                unit="1",
                description="SIMULATED memory usage / capacity from the workload model.",
            ),
            export_errors=meter.create_counter(
                EXPORT_ERRORS, unit="{error}", description="Failed telemetry export attempts."
            ),
            export_duration=hist(EXPORT_DURATION, "s", "Duration of telemetry export calls.", lat),
            telemetry_dropped=meter.create_counter(
                TELEMETRY_DROPPED,
                unit="{item}",
                description="Spans or metric batches dropped because export failed.",
            ),
            instrumentation_errors=meter.create_counter(
                INSTRUMENTATION_ERRORS,
                unit="{error}",
                description="Internal profiler errors swallowed in non-strict mode.",
            ),
        )

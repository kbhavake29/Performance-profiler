from __future__ import annotations

import asyncio
import threading
import time

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

from inference_profiler import Profiler, ProfilerConfig, RequestCancelled, current_request
from inference_profiler import metrics as m
from inference_profiler.instrumentation import InstrumentedSpanExporter, _ExportStats

from .conftest import Harness, make_harness


def test_request_lifecycle_creates_nested_spans(harness: Harness) -> None:
    with harness.profiler.inference_request(
        model="demo-model", request_id="req-1", prompt_tokens=128
    ) as req:
        with req.queue():
            pass
        with req.prefill():
            time.sleep(0.002)
        with req.generation():
            for _ in range(4):
                req.record_token()
        req.record_generation(generated_tokens=4)

    root = harness.span("inference.request")
    names = {s.name for s in harness.finished()}
    assert {
        "inference.queue",
        "inference.prefill",
        "inference.generation",
        "inference.time_to_first_token",
    } <= names
    for child in harness.finished():
        if child.name != "inference.request":
            assert child.parent is not None
            assert child.parent.span_id == root.context.span_id
            assert child.context.trace_id == root.context.trace_id
    assert root.attributes["gen_ai.request.model"] == "demo-model"
    assert root.attributes["inference.request.id"] == "req-1"
    assert root.attributes["gen_ai.usage.input_tokens"] == 128
    assert root.attributes["gen_ai.usage.output_tokens"] == 4
    assert root.attributes["inference.runtime.version"] == "test-1.0"
    assert root.attributes["inference.status"] == "ok"
    assert root.status.status_code == StatusCode.UNSET
    assert req.duration_s is not None and req.duration_s >= 0.002


def test_phase_nesting_inside_phase(harness: Harness) -> None:
    with harness.profiler.inference_request(model="m") as req, req.generation() as gen:
        with gen.phase("sampling"):
            pass
    gen_span = harness.span("inference.generation")
    sampling = harness.span("inference.sampling")
    assert sampling.parent.span_id == gen_span.context.span_id


def test_metrics_recorded_with_bounded_attributes(harness: Harness) -> None:
    with harness.profiler.inference_request(
        model="demo", request_id="unique-123", prompt_tokens=10
    ) as req:
        with req.queue():
            pass
        with req.prefill():
            pass
        with req.generation():
            req.record_token()
            req.record_token()
            req.record_token()
        req.record_generation(3)

    points = harness.metrics()
    for name in (
        m.REQUEST_DURATION,
        m.QUEUE_DURATION,
        m.PREFILL_DURATION,
        m.GENERATION_DURATION,
        m.TIME_TO_FIRST_TOKEN,
        m.INTER_TOKEN_LATENCY,
        m.GENERATION_THROUGHPUT,
    ):
        assert name in points, name
    assert harness.total(m.REQUESTS, **{m.ATTR_STATUS: "ok"}) == 1
    assert harness.total(m.TOKENS, **{m.ATTR_TOKEN_TYPE: "input"}) == 10
    assert harness.total(m.TOKENS, **{m.ATTR_TOKEN_TYPE: "output"}) == 3
    assert harness.total(m.INTER_TOKEN_LATENCY) == 2  # 3 tokens -> 2 gaps
    for pts in points.values():
        for p in pts:
            assert "inference.request.id" not in p.attributes
            assert "unique-123" not in p.attributes.values()
    # gauges return to zero after the request finishes
    assert harness.total(m.REQUESTS_IN_FLIGHT) == 0
    assert harness.total(m.REQUESTS_ACTIVE) == 0
    assert harness.total(m.QUEUE_DEPTH) == 0


def test_active_and_queue_depth_during_request(harness: Harness) -> None:
    with harness.profiler.inference_request(model="demo") as req:
        with req.queue():
            assert harness.total(m.QUEUE_DEPTH) == 1
            assert harness.total(m.REQUESTS_ACTIVE) == 0
            assert harness.total(m.REQUESTS_IN_FLIGHT) == 1
        assert harness.total(m.QUEUE_DEPTH) == 0
        assert harness.total(m.REQUESTS_ACTIVE) == 1


def test_exception_marks_request_failed_and_is_reraised(harness: Harness) -> None:
    with pytest.raises(ValueError, match="boom"):
        with harness.profiler.inference_request(model="demo") as req:
            with req.prefill():
                raise ValueError("boom")
    root = harness.span("inference.request")
    prefill = harness.span("inference.prefill")
    assert root.status.status_code == StatusCode.ERROR
    assert root.attributes["inference.status"] == "error"
    assert root.attributes["error.type"] == "ValueError"
    assert prefill.status.status_code == StatusCode.ERROR
    assert any(e.name == "exception" for e in root.events)
    assert harness.total(m.REQUESTS, **{m.ATTR_STATUS: "error"}) == 1


def test_cancellation_is_not_an_error(harness: Harness) -> None:
    with pytest.raises(RequestCancelled), harness.profiler.inference_request(model="demo"):
        raise RequestCancelled("client went away")
    root = harness.span("inference.request")
    assert root.attributes["inference.status"] == "cancelled"
    assert root.status.status_code != StatusCode.ERROR
    assert harness.total(m.REQUESTS, **{m.ATTR_STATUS: "cancelled"}) == 1


def test_asyncio_cancellation_marks_cancelled(harness: Harness) -> None:
    async def handler() -> None:
        async with harness.profiler.inference_request(model="demo") as req:
            with req.generation():
                await asyncio.sleep(10)

    async def main() -> None:
        task = asyncio.create_task(handler())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert harness.span("inference.request").attributes["inference.status"] == "cancelled"


def test_explicit_fail_without_exception(harness: Harness) -> None:
    with harness.profiler.inference_request(model="demo") as req:
        req.fail("upstream_timeout")
    root = harness.span("inference.request")
    assert root.attributes["inference.status"] == "error"
    assert root.attributes["error.type"] == "upstream_timeout"


def test_ttft_from_explicit_value_creates_span_with_duration(harness: Harness) -> None:
    with harness.profiler.inference_request(model="demo") as req:
        req.record_generation(10, time_to_first_token_ms=35.0, generation_duration_ms=100.0)
    ttft = harness.span("inference.time_to_first_token")
    assert (ttft.end_time - ttft.start_time) == pytest.approx(35_000_000, abs=1_000)
    root = harness.span("inference.request")
    assert ttft.start_time == root.start_time
    assert root.attributes["inference.generation.tokens_per_second"] == pytest.approx(100.0)


def test_cache_operations_record_hits_misses_and_evictions(harness: Harness) -> None:
    with harness.profiler.inference_request(model="demo") as req:
        with req.cache_lookup() as lookup:
            lookup.hit = False
        with req.cache_insert():
            pass
        with req.cache_evict(entries=3):
            pass
    with harness.profiler.inference_request(model="demo") as req, req.cache_lookup() as lookup:
        lookup.hit = True
    assert harness.total(m.CACHE_HITS) == 1
    assert harness.total(m.CACHE_MISSES) == 1
    assert harness.total(m.CACHE_INSERTIONS) == 1
    assert harness.total(m.CACHE_EVICTIONS) == 3
    assert harness.total(m.CACHE_OPERATION_DURATION, **{m.ATTR_CACHE_OPERATION: "lookup"}) == 2
    names = [s.name for s in harness.finished()]
    assert names.count("inference.cache_lookup") == 2
    assert "inference.cache_insert" in names and "inference.cache_evict" in names


def test_context_propagation_from_incoming_headers(harness: Harness) -> None:
    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    parent_id = "00f067aa0ba902b7"
    headers = {"traceparent": f"00-{trace_id}-{parent_id}-01"}
    with harness.profiler.inference_request(model="demo", parent=headers) as req:
        outgoing: dict[str, str] = {}
        req.inject(outgoing)
    root = harness.span("inference.request")
    assert format(root.context.trace_id, "032x") == trace_id
    assert format(root.parent.span_id, "016x") == parent_id
    assert outgoing["traceparent"].split("-")[1] == trace_id
    assert req.trace_id == trace_id


def test_current_request_and_decorators(harness: Harness) -> None:
    prof = harness.profiler

    @prof.phase("prefill")
    def prefill() -> str:
        req = current_request()
        assert req is not None
        return "done"

    @prof.trace_request(model="decorated")
    def handler() -> str:
        return prefill()

    assert current_request() is None
    assert handler() == "done"
    assert current_request() is None
    prefill_span = harness.span("inference.prefill")
    root = harness.span("inference.request")
    assert prefill_span.parent.span_id == root.context.span_id
    assert root.attributes["gen_ai.request.model"] == "decorated"


def test_async_decorators_isolate_concurrent_requests(harness: Harness) -> None:
    prof = harness.profiler

    @prof.trace_request(model="async")
    async def handler(i: int) -> str:
        req = current_request()
        assert req is not None
        await asyncio.sleep(0.001 * (5 - i))
        assert current_request() is req
        async with req.phase("generation"):
            await asyncio.sleep(0.001)
        return req.trace_id or ""

    async def main() -> list[str]:
        return await asyncio.gather(*(handler(i) for i in range(5)))

    trace_ids = asyncio.run(main())
    assert len(set(trace_ids)) == 5
    for gen in [s for s in harness.finished() if s.name == "inference.generation"]:
        assert format(gen.context.trace_id, "032x") in trace_ids


def test_phase_in_worker_thread_keeps_parent(harness: Harness) -> None:
    with harness.profiler.inference_request(model="demo") as req:

        def work() -> None:
            with req.prefill():
                pass

        t = threading.Thread(target=work)
        t.start()
        t.join()
    assert (
        harness.span("inference.prefill").parent.span_id
        == harness.span("inference.request").context.span_id
    )


def test_disabled_profiler_is_noop() -> None:
    h = make_harness(enabled=False)
    with h.profiler.inference_request(model="demo") as req:
        with req.prefill():
            pass
        req.record_generation(5)
        assert not req.is_recording
        assert req.trace_id is None
    assert h.finished() == []
    assert h.metrics() == {}
    assert not h.profiler.enabled
    assert h.profiler.shutdown() is True


def test_sampling_zero_drops_spans_but_keeps_metrics() -> None:
    h = make_harness(sample_rate=0.0)
    try:
        for _ in range(5):
            with h.profiler.inference_request(model="demo") as req, req.prefill():
                pass
        assert h.finished() == []
        assert h.total(m.REQUESTS) == 5
        assert h.total(m.PREFILL_DURATION) == 5
    finally:
        h.profiler.shutdown()


def test_sampling_ratio_is_approximately_respected() -> None:
    h = make_harness(sample_rate=0.25)
    try:
        for _ in range(2000):
            with h.profiler.inference_request(model="demo"):
                pass
        sampled = len([s for s in h.finished() if s.name == "inference.request"])
        assert 380 <= sampled <= 620  # 500 expected; generous binomial bound
    finally:
        h.profiler.shutdown()


def test_cardinality_limit_collapses_excess_models() -> None:
    h = make_harness(max_attribute_values=3)
    try:
        for i in range(10):
            with h.profiler.inference_request(model=f"model-{i}"):
                pass
        models = {p.attributes[m.ATTR_MODEL] for p in h.metrics()[m.REQUESTS]}
        assert models == {"model-0", "model-1", "model-2", m.OTHER_VALUE}
        assert m.ATTR_MODEL in h.profiler.status()["overflowed_attributes"]
    finally:
        h.profiler.shutdown()


def test_token_spans_are_capped() -> None:
    h = make_harness(token_spans=True, token_span_sample_rate=1.0, max_token_spans_per_request=5)
    try:
        with h.profiler.inference_request(model="demo") as req:
            for _ in range(50):
                req.record_token()
        assert len([s for s in h.finished() if s.name == "inference.token"]) == 5
    finally:
        h.profiler.shutdown()


def test_no_token_spans_by_default(harness: Harness) -> None:
    with harness.profiler.inference_request(model="demo") as req:
        for _ in range(20):
            req.record_token()
    assert not [s for s in harness.finished() if s.name == "inference.token"]


def test_invalid_input_raises_in_strict_mode(harness: Harness) -> None:
    with pytest.raises(ValueError, match="model"):
        harness.profiler.inference_request(model="")


def test_invalid_input_is_swallowed_in_non_strict_mode() -> None:
    h = make_harness(strict=False)
    try:
        req = h.profiler.inference_request(model="")
        with req:  # no-op handle, application keeps working
            req.record_generation(1)
        with h.profiler.inference_request(model="ok") as good:
            good.phase("Not A Valid Name")  # logged + counted, not raised
            good.record_generation(-5)
        assert h.total(m.INSTRUMENTATION_ERRORS) >= 3
    finally:
        h.profiler.shutdown()


def test_end_is_idempotent(harness: Harness) -> None:
    req = harness.profiler.start_request(model="demo").start()
    req.end()
    req.end()
    assert len(harness.finished()) == 1
    assert harness.total(m.REQUESTS) == 1


class _Exploding(SpanExporter):
    def export(self, spans: object) -> SpanExportResult:
        raise ConnectionError("collector down")


def test_export_failures_are_contained_and_counted() -> None:
    cfg = ProfilerConfig(exporter="none", memory_interval_ms=0)
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    prof = Profiler(cfg, span_exporter=_Exploding(), metric_reader=reader)
    with prof.inference_request(model="demo") as req:  # must not raise
        with req.prefill():
            pass
    assert prof.export_failures == 2  # prefill span + request span (simple processor)
    assert prof.dropped_items == 2
    points = {
        mm.name: mm
        for rm in reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for mm in sm.metrics
    }
    assert sum(p.value for p in points[m.EXPORT_ERRORS].data.data_points) == 2
    assert prof.shutdown() is True


def test_instrumented_exporter_passes_success_through() -> None:
    class Ok(SpanExporter):
        def export(self, spans: object) -> SpanExportResult:
            return SpanExportResult.SUCCESS

    stats = _ExportStats()
    assert InstrumentedSpanExporter(Ok(), stats).export([]) is SpanExportResult.SUCCESS
    assert stats.failures == 0


def test_shutdown_is_bounded_when_collector_unreachable() -> None:
    cfg = ProfilerConfig(
        otlp_endpoint="http://127.0.0.1:9",
        export_timeout_ms=30_000,
        shutdown_timeout_ms=1_000,
        memory_interval_ms=0,
    )
    prof = Profiler(cfg)
    with prof.inference_request(model="demo"):
        pass
    start = time.monotonic()
    prof.shutdown()
    assert time.monotonic() - start < 3.0
    assert prof.shutdown() is True  # idempotent


def test_global_providers_not_touched_by_default(harness: Harness) -> None:
    assert trace.get_tracer_provider() is not harness.profiler._tracer_provider


def test_status_snapshot(harness: Harness) -> None:
    status = harness.profiler.status()
    assert status["enabled"] is True
    assert status["service_name"] == "test-service"
    assert status["export_failures"] == 0


def test_memory_sampler_reports_rss() -> None:
    h = make_harness(memory_interval_ms=50)
    try:
        sampler = h.profiler.memory_sampler
        assert sampler is not None
        assert sampler.latest_bytes > 0
        points = h.metrics()[m.PROCESS_MEMORY_USAGE]
        assert points[0].value > 1_000_000
    finally:
        h.profiler.shutdown()


def test_simulated_memory_is_labelled(harness: Harness) -> None:
    harness.profiler.set_simulated_memory(1024, 0.5, model="demo")
    harness.profiler.set_cache_utilization(1.7, model="demo")
    points = harness.metrics()
    sim = points[m.SIMULATED_MEMORY_USAGE][0]
    assert sim.attributes[m.ATTR_SIMULATED] == "true"
    assert points[m.CACHE_UTILIZATION][0].value == 1.0  # clamped

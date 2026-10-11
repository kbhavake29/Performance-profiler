from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from inference_profiler import Profiler, ProfilerConfig

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


class Harness:
    def __init__(
        self, profiler: Profiler, spans: InMemorySpanExporter, reader: InMemoryMetricReader
    ) -> None:
        self.profiler = profiler
        self.spans = spans
        self.reader = reader

    def finished(self) -> list[Any]:
        return list(self.spans.get_finished_spans())

    def span(self, name: str) -> Any:
        matches = [s for s in self.finished() if s.name == name]
        assert len(matches) == 1, (
            f"expected one {name} span, got {[s.name for s in self.finished()]}"
        )
        return matches[0]

    def metrics(self) -> dict[str, list[Any]]:
        """metric name -> data points from a fresh collection."""
        data = self.reader.get_metrics_data()
        out: dict[str, list[Any]] = {}
        if data is None:
            return out
        for rm in data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    out.setdefault(metric.name, []).extend(metric.data.data_points)
        return out

    def total(self, name: str, **attrs: str) -> float:
        points = self.metrics().get(name, [])
        total = 0.0
        for p in points:
            if all(p.attributes.get(k) == v for k, v in attrs.items()):
                total += p.count if hasattr(p, "bucket_counts") else p.value
        return total


def make_harness(**overrides: Any) -> Harness:
    cfg = ProfilerConfig(
        service_name="test-service",
        exporter="none",
        memory_interval_ms=0,
        strict=True,
        runtime_version="test-1.0",
    ).with_overrides(**overrides)
    spans = InMemorySpanExporter()
    reader = InMemoryMetricReader()
    profiler = Profiler(cfg, span_exporter=spans, metric_reader=reader)
    return Harness(profiler, spans, reader)


@pytest.fixture
def harness() -> Iterator[Harness]:
    h = make_harness()
    yield h
    h.profiler.shutdown()


@pytest.fixture
def example_report() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / "report_small.json").read_text())
    return data

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from inference_profiler.benchmark import (
    percentile,
    read_report,
    run_benchmark,
    summarize,
    validate_report,
    write_report,
)
from inference_profiler.simulator import ScenarioConfig, load_scenarios

from .conftest import ROOT


def test_percentile_matches_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert percentile(values, 50) == 2.5
    assert percentile(list(range(1, 101)), 99) == pytest.approx(99.01)
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summarize_shape() -> None:
    s = summarize([5.0, 1.0, 3.0])
    assert s is not None
    assert (s["count"], s["min"], s["max"], s["p50"]) == (3, 1.0, 5.0, 3.0)
    assert summarize([]) is None


def test_virtual_benchmark_report_is_valid_and_deterministic() -> None:
    scenarios = [
        ScenarioConfig(name="a", num_requests=40),
        ScenarioConfig(name="b", num_requests=40, concurrency=2),
    ]
    r1 = run_benchmark(scenarios)
    r2 = run_benchmark(scenarios)
    validate_report(r1)
    assert r1["scenarios"] == r2["scenarios"]
    assert r1["measurement"]["clock"] == "virtual"
    assert r1["measurement"]["workload"] == "simulated"
    scn = r1["scenarios"][0]
    assert scn["requests"]["total"] == 40
    assert scn["memory"]["simulated_kv_cache"]["simulated"] is True
    assert scn["cache"]["simulated"] is True
    lat = scn["latency_ms"]["end_to_end"]
    assert lat["p50"] <= lat["p95"] <= lat["p99"] <= lat["max"]


def test_wall_clock_benchmark_report(tmp_path: Path) -> None:
    scn = ScenarioConfig(name="w", num_requests=10, concurrency=5, time_scale=0.02)
    report = run_benchmark([scn], clock="wall")
    path = write_report(report, tmp_path / "r.json")
    loaded = read_report(path)
    assert loaded["measurement"]["clock"] == "wall"
    rss = loaded["scenarios"][0]["memory"]["process_rss"]
    assert rss["simulated"] is False and rss["peak_bytes"] > 0


def test_schema_rejects_bad_reports(example_report: dict[str, Any]) -> None:
    validate_report(example_report)
    broken = copy.deepcopy(example_report)
    del broken["scenarios"][0]["latency_ms"]
    with pytest.raises(jsonschema.ValidationError):
        validate_report(broken)
    broken = copy.deepcopy(example_report)
    broken["scenarios"][0]["memory"]["simulated_kv_cache"]["simulated"] = False
    with pytest.raises(jsonschema.ValidationError):
        validate_report(broken)
    broken = copy.deepcopy(example_report)
    broken["schema_version"] = "9.9"
    with pytest.raises(jsonschema.ValidationError):
        validate_report(broken)


def test_example_baseline_validates_and_is_labelled() -> None:
    data = json.loads((ROOT / "benchmarks" / "baseline.example.json").read_text())
    validate_report(data)
    assert data["example"] is True
    assert data["measurement"]["workload"] == "example"


def test_all_example_scenarios_run() -> None:
    report = run_benchmark(load_scenarios(ROOT / "examples" / "benchmark_scenarios.yaml"))
    validate_report(report)
    by_name = {s["name"]: s for s in report["scenarios"]}
    assert by_name["unreliable"]["requests"]["failed"] > 0
    assert by_name["cache-thrash"]["cache"]["evictions"] > 0
    assert (
        by_name["congestion"]["latency_ms"]["queue"]["p50"]
        > 10 * by_name["baseline"]["latency_ms"]["queue"]["p50"]
    )


def test_overhead_measurement_reports_all_modes() -> None:
    from inference_profiler.benchmark import measure_instrumentation_overhead

    result = measure_instrumentation_overhead(iterations=200, repeats=1)
    for mode in ("disabled", "enabled_sampled", "enabled_unsampled"):
        assert result[mode]["ns_per_request"] > 0
    assert result["enabled_sampled"]["ns_per_request"] > result["disabled"]["ns_per_request"]

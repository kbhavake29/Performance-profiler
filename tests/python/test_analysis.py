from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inference_profiler.analysis import (
    AnalysisThresholds,
    analyze_report,
    report_from_prometheus,
)


def dist(
    p50: float, p95: float | None = None, p99: float | None = None, count: int = 100
) -> dict[str, Any]:
    p95 = p95 if p95 is not None else p50 * 1.2
    p99 = p99 if p99 is not None else p95 * 1.1
    return {
        "count": count,
        "min": p50 / 2,
        "mean": p50,
        "p50": p50,
        "p90": p95,
        "p95": p95,
        "p99": p99,
        "max": p99,
    }


def scenario(**over: Any) -> dict[str, Any]:
    """A healthy scenario; tests perturb one aspect at a time."""
    base: dict[str, Any] = {
        "name": "s",
        "group": None,
        "concurrency": 8,
        "requests": {
            "total": 200,
            "completed": 200,
            "failed": 0,
            "cancelled": 0,
            "error_rate": 0.0,
        },
        "tokens": {"prompt_mean": 500},
        "throughput": {
            "requests_per_second": 10.0,
            "generated_tokens_per_second": 900.0,
            "per_request_tokens_per_second": dist(120.0),
        },
        "latency_ms": {
            "end_to_end": dist(500.0, 650.0, 700.0),
            "queue": dist(2.0),
            "prefill": dist(20.0),
            "time_to_first_token": dist(25.0),
            "generation": dist(450.0),
            "inter_token": dist(7.0, 9.0),
        },
        "cache": {
            "enabled": True,
            "lookups": 200,
            "hits": 150,
            "misses": 50,
            "insertions": 50,
            "evictions": 5,
            "hit_ratio": 0.75,
            "eviction_ratio": 0.1,
        },
        "memory": {
            "process_rss": None,
            "simulated_kv_cache": {
                "peak_bytes": 1,
                "capacity_bytes": 10,
                "peak_pressure": 0.4,
                "simulated": True,
            },
        },
    }
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


def rules(*scenarios: dict[str, Any], t: AnalysisThresholds | None = None) -> dict[str, str]:
    res = analyze_report({"measurement": {"clock": "wall"}, "scenarios": list(scenarios)}, t)
    return {f.rule: f.severity for f in res.findings}


def test_healthy_scenario_has_no_findings() -> None:
    assert rules(scenario()) == {}


@pytest.mark.parametrize(
    ("queue_p50", "expected"),
    [(100.0, None), (124.0, None), (126.0, "warning"), (260.0, "critical")],
)
def test_queue_share_boundaries(queue_p50: float, expected: str | None) -> None:
    s = scenario(latency_ms={"queue": dist(queue_p50), "end_to_end": dist(500.0, 650.0, 700.0)})
    assert rules(s).get("high_queue_latency") == expected


def test_excessive_prefill_per_token() -> None:
    assert "excessive_prefill" not in rules(scenario(latency_ms={"prefill": dist(240.0)}))
    assert rules(scenario(latency_ms={"prefill": dist(260.0)}))["excessive_prefill"] == "warning"


def test_inter_token_latency_boundaries() -> None:
    assert "high_inter_token_latency" not in rules(
        scenario(latency_ms={"inter_token": dist(50, 74.9)})
    )
    assert (
        rules(scenario(latency_ms={"inter_token": dist(50, 80)}))["high_inter_token_latency"]
        == "warning"
    )
    assert (
        rules(scenario(latency_ms={"inter_token": dist(50, 151)}))["high_inter_token_latency"]
        == "critical"
    )


def test_slow_generation() -> None:
    s = scenario(throughput={"per_request_tokens_per_second": dist(12.0)})
    assert rules(s)["slow_generation"] == "warning"


def test_long_tail() -> None:
    assert "long_tail_latency" not in rules(
        scenario(latency_ms={"end_to_end": dist(100, 200, 290)})
    )
    assert "long_tail_latency" in rules(scenario(latency_ms={"end_to_end": dist(100, 200, 310)}))


def test_error_rate() -> None:
    reqs = {"total": 200, "completed": 196, "failed": 4, "cancelled": 0, "error_rate": 0.02}
    assert rules(scenario(requests=reqs))["error_rate"] == "warning"
    reqs = {**reqs, "failed": 12, "error_rate": 0.06}
    assert rules(scenario(requests=reqs))["error_rate"] == "critical"
    reqs = {**reqs, "failed": 1, "error_rate": 0.005}
    assert "error_rate" not in rules(scenario(requests=reqs))


def test_memory_pressure_and_rss() -> None:
    sim = {"peak_bytes": 9, "capacity_bytes": 10, "peak_pressure": 0.9, "simulated": True}
    s = scenario(memory={"simulated_kv_cache": sim, "process_rss": {"peak_bytes": 2 * 2**30}})
    found = rules(s, t=AnalysisThresholds(rss_bytes_warning=2**30))
    assert found["memory_pressure"] == "warning"
    assert found["process_rss"] == "warning"
    sim = {**sim, "peak_pressure": 1.2}
    assert rules(scenario(memory={"simulated_kv_cache": sim}))["memory_pressure"] == "critical"


def test_cache_rules() -> None:
    cache = {
        "enabled": True,
        "lookups": 100,
        "hits": 20,
        "misses": 80,
        "insertions": 80,
        "evictions": 60,
        "hit_ratio": 0.2,
        "eviction_ratio": 0.75,
    }
    found = rules(scenario(cache=cache))
    assert found["low_cache_hit_ratio"] == "warning"
    assert found["cache_eviction_churn"] == "warning"
    disabled = {**cache, "enabled": False, "hit_ratio": None, "eviction_ratio": None}
    assert "low_cache_hit_ratio" not in rules(scenario(cache=disabled))


def _sweep(c: int, tps: float, p95: float) -> dict[str, Any]:
    return scenario(
        name=f"c{c}",
        group="g",
        concurrency=c,
        throughput={"generated_tokens_per_second": tps},
        latency_ms={"end_to_end": dist(p95 / 1.3, p95, p95 * 1.1)},
    )


def test_saturation_detected_only_when_throughput_flattens() -> None:
    scaling = [_sweep(1, 100, 500), _sweep(2, 190, 520), _sweep(4, 360, 560)]
    assert "throughput_saturation" not in rules(*scaling)
    saturated = [_sweep(4, 400, 600), _sweep(16, 420, 2200)]
    assert rules(*saturated)["throughput_saturation"] == "warning"


def test_limitations_reported_for_small_or_missing_data() -> None:
    s = scenario(
        requests={"total": 5, "completed": 5, "failed": 0, "cancelled": 0, "error_rate": 0.0}
    )
    s["latency_ms"]["queue"] = None
    s["latency_ms"]["inter_token"] = None
    res = analyze_report({"measurement": {"clock": "virtual"}, "scenarios": [s]})
    text = " ".join(res.limitations)
    assert "unreliable" in text and "queue" in text and "inter-token" in text
    assert "virtual clock" in text


def test_thresholds_from_file(tmp_path: Path) -> None:
    p = tmp_path / "t.yaml"
    p.write_text("analysis:\n  queue_share_warning: 0.001\n")
    t = AnalysisThresholds.from_file(p)
    assert rules(scenario(), t=t)["high_queue_latency"] == "warning"
    p.write_text("nope: 1\n")
    with pytest.raises(ValueError, match="unknown"):
        AnalysisThresholds.from_file(p)


def test_report_from_prometheus_uses_histogram_quantile() -> None:
    seen: list[str] = []

    def fake(expr: str) -> float | None:
        seen.append(expr)
        if "histogram_quantile(0.95" in expr and "inference_queue_duration" in expr:
            return 0.4
        if "histogram_quantile" in expr:
            return 0.05
        if "_count" in expr:
            return 100.0
        if "_sum" in expr:
            return 10.0
        if 'inference_status="ok"' in expr:
            return 95.0
        if 'inference_status="error"' in expr:
            return 5.0
        return None

    report = report_from_prometheus(fake, window="5m", service="svc")
    scn = report["scenarios"][0]
    assert scn["requests"]["total"] == 100
    assert scn["requests"]["error_rate"] == pytest.approx(0.05)
    assert scn["latency_ms"]["queue"]["p95"] == pytest.approx(400.0)
    assert any("sum by (le)" in q for q in seen)
    assert all('job="svc"' in q for q in seen if "inference_" in q)
    res = analyze_report(report)
    assert "error_rate" in {f.rule for f in res.findings}

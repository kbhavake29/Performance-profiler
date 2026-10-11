from __future__ import annotations

import copy
from typing import Any

import pytest

from inference_profiler.benchmark import run_benchmark
from inference_profiler.regression import RegressionThresholds, compare_reports
from inference_profiler.simulator import RegressionInjection, ScenarioConfig


def _report(**kw: Any) -> dict[str, Any]:
    return run_benchmark([ScenarioConfig(name="s", num_requests=100, **kw)])


def test_identical_reports_pass() -> None:
    r = _report()
    c = compare_reports(r, copy.deepcopy(r))
    assert c.passed()
    assert not c.regressions and not c.improvements


def test_injected_prefill_regression_detected() -> None:
    base = _report()
    cur = _report(regression=RegressionInjection(prefill_multiplier=1.5))
    c = compare_reports(base, cur)
    assert not c.passed()
    metrics = {d.metric for d in c.regressions}
    assert "latency_ms.prefill.p95" in metrics
    assert "latency_ms.time_to_first_token.p95" in metrics
    prefill = next(d for d in c.regressions if d.metric == "latency_ms.prefill.p50")
    assert prefill.change_pct == pytest.approx(50.0, abs=0.5)


def test_improvement_is_not_a_regression() -> None:
    base = _report(regression=RegressionInjection(inter_token_multiplier=1.5))
    cur = _report()
    c = compare_reports(base, cur)
    assert c.passed()
    assert any(d.metric == "throughput.generated_tokens_per_second" for d in c.improvements)


def test_threshold_boundary() -> None:
    base = _report()
    cur = _report(regression=RegressionInjection(prefill_multiplier=1.08))
    assert compare_reports(base, cur, RegressionThresholds(latency_pct=10)).passed()
    assert not compare_reports(base, cur, RegressionThresholds(latency_pct=5)).passed()


def test_noise_floor_ignores_tiny_absolute_changes() -> None:
    base = _report()
    cur = copy.deepcopy(base)
    q = cur["scenarios"][0]["latency_ms"]["queue"]
    q["p50"] = base["scenarios"][0]["latency_ms"]["queue"]["p50"] + 0.3  # >10% but < 0.5 ms
    assert compare_reports(base, cur).passed()
    assert not compare_reports(base, cur, RegressionThresholds(min_latency_delta_ms=0.1)).passed()


def test_per_metric_override() -> None:
    base = _report()
    cur = _report(regression=RegressionInjection(prefill_multiplier=1.5))
    t = RegressionThresholds().with_cli_overrides(
        [
            "latency_ms.prefill.p50=80",
            "latency_ms.prefill.p95=80",
            "latency_ms.prefill.p99=80",
            "latency_ms.time_to_first_token.p50=80",
            "latency_ms.time_to_first_token.p95=80",
            "latency_ms.time_to_first_token.p99=80",
            "latency_ms.end_to_end.p50=80",
            "latency_ms.end_to_end.p95=80",
            "latency_ms.end_to_end.p99=80",
        ]
    )
    assert compare_reports(base, cur, t).passed()


def test_error_rate_absolute_threshold() -> None:
    base = _report()
    cur = _report(failure_rate=0.05)
    c = compare_reports(base, cur)
    assert any(d.metric == "requests.error_rate" for d in c.regressions)


def test_missing_scenario_and_config_drift_warnings() -> None:
    base = run_benchmark(
        [ScenarioConfig(name="a", num_requests=30), ScenarioConfig(name="b", num_requests=30)]
    )
    cur = run_benchmark([ScenarioConfig(name="a", num_requests=30, concurrency=2)])
    c = compare_reports(base, cur)
    assert c.missing_scenarios == ["b"]
    assert not c.passed(fail_on_missing=True)
    assert any("config differs" in w for w in c.warnings)


def test_clock_mismatch_warns() -> None:
    base = _report()
    cur = copy.deepcopy(base)
    cur["measurement"]["clock"] = "wall"
    assert any("clock differs" in w for w in compare_reports(base, cur).warnings)


def test_cli_override_validation() -> None:
    with pytest.raises(ValueError):
        RegressionThresholds().with_cli_overrides(["latency_pct"])
    with pytest.raises(ValueError):
        RegressionThresholds().with_cli_overrides(["bogus=3"])
    with pytest.raises(ValueError):
        RegressionThresholds(latency_pct=-1)
    assert RegressionThresholds().with_cli_overrides(["latency_pct=3"]).latency_pct == 3

"""Compare a benchmark report against a baseline and flag regressions.

Each compared metric has a direction (lower or higher is better) and a
percentage threshold. A change counts as a regression only when it is worse
than the threshold *and* larger than the absolute noise floor, so tiny
sub-millisecond jitter on fast phases cannot fail a build.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Status = Literal["regression", "improvement", "unchanged", "missing"]

LATENCY_METRICS = (
    "end_to_end",
    "time_to_first_token",
    "prefill",
    "queue",
    "inter_token",
    "generation",
)
PERCENTILES = ("p50", "p95", "p99")


@dataclass(frozen=True, slots=True)
class RegressionThresholds:
    latency_pct: float = 10.0
    throughput_pct: float = 10.0
    memory_pct: float = 15.0
    #: Absolute increase in error rate (0.01 = one percentage point).
    error_rate_abs: float = 0.01
    #: Absolute drop in cache hit ratio.
    cache_hit_ratio_abs: float = 0.05
    #: Latency changes smaller than this (ms) are treated as noise.
    min_latency_delta_ms: float = 0.5
    #: Per-metric percentage overrides, e.g. {"latency_ms.queue.p99": 50}.
    overrides: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("latency_pct", "throughput_pct", "memory_pct"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.error_rate_abs < 0 or self.cache_hit_ratio_abs < 0 or self.min_latency_delta_ms < 0:
            raise ValueError("absolute thresholds must be >= 0")

    @classmethod
    def from_file(cls, path: str | Path) -> RegressionThresholds:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        section = data.get("regression", data)
        if not isinstance(section, Mapping):
            raise ValueError(f"{path}: expected a mapping of thresholds")
        return cls(**section)

    def with_cli_overrides(self, items: list[str]) -> RegressionThresholds:
        """Apply ``key=value`` pairs: global keys or dotted metric paths."""
        kwargs = asdict(self)
        overrides = dict(self.overrides)
        for item in items:
            key, sep, raw = item.partition("=")
            if not sep:
                raise ValueError(f"threshold override must be key=value (got {item!r})")
            value = float(raw)
            if key in kwargs and key != "overrides":
                kwargs[key] = value
            elif "." in key:
                overrides[key] = value
            else:
                raise ValueError(f"unknown threshold {key!r}")
        kwargs["overrides"] = overrides
        return RegressionThresholds(**kwargs)


@dataclass(slots=True)
class MetricDelta:
    scenario: str
    metric: str
    baseline: float | None
    current: float | None
    change_pct: float | None
    abs_change: float | None
    better: Literal["lower", "higher"]
    threshold: str
    status: Status

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Comparison:
    deltas: list[MetricDelta]
    warnings: list[str]
    missing_scenarios: list[str]
    baseline_label: str
    current_label: str
    thresholds: RegressionThresholds

    @property
    def regressions(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.status == "regression"]

    @property
    def improvements(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.status == "improvement"]

    def passed(self, fail_on_missing: bool = False) -> bool:
        return not self.regressions and not (fail_on_missing and self.missing_scenarios)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline_label,
            "current": self.current_label,
            "passed": self.passed(),
            "regression_count": len(self.regressions),
            "improvement_count": len(self.improvements),
            "missing_scenarios": self.missing_scenarios,
            "warnings": self.warnings,
            "thresholds": {**asdict(self.thresholds), "overrides": dict(self.thresholds.overrides)},
            "deltas": [d.to_dict() for d in self.deltas],
        }


def _label(report: Mapping[str, Any]) -> str:
    scn = report.get("scenarios") or [{}]
    runtimes = sorted({str(s.get("runtime_version")) for s in scn})
    return str(report.get("label") or ",".join(runtimes))


def _pct(base: float, cur: float) -> float | None:
    if base == 0:
        return None if cur == 0 else float("inf")
    return (cur - base) / abs(base) * 100.0


def compare_reports(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    thresholds: RegressionThresholds | None = None,
) -> Comparison:
    t = thresholds or RegressionThresholds()
    warnings: list[str] = []
    bm, cm = baseline.get("measurement", {}), current.get("measurement", {})
    if bm.get("clock") != cm.get("clock"):
        warnings.append(
            f"clock differs (baseline={bm.get('clock')}, current={cm.get('clock')}); "
            "virtual and wall-clock results are not comparable."
        )
    be, ce = baseline.get("environment", {}), current.get("environment", {})
    if bm.get("clock") == "wall" and (be.get("machine"), be.get("cpu_count")) != (
        ce.get("machine"),
        ce.get("cpu_count"),
    ):
        warnings.append("wall-clock results were measured on different hardware.")

    base_by_name = {s["name"]: s for s in baseline.get("scenarios", [])}
    deltas: list[MetricDelta] = []
    missing: list[str] = []
    for cur in current.get("scenarios", []):
        base = base_by_name.get(cur["name"])
        if base is None:
            warnings.append(f"scenario {cur['name']!r} has no baseline; skipped.")
            continue
        _check_config_drift(base, cur, warnings)
        deltas.extend(_compare_scenario(cur["name"], base, cur, t))
    current_names = {s["name"] for s in current.get("scenarios", [])}
    missing = sorted(set(base_by_name) - current_names)
    for name in missing:
        warnings.append(f"baseline scenario {name!r} missing from current report.")
    return Comparison(
        deltas=deltas,
        warnings=warnings,
        missing_scenarios=missing,
        baseline_label=_label(baseline),
        current_label=_label(current),
        thresholds=t,
    )


def _check_config_drift(
    base: Mapping[str, Any], cur: Mapping[str, Any], warnings: list[str]
) -> None:
    ignore = {"runtime_version", "regression", "description"}
    bc = {k: v for k, v in (base.get("config") or {}).items() if k not in ignore}
    cc = {k: v for k, v in (cur.get("config") or {}).items() if k not in ignore}
    if bc and cc and bc != cc:
        changed = sorted(k for k in set(bc) | set(cc) if bc.get(k) != cc.get(k))
        warnings.append(
            f"scenario {cur['name']!r}: workload config differs in {changed}; "
            "differences may not be caused by the runtime."
        )


def _compare_scenario(
    name: str, base: Mapping[str, Any], cur: Mapping[str, Any], t: RegressionThresholds
) -> list[MetricDelta]:
    out: list[MetricDelta] = []
    for metric in LATENCY_METRICS:
        bd = (base.get("latency_ms") or {}).get(metric)
        cd = (cur.get("latency_ms") or {}).get(metric)
        for p in PERCENTILES:
            path = f"latency_ms.{metric}.{p}"
            bv = bd.get(p) if bd else None
            cv = cd.get(p) if cd else None
            pct = t.overrides.get(path, t.latency_pct)
            out.append(_relative(name, path, bv, cv, "lower", pct, t.min_latency_delta_ms))
    for path, getter in (
        (
            "throughput.generated_tokens_per_second",
            lambda s: s["throughput"]["generated_tokens_per_second"],
        ),
        ("throughput.requests_per_second", lambda s: s["throughput"]["requests_per_second"]),
    ):
        pct = t.overrides.get(path, t.throughput_pct)
        out.append(_relative(name, path, _get(getter, base), _get(getter, cur), "higher", pct, 0.0))
    mem_path = "memory.simulated_kv_cache.peak_bytes"
    out.append(
        _relative(
            name,
            mem_path,
            _get(lambda s: s["memory"]["simulated_kv_cache"]["peak_bytes"], base),
            _get(lambda s: s["memory"]["simulated_kv_cache"]["peak_bytes"], cur),
            "lower",
            t.overrides.get(mem_path, t.memory_pct),
            0.0,
        )
    )
    rss_path = "memory.process_rss.peak_bytes"
    b_rss = _get(lambda s: s["memory"]["process_rss"]["peak_bytes"], base)
    c_rss = _get(lambda s: s["memory"]["process_rss"]["peak_bytes"], cur)
    if b_rss is not None and c_rss is not None:
        out.append(
            _relative(
                name, rss_path, b_rss, c_rss, "lower", t.overrides.get(rss_path, t.memory_pct), 0.0
            )
        )
    out.append(
        _absolute(
            name,
            "requests.error_rate",
            _get(lambda s: s["requests"]["error_rate"], base),
            _get(lambda s: s["requests"]["error_rate"], cur),
            "lower",
            t.overrides.get("requests.error_rate", t.error_rate_abs),
        )
    )
    b_hit = _get(lambda s: s["cache"]["hit_ratio"], base)
    c_hit = _get(lambda s: s["cache"]["hit_ratio"], cur)
    if b_hit is not None or c_hit is not None:
        out.append(
            _absolute(
                name,
                "cache.hit_ratio",
                b_hit,
                c_hit,
                "higher",
                t.overrides.get("cache.hit_ratio", t.cache_hit_ratio_abs),
            )
        )
    return out


def _get(getter: Any, data: Mapping[str, Any]) -> float | None:
    try:
        value = getter(data)
    except (KeyError, TypeError):
        return None
    return None if value is None else float(value)


def _relative(
    scenario: str,
    metric: str,
    base: float | None,
    cur: float | None,
    better: Literal["lower", "higher"],
    pct_threshold: float,
    noise_floor: float,
) -> MetricDelta:
    threshold = f"{pct_threshold:g}%" + (f" and > {noise_floor:g} ms" if noise_floor else "")
    if base is None or cur is None:
        return MetricDelta(scenario, metric, base, cur, None, None, better, threshold, "missing")
    change = _pct(base, cur)
    diff = cur - base
    worse = diff > 0 if better == "lower" else diff < 0
    status: Status = "unchanged"
    if change is not None and abs(diff) > noise_floor and abs(change) > pct_threshold:
        status = "regression" if worse else "improvement"
    return MetricDelta(
        scenario,
        metric,
        base,
        cur,
        None if change is None or change == float("inf") else round(change, 3),
        round(diff, 6),
        better,
        threshold,
        status,
    )


def _absolute(
    scenario: str,
    metric: str,
    base: float | None,
    cur: float | None,
    better: Literal["lower", "higher"],
    abs_threshold: float,
) -> MetricDelta:
    threshold = f"absolute {abs_threshold:g}"
    if base is None or cur is None:
        return MetricDelta(scenario, metric, base, cur, None, None, better, threshold, "missing")
    diff = cur - base
    worse = diff > 0 if better == "lower" else diff < 0
    status: Status = "unchanged"
    if abs(diff) > abs_threshold:
        status = "regression" if worse else "improvement"
    return MetricDelta(
        scenario,
        metric,
        base,
        cur,
        _pct(base, cur) if base else None,
        round(diff, 6),
        better,
        threshold,
        status,
    )


def publish_comparison(
    comparison: Comparison, endpoint: str, service_name: str, timeout_ms: int = 10_000
) -> bool:
    """Export comparison results as OTLP gauges for the Performance Regressions dashboard."""
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource

    exporter = OTLPMetricExporter(
        endpoint=endpoint.rstrip("/") + "/v1/metrics", timeout=timeout_ms / 1000.0
    )
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=3_600_000)
    provider = MeterProvider(
        resource=Resource.create({"service.name": service_name}),
        metric_readers=[reader],
        shutdown_on_exit=False,
    )
    meter = provider.get_meter("inference_profiler.regression")
    value = meter.create_gauge(
        "inference.benchmark.value",
        unit="",
        description="Benchmark metric value (ms, tokens/s, bytes or ratio).",
    )
    change = meter.create_gauge(
        "inference.benchmark.change",
        unit="%",
        description="Percentage change of current vs baseline.",
    )
    failed = meter.create_gauge(
        "inference.benchmark.regression",
        unit="",
        description="1 if the metric exceeded its regression threshold.",
    )
    total = meter.create_gauge(
        "inference.benchmark.regressions",
        unit="",
        description="Number of metrics exceeding regression thresholds.",
    )
    for d in comparison.deltas:
        if d.status == "missing":
            continue
        common = {"scenario": d.scenario, "metric": d.metric}
        if d.baseline is not None:
            value.set(
                d.baseline,
                {**common, "run": "baseline", "runtime_version": comparison.baseline_label},
            )
        if d.current is not None:
            value.set(
                d.current, {**common, "run": "current", "runtime_version": comparison.current_label}
            )
        if d.change_pct is not None:
            change.set(d.change_pct, {**common, "status": d.status})
        failed.set(1 if d.status == "regression" else 0, common)
    total.set(
        len(comparison.regressions),
        {"baseline": comparison.baseline_label, "current": comparison.current_label},
    )
    ok = provider.force_flush(timeout_ms)
    provider.shutdown(timeout_millis=timeout_ms)
    return bool(ok)

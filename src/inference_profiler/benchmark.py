"""Benchmark runner and machine-readable report format.

Percentiles are computed from raw per-request samples (linear interpolation
between closest ranks, the same definition as ``numpy.percentile``'s default).
They are never averaged across requests, scenarios or machines.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema

from inference_profiler.config import ProfilerConfig
from inference_profiler.instrumentation import Profiler
from inference_profiler.memory import process_rss_bytes
from inference_profiler.simulator import (
    SIMULATOR_VERSION,
    RequestOutcome,
    ScenarioConfig,
    SimulationRun,
    build_plan,
    run_wall_clock,
    simulate_virtual,
)
from inference_profiler.version import __version__

SCHEMA_VERSION = "1.0"
LATENCY_KEYS = (
    "end_to_end",
    "queue",
    "prefill",
    "time_to_first_token",
    "generation",
    "inter_token",
)


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of pre-sorted values, ``q`` in [0, 100]."""
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))


def summarize(values: Sequence[float]) -> dict[str, float | int] | None:
    """Distribution summary in the unit of ``values``; ``None`` when there are no samples."""
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mean = sum(ordered) / n
    var = sum((v - mean) ** 2 for v in ordered) / n
    return {
        "count": n,
        "min": round(ordered[0], 4),
        "mean": round(mean, 4),
        "stddev": round(math.sqrt(var), 4),
        "p50": round(percentile(ordered, 50), 4),
        "p90": round(percentile(ordered, 90), 4),
        "p95": round(percentile(ordered, 95), 4),
        "p99": round(percentile(ordered, 99), 4),
        "max": round(ordered[-1], 4),
    }


def scenario_result(run: SimulationRun) -> dict[str, Any]:
    s = run.scenario
    outs = run.outcomes
    ok = [o for o in outs if o.status == "ok"]
    failed = [o for o in outs if o.status == "error"]
    cancelled = [o for o in outs if o.status == "cancelled"]
    generated_total = sum(o.generated_tokens for o in outs)
    with_tokens = [o for o in outs if o.generated_tokens > 0]
    per_request_tps = [
        o.generated_tokens / (o.generation_ms / 1000.0)
        for o in ok
        if o.generation_ms > 0 and o.generated_tokens > 0
    ]
    hits = sum(1 for o in outs if o.cache_hit) if s.cache.enabled else 0
    lookups = len(outs) if s.cache.enabled else 0
    misses = lookups - hits
    insertions = sum(1 for o in ok if not o.cache_hit) if s.cache.enabled else 0
    evictions = sum(o.cache_evictions for o in ok) if s.cache.enabled else 0
    itl_all = [v for o in outs for v in o.inter_token_ms]

    return {
        "name": s.name,
        "description": s.description,
        "group": s.group,
        "model": s.model,
        "runtime_version": s.runtime_version,
        "config": s.to_dict(),
        "regression_injected": s.regression.active,
        "concurrency": s.concurrency,
        "duration_s": round(run.duration_s, 6),
        "requests": {
            "total": len(outs),
            "completed": len(ok),
            "failed": len(failed),
            "cancelled": len(cancelled),
            "error_rate": round(len(failed) / len(outs), 6) if outs else 0.0,
        },
        "tokens": {
            "prompt_total": sum(o.prompt_tokens for o in outs),
            "generated_total": generated_total,
            "prompt_mean": round(sum(o.prompt_tokens for o in outs) / len(outs), 3) if outs else 0,
        },
        "throughput": {
            "requests_per_second": round(len(ok) / run.duration_s, 4),
            "generated_tokens_per_second": round(generated_total / run.duration_s, 4),
            "per_request_tokens_per_second": summarize(per_request_tps),
        },
        "latency_ms": {
            "end_to_end": summarize([o.e2e_ms for o in ok]),
            "queue": summarize([o.queue_ms for o in outs]),
            "prefill": summarize([o.prefill_ms for o in outs if o.prefill_ms > 0]),
            "time_to_first_token": summarize(
                [o.ttft_ms for o in with_tokens if o.ttft_ms is not None]
            ),
            "generation": summarize([o.generation_ms for o in ok]),
            "inter_token": summarize(itl_all),
        },
        "cache": {
            "enabled": s.cache.enabled,
            "lookups": lookups,
            "hits": hits,
            "misses": misses,
            "insertions": insertions,
            "evictions": evictions,
            "hit_ratio": round(hits / lookups, 6) if lookups else None,
            "eviction_ratio": round(evictions / insertions, 6) if insertions else None,
            "simulated": True,
        },
        "memory": {
            "process_rss": (
                {**run.process_rss, "source": run.process_rss_source, "simulated": False}
                if run.process_rss
                else None
            ),
            "simulated_kv_cache": {
                "peak_bytes": run.simulated_memory_peak_bytes,
                "capacity_bytes": run.simulated_memory_capacity_bytes,
                "peak_pressure": round(
                    run.simulated_memory_peak_bytes / run.simulated_memory_capacity_bytes, 6
                ),
                "requests_slowed_by_pressure": sum(1 for o in outs if o.memory_pressure_slowdown),
                "simulated": True,
            },
        },
        "errors": _error_summary(failed, cancelled),
    }


def _error_summary(
    failed: list[RequestOutcome], cancelled: list[RequestOutcome]
) -> list[dict[str, Any]]:
    errors = []
    if failed:
        errors.append(
            {
                "type": "SimulatedInferenceError",
                "count": len(failed),
                "request_indexes": [o.index for o in failed[:20]],
            }
        )
    if cancelled:
        errors.append(
            {
                "type": "RequestCancelled",
                "count": len(cancelled),
                "request_indexes": [o.index for o in cancelled[:20]],
            }
        )
    return errors


def git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and len(sha) == 40 else None


def environment_metadata() -> dict[str, Any]:
    """Environment facts that affect timings. Hostnames and usernames are deliberately omitted."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(terse=True),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "ci": bool(os.environ.get("CI")),
    }


def build_report(
    runs: list[SimulationRun],
    *,
    clock: str,
    overhead: dict[str, Any] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": label,
        "git_commit": git_commit(),
        "sdk_version": __version__,
        "simulator_version": SIMULATOR_VERSION,
        "environment": environment_metadata(),
        "measurement": {
            "clock": clock,
            "workload": "simulated",
            "latency_source": (
                "virtual_clock_model"
                if clock == "virtual"
                else "wall_clock_measurement_of_simulated_workload"
            ),
            "notes": (
                "Latencies come from a deterministic model; they are not measurements."
                if clock == "virtual"
                else "Latencies are real wall-clock measurements of a synthetic workload "
                "(asyncio sleeps), not of a real model."
            ),
            "cache_statistics": "simulated",
            "simulated_memory": "simulated KV-cache model; not a real measurement",
            "process_rss": "real measurement (psutil)" if clock == "wall" else "not collected",
        },
        "scenarios": [scenario_result(r) for r in runs],
        "instrumentation_overhead": overhead,
    }


def run_benchmark(
    scenarios: list[ScenarioConfig],
    *,
    clock: str = "virtual",
    profiler: Profiler | None = None,
    measure_overhead: bool = False,
    label: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if clock not in ("virtual", "wall"):
        raise ValueError("clock must be 'virtual' or 'wall'")
    runs: list[SimulationRun] = []
    own_profiler = None
    if clock == "wall" and profiler is None:
        own_profiler = profiler = Profiler(ProfilerConfig.from_env(exporter="none"))
    try:
        for s in scenarios:
            if progress:
                progress(f"running scenario {s.name} ({s.num_requests} requests, {clock} clock)")
            plan = build_plan(s)
            if clock == "virtual":
                run = simulate_virtual(s, plan)
            else:
                assert profiler is not None
                run = asyncio.run(run_wall_clock(s, profiler, plan))
            runs.append(run)
    finally:
        if own_profiler is not None:
            own_profiler.shutdown()
    overhead = measure_instrumentation_overhead() if measure_overhead else None
    return build_report(runs, clock=clock, overhead=overhead, label=label)


# ======================================================================= schema


def load_schema() -> dict[str, Any]:
    text = resources.files("inference_profiler").joinpath("schemas/benchmark-report.schema.json")
    data: dict[str, Any] = json.loads(text.read_text(encoding="utf-8"))
    return data


def validate_report(report: dict[str, Any]) -> None:
    """Raise ``jsonschema.ValidationError`` if ``report`` does not match the schema."""
    jsonschema.validate(report, load_schema())


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    validate_report(report)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return out


def read_report(path: str | Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_report(data)
    return data


# ===================================================================== overhead


def measure_instrumentation_overhead(iterations: int = 5_000, repeats: int = 5) -> dict[str, Any]:
    """Measure SDK cost per request (3 phases + 32 tokens) with instrumentation off vs on.

    "enabled" uses the full SDK with an in-process exporter that discards
    spans, so network/export cost is excluded; it measures the work added to the
    application's request path. The best of ``repeats`` runs is reported to reduce
    scheduler noise.
    """
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class _Discard(SpanExporter):
        def export(self, spans: Any) -> SpanExportResult:
            return SpanExportResult.SUCCESS

    def workload(prof: Profiler) -> float:
        start = time.perf_counter_ns()
        for i in range(iterations):
            with prof.inference_request(
                model="overhead", request_id=str(i), prompt_tokens=128
            ) as r:
                with r.queue():
                    pass
                with r.prefill():
                    pass
                with r.generation():
                    for _ in range(32):
                        r.record_token()
                r.record_generation(32)
        return (time.perf_counter_ns() - start) / iterations

    base = ProfilerConfig(exporter="none", memory_interval_ms=0, service_name="overhead")
    results: dict[str, Any] = {
        "iterations": iterations,
        "repeats": repeats,
        "operations_per_request": "1 request, 3 phases, 32 tokens",
    }
    for label, cfg, sample_rate in (
        ("disabled", base.with_overrides(enabled=False), 1.0),
        ("enabled_sampled", base, 1.0),
        ("enabled_unsampled", base.with_overrides(sample_rate=0.0), 0.0),
    ):
        best = math.inf
        for _ in range(repeats):
            prof = Profiler(
                cfg,
                span_exporter=_Discard(),
                metric_reader=InMemoryMetricReader(),
                batch_spans=True,
            )
            try:
                workload(prof)  # warm-up (caches, attribute admission)
                best = min(best, workload(prof))
            finally:
                prof.shutdown()
        results[label] = {"ns_per_request": round(best, 1), "sample_rate": sample_rate}
    # Cost of record_token() alone, i.e. what the decode loop pays per token.
    prof = Profiler(base, span_exporter=_Discard(), metric_reader=InMemoryMetricReader())
    try:
        with prof.inference_request(model="overhead") as req:
            req.record_token()
            n = iterations * 10
            start = time.perf_counter_ns()
            for _ in range(n):
                req.record_token()
            results["record_token_ns"] = round((time.perf_counter_ns() - start) / n, 1)
    finally:
        prof.shutdown()
    disabled = results["disabled"]["ns_per_request"]
    for label in ("enabled_sampled", "enabled_unsampled"):
        results[label]["added_us_per_request"] = round(
            (results[label]["ns_per_request"] - disabled) / 1000.0, 3
        )
    results["environment"] = environment_metadata()
    results["process_rss_bytes"] = process_rss_bytes()
    return results

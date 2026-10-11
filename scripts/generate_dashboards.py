#!/usr/bin/env python3
"""Generate the provisioned Grafana dashboards in monitoring/grafana/dashboards/.

Dashboards are generated (not hand-edited) so every PromQL expression is built
from the metric names below, which tests/python/test_dashboards.py checks
against the names the SDK emits. Run after changing metrics:

    python scripts/generate_dashboards.py          # write
    python scripts/generate_dashboards.py --check  # fail if files are stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parents[1] / "monitoring" / "grafana" / "dashboards"

PROM = {"type": "prometheus", "uid": "prometheus"}
TEMPO = {"type": "tempo", "uid": "tempo"}

# Prometheus names produced by the collector's prometheus exporter for the SDK metrics.
M = {
    "request": "inference_request_duration_seconds",
    "queue": "inference_queue_duration_seconds",
    "prefill": "inference_prefill_duration_seconds",
    "ttft": "inference_time_to_first_token_seconds",
    "itl": "inference_inter_token_latency_seconds",
    "generation": "inference_generation_duration_seconds",
    "throughput": "inference_generation_throughput_per_second",
    "cache_op": "inference_cache_operation_duration_seconds",
    "requests": "inference_requests_total",
    "tokens": "inference_tokens_total",
    "in_flight": "inference_requests_in_flight",
    "active": "inference_requests_active",
    "queue_depth": "inference_queue_depth",
    "hits": "inference_cache_hits_total",
    "misses": "inference_cache_misses_total",
    "insertions": "inference_cache_insertions_total",
    "evictions": "inference_cache_evictions_total",
    "cache_util": "inference_cache_utilization_ratio",
    "rss": "process_memory_usage_bytes",
    "sim_mem": "inference_simulated_memory_usage_bytes",
    "sim_pressure": "inference_simulated_memory_pressure_ratio",
    "export_errors": "inference_profiler_export_errors_total",
    "dropped": "inference_profiler_telemetry_dropped_total",
    "instr_errors": "inference_profiler_instrumentation_errors_total",
    "export_duration": "inference_profiler_export_duration_seconds",
    "enabled": "inference_profiler_enabled",
    "bench_value": "inference_benchmark_value",
    "bench_change": "inference_benchmark_change_percent",
    "bench_regression": "inference_benchmark_regression",
    "bench_regressions": "inference_benchmark_regressions",
}

SEL = 'job=~"$job", gen_ai_request_model=~"$model", inference_runtime_version=~"$runtime"'
JOB = 'job=~"$job"'


# ----------------------------------------------------------------- panel helpers


class Layout:
    def __init__(self) -> None:
        self.y = 0
        self.x = 0
        self.row_h = 0
        self.next_id = 1

    def place(self, w: int, h: int) -> dict[str, int]:
        if self.x + w > 24:
            self.x = 0
            self.y += self.row_h
            self.row_h = 0
        pos = {"x": self.x, "y": self.y, "w": w, "h": h}
        self.x += w
        self.row_h = max(self.row_h, h)
        return pos

    def newline(self) -> None:
        if self.x:
            self.x = 0
            self.y += self.row_h
            self.row_h = 0

    def pid(self) -> int:
        self.next_id += 1
        return self.next_id


def target(expr: str, legend: str = "", ref: str = "A", **extra: Any) -> dict[str, Any]:
    t = {
        "refId": ref,
        "datasource": PROM,
        "expr": expr,
        "legendFormat": legend or "__auto",
        "range": True,
    }
    t.update(extra)
    return t


def row(lay: Layout, title: str) -> dict[str, Any]:
    lay.newline()
    panel = {
        "type": "row",
        "title": title,
        "id": lay.pid(),
        "collapsed": False,
        "gridPos": {"x": 0, "y": lay.y, "w": 24, "h": 1},
        "panels": [],
    }
    lay.y += 1
    return panel


def timeseries(
    lay: Layout,
    title: str,
    targets: list[dict[str, Any]],
    unit: str,
    desc: str,
    w: int = 12,
    h: int = 8,
    stack: bool = False,
    exemplars: bool = False,
) -> dict[str, Any]:
    for i, t in enumerate(targets):
        t["refId"] = chr(ord("A") + i)
        if exemplars:
            t["exemplar"] = True
    return {
        "type": "timeseries",
        "title": title,
        "description": desc,
        "id": lay.pid(),
        "gridPos": lay.place(w, h),
        "datasource": PROM,
        "targets": targets,
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "custom": {
                    "drawStyle": "line",
                    "lineWidth": 1,
                    "fillOpacity": 10 if stack else 0,
                    "showPoints": "never",
                    "spanNulls": True,
                    "stacking": {"mode": "normal" if stack else "none"},
                },
            },
            "overrides": [],
        },
        "options": {
            "legend": {
                "displayMode": "table",
                "placement": "bottom",
                "calcs": ["lastNotNull", "max"],
            },
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def stat(
    lay: Layout,
    title: str,
    expr: str,
    unit: str,
    desc: str,
    w: int = 4,
    thresholds: list[tuple[str, float | None]] | None = None,
) -> dict[str, Any]:
    steps = [{"color": c, "value": v} for c, v in (thresholds or [("green", None)])]
    return {
        "type": "stat",
        "title": title,
        "description": desc,
        "id": lay.pid(),
        "gridPos": lay.place(w, 4),
        "datasource": PROM,
        "targets": [target(expr, instant=True, range=False)],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "decimals": 2,
                "thresholds": {"mode": "absolute", "steps": steps},
            },
            "overrides": [],
        },
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "value",
            "graphMode": "area",
            "textMode": "auto",
        },
    }


def quantiles(metric: str, sel: str, by: str = "", legend_prefix: str = "") -> list[dict[str, Any]]:
    group = f"le{', ' + by if by else ''}"
    out = []
    for q in ("0.5", "0.95", "0.99"):
        label = f"{legend_prefix}p{int(float(q) * 100)}" + (f" {{{{{by}}}}}" if by else "")
        out.append(
            target(
                f"histogram_quantile({q}, sum by ({group}) (rate({metric}_bucket{{{sel}}}[$__rate_interval])))",
                label,
            )
        )
    return out


def variables(*, with_model: bool = True) -> dict[str, Any]:
    def var(name: str, label: str, query: str) -> dict[str, Any]:
        return {
            "name": name,
            "label": label,
            "type": "query",
            "datasource": PROM,
            "query": {"query": query, "refId": "PrometheusVariableQueryEditor-VariableQuery"},
            "definition": query,
            "refresh": 2,
            "includeAll": True,
            "multi": True,
            "allValue": ".*",
            "current": {"selected": True, "text": ["All"], "value": ["$__all"]},
            "sort": 1,
        }

    items = [var("job", "Service", f"label_values({M['requests']}, job)")]
    if with_model:
        items.append(
            var("model", "Model", f"label_values({M['requests']}{{{JOB}}}, gen_ai_request_model)")
        )
        items.append(
            var(
                "runtime",
                "Runtime version",
                f"label_values({M['requests']}{{{JOB}}}, inference_runtime_version)",
            )
        )
    return {"list": items}


def dashboard(
    uid: str,
    title: str,
    desc: str,
    panels: list[dict[str, Any]],
    templating: dict[str, Any],
    time_from: str = "now-30m",
) -> dict[str, Any]:
    return {
        "uid": uid,
        "title": title,
        "description": desc,
        "tags": ["inference-profiler"],
        "editable": False,
        "graphTooltip": 1,
        "schemaVersion": 39,
        "version": 1,
        "refresh": "10s",
        "time": {"from": time_from, "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "templating": templating,
        "annotations": {"list": []},
        "panels": panels,
        "links": [
            {
                "title": "Inference Profiler",
                "type": "dashboards",
                "tags": ["inference-profiler"],
                "asDropdown": True,
                "includeVars": True,
                "keepTime": True,
            }
        ],
    }


# ------------------------------------------------------------------- dashboards


def overview() -> dict[str, Any]:
    lay = Layout()
    req_rate = f"sum(rate({M['requests']}{{{SEL}}}[$__rate_interval]))"
    p = [
        stat(
            lay, "Request rate", req_rate, "reqps", "Finished requests per second (all statuses)."
        ),
        stat(
            lay,
            "Error ratio",
            f'sum(rate({M["requests"]}{{{SEL}, inference_status="error"}}[$__rate_interval])) / {req_rate}',
            "percentunit",
            "Share of finished requests with status=error.",
            thresholds=[("green", None), ("orange", 0.01), ("red", 0.05)],
        ),
        stat(
            lay,
            "p95 latency",
            f"histogram_quantile(0.95, sum by (le) (rate({M['request']}_bucket{{{SEL}}}[$__rate_interval])))",
            "s",
            "End-to-end request latency p95 from histogram buckets.",
        ),
        stat(
            lay,
            "Output tokens/s",
            f'sum(rate({M["tokens"]}{{{SEL}, gen_ai_token_type="output"}}[$__rate_interval]))',
            "short",
            "Generated tokens per second across all requests.",
        ),
        stat(
            lay,
            "In-flight requests",
            f"sum({M['in_flight']}{{{SEL}}})",
            "short",
            "Admitted requests not yet finished (queued + executing).",
        ),
        stat(
            lay,
            "Profiler export errors",
            f"sum(increase({M['export_errors']}{{{JOB}}}[$__range]))",
            "short",
            "Failed OTLP export calls in the selected range (SDK health).",
            thresholds=[("green", None), ("red", 1)],
        ),
        timeseries(
            lay,
            "Requests by status",
            [
                target(
                    f"sum by (inference_status) (rate({M['requests']}{{{SEL}}}[$__rate_interval]))",
                    "{{inference_status}}",
                )
            ],
            "reqps",
            "Completed, failed and cancelled requests per second.",
            stack=True,
        ),
        timeseries(
            lay,
            "End-to-end latency percentiles",
            quantiles(M["request"], SEL),
            "s",
            "p50/p95/p99 computed with histogram_quantile over summed bucket rates. "
            "Diamonds are exemplars linking to traces.",
            exemplars=True,
        ),
        timeseries(
            lay,
            "Token throughput",
            [
                target(
                    f"sum by (gen_ai_token_type) (rate({M['tokens']}{{{SEL}}}[$__rate_interval]))",
                    "{{gen_ai_token_type}} tokens/s",
                )
            ],
            "short",
            "Prompt (input) and generated (output) tokens per second.",
        ),
        timeseries(
            lay,
            "Concurrency",
            [
                target(f"sum({M['in_flight']}{{{SEL}}})", "in flight"),
                target(f"sum({M['active']}{{{SEL}}})", "active (executing)"),
                target(f"sum({M['queue_depth']}{{{SEL}}})", "queue depth"),
            ],
            "short",
            "In flight = queued + executing. Active excludes queued requests.",
        ),
        timeseries(
            lay,
            "Latency p95 by runtime version",
            quantiles(M["request"], SEL, by="inference_runtime_version")[1:2],
            "s",
            "Compare runtime versions running side by side.",
        ),
        timeseries(
            lay,
            "Instrumentation health",
            [
                target(
                    f"sum by (inference_profiler_signal) (rate({M['export_errors']}{{{JOB}}}[$__rate_interval]))",
                    "export errors {{inference_profiler_signal}}",
                ),
                target(f"sum(rate({M['dropped']}{{{JOB}}}[$__rate_interval]))", "dropped items"),
                target(
                    f"sum(rate({M['instr_errors']}{{{JOB}}}[$__rate_interval]))",
                    "instrumentation errors",
                ),
                target(f"sum by (job) ({M['enabled']}{{{JOB}}})", "SDK enabled {{job}}"),
            ],
            "short",
            "Export failures, dropped telemetry and whether the SDK is enabled.",
        ),
    ]
    return dashboard(
        "ipp-overview",
        "Inference Overview",
        "Request rate, errors, latency percentiles, token throughput and concurrency.",
        p,
        variables(),
    )


def latency() -> dict[str, Any]:
    lay = Layout()
    phases = [
        ("Queue", M["queue"]),
        ("Prefill", M["prefill"]),
        ("Time to first token", M["ttft"]),
        ("Generation", M["generation"]),
        ("Inter-token latency", M["itl"]),
    ]
    p: list[dict[str, Any]] = [
        timeseries(
            lay,
            "Mean time per request by phase",
            [
                target(
                    f"sum(rate({m}_sum{{{SEL}}}[$__rate_interval])) / "
                    f"sum(rate({m}_count{{{SEL}}}[$__rate_interval]))",
                    name,
                )
                for name, m in phases
                if m not in (M["ttft"], M["itl"])
            ],
            "s",
            "Average seconds spent in queue, prefill and generation per request. "
            "TTFT overlaps queue+prefill, so it is shown separately.",
            w=24,
            stack=True,
        ),
    ]
    for name, metric in phases:
        p.append(
            timeseries(
                lay,
                f"{name} percentiles",
                quantiles(metric, SEL),
                "s",
                f"p50/p95/p99 of {metric} from histogram buckets.",
                w=12,
                exemplars=metric in (M["ttft"], M["queue"]),
            )
        )
    p.append(
        timeseries(
            lay,
            "Per-request decode throughput",
            quantiles(M["throughput"], SEL),
            "short",
            "Generated tokens per second per request (p50/p95/p99).",
        )
    )
    p.append(
        {
            "type": "heatmap",
            "title": "Request latency distribution",
            "id": lay.pid(),
            "description": "Histogram buckets of end-to-end latency over time.",
            "gridPos": lay.place(24, 9),
            "datasource": PROM,
            "targets": [
                target(
                    f"sum by (le) (increase({M['request']}_bucket{{{SEL}}}[$__rate_interval]))",
                    "{{le}}",
                    format="heatmap",
                )
            ],
            "options": {
                "calculate": False,
                "yAxis": {"unit": "s"},
                "color": {"scheme": "Oranges"},
                "cellGap": 1,
            },
        }
    )
    return dashboard(
        "ipp-latency",
        "Latency Breakdown",
        "Where request time goes: queue, prefill, TTFT, generation and inter-token latency.",
        p,
        variables(),
    )


def memory_cache() -> dict[str, Any]:
    lay = Layout()
    hits = f"sum(rate({M['hits']}{{{SEL}}}[$__rate_interval]))"
    misses = f"sum(rate({M['misses']}{{{SEL}}}[$__rate_interval]))"
    p = [
        timeseries(
            lay,
            "Process resident memory (real)",
            [target(f"max by (job) ({M['rss']}{{{JOB}}})", "{{job}}")],
            "bytes",
            "RSS of each instrumented process, measured with psutil / OS APIs.",
        ),
        timeseries(
            lay,
            "SIMULATED KV-cache memory",
            [
                target(
                    f"max by (job, gen_ai_request_model) ({M['sim_mem']}{{{JOB}}})",
                    "{{job}} {{gen_ai_request_model}} (simulated)",
                )
            ],
            "bytes",
            "SIMULATED: computed by the workload simulator from token counts. "
            "Not a real memory measurement.",
        ),
        timeseries(
            lay,
            "SIMULATED memory pressure",
            [target(f"max by (job) ({M['sim_pressure']}{{{JOB}}})", "{{job}} (simulated)")],
            "percentunit",
            "SIMULATED KV demand / capacity. Values above 100% mean the "
            "simulated demand exceeds capacity.",
        ),
        timeseries(
            lay,
            "Cache hits and misses",
            [target(hits, "hits/s"), target(misses, "misses/s")],
            "ops",
            "Cache lookups per second by result.",
        ),
        timeseries(
            lay,
            "Cache hit ratio",
            [target(f"{hits} / ({hits} + {misses})", "hit ratio")],
            "percentunit",
            "Hits / lookups.",
        ),
        timeseries(
            lay,
            "Cache insertions and evictions",
            [
                target(f"sum(rate({M['insertions']}{{{SEL}}}[$__rate_interval]))", "insertions/s"),
                target(f"sum(rate({M['evictions']}{{{SEL}}}[$__rate_interval]))", "evictions/s"),
            ],
            "ops",
            "High evictions relative to insertions indicate churn.",
        ),
        timeseries(
            lay,
            "Cache operation latency p95",
            quantiles(M["cache_op"], SEL, by="inference_cache_operation")[1:2],
            "s",
            "p95 of lookup/insert/evict duration.",
        ),
        timeseries(
            lay,
            "Cache utilization",
            [target(f"max by (job) ({M['cache_util']}{{{JOB}}})", "{{job}}")],
            "percentunit",
            "Reported by integrations that call Profiler.set_cache_utilization(); empty otherwise.",
        ),
    ]
    return dashboard(
        "ipp-memory-cache",
        "Memory and Cache",
        "Real process memory, clearly labelled simulated KV memory, and cache behaviour.",
        p,
        variables(),
    )


def regressions() -> dict[str, Any]:
    lay = Layout()
    p = [
        stat(
            lay,
            "Metrics over threshold",
            f"sum({M['bench_regressions']})",
            "short",
            "From the last `inference-profiler compare --publish` run.",
            w=6,
            thresholds=[("green", None), ("red", 1)],
        ),
        {
            "type": "table",
            "title": "Baseline vs current",
            "id": lay.pid(),
            "description": "Published by `inference-profiler compare --publish`. Values are ms "
            "(latency), tokens/s or req/s (throughput), bytes or ratios.",
            "gridPos": lay.place(18, 10),
            "datasource": PROM,
            "targets": [
                target(
                    f'max by (scenario, metric) ({M["bench_value"]}{{run="baseline"}})',
                    "",
                    format="table",
                    instant=True,
                    range=False,
                ),
                target(
                    f'max by (scenario, metric) ({M["bench_value"]}{{run="current"}})',
                    "",
                    ref="B",
                    format="table",
                    instant=True,
                    range=False,
                ),
                target(
                    f"max by (scenario, metric, status) ({M['bench_change']})",
                    "",
                    ref="C",
                    format="table",
                    instant=True,
                    range=False,
                ),
            ],
            "transformations": [
                {"id": "merge", "options": {}},
                {
                    "id": "organize",
                    "options": {
                        "excludeByName": {"Time": True},
                        "renameByName": {
                            "Value #A": "baseline",
                            "Value #B": "current",
                            "Value #C": "change %",
                        },
                    },
                },
                {"id": "sortBy", "options": {"sort": [{"field": "change %", "desc": True}]}},
            ],
            "fieldConfig": {
                "defaults": {"decimals": 2},
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "status"},
                        "properties": [
                            {"id": "custom.cellOptions", "value": {"type": "color-text"}},
                            {
                                "id": "mappings",
                                "value": [
                                    {
                                        "type": "value",
                                        "options": {
                                            "regression": {"color": "red", "index": 0},
                                            "improvement": {"color": "green", "index": 1},
                                            "unchanged": {"color": "text", "index": 2},
                                        },
                                    }
                                ],
                            },
                        ],
                    }
                ],
            },
            "options": {"showHeader": True},
        },
        {
            "type": "bargauge",
            "title": "Change vs baseline (%)",
            "id": lay.pid(),
            "description": "Positive = higher than baseline. Whether that is worse depends on "
            "the metric (latency: worse; throughput: better).",
            "gridPos": lay.place(24, 10),
            "datasource": PROM,
            "targets": [
                target(
                    f"max by (scenario, metric) ({M['bench_change']})",
                    "{{scenario}} {{metric}}",
                    instant=True,
                    range=False,
                )
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "min": -100,
                    "max": 100,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "orange", "value": 10},
                            {"color": "red", "value": 25},
                        ],
                    },
                },
                "overrides": [],
            },
            "options": {
                "orientation": "horizontal",
                "displayMode": "basic",
                "reduceOptions": {"calcs": ["lastNotNull"], "values": False},
            },
        },
        {
            "type": "table",
            "title": "Failed regression thresholds",
            "id": lay.pid(),
            "description": "Metrics whose change exceeded the configured threshold.",
            "gridPos": lay.place(24, 8),
            "datasource": PROM,
            "targets": [
                target(
                    f"max by (scenario, metric) ({M['bench_regression']}) == 1",
                    "",
                    format="table",
                    instant=True,
                    range=False,
                )
            ],
            "transformations": [
                {"id": "organize", "options": {"excludeByName": {"Time": True, "Value": True}}}
            ],
        },
        timeseries(
            lay,
            "Live p95 latency by runtime version",
            quantiles(M["request"], SEL, by="inference_runtime_version")[1:2],
            "s",
            "Live comparison when two runtime versions emit telemetry at the same time.",
            w=12,
        ),
        timeseries(
            lay,
            "Live TTFT p95 by runtime version",
            quantiles(M["ttft"], SEL, by="inference_runtime_version")[1:2],
            "s",
            "Live comparison of time to first token by runtime version.",
            w=12,
        ),
    ]
    return dashboard(
        "ipp-regressions",
        "Performance Regressions",
        "Benchmark comparisons published by the regression checker, plus live "
        "per-runtime-version latency.",
        p,
        variables(),
        time_from="now-6h",
    )


def traces() -> dict[str, Any]:
    lay = Layout()

    def trace_table(title: str, query: str, desc: str, h: int = 10) -> dict[str, Any]:
        return {
            "type": "table",
            "title": title,
            "description": desc,
            "id": lay.pid(),
            "gridPos": lay.place(24, h),
            "datasource": TEMPO,
            "targets": [
                {
                    "refId": "A",
                    "datasource": TEMPO,
                    "queryType": "traceql",
                    "query": query,
                    "limit": 50,
                    "tableType": "traces",
                }
            ],
        }

    svc = 'resource.service.name=~"${job:regex}"'
    p = [
        trace_table(
            "Recent inference requests",
            f'{{ {svc} && name = "inference.request" }}',
            "Click a trace ID to see the request and its queue/prefill/generation spans.",
        ),
        trace_table(
            "Slowest requests (> 1 s)",
            f'{{ {svc} && name = "inference.request" && duration > 1s }}',
            "Requests above 1 s end-to-end. Compare their phase breakdown with fast ones.",
        ),
        trace_table(
            "Failed requests",
            f'{{ {svc} && name = "inference.request" && status = error }}',
            "Requests with status=error; open to see the failing phase and error.type.",
        ),
        trace_table(
            "Slow prefill phases (> 50 ms)",
            f'{{ {svc} && name = "inference.prefill" && duration > 50ms }}',
            "Prefill spans over 50 ms; open the parent request for context.",
            h=8,
        ),
        timeseries(
            lay,
            "End-to-end latency with exemplars",
            quantiles(M["request"], SEL)[1:3],
            "s",
            "Exemplar points link directly to the trace that produced them.",
            w=24,
            exemplars=True,
        ),
    ]
    return dashboard(
        "ipp-traces",
        "Distributed Traces",
        "Search and inspect request traces in Tempo; jump from latency to traces.",
        p,
        variables(),
    )


DASHBOARDS = {
    "inference-overview.json": overview,
    "latency-breakdown.json": latency,
    "memory-cache.json": memory_cache,
    "performance-regressions.json": regressions,
    "distributed-traces.json": traces,
}


def render() -> dict[str, str]:
    return {name: json.dumps(fn(), indent=2) + "\n" for name, fn in DASHBOARDS.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if generated files differ")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stale = []
    for name, text in render().items():
        path = OUT / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(name)
        else:
            path.write_text(text)
    if stale:
        print(f"stale dashboards (run scripts/generate_dashboards.py): {stale}", file=sys.stderr)
        return 1
    if not args.check:
        print(f"wrote {len(DASHBOARDS)} dashboards to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Human-readable rendering of reports, analysis findings and comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from inference_profiler.analysis import AnalysisResult
from inference_profiler.regression import Comparison

_LATENCY_LABELS = {
    "end_to_end": "End-to-end",
    "queue": "Queue",
    "prefill": "Prefill",
    "time_to_first_token": "TTFT",
    "generation": "Generation",
    "inter_token": "Inter-token",
}


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def _mib(value: Any) -> str:
    return "-" if value is None else f"{value / 2**20:,.1f} MiB"


def latency_table(scn: Mapping[str, Any]) -> list[str]:
    rows = [
        "| Phase | count | p50 ms | p95 ms | p99 ms | max ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in _LATENCY_LABELS.items():
        d = (scn.get("latency_ms") or {}).get(key)
        if not d:
            rows.append(f"| {label} | 0 | - | - | - | - |")
            continue
        rows.append(
            f"| {label} | {d['count']} | {_fmt(d['p50'])} | {_fmt(d['p95'])} | "
            f"{_fmt(d['p99'])} | {_fmt(d['max'])} |"
        )
    return rows


def render_markdown(
    report: Mapping[str, Any],
    analysis: AnalysisResult | None = None,
    comparison: Comparison | None = None,
) -> str:
    m = report.get("measurement", {})
    lines = [
        "# Inference Profiler benchmark report",
        "",
        f"- Generated: {report.get('generated_at', '-')}",
        f"- SDK version: {report.get('sdk_version', '-')}",
        f"- Git commit: {report.get('git_commit') or 'unavailable'}",
        f"- Clock: **{m.get('clock')}** ({m.get('latency_source')})",
        f"- Workload: **{m.get('workload')}**",
        f"- Environment: {_env(report.get('environment', {}))}",
        "",
    ]
    if report.get("example"):
        lines += [
            "> **EXAMPLE DATA.** This report illustrates the schema and is not a measurement.",
            "",
        ]
    if m.get("notes"):
        lines += [f"> {m['notes']}", ""]

    for scn in report.get("scenarios", []):
        req = scn["requests"]
        thr = scn["throughput"]
        cache = scn["cache"]
        sim = scn["memory"]["simulated_kv_cache"]
        rss = scn["memory"].get("process_rss") or {}
        lines += [
            f"## Scenario `{scn['name']}`",
            "",
            scn.get("description") or "",
            "",
            f"- Model / runtime: `{scn['model']}` / `{scn['runtime_version']}`"
            + (" (regression injected)" if scn.get("regression_injected") else ""),
            f"- Concurrency: {scn['concurrency']}; duration {_fmt(scn['duration_s'], 3)} s",
            f"- Requests: {req['total']} total, {req['completed']} completed, "
            f"{req['failed']} failed, {req['cancelled']} cancelled "
            f"(error rate {req['error_rate']:.2%})",
            f"- Throughput: {_fmt(thr['requests_per_second'])} req/s, "
            f"{_fmt(thr['generated_tokens_per_second'])} generated tokens/s",
            "",
            *latency_table(scn),
            "",
            f"- Cache (simulated): {cache['hits']} hits / {cache['misses']} misses "
            f"(hit ratio {_fmt(cache['hit_ratio'], 3)}), {cache['evictions']} evictions",
            f"- Simulated KV memory: peak {_mib(sim['peak_bytes'])} of "
            f"{_mib(sim['capacity_bytes'])} (pressure {sim['peak_pressure']:.1%}) - *simulated*",
            f"- Process RSS (real): start {_mib(rss.get('start_bytes'))}, "
            f"peak {_mib(rss.get('peak_bytes'))}"
            if rss
            else "- Process RSS: not collected",
            "",
        ]

    if analysis is not None:
        lines += render_findings_markdown(analysis)
    if comparison is not None:
        lines += render_comparison_markdown(comparison)
    overhead = report.get("instrumentation_overhead")
    if overhead:
        lines += [
            "## Instrumentation overhead",
            "",
            f"Measured over {overhead['iterations']} iterations "
            f"({overhead['operations_per_request']}), best of {overhead['repeats']}.",
            "",
            "| Mode | ns / request | added µs / request |",
            "|---|---:|---:|",
        ]
        for mode in ("disabled", "enabled_unsampled", "enabled_sampled"):
            o = overhead[mode]
            lines.append(
                f"| {mode} | {_fmt(o['ns_per_request'], 0)} | "
                f"{_fmt(o.get('added_us_per_request'), 2)} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_findings_markdown(analysis: AnalysisResult) -> list[str]:
    lines = ["## Bottleneck analysis", ""]
    if not analysis.findings:
        lines += ["No rule exceeded its threshold.", ""]
    for f in analysis.findings:
        lines += [
            f"### [{f.severity.upper()}] {f.title}" + (f" - `{f.scenario}`" if f.scenario else ""),
            "",
            f"- Observed: {f.observed}",
            f"- Threshold: {f.threshold}",
            f"- Why it matters: {f.explanation}",
            f"- Investigate: {f.recommendation}",
            "",
        ]
    if analysis.limitations:
        lines += ["### Limitations", "", *[f"- {item}" for item in analysis.limitations], ""]
    return lines


def render_comparison_markdown(c: Comparison) -> list[str]:
    verdict = "PASSED" if c.passed() else f"FAILED ({len(c.regressions)} regressions)"
    lines = [
        f"## Regression check: {verdict}",
        "",
        f"Baseline `{c.baseline_label}` vs current `{c.current_label}`.",
        "",
    ]
    interesting = [d for d in c.deltas if d.status in ("regression", "improvement")]
    if interesting:
        lines += [
            "| Scenario | Metric | Baseline | Current | Change | Threshold | Status |",
            "|---|---|---:|---:|---:|---|---|",
        ]
        for d in interesting:
            lines.append(
                f"| {d.scenario} | {d.metric} | {_fmt(d.baseline)} | {_fmt(d.current)} | "
                f"{_fmt(d.change_pct)}% | {d.threshold} | **{d.status}** |"
            )
    else:
        lines.append("No metric changed beyond its threshold.")
    if c.warnings:
        lines += ["", "Warnings:", *[f"- {w}" for w in c.warnings]]
    return [*lines, ""]


def render_comparison_text(c: Comparison) -> str:
    out = [f"baseline: {c.baseline_label}   current: {c.current_label}"]
    width = max((len(d.metric) for d in c.deltas), default=10)
    for d in c.deltas:
        if d.status in ("regression", "improvement"):
            mark = "REGRESSION " if d.status == "regression" else "improvement"
            out.append(
                f"  {mark}  {d.scenario:<20} {d.metric:<{width}} "
                f"{_fmt(d.baseline):>12} -> {_fmt(d.current):>12} "
                f"({_fmt(d.change_pct)}%, threshold {d.threshold})"
            )
    for w in c.warnings:
        out.append(f"  warning: {w}")
    unchanged = sum(1 for d in c.deltas if d.status == "unchanged")
    out.append(
        f"{len(c.regressions)} regression(s), {len(c.improvements)} improvement(s), "
        f"{unchanged} unchanged"
    )
    out.append("RESULT: " + ("PASS" if c.passed() else "FAIL"))
    return "\n".join(out)


def render_findings_text(analysis: AnalysisResult) -> str:
    if not analysis.findings:
        out = ["No findings: no rule exceeded its threshold."]
    else:
        out = []
        for f in analysis.findings:
            out += [
                f"[{f.severity.upper():8}] {f.scenario or '-'}: {f.title}",
                f"           observed:  {f.observed}",
                f"           threshold: {f.threshold}",
                f"           investigate: {f.recommendation}",
            ]
    for item in analysis.limitations:
        out.append(f"limitation: {item}")
    return "\n".join(out)


def render_summary_text(report: Mapping[str, Any]) -> str:
    out = []
    for scn in report.get("scenarios", []):
        e2e = scn["latency_ms"].get("end_to_end") or {}
        ttft = scn["latency_ms"].get("time_to_first_token") or {}
        out.append(
            f"{scn['name']:<24} reqs={scn['requests']['total']:<5} "
            f"e2e p50/p95/p99={_fmt(e2e.get('p50'), 1)}/{_fmt(e2e.get('p95'), 1)}/"
            f"{_fmt(e2e.get('p99'), 1)} ms  ttft p95={_fmt(ttft.get('p95'), 1)} ms  "
            f"tok/s={_fmt(scn['throughput']['generated_tokens_per_second'], 0)}"
        )
    return "\n".join(out)


def _env(env: Mapping[str, Any]) -> str:
    return (
        f"Python {env.get('python', '?')} on {env.get('platform', '?')} "
        f"({env.get('machine', '?')}, {env.get('cpu_count', '?')} CPUs)"
    )

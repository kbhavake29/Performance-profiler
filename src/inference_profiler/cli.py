"""``inference-profiler`` command-line interface.

Exit codes: 0 success, 1 check failed (regression or findings at --fail-on
severity), 2 invalid input or usage error.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import platform
import socket
import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import jsonschema

from inference_profiler.analysis import (
    AnalysisThresholds,
    analyze_report,
    prometheus_query_fn,
    report_from_prometheus,
)
from inference_profiler.benchmark import (
    build_report,
    measure_instrumentation_overhead,
    read_report,
    run_benchmark,
    write_report,
)
from inference_profiler.config import ENV_VARIABLES, ConfigError, ProfilerConfig
from inference_profiler.instrumentation import Profiler
from inference_profiler.regression import (
    RegressionThresholds,
    compare_reports,
    publish_comparison,
)
from inference_profiler.reporting import (
    render_comparison_text,
    render_findings_text,
    render_markdown,
    render_summary_text,
)
from inference_profiler.simulator import (
    RegressionInjection,
    ScenarioConfig,
    ScenarioError,
    build_plan,
    load_scenarios,
    run_wall_clock,
)
from inference_profiler.version import __version__

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2
_SEVERITY = {"info": 0, "warning": 1, "critical": 2}


class UsageError(Exception):
    pass


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr, flush=True)


# ======================================================================= doctor


def cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    check("python", sys.version_info >= (3, 11), platform.python_version())
    for dist in (
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-otlp-proto-http",
    ):
        try:
            check(dist, True, metadata.version(dist))
        except metadata.PackageNotFoundError:
            check(dist, False, "not installed")
    try:
        cfg = ProfilerConfig.from_env()
        check("configuration", True, "valid")
    except ConfigError as exc:
        check("configuration", False, str(exc))
        cfg = None
    env = {k: os.environ[k] for k in ENV_VARIABLES if k in os.environ}
    if cfg is not None:
        check("instrumentation", cfg.enabled, "enabled" if cfg.enabled else "disabled")
        parsed = urlparse(cfg.otlp_endpoint)
        host, port = (
            parsed.hostname or "localhost",
            parsed.port or (443 if parsed.scheme == "https" else 80),
        )
        if cfg.exporter == "otlp":
            try:
                with socket.create_connection((host, port), timeout=args.timeout):
                    check("collector reachable", True, f"{host}:{port} accepts connections")
            except OSError as exc:
                check(
                    "collector reachable",
                    False,
                    f"{host}:{port} unreachable ({exc}); start it with `make stack-up`",
                )
    build_dir = Path(args.cpp_build_dir)
    demo = next(
        (
            p
            for p in (
                build_dir / "cpp_inference_demo",
                build_dir / "Release" / "cpp_inference_demo.exe",
            )
            if p.exists()
        ),
        None,
    )
    check(
        "native demo built",
        demo is not None,
        str(demo) if demo else f"not found in {build_dir} (run `make build`)",
    )

    result = {
        "version": __version__,
        "config": _config_dict(cfg) if cfg else None,
        "environment_overrides": env,
        "checks": checks,
    }
    if args.json:
        _print(json.dumps(result, indent=2))
    else:
        _print(f"inference-profiler {__version__}")
        for c in checks:
            _print(f"  [{'ok' if c['ok'] else '!!'}] {c['check']}: {c['detail']}")
        if cfg:
            _print("effective configuration (defaults < environment < explicit):")
            for k, v in _config_dict(cfg).items():
                _print(f"  {k} = {v}")
    essential = {"python", "opentelemetry-sdk", "configuration"}
    return EXIT_OK if all(c["ok"] for c in checks if c["check"] in essential) else EXIT_FAILED


def _config_dict(cfg: ProfilerConfig) -> dict[str, Any]:
    out = dataclasses.asdict(cfg)
    out["resource_attributes"] = dict(cfg.resource_attributes)
    return out


# ========================================================================= demo

DEMO_SCENARIO = ScenarioConfig(
    name="demo",
    description="Short wall-clock demo emitting real telemetry for a simulated model.",
    num_requests=60,
    concurrency=6,
    arrival_rate_rps=20.0,
    failure_rate=0.03,
    cancel_rate=0.02,
)


def _profiler_for_run(args: argparse.Namespace, service: str) -> Profiler:
    overrides: dict[str, Any] = {}
    if args.endpoint:
        overrides["otlp_endpoint"] = args.endpoint
    if not os.environ.get("OTEL_SERVICE_NAME"):
        overrides["service_name"] = service
    if getattr(args, "no_export", False):
        overrides["exporter"] = "none"
    cfg = ProfilerConfig.from_env(**overrides)
    return Profiler(cfg)


def cmd_demo(args: argparse.Namespace) -> int:
    scenario = DEMO_SCENARIO
    if args.scenario:
        found = load_scenarios(args.scenario, [args.name] if args.name else None)
        scenario = found[0]
    changes: dict[str, Any] = {}
    if args.requests:
        changes["num_requests"] = args.requests
    if args.concurrency:
        changes["concurrency"] = args.concurrency
    if args.runtime_version:
        changes["runtime_version"] = args.runtime_version
    if changes:
        scenario = dataclasses.replace(scenario, **changes)

    profiler = _profiler_for_run(args, "inference-profiler-demo")
    cfg = profiler.config
    _print(
        f"demo: {scenario.num_requests} simulated requests, concurrency {scenario.concurrency}, "
        f"exporting to {cfg.otlp_endpoint if cfg.exporter == 'otlp' else cfg.exporter} "
        f"as service '{cfg.service_name}'"
    )
    run = asyncio.run(run_wall_clock(scenario, profiler, build_plan(scenario)))
    report = build_report([run], clock="wall", label="demo")
    _print(render_summary_text(report))
    flushed = profiler.force_flush(cfg.export_timeout_ms)
    profiler.shutdown()
    failures = profiler.export_failures
    if cfg.exporter == "otlp":
        if failures or not flushed:
            _print(
                f"warning: {failures} export attempt(s) failed - is the collector running at "
                f"{cfg.otlp_endpoint}? (telemetry was dropped; the workload was unaffected)"
            )
        else:
            _print(
                "telemetry exported. Open Grafana at http://localhost:3000 "
                "(dashboard folder 'Inference Profiler')."
            )
    if args.output:
        write_report(report, args.output)
        _print(f"report written to {args.output}")
    return EXIT_OK


# ==================================================================== benchmark


def _parse_injection(items: Sequence[str]) -> RegressionInjection | None:
    if not items:
        return None
    allowed = {f.name for f in dataclasses.fields(RegressionInjection)}
    kwargs: dict[str, float] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or key not in allowed:
            raise UsageError(f"--inject expects KEY=VALUE with KEY in {sorted(allowed)}")
        kwargs[key] = float(raw)
    return RegressionInjection(**kwargs)


def cmd_benchmark(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(args.scenario, args.name or None)
    injection = _parse_injection(args.inject or [])
    updated = []
    for s in scenarios:
        changes: dict[str, Any] = {}
        if injection is not None:
            changes["regression"] = injection
        if args.runtime_version:
            changes["runtime_version"] = args.runtime_version
        if args.requests:
            changes["num_requests"] = args.requests
        if args.time_scale:
            changes["time_scale"] = args.time_scale
        updated.append(dataclasses.replace(s, **changes) if changes else s)

    profiler = None
    if args.clock == "wall":
        profiler = _profiler_for_run(args, "inference-profiler-benchmark")
    try:
        report = run_benchmark(
            updated,
            clock=args.clock,
            profiler=profiler,
            measure_overhead=args.overhead,
            label=args.label,
            progress=lambda msg: print(msg, file=sys.stderr, flush=True),
        )
    finally:
        if profiler is not None:
            profiler.shutdown()
    write_report(report, args.output)
    _print(render_summary_text(report))
    _print(f"report written to {args.output} (schema-validated)")
    return EXIT_OK


# ====================================================================== analyze


def cmd_analyze(args: argparse.Namespace) -> int:
    thresholds = AnalysisThresholds.from_file(args.thresholds) if args.thresholds else None
    if args.prometheus_url:
        report = report_from_prometheus(
            prometheus_query_fn(args.prometheus_url), window=args.window, service=args.service
        )
    elif args.input:
        report = read_report(args.input)
    else:
        raise UsageError("analyze needs --input FILE or --prometheus-url URL")
    result = analyze_report(report, thresholds)
    if args.json:
        _print(json.dumps(result.to_dict(), indent=2))
    else:
        _print(render_findings_text(result))
    if (
        args.fail_on
        and result.max_severity is not None
        and (_SEVERITY[result.max_severity] >= _SEVERITY[args.fail_on])
    ):
        return EXIT_FAILED
    return EXIT_OK


# ====================================================================== compare


def _thresholds(args: argparse.Namespace) -> RegressionThresholds:
    t = (
        RegressionThresholds.from_file(args.thresholds)
        if args.thresholds
        else RegressionThresholds()
    )
    return t.with_cli_overrides(args.threshold or [])


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = read_report(args.baseline)
    current = read_report(args.current)
    comparison = compare_reports(baseline, current, _thresholds(args))
    _print(render_comparison_text(comparison))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(comparison.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        _print(f"comparison written to {args.output}")
    if args.publish:
        cfg = ProfilerConfig.from_env(**({"otlp_endpoint": args.endpoint} if args.endpoint else {}))
        ok = publish_comparison(comparison, cfg.otlp_endpoint, "inference-profiler-regression")
        _print(
            "comparison published to the collector"
            if ok
            else f"warning: could not publish comparison to {cfg.otlp_endpoint}"
        )
    return EXIT_OK if comparison.passed(args.fail_on_missing) else EXIT_FAILED


# ================================================================ export-report


def cmd_export_report(args: argparse.Namespace) -> int:
    report = read_report(args.input)
    thresholds = AnalysisThresholds.from_file(args.thresholds) if args.thresholds else None
    analysis = analyze_report(report, thresholds)
    comparison = None
    if args.baseline:
        comparison = compare_reports(
            read_report(args.baseline),
            report,
            RegressionThresholds().with_cli_overrides(args.threshold or []),
        )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(report, analysis, comparison), encoding="utf-8")
    _print(f"markdown report written to {out}")
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    read_report(args.input)
    _print(f"{args.input}: valid benchmark report (schema 1.0)")
    return EXIT_OK


def cmd_overhead(args: argparse.Namespace) -> int:
    result = measure_instrumentation_overhead(iterations=args.iterations, repeats=args.repeats)
    _print(
        f"instrumentation overhead ({result['operations_per_request']}, "
        f"{args.iterations} iterations, best of {args.repeats}):"
    )
    for mode in ("disabled", "enabled_unsampled", "enabled_sampled"):
        r = result[mode]
        extra = r.get("added_us_per_request")
        _print(
            f"  {mode:<18} {r['ns_per_request'] / 1000:9.2f} µs/request"
            + (f"   (+{extra:.2f} µs vs disabled)" if extra is not None else "")
        )
    _print(f"  record_token() in the decode loop: {result['record_token_ns']:.0f} ns/token")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _print(f"written to {args.output}")
    return EXIT_OK


# ======================================================================= parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inference-profiler",
        description="Instrument, benchmark and analyse LLM inference performance.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    d = sub.add_parser(
        "doctor", help="check installation, configuration and collector reachability"
    )
    d.add_argument("--json", action="store_true")
    d.add_argument("--timeout", type=float, default=1.0, help="connect timeout in seconds")
    d.add_argument("--cpp-build-dir", default="build", help="where to look for the native demo")
    d.set_defaults(func=cmd_doctor)

    demo = sub.add_parser("demo", help="run a short simulated workload that exports real telemetry")
    demo.add_argument(
        "--endpoint", help="OTLP/HTTP base URL (default: env or http://localhost:4318)"
    )
    demo.add_argument("--scenario", help="scenario YAML to use instead of the built-in demo")
    demo.add_argument("--name", help="scenario name within --scenario")
    demo.add_argument("--requests", type=int)
    demo.add_argument("--concurrency", type=int)
    demo.add_argument("--runtime-version")
    demo.add_argument("--no-export", action="store_true", help="run without exporting telemetry")
    demo.add_argument("--output", help="also write a JSON report here")
    demo.set_defaults(func=cmd_demo)

    b = sub.add_parser("benchmark", help="run benchmark scenarios and write a JSON report")
    b.add_argument("--scenario", required=True, help="scenario YAML file")
    b.add_argument("--name", action="append", help="run only this scenario (repeatable)")
    b.add_argument(
        "--clock",
        choices=("virtual", "wall"),
        default="virtual",
        help="virtual: deterministic model (default); wall: real sleeps + telemetry",
    )
    b.add_argument("--output", default="benchmarks/results/latest.json")
    b.add_argument("--label", help="free-form label stored in the report")
    b.add_argument("--runtime-version", help="override runtime_version for all scenarios")
    b.add_argument("--requests", type=int, help="override num_requests for all scenarios")
    b.add_argument("--time-scale", type=float, help="override time_scale (wall clock only)")
    b.add_argument(
        "--inject",
        action="append",
        metavar="KEY=VALUE",
        help="inject a regression, e.g. prefill_multiplier=1.5 (repeatable)",
    )
    b.add_argument("--overhead", action="store_true", help="also measure instrumentation overhead")
    b.add_argument("--endpoint", help="OTLP endpoint for --clock wall")
    b.add_argument("--no-export", action="store_true", help="--clock wall without exporting")
    b.set_defaults(func=cmd_benchmark)

    a = sub.add_parser("analyze", help="detect bottlenecks in a report or live Prometheus data")
    a.add_argument("--input", help="benchmark report JSON")
    a.add_argument("--prometheus-url", help="analyse live metrics, e.g. http://localhost:9090")
    a.add_argument("--window", default="15m", help="PromQL range for --prometheus-url")
    a.add_argument("--service", help="restrict --prometheus-url to this service (job label)")
    a.add_argument("--thresholds", help="YAML file overriding analysis thresholds")
    a.add_argument("--json", action="store_true")
    a.add_argument(
        "--fail-on",
        choices=("info", "warning", "critical"),
        help="exit 1 if a finding at or above this severity exists",
    )
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("compare", help="compare a report with a baseline; exit 1 on regression")
    c.add_argument("--baseline", required=True)
    c.add_argument("--current", required=True)
    c.add_argument("--thresholds", help="YAML file with regression thresholds")
    c.add_argument(
        "--threshold",
        action="append",
        metavar="KEY=VALUE",
        help="e.g. latency_pct=5 or latency_ms.queue.p99=50 (repeatable)",
    )
    c.add_argument("--output", help="write the comparison as JSON")
    c.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="fail when a baseline scenario is missing from the current report",
    )
    c.add_argument(
        "--publish",
        action="store_true",
        help="export results as OTLP metrics for the Grafana regressions dashboard",
    )
    c.add_argument("--endpoint", help="OTLP endpoint for --publish")
    c.set_defaults(func=cmd_compare)

    e = sub.add_parser("export-report", help="render a Markdown report with analysis")
    e.add_argument("--input", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--baseline", help="include a regression comparison against this report")
    e.add_argument("--threshold", action="append", metavar="KEY=VALUE")
    e.add_argument("--thresholds", help="YAML file overriding analysis thresholds")
    e.set_defaults(func=cmd_export_report)

    v = sub.add_parser("validate", help="validate a report against the JSON schema")
    v.add_argument("--input", required=True)
    v.set_defaults(func=cmd_validate)

    o = sub.add_parser("overhead", help="measure instrumentation overhead (enabled vs disabled)")
    o.add_argument("--iterations", type=int, default=5_000)
    o.add_argument("--repeats", type=int, default=5)
    o.add_argument("--output", help="write results as JSON")
    o.set_defaults(func=cmd_overhead)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        code: int = args.func(args)
        return code
    except (UsageError, ConfigError, ScenarioError, ValueError) as exc:
        _err(str(exc))
        return EXIT_USAGE
    except jsonschema.ValidationError as exc:
        _err(
            f"report failed schema validation: {exc.message} at "
            f"{'/'.join(str(p) for p in exc.absolute_path) or '<root>'}"
        )
        return EXIT_USAGE
    except FileNotFoundError as exc:
        _err(f"file not found: {exc.filename}")
        return EXIT_USAGE
    except json.JSONDecodeError as exc:
        _err(f"invalid JSON: {exc}")
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())

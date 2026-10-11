from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_profiler.cli import main

from .conftest import ROOT

SCENARIOS = str(ROOT / "examples" / "benchmark_scenarios.yaml")


def run(*args: str) -> int:
    return main(list(args))


def test_help_lists_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        run("--help")
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for cmd in (
        "doctor",
        "demo",
        "benchmark",
        "analyze",
        "compare",
        "export-report",
        "validate",
        "overhead",
    ):
        assert cmd in out


def test_benchmark_compare_and_report_flow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = tmp_path / "baseline.json"
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    assert (
        run(
            "benchmark",
            "--scenario",
            SCENARIOS,
            "--name",
            "baseline",
            "--requests",
            "80",
            "--output",
            str(base),
        )
        == 0
    )
    assert (
        run(
            "benchmark",
            "--scenario",
            SCENARIOS,
            "--name",
            "baseline",
            "--requests",
            "80",
            "--output",
            str(good),
        )
        == 0
    )
    assert (
        run(
            "benchmark",
            "--scenario",
            SCENARIOS,
            "--name",
            "baseline",
            "--requests",
            "80",
            "--inject",
            "prefill_multiplier=1.5",
            "--runtime-version",
            "sim-1.1",
            "--output",
            str(bad),
        )
        == 0
    )

    assert run("compare", "--baseline", str(base), "--current", str(good)) == 0
    out_json = tmp_path / "cmp.json"
    assert (
        run("compare", "--baseline", str(base), "--current", str(bad), "--output", str(out_json))
        == 1
    )
    cmp = json.loads(out_json.read_text())
    assert cmp["passed"] is False and cmp["regression_count"] > 0
    assert (
        run(
            "compare",
            "--baseline",
            str(base),
            "--current",
            str(bad),
            "--threshold",
            "latency_pct=500",
        )
        == 0
    )

    md = tmp_path / "report.md"
    assert (
        run("export-report", "--input", str(bad), "--baseline", str(base), "--output", str(md)) == 0
    )
    text = md.read_text()
    assert "Regression check: FAILED" in text and "Bottleneck analysis" in text
    capsys.readouterr()

    assert run("analyze", "--input", str(base), "--json") == 0
    analysis = json.loads(capsys.readouterr().out)
    assert analysis["scenarios_analyzed"] == 1


def test_analyze_fail_on(tmp_path: Path) -> None:
    out = tmp_path / "c.json"
    assert (
        run("benchmark", "--scenario", SCENARIOS, "--name", "congestion", "--output", str(out)) == 0
    )
    assert run("analyze", "--input", str(out), "--fail-on", "critical") == 1
    assert run("analyze", "--input", str(out)) == 0


def test_usage_errors_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("validate", "--input", str(tmp_path / "missing.json")) == 2
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": "1.0"}')
    assert run("validate", "--input", str(bad)) == 2
    assert "schema validation" in capsys.readouterr().err
    assert run("benchmark", "--scenario", SCENARIOS, "--name", "nope") == 2
    assert run("benchmark", "--scenario", SCENARIOS, "--inject", "warp=9") == 2
    assert run("analyze") == 2


def test_doctor_json(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
    code = run("doctor", "--json", "--timeout", "0.2")
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    checks = {c["check"]: c for c in data["checks"]}
    assert checks["configuration"]["ok"] is True
    assert checks["collector reachable"]["ok"] is False
    assert data["environment_overrides"]["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://127.0.0.1:9"


def test_doctor_reports_invalid_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("INFERENCE_PROFILER_SAMPLE_RATE", "7")
    assert run("doctor") == 1
    assert "sample_rate" in capsys.readouterr().out


def test_demo_without_export(tmp_path: Path) -> None:
    out = tmp_path / "demo.json"
    assert run("demo", "--no-export", "--requests", "8", "--output", str(out)) == 0
    assert json.loads(out.read_text())["measurement"]["clock"] == "wall"

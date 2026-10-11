"""End-to-end tests against a running local stack (`make stack-up` or the native stack).

Run with:  pytest tests/integration -m integration
Every test is skipped (not failed) when the collector is unreachable, so the
unit-test job never depends on Docker.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import pytest

from inference_profiler import Profiler, ProfilerConfig

ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
PROMETHEUS = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
TEMPO = os.environ.get("TEMPO_URL", "http://localhost:3200")
GRAFANA = os.environ.get("GRAFANA_URL", "http://localhost:3000")
GRAFANA_AUTH = os.environ.get("GRAFANA_AUTH", "admin:admin")
CPP_DEMO = ROOT / os.environ.get("BUILD_DIR", "build") / "cpp_inference_demo"

pytestmark = pytest.mark.integration


def _reachable(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=1):
            return True
    except OSError:
        return False


if not _reachable(COLLECTOR):
    pytest.skip(
        f"collector not reachable at {COLLECTOR}; start the stack first", allow_module_level=True
    )


def _get(url: str, auth: str | None = None) -> Any:
    req = urllib.request.Request(url)
    if auth:
        import base64

        req.add_header("Authorization", "Basic " + base64.b64encode(auth.encode()).decode())
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _prom(expr: str) -> list[dict[str, Any]]:
    url = f"{PROMETHEUS}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    result: list[dict[str, Any]] = _get(url)["data"]["result"]
    return result


def _tempo_search(query: str) -> list[dict[str, Any]]:
    now = int(time.time())
    params = urllib.parse.urlencode({"q": query, "start": now - 3600, "end": now, "limit": 50})
    traces: list[dict[str, Any]] = _get(f"{TEMPO}/api/search?{params}").get("traces", [])
    return traces


def _eventually(fn: Any, timeout: float = 60.0, interval: float = 2.0) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = fn()
        if last:
            return last
        time.sleep(interval)
    return last


@pytest.fixture(scope="module")
def python_service() -> str:
    """Emit a few requests from the Python SDK under a unique service name."""
    service = f"it-python-{uuid.uuid4().hex[:8]}"
    cfg = ProfilerConfig.from_env(
        service_name=service,
        otlp_endpoint=COLLECTOR,
        export_interval_ms=1000,
        runtime_version="it-1.0",
    )
    with Profiler(cfg) as profiler:
        for i in range(5):
            with profiler.inference_request(model="it-model", prompt_tokens=32) as req:
                with req.queue():
                    time.sleep(0.001)
                with req.prefill():
                    time.sleep(0.002)
                with req.generation():
                    for _ in range(4):
                        time.sleep(0.001)
                        req.record_token()
                req.record_generation(4)
            if i == 4:
                assert profiler.force_flush(5000)
        assert profiler.export_failures == 0
    return service


@pytest.fixture(scope="module")
def cpp_service() -> str:
    if not CPP_DEMO.exists():
        pytest.skip(f"{CPP_DEMO} not built (run `make build`)")
    service = f"it-cpp-{uuid.uuid4().hex[:8]}"
    env = {**os.environ, "OTEL_SERVICE_NAME": service, "OTEL_EXPORTER_OTLP_ENDPOINT": COLLECTOR}
    out = subprocess.run(
        [str(CPP_DEMO), "--requests", "8"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert "export_failures=0" in out.stdout, out.stdout + out.stderr
    return service


def test_python_metrics_reach_prometheus(python_service: str) -> None:
    result = _eventually(lambda: _prom(f'sum(inference_requests_total{{job="{python_service}"}})'))
    assert result and float(result[0]["value"][1]) == 5
    hist = _eventually(
        lambda: _prom(f'sum(inference_prefill_duration_seconds_count{{job="{python_service}"}})')
    )
    assert hist and float(hist[0]["value"][1]) == 5


def test_python_traces_reach_tempo(python_service: str) -> None:
    traces = _eventually(
        lambda: _tempo_search(
            f'{{resource.service.name="{python_service}" && name="inference.request"}}'
        )
    )
    assert len(traces) == 5
    trace = _get(f"{TEMPO}/api/v2/traces/{traces[0]['traceID']}")
    spans = [
        s
        for rs in trace.get("trace", trace)["resourceSpans"]
        for ss in rs["scopeSpans"]
        for s in ss["spans"]
    ]
    names = {s["name"] for s in spans}
    assert {
        "inference.request",
        "inference.queue",
        "inference.prefill",
        "inference.generation",
        "inference.time_to_first_token",
    } <= names


def test_cpp_metrics_and_traces(cpp_service: str) -> None:
    result = _eventually(lambda: _prom(f'sum(inference_requests_total{{job="{cpp_service}"}})'))
    assert result and float(result[0]["value"][1]) == 8
    traces = _eventually(
        lambda: _tempo_search(
            f'{{resource.service.name="{cpp_service}" && name="inference.request"}}'
        )
    )
    assert len(traces) == 8


def test_grafana_provisioning() -> None:
    for uid in ("prometheus", "tempo"):
        health = _get(f"{GRAFANA}/api/datasources/uid/{uid}/health", GRAFANA_AUTH)
        assert health["status"] == "OK", health
    dashboards = _get(f"{GRAFANA}/api/search?tag=inference-profiler", GRAFANA_AUTH)
    assert {d["uid"] for d in dashboards} >= {
        "ipp-overview",
        "ipp-latency",
        "ipp-memory-cache",
        "ipp-regressions",
        "ipp-traces",
    }


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "inference_profiler.cli", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)


def test_benchmark_and_regression_checker(tmp_path: Path) -> None:
    scenarios = str(ROOT / "examples" / "benchmark_scenarios.yaml")
    base, good, bad = tmp_path / "base.json", tmp_path / "good.json", tmp_path / "bad.json"
    for out, extra in (
        (base, []),
        (good, []),
        (bad, ["--inject", "prefill_multiplier=1.5", "--runtime-version", "sim-1.1"]),
    ):
        r = _cli(
            "benchmark", "--scenario", scenarios, "--name", "baseline", "--output", str(out), *extra
        )
        assert r.returncode == 0, r.stderr
        assert _cli("validate", "--input", str(out)).returncode == 0
    assert _cli("compare", "--baseline", str(base), "--current", str(good)).returncode == 0
    regress = _cli(
        "compare",
        "--baseline",
        str(base),
        "--current",
        str(bad),
        "--publish",
        "--endpoint",
        COLLECTOR,
    )
    assert regress.returncode == 1
    assert "latency_ms.prefill.p95" in regress.stdout
    published = _eventually(lambda: _prom("max(inference_benchmark_regressions)"))
    assert published and float(published[0]["value"][1]) > 0

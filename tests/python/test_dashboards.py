"""Dashboards must only query metrics the SDK actually emits."""

from __future__ import annotations

import json
import re
import subprocess
import sys

import yaml

from inference_profiler import metrics as m
from inference_profiler.config import DEFAULT_OTLP_ENDPOINT

from .conftest import ROOT

DASHBOARDS = ROOT / "monitoring" / "grafana" / "dashboards"

# OpenTelemetry -> Prometheus translation used by the collector's prometheus exporter
# (verified against otelcol-contrib 0.161.0): dots become underscores, units become
# suffixes, monotonic counters gain _total, histograms expose _bucket/_sum/_count.
UNIT_SUFFIX = {
    "s": "_seconds",
    "By": "_bytes",
    "1": "_ratio",
    "{token}/s": "_per_second",
    "%": "_percent",
}


def prometheus_names() -> set[str]:
    hist = {
        m.REQUEST_DURATION: "s",
        m.QUEUE_DURATION: "s",
        m.PREFILL_DURATION: "s",
        m.TIME_TO_FIRST_TOKEN: "s",
        m.INTER_TOKEN_LATENCY: "s",
        m.GENERATION_DURATION: "s",
        m.PHASE_DURATION: "s",
        m.GENERATION_THROUGHPUT: "{token}/s",
        m.CACHE_OPERATION_DURATION: "s",
        m.EXPORT_DURATION: "s",
    }
    counters = [
        m.REQUESTS,
        m.TOKENS,
        m.CACHE_HITS,
        m.CACHE_MISSES,
        m.CACHE_INSERTIONS,
        m.CACHE_EVICTIONS,
        m.EXPORT_ERRORS,
        m.TELEMETRY_DROPPED,
        m.INSTRUMENTATION_ERRORS,
    ]
    gauges = {
        m.REQUESTS_IN_FLIGHT: "",
        m.REQUESTS_ACTIVE: "",
        m.QUEUE_DEPTH: "",
        m.CACHE_UTILIZATION: "1",
        m.PROCESS_MEMORY_USAGE: "By",
        m.SIMULATED_MEMORY_USAGE: "By",
        m.SIMULATED_MEMORY_PRESSURE: "1",
        m.PROFILER_ENABLED: "",
        # published by `inference-profiler compare --publish`
        "inference.benchmark.value": "",
        "inference.benchmark.change": "%",
        "inference.benchmark.regression": "",
        "inference.benchmark.regressions": "",
    }

    def base(name: str, unit: str) -> str:
        return name.replace(".", "_") + UNIT_SUFFIX.get(unit, "")

    names: set[str] = set()
    for name, unit in hist.items():
        b = base(name, unit)
        names |= {f"{b}_bucket", f"{b}_sum", f"{b}_count"}
    names |= {base(n, "") + "_total" for n in counters}
    names |= {base(n, u) for n, u in gauges.items()}
    return names


def referenced_metrics() -> set[str]:
    found: set[str] = set()
    for path in DASHBOARDS.glob("*.json"):
        for panel in json.loads(path.read_text())["panels"]:
            for t in panel.get("targets", []):
                # drop label matchers and grouping clauses so only metric names remain
                expr = re.sub(r"\{[^}]*\}|by \([^)]*\)", "", t.get("expr", ""))
                found |= set(re.findall(r"\b(inference_[a-z_]+|process_memory_[a-z_]+)", expr))
    return found


def test_dashboards_reference_only_emitted_metrics() -> None:
    unknown = referenced_metrics() - prometheus_names()
    assert not unknown, f"dashboards reference metrics the SDK does not emit: {sorted(unknown)}"


def test_all_five_dashboards_exist_with_unique_uids() -> None:
    uids = {json.loads(p.read_text())["uid"] for p in DASHBOARDS.glob("*.json")}
    assert uids == {
        "ipp-overview",
        "ipp-latency",
        "ipp-memory-cache",
        "ipp-regressions",
        "ipp-traces",
    }


def test_generated_dashboards_are_up_to_date() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_dashboards.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_datasource_uids_match_dashboards() -> None:
    ds = yaml.safe_load(
        (
            ROOT / "monitoring" / "grafana" / "provisioning" / "datasources" / "datasources.yaml"
        ).read_text()
    )
    uids = {d["uid"] for d in ds["datasources"]}
    for path in DASHBOARDS.glob("*.json"):
        for panel in json.loads(path.read_text())["panels"]:
            if "datasource" in panel:
                assert panel["datasource"]["uid"] in uids


def test_ports_consistent_across_configs() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    collector_ports = {p.split(":")[-1] for p in compose["services"]["otel-collector"]["ports"]}
    assert {"4317", "4318", "8889", "13133"} <= collector_ports
    prom = yaml.safe_load((ROOT / "monitoring" / "prometheus.yml").read_text())
    targets = {
        t for job in prom["scrape_configs"] for sc in job["static_configs"] for t in sc["targets"]
    }
    assert "otel-collector:8889" in targets
    col = (ROOT / "monitoring" / "otel-collector-config.yaml").read_text()
    assert "0.0.0.0:4318" in col and "0.0.0.0:8889" in col and "tempo:4317" in col
    assert DEFAULT_OTLP_ENDPOINT.endswith(":4318")

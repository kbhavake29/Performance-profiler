#!/usr/bin/env python3
"""Evaluate every PromQL query in the provisioned dashboards against a live Prometheus.

Reports queries that fail to parse and queries that return no data. Run after
`make demo` so the stack contains telemetry:

    python scripts/check_dashboard_queries.py --prometheus http://localhost:9090

Exit code 1 if any query errors, or (with --require-data) returns no series.
Panels that depend on optional data (cache utilization, benchmark comparisons)
are listed under --optional and only reported.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DASHBOARDS = Path(__file__).resolve().parents[1] / "monitoring" / "grafana" / "dashboards"
OPTIONAL = (
    "inference_cache_utilization",
    "inference_cache_evictions",
    "inference_benchmark_regression)",
    "== 1",
)


def substitute(expr: str) -> str:
    for var in ("$job", "$model", "$runtime"):
        expr = expr.replace(var, ".*")
    return (
        expr.replace("$__rate_interval", "1m")
        .replace("$__range", "1h")
        .replace("${job:regex}", ".*")
    )


def query(base: str, expr: str) -> tuple[str, int]:
    url = f"{base.rstrip('/')}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return f"error: {json.loads(exc.read()).get('error')}", 0
    return payload["status"], len(payload["data"]["result"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--require-data", action="store_true")
    args = parser.parse_args()
    errors = empty = total = 0
    for path in sorted(DASHBOARDS.glob("*.json")):
        dash = json.loads(path.read_text())
        for panel in dash["panels"]:
            for t in panel.get("targets", []):
                if "expr" not in t:
                    continue
                total += 1
                status, n = query(args.prometheus, substitute(t["expr"]))
                optional = any(o in t["expr"] for o in OPTIONAL)
                if status != "success":
                    errors += 1
                    print(f"ERROR  {path.name} / {panel['title']}: {status}\n       {t['expr']}")
                elif n == 0:
                    tag = "optional-empty" if optional else "EMPTY"
                    empty += 0 if optional else 1
                    print(f"{tag:6} {path.name} / {panel['title']}: {t['expr'][:110]}")
    print(f"{total} queries checked: {errors} errors, {empty} empty (excluding optional)")
    if errors or (args.require_data and empty):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

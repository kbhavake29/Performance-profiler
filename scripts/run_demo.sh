#!/usr/bin/env bash
# Send telemetry from the Python example, the native C++ example and the CLI demo
# to the collector (default http://localhost:4318).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-.venv/bin/python}"
BUILD_DIR="${BUILD_DIR:-build}"
export OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-http://localhost:4318}"
export INFERENCE_PROFILER_EXPORT_INTERVAL="${INFERENCE_PROFILER_EXPORT_INTERVAL:-1000}"

echo "==> Python example (service: python-inference-demo)"
"$PY" examples/python_inference_demo.py --requests "${REQUESTS:-40}"

if [[ -x "$BUILD_DIR/cpp_inference_demo" ]]; then
  echo "==> C++ example (service: cpp-inference-demo)"
  "$BUILD_DIR/cpp_inference_demo" --requests "${REQUESTS:-40}"
else
  echo "==> C++ example skipped: $BUILD_DIR/cpp_inference_demo not built (run 'make build')" >&2
fi

echo "==> CLI simulated workload (service: inference-profiler-demo)"
"$PY" -m inference_profiler.cli demo --requests "${REQUESTS:-60}"
echo "==> Regression comparison (deterministic; a 1.5x prefill slowdown is injected on purpose)"
TMP="$(mktemp -d)"
"$PY" -m inference_profiler.cli benchmark --scenario examples/benchmark_scenarios.yaml \
  --name baseline --output "$TMP/baseline.json" >/dev/null
"$PY" -m inference_profiler.cli benchmark --scenario examples/benchmark_scenarios.yaml \
  --name baseline --inject prefill_multiplier=1.5 --runtime-version sim-1.1 \
  --output "$TMP/current.json" >/dev/null
# Exit code 1 (regression found) is the expected outcome here.
"$PY" -m inference_profiler.cli compare --baseline "$TMP/baseline.json" \
  --current "$TMP/current.json" --publish | tail -n 3 || true
rm -rf "$TMP"
echo "Open Grafana: http://localhost:3000  (Prometheus: http://localhost:9090)"

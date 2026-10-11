#!/usr/bin/env bash
# Run the benchmark scenarios, analyse them, render a Markdown report, and compare
# with benchmarks/baseline.json when it exists.
#
#   scripts/run_benchmarks.sh               # deterministic virtual clock (default)
#   CLOCK=wall scripts/run_benchmarks.sh    # real sleeps, exports telemetry
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CLI="${CLI:-.venv/bin/inference-profiler}"
CLOCK="${CLOCK:-virtual}"
SCENARIOS="${SCENARIOS:-examples/benchmark_scenarios.yaml}"
OUT="${OUT:-benchmarks/results/latest.json}"
BASELINE="${BASELINE:-benchmarks/baseline.json}"

read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
"$CLI" benchmark --scenario "$SCENARIOS" --clock "$CLOCK" --output "$OUT" ${EXTRA[@]+"${EXTRA[@]}"}
"$CLI" analyze --input "$OUT"
REPORT_ARGS=(--input "$OUT" --output "${OUT%.json}.md")
if [[ -f "$BASELINE" ]]; then
  REPORT_ARGS+=(--baseline "$BASELINE")
  "$CLI" export-report "${REPORT_ARGS[@]}"
  echo "==> regression check against $BASELINE"
  "$CLI" compare --baseline "$BASELINE" --current "$OUT" --output "${OUT%.json}.comparison.json"
else
  "$CLI" export-report "${REPORT_ARGS[@]}"
  echo "no $BASELINE yet; create one with: make baseline"
fi

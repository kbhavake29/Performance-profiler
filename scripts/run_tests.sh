#!/usr/bin/env bash
# Unit tests: Python (pytest) and C++ (CTest). Integration tests: `make integration`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-.venv/bin/python}"
BUILD_DIR="${BUILD_DIR:-build}"

echo "==> Python unit tests"
"$PY" -m pytest tests/python -q

if [[ -f "$BUILD_DIR/CTestTestfile.cmake" ]]; then
  echo "==> C++ unit tests ($BUILD_DIR)"
  ctest --test-dir "$BUILD_DIR" --output-on-failure
else
  echo "==> C++ tests skipped: $BUILD_DIR is not configured (run 'make build')" >&2
  exit 1
fi

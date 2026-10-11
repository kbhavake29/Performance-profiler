#!/usr/bin/env bash
# Create .venv, install the Python package with dev extras, and check native prerequisites.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
      PYTHON="$candidate"; break
    fi
  done
fi
[[ -n "$PYTHON" ]] || { echo "error: Python >= 3.11 is required" >&2; exit 1; }

echo "==> creating .venv with $($PYTHON --version)"
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -e ".[dev]"
.venv/bin/inference-profiler --version

echo "==> native prerequisites"
command -v cmake >/dev/null && cmake --version | head -n1 || echo "  cmake: not on PATH (the venv provides one: .venv/bin/cmake)"
if command -v brew >/dev/null && brew list opentelemetry-cpp >/dev/null 2>&1; then
  echo "  opentelemetry-cpp: $(brew list --versions opentelemetry-cpp)"
else
  echo "  opentelemetry-cpp: not found via Homebrew."
  echo "    macOS:  brew install opentelemetry-cpp"
  echo "    other:  cmake --preset fetch   (builds it from source; needs protobuf + libcurl dev packages)"
fi
command -v docker >/dev/null && echo "  docker: $(docker --version)" \
  || echo "  docker: not found (use scripts/native_stack.sh to run the stack without Docker)"
echo "==> done. Activate with: source .venv/bin/activate"

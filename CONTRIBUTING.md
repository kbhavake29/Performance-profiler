# Contributing

Thanks for helping. Bug reports, docs fixes, new analysis rules and engine integrations
are all welcome.

## Development setup

```bash
make setup     # .venv with pinned dev dependencies
make build     # needs opentelemetry-cpp (brew install opentelemetry-cpp) or: cmake --preset fetch
make lint test
```

Optional: `pre-commit install` runs ruff, mypy and the dashboard check on every commit.

## Before opening a pull request

- `make lint test` passes (ruff, `ruff format --check`, mypy strict, pytest, CTest).
- New behaviour has tests. Python tests use the in-memory exporters from
  `tests/python/conftest.py`. C++ tests use `tests/cpp/test_harness.hpp`, and each new
  case must be added to `IPP_TEST_CASES` in `CMakeLists.txt`.
- If you change a metric or attribute name, update `src/inference_profiler/metrics.py`,
  `cpp/include/inference_profiler/metrics.hpp`, `docs/metrics.md` and
  `scripts/generate_dashboards.py`, then run `make dashboards`.
- If you touch the stack configs, run `make stack-up demo verify` (or the native stack)
  and say so in the PR.
- Never add prompt text, generated text, credentials or high-cardinality identifiers to
  telemetry.
- Do not commit benchmark numbers as if they were measurements. Label example data.

## Style

- Python: ruff (`make format`), full type hints (mypy `--strict`), 100-column lines.
- C++17: RAII, no exceptions escaping destructors, `-Wall -Wextra -Wpedantic -Wshadow
  -Wconversion` clean (CI builds with `-Werror`).
- Commit messages: imperative mood, with a short summary line.

## Releasing

Update `CHANGELOG.md`, bump the version in `pyproject.toml`,
`src/inference_profiler/version.py` and `CMakeLists.txt`, and tag `vX.Y.Z`.

# Developer entry points. Run `make help` for a summary.
SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

PY        ?= .venv/bin/python
CLI       ?= .venv/bin/inference-profiler
PRESET    ?= default
BUILD_DIR ?= build
COMPOSE   ?= docker compose

.PHONY: help setup build build-python test lint format stack-up stack-down stack-logs \
        native-stack-up native-stack-down demo benchmark baseline verify integration \
        overhead dashboards clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

setup: ## Create .venv and install the package with dev dependencies
	scripts/setup.sh

build: ## Configure and build the C++ library, demo and tests (CMake preset $(PRESET))
	cmake --preset $(PRESET)
	cmake --build --preset $(PRESET)

build-python: ## Build the Python sdist and wheel into dist/
	$(PY) -m build

test: ## Run Python and C++ unit tests
	BUILD_DIR=$(BUILD_DIR) PY=$(PY) scripts/run_tests.sh

lint: ## Ruff lint + format check, mypy, dashboard freshness
	$(PY) -m ruff check src tests examples scripts
	$(PY) -m ruff format --check src tests examples scripts
	$(PY) -m mypy
	$(PY) scripts/generate_dashboards.py --check

format: ## Apply ruff formatting and fixes
	$(PY) -m ruff check --fix src tests examples scripts
	$(PY) -m ruff format src tests examples scripts

dashboards: ## Regenerate Grafana dashboards from scripts/generate_dashboards.py
	$(PY) scripts/generate_dashboards.py

stack-up: ## Start collector, Tempo, Prometheus and Grafana with Docker Compose
	$(COMPOSE) up -d
	WAIT_SECONDS=120 scripts/verify_stack.sh

stack-down: ## Stop the Docker Compose stack (keeps volumes; add -v to remove data)
	$(COMPOSE) down

stack-logs: ## Follow Docker Compose logs
	$(COMPOSE) logs -f

native-stack-up: ## Start the stack without Docker (BIN_DIR must contain the binaries)
	scripts/native_stack.sh start
	WAIT_SECONDS=120 scripts/verify_stack.sh

native-stack-down: ## Stop the Docker-free stack
	scripts/native_stack.sh stop

demo: ## Emit telemetry from the Python, C++ and CLI demos to localhost:4318
	PY=$(PY) BUILD_DIR=$(BUILD_DIR) scripts/run_demo.sh

benchmark: ## Run benchmark scenarios -> benchmarks/results/latest.{json,md}
	CLI=$(CLI) scripts/run_benchmarks.sh

baseline: ## Generate benchmarks/baseline.json from a local (virtual-clock) run
	$(CLI) benchmark --scenario examples/benchmark_scenarios.yaml --label baseline \
	  --output benchmarks/baseline.json

verify: ## Verify the running stack received telemetry from both SDKs (run `make demo` first)
	scripts/verify_stack.sh --telemetry
	$(PY) scripts/check_dashboard_queries.py --require-data
	$(PY) -m pytest tests/integration -m integration -q

integration: verify ## Alias for verify

overhead: ## Measure instrumentation overhead (enabled vs disabled)
	$(CLI) overhead --output benchmarks/results/overhead.json

clean: ## Remove build outputs and caches (keeps .venv)
	rm -rf $(BUILD_DIR) build-* dist *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -not -path './.venv/*' -exec rm -rf {} +

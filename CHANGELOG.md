# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-10

### Added
- Python SDK (`inference_profiler`): request and phase scopes, cache operations,
  TTFT and inter-token latency, W3C propagation, sampling, cardinality limits, bounded
  shutdown, export-health metrics.
- C++17 SDK on opentelemetry-cpp with RAII request and phase scopes, OTLP/HTTP export,
  a native demo and 25 CTest cases.
- Deterministic workload simulator with virtual and wall-clock execution.
- Benchmark runner with a JSON schema, bottleneck analysis, regression checker, and
  Markdown reports. CLI: `doctor`, `demo`, `benchmark`, `analyze`, `compare`,
  `export-report`, `validate`, `overhead`.
- Docker Compose stack (OpenTelemetry Collector, Tempo, Prometheus, Grafana) with five
  provisioned dashboards, plus a Docker-free `scripts/native_stack.sh`.
- GitHub Actions: CI, integration, performance regression.

# Metrics and trace attributes

Both SDKs emit the same names. Prometheus names are what the collector's Prometheus
exporter produces (verified with otelcol-contrib 0.161.0).

## Metrics

| OTel name | Type | Unit | Prometheus name | Labels |
|---|---|---|---|---|
| `inference.request.duration` | histogram | s | `inference_request_duration_seconds_*` | model, runtime, status |
| `inference.queue.duration` | histogram | s | `inference_queue_duration_seconds_*` | model, runtime |
| `inference.prefill.duration` | histogram | s | `inference_prefill_duration_seconds_*` | model, runtime |
| `inference.time_to_first_token` | histogram | s | `inference_time_to_first_token_seconds_*` | model, runtime |
| `inference.inter_token.latency` | histogram | s | `inference_inter_token_latency_seconds_*` | model, runtime |
| `inference.generation.duration` | histogram | s | `inference_generation_duration_seconds_*` | model, runtime |
| `inference.phase.duration` | histogram | s | `inference_phase_duration_seconds_*` | model, runtime, phase |
| `inference.generation.throughput` | histogram | {token}/s | `inference_generation_throughput_per_second_*` | model, runtime |
| `inference.cache.operation.duration` | histogram | s | `inference_cache_operation_duration_seconds_*` | model, runtime, cache operation |
| `inference.requests` | counter | {request} | `inference_requests_total` | model, runtime, status |
| `inference.tokens` | counter | {token} | `inference_tokens_total` | model, runtime, `gen_ai.token.type` (input/output) |
| `inference.cache.hits` / `.misses` | counter | {lookup} | `inference_cache_hits_total`, `inference_cache_misses_total` | model, runtime |
| `inference.cache.insertions` / `.evictions` | counter | {entry} | `inference_cache_insertions_total`, `inference_cache_evictions_total` | model, runtime |
| `inference.requests.in_flight` | up-down counter | {request} | `inference_requests_in_flight` | model, runtime |
| `inference.requests.active` | up-down counter | {request} | `inference_requests_active` | model, runtime |
| `inference.queue.depth` | up-down counter | {request} | `inference_queue_depth` | model, runtime |
| `inference.cache.utilization` | gauge | 1 | `inference_cache_utilization_ratio` | model |
| `process.memory.usage` | gauge | By | `process_memory_usage_bytes` | (resource only) |
| `inference.simulated.memory.usage` | gauge | By | `inference_simulated_memory_usage_bytes` | model, `inference.simulated="true"` |
| `inference.simulated.memory.pressure` | gauge | 1 | `inference_simulated_memory_pressure_ratio` | model, `inference.simulated="true"` |
| `inference_profiler.export.errors` | counter | {error} | `inference_profiler_export_errors_total` | signal |
| `inference_profiler.export.duration` | histogram | s | `inference_profiler_export_duration_seconds_*` | signal |
| `inference_profiler.telemetry.dropped` | counter | {item} | `inference_profiler_telemetry_dropped_total` | signal, reason |
| `inference_profiler.instrumentation.errors` | counter | {error} | `inference_profiler_instrumentation_errors_total` | where (Python) |
| `inference_profiler.enabled` | gauge | | `inference_profiler_enabled` | |
| `inference.benchmark.value` / `.change` / `.regression` / `.regressions` | gauge | / % / / | `inference_benchmark_value`, `inference_benchmark_change_percent`, `inference_benchmark_regression`, `inference_benchmark_regressions` | scenario, metric, run, status (published by `compare --publish`) |

Label keys in Prometheus: `gen_ai_request_model`, `inference_runtime_version`,
`inference_status`, `inference_phase`, `inference_cache_operation`, `gen_ai_token_type`.
`job` is the OTel `service.name`.

**Definitions**

- *End-to-end latency* runs from request start to request end and includes queueing.
- *TTFT* runs from request start (arrival) to the first output token, so it includes
  queue and prefill.
- *Active* requests are executing. *In-flight* requests are active or queued.
- Gauges built on up-down counters stay correct across concurrent requests because each
  request adds and removes exactly once.

Histogram buckets (seconds): `0.0005 … 60` for latencies, `0.001 … 1` for inter-token
latency, `1 … 1000` tokens/s for throughput. They are identical in Python and C++.

## Percentiles

- Dashboards use `histogram_quantile(q, sum by (le) (rate(<name>_bucket[...])))`, which
  aggregates buckets across instances **before** computing the quantile. Never average
  per-instance percentiles.
- Benchmark reports compute percentiles from raw per-request samples with linear
  interpolation (numpy's default). They are never averaged across requests or scenarios.

## Span names and attributes

| Span | Attributes |
|---|---|
| `inference.request` (SERVER) | `gen_ai.request.model`, `inference.runtime.version`, `inference.request.id`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `inference.status`, `inference.queue.duration_ms`, `inference.time_to_first_token_ms`, `inference.generation.tokens_per_second`, `error.type`, `inference.cancel.reason`, user attributes |
| `inference.queue`, `inference.prefill`, `inference.generation`, custom `inference.<phase>` | `inference.phase` |
| `inference.time_to_first_token` | `inference.phase` (starts at request start, ends at first token) |
| `inference.cache_lookup` / `_insert` / `_evict` | `inference.cache.name`, `inference.cache.hit`, `inference.cache.entries` |
| `inference.token` (opt-in, sampled, capped) | `inference.token.index` |

Resource attributes: `service.name`, `deployment.environment.name`,
`inference.runtime.version`, `telemetry.distro.name`, `telemetry.distro.version`, plus
`OTEL_RESOURCE_ATTRIBUTES`.

## Memory measurement limitations

- `process.memory.usage` is resident set size: psutil in Python, `task_info` on macOS
  or `/proc/self/statm` on Linux in C++. It includes the interpreter, allocator caches
  and shared pages. It is not GPU memory, and it does not reflect a KV cache that lives
  on an accelerator.
- Python samples RSS on a background thread every `memory_interval_ms` and reports the
  latest value. C++ reads RSS when metrics are collected.
- `inference.simulated.memory.*` is computed by the simulator from token counts. It is
  **not a measurement** and is labelled `inference.simulated="true"` on metrics and
  `"simulated": true` in reports. Pressure above 1.0 means simulated demand exceeds the
  configured capacity.

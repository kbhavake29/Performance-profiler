# Benchmarking, analysis and regression checks

## Scenarios

Scenarios are YAML (`examples/benchmark_scenarios.yaml`). `defaults` is deep-merged
into each entry, and unknown keys are rejected.

| Key | Meaning |
|---|---|
| `seed` | RNG seed; same seed + same config → identical plan |
| `num_requests`, `concurrency` | Request count; execution slots (open loop) or clients (closed loop) |
| `arrival_rate_rps` | Poisson arrivals if > 0; `0` = closed-loop clients sending back-to-back |
| `prompt_tokens`, `generated_tokens`, `scheduler_delay_ms`, `inter_token_ms` | `{mean, stddev, min, max}` clipped normal (or a constant number) |
| `prefill_base_ms`, `prefill_ms_per_token` | prefill = base + per_token × prompt_tokens |
| `first_token_overhead_ms` | First decode step after prefill |
| `batch_contention` | Inter-token latency × (1 + c × (concurrency − 1)) |
| `failure_rate`, `cancel_rate` | Injected failures / client cancellations at a random token |
| `cache.{enabled, capacity_entries, prefix_pool, prefix_skew, hit_prefill_fraction, lookup_ms, insert_ms, evict_ms}` | LRU prefix cache; hits cut prefill cost |
| `memory.{bytes_per_token, capacity_bytes, pressure_threshold, pressure_slowdown, real_allocation_mb}` | SIMULATED KV memory; above the threshold, inter-token latency is multiplied by the slowdown. `real_allocation_mb` (≤ 64) holds a real bounded allocation during wall-clock runs |
| `regression.{prefill_multiplier, inter_token_multiplier, queue_delay_add_ms, first_token_add_ms}` | Deliberate slowdowns for testing detection |
| `time_scale` | Multiplies every sleep in wall-clock mode |
| `group` | Scenarios sharing a group form a concurrency sweep for saturation analysis |

Bundled scenarios: `baseline`, `congestion`, `memory-pressure`, `cache-thrash`,
`unreliable`, and the sweep `sweep-c1/c4/c16/c64`. Each one is designed to trigger the
analysis rule its name suggests, and `baseline` triggers none.

## Clocks

| | `--clock virtual` (default) | `--clock wall` |
|---|---|---|
| Execution | Discrete-event simulation, no sleeping | Real `asyncio.sleep` through the SDK |
| Telemetry | none | real spans and metrics (unless `--no-export`) |
| Determinism | exact | subject to scheduler noise |
| Use for | CI correctness, regression-checker tests | dashboards, overhead, timing comparisons on one machine |

Memory pressure is applied in two passes in virtual mode. The first pass finds
requests that start while simulated pressure is above the threshold, and the second
pass slows them down. Wall-clock mode applies the slowdown live.

## Report

`benchmark` writes JSON validated against
`src/inference_profiler/schemas/benchmark-report.schema.json`:

- `schema_version`, `generated_at`, `git_commit` (if available), `sdk_version`,
  `simulator_version`, `label`, `environment` (Python, platform, machine, CPU count, CI
  flag; hostnames and usernames are deliberately omitted)
- `measurement`: `clock`, `workload` (`simulated`), `latency_source`, notes on which
  values are simulated
- per scenario: `config`, model/runtime, `requests` (total/completed/failed/cancelled/
  error_rate), `tokens`, `throughput`, `latency_ms.{end_to_end, queue, prefill,
  time_to_first_token, generation, inter_token}` (count/min/mean/stddev/p50/p90/p95/
  p99/max), `cache` (simulated), `memory.process_rss` (real, wall clock only),
  `memory.simulated_kv_cache` (simulated), `errors`
- optional `instrumentation_overhead` (`--overhead`)

End-to-end and generation latency include only completed requests. Queue, prefill and
TTFT include every request that reached that phase.

## Analysis rules

Every rule's threshold is configurable (`--thresholds file.yaml` with an `analysis:`
mapping of `AnalysisThresholds` fields).

| Rule | Fires when (default) | Severity |
|---|---|---|
| `high_queue_latency` | queue p50 / e2e p50 ≥ 25% (critical ≥ 50%) | warning/critical |
| `excessive_prefill` | prefill p50 / mean prompt tokens ≥ 0.5 ms/token | warning |
| `slow_ttft` | TTFT p95 ≥ 1000 ms | warning |
| `slow_generation` | per-request decode p50 < 20 tokens/s | warning |
| `high_inter_token_latency` | ITL p95 ≥ 75 ms (critical ≥ 150 ms) | warning/critical |
| `long_tail_latency` | e2e p99 / p50 ≥ 3 | warning |
| `error_rate` | failed/total ≥ 1% (critical ≥ 5%) | warning/critical |
| `memory_pressure` | simulated peak pressure ≥ 85% (critical ≥ 95%) | warning/critical |
| `process_rss` | peak RSS ≥ `rss_bytes_warning` (off by default) | warning |
| `low_cache_hit_ratio` | hit ratio < 50% | warning |
| `cache_eviction_churn` | evictions / insertions ≥ 0.5 | warning |
| `throughput_saturation` | in a `group`, tokens/s gain < 15% **and** p95 growth > 25% per doubling of concurrency | warning |

Findings are hypotheses, not root causes. Limitations are reported explicitly, for
example fewer than `min_samples` requests (default 20), missing phases, or a
virtual-clock report. `analyze --prometheus-url` builds the same structure from live
histograms (`histogram_quantile` over `sum by (le)`), so the same rules apply to a
running system.

## Regression checker

`compare` matches scenarios by name and compares p50/p95/p99 of every latency phase,
throughput, simulated peak memory, process RSS (when both reports have it), error rate
and cache hit ratio.

A latency or throughput change counts as a regression only if it is worse than
`latency_pct` / `throughput_pct` **and** larger than `min_latency_delta_ms`, which keeps
sub-millisecond noise on fast phases from failing builds. Error rate and hit ratio use
absolute thresholds. The checker also warns when the clock types differ, when
wall-clock results come from different hardware, when scenario configuration drifted,
or when scenarios are missing. `--fail-on-missing` turns a missing scenario into a
failure.

Exit code: `0` pass, `1` regression, `2` invalid input.

### In CI

- `ci.yml` runs the **deterministic** check: virtual-clock benchmarks, one comparison
  that must pass and one with an injected regression that must exit 1.
- `performance.yml` (manual or weekly) runs wall-clock benchmarks of the baseline ref
  and the current ref **back to back on the same runner** with generous thresholds.
  Shared runners are noisy, so absolute timing is never asserted. Use a dedicated runner
  if you need tight thresholds.

### Creating a baseline

```bash
make baseline                                   # virtual clock → benchmarks/baseline.json
inference-profiler benchmark --scenario examples/benchmark_scenarios.yaml --clock wall \
    --no-export --output benchmarks/baseline.wall.json      # wall-clock baseline on this machine
```

`benchmarks/baseline.json` is gitignored by default. Commit it deliberately if your CI
should compare against a fixed baseline. Never commit `baseline.example.json` values as
real measurements.

### Publishing to Grafana

`compare --publish` exports gauges for the Performance Regressions dashboard. The
collector keeps them visible for `metric_expiration` (180 min).

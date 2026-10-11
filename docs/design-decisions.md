# Design decisions

## OTLP over HTTP/protobuf, port 4318

HTTP needs no gRPC runtime in Python and only libcurl in C++. Both SDKs default to
`http://localhost:4318`. The collector also accepts gRPC on 4317 for other clients.

## Own providers, not globals

Libraries that call `set_tracer_provider` can silently break the host application's
telemetry. The profiler builds private tracer and meter providers and exposes
`set_global_providers` as an opt-in. Spans still nest under the application's active
context, because request scopes read and attach the current OpenTelemetry context.

## Explicit parents for phases

Phases are started with the request span as an explicit parent instead of relying only
on the ambient context. Scheduler threads, executor callbacks and other asyncio tasks
can therefore open phases for a request and still produce a correct tree.

## Metrics are never sampled

Sampling applies to traces only. Rates, error ratios and percentiles come from
unsampled metrics. Exemplars link the sampled traces to them.

## Bounded cardinality

Metric labels are limited to model, runtime version, status, phase, cache operation and
token type, and each label key is capped (default 64 values; extra values become
`_other`). Request IDs are span attributes only. The limiter's fast path reads a dict
without locking; only admitting a new value takes a lock.

## Errors: the application's exceptions vs. ours

Application exceptions are recorded and re-raised unchanged. Instrumentation faults are
logged once and counted, never raised, unless `strict=True`. Exporter wrappers convert
exceptions to failed exports, so a broken collector can never surface in request
handling.

## Bounded shutdown

OpenTelemetry Python's `TracerProvider.shutdown()` has no timeout. The profiler runs
shutdown on a helper thread and stops waiting after `shutdown_timeout_ms`. C++ does the
same with a detached worker that keeps the pipeline alive until it finishes.

## Token-level telemetry

A span per token would multiply span volume by output length and distort the system
being measured. Inter-token latency is therefore a histogram fed from timestamps.
Python buffers the samples during decoding and records them when the request ends.
Optional token spans are sampled per request and capped.

## TTFT definition

TTFT is measured from request start, which includes queueing, because that is what a
client experiences. Queue and prefill are recorded separately so the components can be
separated.

## Percentiles

Benchmarks keep raw samples and compute percentiles directly. Dashboards aggregate
histogram buckets with `sum by (le)` before `histogram_quantile`. Averaging percentiles
is never done.

## Simulated vs. measured

Every simulated quantity carries its label: `measurement.clock`, `workload:
"simulated"`, `"simulated": true` on cache and KV-memory blocks, the
`inference.simulated` attribute, and "SIMULATED" in dashboard titles. Real process RSS
is reported separately. Memory-pressure scenarios never allocate large amounts of RAM.
The optional real allocation is capped at 64 MiB.

## Virtual and wall clocks

CI correctness gates must not depend on shared-runner timing. The same deterministic
plan can be executed on a virtual clock (exact, fast) or a wall clock (real telemetry).
Timing comparisons are isolated in `performance.yml`.

## Dashboards are generated

Dashboard JSON is produced by `scripts/generate_dashboards.py` from one table of metric
names. A unit test checks every referenced metric against the SDK's names, and a script
runs every query against a live Prometheus.

## Native stack without Docker

`scripts/native_stack.sh` runs the same configuration files with local binaries. Every
host-specific address in the configs is an environment variable with a Compose default,
so nothing is duplicated.

## Overhead

`inference-profiler overhead` compares disabled, enabled-unsampled and enabled-sampled
modes with an exporter that discards spans, which isolates in-process cost. On the
development machine (Apple M5, Python 3.12.3, OpenTelemetry Python 1.45.1), one request
with 3 phases and 32 tokens added about 204 µs unsampled and 259 µs sampled, and
`record_token()` cost about 137 ns per token. The OpenTelemetry Python SDK's
per-measurement cost dominates. Results depend on hardware and SDK versions, so measure
your own.

# Instrumentation API

## Python

### Profiler

```python
Profiler(config: ProfilerConfig | None = None, *, span_exporter=None, metric_reader=None,
         batch_spans: bool | None = None)
```

- `config=None` → `ProfilerConfig.from_env()`.
- `span_exporter` / `metric_reader` replace the OTLP pipeline (tests, custom backends).
  An injected span exporter uses a synchronous `SimpleSpanProcessor` unless `batch_spans=True`.
- The profiler creates **its own** tracer and meter providers. It does not replace the
  global OpenTelemetry providers unless `set_global_providers=True`, so it cannot
  clobber instrumentation the host application already configured.

| Method | Purpose |
|---|---|
| `inference_request(*, model, request_id=None, prompt_tokens=None, runtime_version=None, attributes=None, parent=None, kind=SpanKind.SERVER)` | Request scope (sync and async context manager). `start_request` is an alias for manual `start()`/`end()` |
| `trace_request(model=...)` | Decorator for sync/async handlers |
| `phase(name)` | Decorator recording a phase on `current_request()` (no-op outside a request) |
| `set_cache_utilization(ratio, model=...)` | Gauge for integrations that know cache occupancy |
| `set_simulated_memory(used_bytes, pressure, model=...)` | Used by the simulator; labelled `inference.simulated=true` |
| `force_flush(timeout_ms=None)` / `shutdown(timeout_ms=None)` | Bounded flush/stop; `shutdown` is idempotent and registered with `atexit` |
| `status()` | Health snapshot (enabled, export failures, dropped items, overflowed labels) |
| `with Profiler(...) as p:` | Shuts down on exit |

### Request scope (`RequestHandle`)

| Method | Span | Metrics |
|---|---|---|
| `queue()` | `inference.queue` | `inference.queue.duration`, `inference.queue.depth` ± |
| `prefill()` | `inference.prefill` | `inference.prefill.duration` |
| `generation()` | `inference.generation` | `inference.generation.duration` |
| `phase(name, **attrs)` | `inference.<name>` | `inference.phase.duration{inference.phase}` for custom names |
| `cache_lookup(cache="kv")` (set `.hit`) | `inference.cache_lookup` | `inference.cache.hits` / `misses`, `inference.cache.operation.duration` |
| `cache_insert(cache, entries)` / `cache_evict(cache, entries)` | `inference.cache_insert` / `_evict` | insertions / evictions counters |
| `mark_first_token()` | `inference.time_to_first_token` | `inference.time_to_first_token` |
| `record_token()` | `inference.token` only in token-span mode | `inference.inter_token.latency` (buffered, recorded at end) |
| `record_generation(n, *, time_to_first_token_ms=None, inter_token_latencies_ms=None, generation_duration_ms=None)` | `gen_ai.usage.output_tokens` | `inference.tokens{output}`, `inference.generation.throughput` |
| `fail(error)` / `cancel(reason)` | status + `error.type` / `inference.cancel.reason` | status label on request metrics |
| `inject(headers)` / `traceparent` / `trace_id` | | |

Phase names must match `^[a-z][a-z0-9_]{0,63}$`. Phases can be nested via the returned
scope's `phase()`.

### Errors and cancellation

- An exception that leaves a request or phase scope is recorded (`record_exception`,
  status ERROR, `error.type=<ExceptionClass>`) and re-raised. Instrumentation never
  swallows application exceptions.
- `asyncio.CancelledError`, `RequestCancelled`, `GeneratorExit` and `KeyboardInterrupt`
  mark the request `cancelled`. Its span status is not set to ERROR.
- **Instrumentation** errors (invalid phase name, negative token counts, SDK failures)
  are logged once at WARNING and counted in `inference_profiler.instrumentation.errors`.
  With `strict=True` they are raised instead, which is useful in tests.

### Context

- Request and phase scopes attach their span to the OpenTelemetry context, so spans you
  create with any OpenTelemetry tracer inside the block become children.
- `current_request()` uses a `contextvars.ContextVar`, which isolates threads and
  asyncio tasks.
- Incoming: `inference_request(parent=headers)` extracts `traceparent`/`tracestate`.
  Outgoing: `request.inject(headers)` or `inference_profiler.inject_context(headers)`.

### Configuration

| Field | Env var | Default | Validation |
|---|---|---|---|
| `service_name` | `OTEL_SERVICE_NAME` (or `service.name` in `OTEL_RESOURCE_ATTRIBUTES`) | `inference-service` | non-empty, ≤128 chars, no control chars |
| `otlp_endpoint` | `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | http(s), host required, no credentials/query |
| `enabled` | `INFERENCE_PROFILER_ENABLED`, `OTEL_SDK_DISABLED` | `true` | boolean |
| `sample_rate` | `INFERENCE_PROFILER_SAMPLE_RATE` | `1.0` | 0–1 |
| `export_interval_ms` | `INFERENCE_PROFILER_EXPORT_INTERVAL` | `5000` | 100 ms – 1 h |
| `export_timeout_ms` | `INFERENCE_PROFILER_EXPORT_TIMEOUT` | `10000` | 100 ms – 10 min |
| `shutdown_timeout_ms` | `INFERENCE_PROFILER_SHUTDOWN_TIMEOUT` | `5000` | 0 – 10 min |
| `memory_interval_ms` | `INFERENCE_PROFILER_MEMORY_INTERVAL` | `5000` | 0 (off) or ≥ 50 |
| `exporter` | `INFERENCE_PROFILER_EXPORTER` | `otlp` | `otlp`, `console`, `none` |
| `runtime_version` | `INFERENCE_PROFILER_RUNTIME_VERSION` | `unknown` | as service_name |
| `deployment_environment` | `INFERENCE_PROFILER_ENVIRONMENT` | `local` | as service_name |
| `strict` | `INFERENCE_PROFILER_STRICT` | `false` | boolean |
| `token_spans` | `INFERENCE_PROFILER_TOKEN_SPANS` | `false` | boolean |
| `token_span_sample_rate` | `INFERENCE_PROFILER_TOKEN_SPAN_SAMPLE_RATE` | `0.01` | 0–1 |
| `max_token_spans_per_request` | n/a | `256` | 0–100000 |
| `max_attribute_values` | n/a | `64` | 1–10000 |
| `max_span_queue_size` | n/a | `2048` | 1–1000000 |
| `set_global_providers` | n/a | `false` | |
| `resource_attributes` | `OTEL_RESOURCE_ATTRIBUTES` | `{}` | printable strings |

Headers for authenticated collectors come from `OTEL_EXPORTER_OTLP_HEADERS`, which the
OTLP exporter reads directly. Never put credentials in the endpoint URL.

### Sampling

- Traces: `ParentBased(TraceIdRatioBased(sample_rate))`. An upstream sampling decision
  in `traceparent` is respected.
- Metrics are recorded for every request regardless of trace sampling, so rates and
  percentiles stay unbiased.
- Token spans: off by default. When enabled, a sampled request emits up to
  `max_token_spans_per_request` `inference.token` spans with probability
  `token_span_sample_rate`. A span per token for every request would multiply span
  volume by the output length, which is why it is opt-in.

## C++

Headers live in `cpp/include/inference_profiler/`. Key types:

```cpp
struct ProfilerConfig { /* same fields as Python, std::chrono durations */
  static ProfilerConfig FromEnv();               // defaults < env; throws ConfigError
  std::vector<std::string> Validate() const;     // empty when valid
};
struct RequestOptions { std::string model, request_id, runtime_version, traceparent;
                        std::optional<int64_t> prompt_tokens;
                        std::map<std::string, std::string> attributes; };

class Profiler {
  static std::shared_ptr<Profiler> Create(ProfilerConfig);                     // validates
  static std::shared_ptr<Profiler> Create(ProfilerConfig, ExporterOverrides);  // testing.hpp
  RequestScope StartRequest(const RequestOptions&);
  RequestScope StartRequest(std::string model, std::optional<int64_t> prompt_tokens = {});
  bool ForceFlush(std::optional<std::chrono::milliseconds> = {});
  bool Shutdown(std::optional<std::chrono::milliseconds> = {});  // idempotent, bounded
  uint64_t export_failures() const, dropped_spans() const, instrumentation_errors() const;
};

class RequestScope {   // move-only RAII
  PhaseScope Queue(), Prefill(), Generation(), Phase(std::string_view);
  PhaseScope CacheLookup(std::string_view cache = "kv");     // .SetCacheHit(bool)
  PhaseScope CacheInsert(std::string_view cache = "kv", int64_t entries = 1);
  PhaseScope CacheEvict(std::string_view cache = "kv", int64_t entries = 1);
  void MarkFirstToken(), RecordToken();
  void RecordGeneration(int64_t tokens, std::optional<double> ttft_ms = {});
  void SetAttribute(std::string_view, std::string_view | int64_t | double | bool);
  void Fail(std::string_view error_type), Cancel(std::string_view reason = {});
  void End();                                     // idempotent; destructor calls it
  std::string TraceId() const, TraceParent() const, Status() const;
};
```

- **RAII**: scopes end exactly once, through `End()` or the destructor. A moved-from
  scope is inert. If a destructor runs while an exception is propagating
  (`std::uncaught_exceptions()` increased since construction), the scope is marked
  `error` with `error.type=exception`.
- **Disabled or after shutdown**: `StartRequest` returns an inert scope. Every method is
  a no-op.
- **Thread safety**: phases may end on other threads. Shared request state is guarded
  by a mutex taken only at phase and request boundaries. `RecordToken()` takes no lock,
  so call it from the thread that produces tokens.
- **Strict mode** rethrows instrumentation errors from `StartRequest` and `Phase`.
  Destructors never throw.

# Integrating a real inference engine

The SDK only needs to see request boundaries, phase boundaries and tokens. Keep the
instrumentation at the server/scheduler layer, not inside kernels.

## Python server (asyncio)

```python
from inference_profiler import Profiler, ProfilerConfig

profiler = Profiler(
    ProfilerConfig.from_env(service_name="my-llm-server", runtime_version=engine.__version__)
)


async def handle(http_request):
    body = await http_request.json()
    async with profiler.inference_request(
        model=body["model"],
        prompt_tokens=len(tokenizer.encode(body["prompt"])),  # count only; text is never sent
        parent=dict(http_request.headers),
    ) as req:
        with req.queue():
            slot = await scheduler.admit()  # time spent waiting for capacity
        with req.cache_lookup() as lookup:
            lookup.hit = engine.prefix_cache_hit(body["prompt"])
        with req.prefill():
            state = await engine.prefill(slot)
        with req.generation():
            async for token in engine.decode(state):
                req.record_token()
                yield token
        req.record_generation(state.generated_tokens)
```

### Continuous-batching engines

When one scheduler step serves many requests (vLLM-style), the scheduler, not the
request coroutine, knows when each request was admitted and when its tokens were
produced. Two options:

1. Keep a `RequestHandle` per sequence in the scheduler and call its methods from the
   scheduler thread. Scopes take explicit parents, so this is safe from any thread.
2. Collect timestamps yourself and report them at the end:
   `req.record_generation(n, time_to_first_token_ms=..., inter_token_latencies_ms=[...],
   generation_duration_ms=...)`.

Report cache occupancy periodically with
`profiler.set_cache_utilization(used_blocks / total_blocks, model=...)`.

## C++ engine

```cmake
add_subdirectory(third_party/inference-performance-profiler)   # or FetchContent
target_link_libraries(my_server PRIVATE inference_profiler::inference_profiler)
```

Configure with `-DIPP_BUILD_TESTS=OFF -DIPP_BUILD_EXAMPLES=OFF` when embedding. Create
one `Profiler` at startup, keep a `RequestScope` in each request's state object, and
end it when the response completes. `RequestScope` is movable, so it can live in a
`std::unique_ptr` or a struct.

## Mixed Python + C++ in one process

If a Python server calls into a C++ engine (pybind11, nanobind), pick **one SDK per
request** and avoid instrumenting the same work twice.

- **Recommended:** the C++ engine owns the whole request. Python passes the incoming
  `traceparent` into C++ (`RequestOptions::traceparent`), C++ creates the request and
  all phase spans, and Python creates no request scope of its own.
- **Alternative:** Python owns the request scope. The C++ engine returns timings (TTFT,
  inter-token gaps, prefill time), and Python reports them with
  `record_generation(..., time_to_first_token_ms=..., inter_token_latencies_ms=...)`.
- Do not start request scopes for the same request in both SDKs. Request metrics such
  as `inference.requests` and `inference.request.duration` would be counted twice.
- Leave `set_global_providers` off in both SDKs. Each SDK uses its own providers and
  exports to the collector independently. Turning it on in both would make two
  providers compete for the global slot.

## Across services

Outgoing calls such as tokenizer, retrieval or tool services can carry the trace with
`request.inject(headers)` (Python) or `request.TraceParent()` (C++). The receiving
service passes the header as `parent=` / `RequestOptions::traceparent`.

## Avoiding pitfalls

- Do not put user text, prompts, completions or user IDs in `attributes`.
- Keep `model` and `runtime_version` low-cardinality (≤ 64 distinct values each by
  default). Beyond that, values collapse to `_other` and `status()` reports the overflow.
- In latency-critical loops, `record_token()` costs about 0.14 µs per token in Python on
  the reference machine, because samples are buffered. The remaining cost of roughly
  0.2 ms per request is paid when the request ends. See the README overhead section.
- Use `exporter="none"` or `enabled=False` to compare performance with and without
  telemetry.

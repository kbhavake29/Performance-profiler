# Troubleshooting

Start with:

```bash
inference-profiler doctor            # versions, effective config, collector reachability
scripts/verify_stack.sh --telemetry  # stack health, provisioning, data per service
```

## No data in Grafana

1. Is the collector reachable? `curl -s localhost:13133` should print `Server available`.
   The demos print `export_failures=N`. Anything above 0 means the collector could not be
   reached.
2. Is Prometheus scraping it? Open http://localhost:9090/targets: `otel-collector-exporter`
   should be **UP**. Then check `curl -s localhost:8889/metrics | grep inference_`.
3. Check the time range and variables. Dashboards default to the last 30 minutes, and the
   `Service` variable must include your `OTEL_SERVICE_NAME`.
4. Metrics are exported every 5 s by default. Short scripts must call
   `profiler.shutdown()` (Python does this at exit) or `ForceFlush()` (C++). Set
   `INFERENCE_PROFILER_EXPORT_INTERVAL=1000` for demos.
5. If traces are missing, Tempo needs a few seconds to make traces searchable (see
   `trace_idle_period` in `monitoring/tempo.yaml`). Look at `docker compose logs tempo`.

## Exporter errors in application logs

`Failed to export ... Connection refused` means the endpoint is wrong or the stack is
down. The application keeps running and failures are counted. Outside Docker, use
`http://localhost:4318`. Inside Compose, use `http://otel-collector:4318`. The SDK
expects the **base** URL; it appends `/v1/traces` and `/v1/metrics` itself.

## `opentelemetry-cpp not found` when configuring CMake

- macOS: `brew install opentelemetry-cpp`
- vcpkg: `vcpkg install opentelemetry-cpp[otlp-http]` plus the vcpkg toolchain file
- Anywhere: `cmake --preset fetch` builds opentelemetry-cpp v1.29.0 from source. It
  needs protobuf (≥ 3.16) and libcurl development packages
  (`apt install libprotobuf-dev protobuf-compiler libcurl4-openssl-dev`).

## C++ demo prints `[Warning] MetricReader::Collect invoked while Shutdown in progress!`

This log line comes from the OpenTelemetry C++ SDK during shutdown and is harmless. The
final metric export still runs.

## Port already in use

Another collector, Prometheus or Grafana is running. Stop it, or change the host side of
the port mapping in `docker-compose.yml` (for example `127.0.0.1:14318:4318`) and set
`OTEL_EXPORTER_OTLP_ENDPOINT` to match.

## Tempo: `permission denied` on `/var/tempo`

The `tempo-init` service sets ownership of the named volume to UID 10001. If you replaced
the volume with a bind mount, run `chown -R 10001:10001` on that directory.

## Benchmark comparison fails on a shared CI runner

Wall-clock results are noisy on shared machines. Use the virtual clock for correctness
gates, raise `--threshold latency_pct=...` and `min_latency_delta_ms`, or run
`performance.yml` on a dedicated runner. The checker warns when the two reports came
from different hardware or clocks.

## Sanitizer builds hang on macOS

On some macOS/Xcode combinations, the AddressSanitizer runtime deadlocks during process
startup (`__asan::InitializeShadowMemory`), before any project code runs. The CI job
`cpp-sanitizers` runs ASan/UBSan on Linux instead.

## High instrumentation overhead

Measure it with `inference-profiler overhead`. Lower `INFERENCE_PROFILER_SAMPLE_RATE`
to reduce span cost (metrics stay complete), keep token spans off, and avoid creating
custom phases inside tight loops.

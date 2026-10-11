"""Minimal example: instrumenting a (fake) inference server with the public API.

This file shows how the SDK is used from application code. The "model" below
just sleeps; replace the bodies of ``prefill``/``decode_step`` with calls into a
real engine.

Run (with the stack up: ``make stack-up``):

    python examples/python_inference_demo.py
    python examples/python_inference_demo.py --requests 50 --endpoint http://localhost:4318

Telemetry goes to ``OTEL_EXPORTER_OTLP_ENDPOINT`` (default http://localhost:4318).
"""

from __future__ import annotations

import argparse
import random
import time

from inference_profiler import (
    Profiler,
    ProfilerConfig,
    RequestCancelled,
    current_request,
    inject_context,
)

# ---------------------------------------------------------------- fake engine


class FakeEngine:
    """Stands in for a real model runtime. Timings scale with token counts."""

    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        self.prefix_cache: dict[int, bool] = {}

    def lookup_prefix(self, prefix: int) -> bool:
        time.sleep(0.0002)
        return prefix in self.prefix_cache

    def insert_prefix(self, prefix: int) -> int:
        time.sleep(0.0002)
        self.prefix_cache[prefix] = True
        evicted = 0
        while len(self.prefix_cache) > 16:
            self.prefix_cache.pop(next(iter(self.prefix_cache)))
            evicted += 1
        return evicted

    def prefill(self, prompt_tokens: int, cached: bool) -> None:
        cost = 0.004 + prompt_tokens * 0.00003
        time.sleep(cost * (0.3 if cached else 1.0))

    def decode_step(self) -> None:
        time.sleep(self.rng.uniform(0.004, 0.008))


# ------------------------------------------------------------ instrumented app

engine = FakeEngine()


def handle_request(
    profiler: Profiler, headers: dict[str, str], prompt_tokens: int, max_tokens: int, prefix: int
) -> int:
    """One request. ``headers`` may carry an upstream W3C traceparent."""
    with profiler.inference_request(
        model="demo-model",
        prompt_tokens=prompt_tokens,
        parent=headers,  # continue the caller's trace if present
    ) as request:
        with request.queue():
            time.sleep(engine.rng.uniform(0.0, 0.003))  # waiting for a batch slot

        with request.cache_lookup() as lookup:
            lookup.hit = engine.lookup_prefix(prefix)

        run_prefill(prompt_tokens, cached=bool(lookup.hit))

        produced = 0
        with request.generation():
            for _ in range(max_tokens):
                engine.decode_step()
                request.record_token()  # first call also records TTFT
                produced += 1
                if engine.rng.random() < 0.002:
                    raise RequestCancelled("client disconnected")

        if not lookup.hit:
            with request.cache_insert():
                evicted = engine.insert_prefix(prefix)
            if evicted:
                with request.cache_evict(entries=evicted):
                    pass

        request.record_generation(generated_tokens=produced)

        # Propagate context to a downstream service (e.g. a tokenizer or tool call).
        outgoing: dict[str, str] = {}
        inject_context(outgoing)
        return produced


def run_prefill(prompt_tokens: int, cached: bool) -> None:
    """Deep helper: finds the active request without it being passed in."""
    req = current_request()
    if req is None:
        engine.prefill(prompt_tokens, cached)
        return
    with req.prefill():
        engine.prefill(prompt_tokens, cached)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", type=int, default=25)
    parser.add_argument("--endpoint", help="OTLP/HTTP base URL")
    args = parser.parse_args()

    overrides = {"service_name": "python-inference-demo", "runtime_version": "demo-1.0"}
    if args.endpoint:
        overrides["otlp_endpoint"] = args.endpoint
    # Environment variables (OTEL_*, INFERENCE_PROFILER_*) apply unless overridden here.
    config = ProfilerConfig.from_env(**overrides)

    with Profiler(config) as profiler:
        ok = failed = cancelled = 0
        for i in range(args.requests):
            # Every 5th request arrives with an upstream trace context.
            headers = (
                {
                    "traceparent": f"00-{random.getrandbits(128):032x}-{random.getrandbits(64):016x}-01"
                }
                if i % 5 == 0
                else {}
            )
            try:
                if i % 17 == 16:
                    raise_inside(profiler)
                handle_request(
                    profiler,
                    headers,
                    prompt_tokens=engine.rng.randint(64, 1024),
                    max_tokens=engine.rng.randint(16, 64),
                    prefix=engine.rng.randint(0, 24),
                )
                ok += 1
            except RequestCancelled:
                cancelled += 1
            except RuntimeError:
                failed += 1
        flushed = profiler.force_flush()
        print(
            f"requests: {ok} ok, {failed} failed, {cancelled} cancelled; "
            f"export failures: {profiler.export_failures}; flushed: {flushed}"
        )


def raise_inside(profiler: Profiler) -> None:
    """Shows that application exceptions are recorded on the span and re-raised."""
    request = profiler.inference_request(model="demo-model", prompt_tokens=10)
    with request, request.prefill():
        raise RuntimeError("simulated engine fault")


if __name__ == "__main__":
    main()

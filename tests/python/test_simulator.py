from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

import pytest

from inference_profiler import metrics as m
from inference_profiler.simulator import (
    Distribution,
    RegressionInjection,
    ScenarioConfig,
    ScenarioError,
    build_plan,
    load_scenarios,
    run_wall_clock,
    scenario_from_dict,
    simulate_virtual,
)

from .conftest import ROOT, make_harness

SCENARIOS = ROOT / "examples" / "benchmark_scenarios.yaml"


def small(**kw: object) -> ScenarioConfig:
    base = {"name": "t", "num_requests": 50, "concurrency": 4}
    base.update(kw)
    return ScenarioConfig(**base)  # type: ignore[arg-type]


def test_plan_is_deterministic_for_same_seed() -> None:
    a, b = build_plan(small(seed=7)), build_plan(small(seed=7))
    assert a == b
    assert build_plan(small(seed=8)) != a


def test_virtual_simulation_is_deterministic() -> None:
    r1, r2 = simulate_virtual(small()), simulate_virtual(small())
    assert [o.e2e_ms for o in r1.outcomes] == [o.e2e_ms for o in r2.outcomes]


def test_queue_delay_increases_queue_latency() -> None:
    low = simulate_virtual(small(scheduler_delay_ms=Distribution(1.0)))
    high = simulate_virtual(small(scheduler_delay_ms=Distribution(20.0)))
    mean = lambda r: sum(o.queue_ms for o in r.outcomes) / len(r.outcomes)
    assert mean(high) == pytest.approx(mean(low) + 19.0, rel=0.01)


def test_more_generated_tokens_increase_generation_time() -> None:
    short = simulate_virtual(small(generated_tokens=Distribution(16)))
    long = simulate_virtual(small(generated_tokens=Distribution(128)))
    gen = lambda r: sum(o.generation_ms for o in r.outcomes)
    assert gen(long) > 5 * gen(short)
    assert all(o.generated_tokens == 128 for o in long.outcomes)


def test_open_loop_overload_builds_queue() -> None:
    calm = simulate_virtual(small(arrival_rate_rps=1.0, num_requests=100))
    busy = simulate_virtual(small(arrival_rate_rps=200.0, num_requests=100))
    assert max(o.queue_ms for o in busy.outcomes) > 10 * max(o.queue_ms for o in calm.outcomes)


def test_contention_slows_tokens_and_saturates_throughput() -> None:
    def tps(c: int) -> float:
        r = simulate_virtual(small(concurrency=c, num_requests=200, batch_contention=0.5))
        return sum(o.generated_tokens for o in r.outcomes) / r.duration_s

    assert tps(4) > tps(1)
    assert tps(64) < tps(16) * 1.2  # near-flat at high concurrency


def test_prefill_scales_with_prompt_and_cache_hits_are_cheaper() -> None:
    plan = build_plan(small(num_requests=300))
    hits = [p for p in plan if p.cache_hit]
    misses = [p for p in plan if not p.cache_hit]
    assert hits and misses
    per_tok = lambda ps: sum(p.prefill_ms for p in ps) / sum(p.prompt_tokens for p in ps)
    assert per_tok(hits) < per_tok(misses)


def test_regression_injection_multiplies_prefill() -> None:
    base = build_plan(small())
    slow = build_plan(small(regression=RegressionInjection(prefill_multiplier=2.0)))
    assert [s.prefill_ms for s in slow] == pytest.approx([2 * b.prefill_ms for b in base])


def test_failure_and_cancel_rates() -> None:
    r = simulate_virtual(small(num_requests=1000, failure_rate=0.1, cancel_rate=0.05))
    statuses = [o.status for o in r.outcomes]
    assert 70 <= statuses.count("error") <= 130
    assert 25 <= statuses.count("cancelled") <= 80


def test_memory_pressure_slows_requests_and_is_reported() -> None:
    from inference_profiler.simulator import MemoryConfig

    tight = small(
        concurrency=16, memory=MemoryConfig(capacity_bytes=200 * 1024**2, pressure_threshold=0.5)
    )
    r = simulate_virtual(tight)
    assert r.simulated_memory_peak_bytes > 0.5 * tight.memory.capacity_bytes
    assert any(o.memory_pressure_slowdown for o in r.outcomes)


def test_cache_capacity_drives_evictions() -> None:
    from inference_profiler.simulator import CacheConfig

    r = build_plan(
        small(
            num_requests=300,
            cache=CacheConfig(capacity_entries=4, prefix_pool=500, prefix_skew=1.0),
        )
    )
    assert sum(p.cache_evictions for p in r) > 200


def test_load_example_scenarios() -> None:
    scenarios = load_scenarios(SCENARIOS)
    names = [s.name for s in scenarios]
    assert {"baseline", "congestion", "memory-pressure", "cache-thrash"} <= set(names)
    only = load_scenarios(SCENARIOS, ["baseline"])
    assert [s.name for s in only] == ["baseline"]
    with pytest.raises(ScenarioError, match="unknown scenario"):
        load_scenarios(SCENARIOS, ["nope"])


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError, match="unknown key"):
        scenario_from_dict({"name": "x", "concurency": 3})
    with pytest.raises(ScenarioError, match="unknown key"):
        scenario_from_dict({"name": "x", "cache": {"size": 3}})
    bad = tmp_path / "s.yaml"
    bad.write_text("scenarios: 3\n")
    with pytest.raises(ScenarioError):
        load_scenarios(bad)


@pytest.mark.parametrize(
    "kw",
    [
        {"num_requests": 0},
        {"concurrency": 0},
        {"failure_rate": 1.5},
        {"failure_rate": 0.6, "cancel_rate": 0.6},
        {"time_scale": 0},
    ],
)
def test_invalid_scenarios_rejected(kw: dict[str, object]) -> None:
    with pytest.raises(ScenarioError):
        small(**kw)


def test_wall_clock_run_emits_matching_telemetry() -> None:
    h = make_harness()
    s = small(
        num_requests=12,
        concurrency=3,
        generated_tokens=Distribution(5),
        failure_rate=0.0,
        time_scale=0.05,
    )
    try:
        run = asyncio.run(run_wall_clock(s, h.profiler))
        assert len(run.outcomes) == 12
        assert all(o.status == "ok" for o in run.outcomes)
        assert h.total(m.REQUESTS, **{m.ATTR_STATUS: "ok"}) == 12
        assert h.total(m.TOKENS, **{m.ATTR_TOKEN_TYPE: "output"}) == 60
        assert h.total(m.INTER_TOKEN_LATENCY) == 12 * 4
        roots = [sp for sp in h.finished() if sp.name == "inference.request"]
        assert len(roots) == 12
        assert all(sp.attributes["inference.simulated"] is True for sp in roots)
        assert run.process_rss["peak_bytes"] >= run.process_rss["start_bytes"] > 0
    finally:
        h.profiler.shutdown()


def test_wall_clock_failures_recorded_as_errors() -> None:
    h = make_harness()
    s = dataclasses.replace(
        small(
            num_requests=40,
            concurrency=8,
            failure_rate=0.5,
            generated_tokens=Distribution(4),
            time_scale=0.02,
        ),
        seed=3,
    )
    try:
        run = asyncio.run(run_wall_clock(s, h.profiler))
        failed = sum(1 for o in run.outcomes if o.status == "error")
        assert failed > 0
        assert h.total(m.REQUESTS, **{m.ATTR_STATUS: "error"}) == failed
    finally:
        h.profiler.shutdown()

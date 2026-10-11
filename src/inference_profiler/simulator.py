"""Deterministic synthetic inference workload.

A scenario is turned into a :class:`RequestPlan` per request using a seeded RNG.
The plan fixes every random choice (token counts, per-token delays, cache
hits, failures), so the same scenario + seed always produces the same plan.

The plan can then be executed two ways:

* :func:`simulate_virtual` - a discrete-event simulation on a virtual clock.
  No sleeping, no telemetry, fully deterministic. Used for correctness tests and
  CI regression checks.
* :func:`run_wall_clock` - executes the plan with real ``asyncio.sleep`` calls
  through the :class:`~inference_profiler.Profiler`, producing real spans and
  metrics. Durations are wall-clock measurements of a *simulated* workload.

Model of the service (all values configurable):

* ``concurrency`` execution slots. With ``arrival_rate_rps > 0`` requests arrive
  as a Poisson process and wait for a free slot (queueing emerges naturally);
  with ``arrival_rate_rps = 0`` ``concurrency`` closed-loop clients send
  back-to-back requests.
* queue time = slot wait + ``scheduler_delay_ms``.
* prefill = ``prefill_base_ms + prefill_ms_per_token * prompt_tokens`` (reduced to
  ``cache.hit_prefill_fraction`` on a prefix-cache hit).
* TTFT = queue + cache lookup + prefill + ``first_token_overhead_ms``.
* inter-token latency is sampled per token and multiplied by
  ``1 + batch_contention * (concurrency - 1)`` to model batching contention, and by
  ``memory.pressure_slowdown`` while simulated KV memory exceeds
  ``memory.pressure_threshold``.
"""

from __future__ import annotations

import asyncio
import heapq
import math
import random
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from inference_profiler.instrumentation import Profiler, RequestCancelled
from inference_profiler.memory import SimulatedMemoryModel, bounded_allocation

SIMULATOR_VERSION = "1"


class ScenarioError(ValueError):
    """Invalid scenario definition."""


class SimulatedInferenceError(RuntimeError):
    """Injected failure raised by the simulated model."""


# ===================================================================== config


@dataclass(frozen=True, slots=True)
class Distribution:
    """Clipped normal distribution. ``stddev=0`` gives a constant."""

    mean: float
    stddev: float = 0.0
    min: float = 0.0
    max: float = math.inf

    def __post_init__(self) -> None:
        if self.stddev < 0 or self.min > self.max or not math.isfinite(self.mean):
            raise ScenarioError(f"invalid distribution {self}")

    def sample(self, rng: random.Random) -> float:
        value = rng.gauss(self.mean, self.stddev) if self.stddev > 0 else self.mean
        return min(self.max, max(self.min, value))

    def sample_int(self, rng: random.Random) -> int:
        return round(self.sample(rng))


@dataclass(frozen=True, slots=True)
class CacheConfig:
    enabled: bool = True
    capacity_entries: int = 256
    #: Number of distinct prompt prefixes requests draw from.
    prefix_pool: int = 64
    #: >1 concentrates traffic on few prefixes (higher hit ratio).
    prefix_skew: float = 2.0
    #: Fraction of prefill cost remaining on a cache hit.
    hit_prefill_fraction: float = 0.3
    lookup_ms: float = 0.2
    insert_ms: float = 0.3
    evict_ms: float = 0.1


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    #: SIMULATED KV-cache bytes per resident token.
    bytes_per_token: int = 131_072
    capacity_bytes: int = 8 * 1024**3
    pressure_threshold: float = 0.9
    #: Inter-token latency multiplier while pressure exceeds the threshold.
    pressure_slowdown: float = 1.5
    #: Optional real, bounded allocation (<= 64 MiB) held during wall-clock runs.
    real_allocation_mb: int = 0


@dataclass(frozen=True, slots=True)
class RegressionInjection:
    """Deliberate slowdowns used to test regression detection."""

    prefill_multiplier: float = 1.0
    inter_token_multiplier: float = 1.0
    queue_delay_add_ms: float = 0.0
    first_token_add_ms: float = 0.0

    @property
    def active(self) -> bool:
        return (
            self.prefill_multiplier,
            self.inter_token_multiplier,
            self.queue_delay_add_ms,
            self.first_token_add_ms,
        ) != (1.0, 1.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    name: str
    description: str = ""
    #: Scenarios with the same group form a concurrency sweep for saturation analysis.
    group: str | None = None
    seed: int = 42
    model: str = "demo-model"
    runtime_version: str = "sim-1.0"
    num_requests: int = 200
    concurrency: int = 8
    #: Poisson arrival rate; 0 means closed-loop clients.
    arrival_rate_rps: float = 0.0
    prompt_tokens: Distribution = Distribution(512, 128, 16, 4096)
    generated_tokens: Distribution = Distribution(64, 16, 1, 1024)
    scheduler_delay_ms: Distribution = Distribution(1.0, 0.5, 0.0, 50.0)
    prefill_base_ms: float = 4.0
    prefill_ms_per_token: float = 0.04
    first_token_overhead_ms: float = 2.0
    inter_token_ms: Distribution = Distribution(6.0, 1.0, 0.5, 100.0)
    batch_contention: float = 0.05
    failure_rate: float = 0.0
    cancel_rate: float = 0.0
    cache: CacheConfig = CacheConfig()
    memory: MemoryConfig = MemoryConfig()
    regression: RegressionInjection = RegressionInjection()
    #: Wall-clock mode multiplies every sleep by this factor.
    time_scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ScenarioError("scenario name is required")
        if not 1 <= self.num_requests <= 1_000_000:
            raise ScenarioError(f"{self.name}: num_requests must be in [1, 1000000]")
        if not 1 <= self.concurrency <= 10_000:
            raise ScenarioError(f"{self.name}: concurrency must be in [1, 10000]")
        if self.arrival_rate_rps < 0:
            raise ScenarioError(f"{self.name}: arrival_rate_rps must be >= 0")
        for label, rate in (("failure_rate", self.failure_rate), ("cancel_rate", self.cancel_rate)):
            if not 0 <= rate <= 1:
                raise ScenarioError(f"{self.name}: {label} must be in [0, 1]")
        if self.failure_rate + self.cancel_rate > 1:
            raise ScenarioError(f"{self.name}: failure_rate + cancel_rate must be <= 1")
        if self.time_scale <= 0 or self.batch_contention < 0:
            raise ScenarioError(f"{self.name}: time_scale must be > 0, batch_contention >= 0")
        if self.cache.capacity_entries < 1 or self.cache.prefix_pool < 1:
            raise ScenarioError(f"{self.name}: cache capacity_entries/prefix_pool must be >= 1")
        if not 0 <= self.cache.hit_prefill_fraction <= 1:
            raise ScenarioError(f"{self.name}: cache.hit_prefill_fraction must be in [0, 1]")
        if self.memory.capacity_bytes <= 0 or self.memory.bytes_per_token < 0:
            raise ScenarioError(f"{self.name}: invalid memory configuration")
        if not 0 <= self.memory.real_allocation_mb <= 64:
            raise ScenarioError(f"{self.name}: memory.real_allocation_mb must be in [0, 64]")
        for label, value in (
            ("prefill_multiplier", self.regression.prefill_multiplier),
            ("inter_token_multiplier", self.regression.inter_token_multiplier),
        ):
            if value <= 0:
                raise ScenarioError(f"{self.name}: regression.{label} must be > 0")

    @property
    def contention_factor(self) -> float:
        return 1.0 + self.batch_contention * (self.concurrency - 1)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = _jsonable(asdict(self))
        return data


_NESTED: dict[str, type] = {
    "prompt_tokens": Distribution,
    "generated_tokens": Distribution,
    "scheduler_delay_ms": Distribution,
    "inter_token_ms": Distribution,
    "cache": CacheConfig,
    "memory": MemoryConfig,
    "regression": RegressionInjection,
}


def scenario_from_dict(data: Mapping[str, Any]) -> ScenarioConfig:
    """Build a scenario, rejecting unknown keys so typos are not silently ignored."""
    _reject_unknown(ScenarioConfig, data, "scenario")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        nested = _NESTED.get(key)
        if nested is None:
            kwargs[key] = value
        elif nested is Distribution and isinstance(value, (int, float)):
            kwargs[key] = Distribution(float(value))
        elif isinstance(value, Mapping):
            _reject_unknown(nested, value, f"{data.get('name', '?')}.{key}")
            kwargs[key] = nested(**value)
        else:
            raise ScenarioError(f"{key} must be a mapping")
    try:
        return ScenarioConfig(**kwargs)
    except TypeError as exc:
        raise ScenarioError(str(exc)) from exc


def load_scenarios(path: str | Path, names: list[str] | None = None) -> list[ScenarioConfig]:
    """Load scenarios from YAML: ``{defaults: {...}, scenarios: [{name: ...}, ...]}``."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("scenarios"), list):
        raise ScenarioError(f"{path}: expected a mapping with a 'scenarios' list")
    defaults = raw.get("defaults") or {}
    scenarios = []
    for item in raw["scenarios"]:
        merged = _deep_merge(defaults, item)
        scenarios.append(scenario_from_dict(merged))
    seen = set()
    for s in scenarios:
        if s.name in seen:
            raise ScenarioError(f"duplicate scenario name {s.name!r}")
        seen.add(s.name)
    if names:
        missing = set(names) - seen
        if missing:
            raise ScenarioError(f"unknown scenario(s): {sorted(missing)}")
        scenarios = [s for s in scenarios if s.name in names]
    return scenarios


# ====================================================================== plans


@dataclass(slots=True)
class RequestPlan:
    index: int
    request_id: str
    #: Arrival offset in seconds (open-loop only; closed-loop ignores it).
    arrival_s: float
    prompt_tokens: int
    generated_tokens: int
    scheduler_delay_ms: float
    prefix_id: int
    cache_hit: bool
    cache_evictions: int
    #: Simulated cache entries resident after this request's lookup/insert.
    cache_entries: int
    prefill_ms: float
    first_token_overhead_ms: float
    inter_token_ms: list[float]
    fail_at_token: int | None = None
    cancel_at_token: int | None = None


def build_plan(s: ScenarioConfig) -> list[RequestPlan]:
    """Deterministically expand a scenario into per-request plans."""
    rng = random.Random(s.seed)
    inj = s.regression
    lru: OrderedDict[int, None] = OrderedDict()
    plans: list[RequestPlan] = []
    t = 0.0
    for i in range(s.num_requests):
        if s.arrival_rate_rps > 0:
            t += rng.expovariate(s.arrival_rate_rps)
        prompt = max(1, s.prompt_tokens.sample_int(rng))
        generated = max(1, s.generated_tokens.sample_int(rng))
        sched = s.scheduler_delay_ms.sample(rng) + inj.queue_delay_add_ms

        prefix_id = int((rng.random() ** s.cache.prefix_skew) * s.cache.prefix_pool)
        hit = False
        evictions = 0
        if s.cache.enabled:
            if prefix_id in lru:
                hit = True
                lru.move_to_end(prefix_id)
            else:
                lru[prefix_id] = None
                while len(lru) > s.cache.capacity_entries:
                    lru.popitem(last=False)
                    evictions += 1

        prefill = s.prefill_base_ms + s.prefill_ms_per_token * prompt
        if hit:
            prefill *= s.cache.hit_prefill_fraction
        prefill *= inj.prefill_multiplier
        itl_scale = s.contention_factor * inj.inter_token_multiplier
        itl = [s.inter_token_ms.sample(rng) * itl_scale for _ in range(generated - 1)]

        fail_at = cancel_at = None
        roll = rng.random()
        if roll < s.failure_rate:
            fail_at = rng.randrange(generated)
        elif roll < s.failure_rate + s.cancel_rate:
            cancel_at = rng.randrange(generated)

        plans.append(
            RequestPlan(
                index=i,
                request_id=f"{s.name}-{s.seed}-{i:06d}",
                arrival_s=t,
                prompt_tokens=prompt,
                generated_tokens=generated,
                scheduler_delay_ms=sched,
                prefix_id=prefix_id,
                cache_hit=hit,
                cache_evictions=evictions,
                cache_entries=len(lru),
                prefill_ms=prefill,
                first_token_overhead_ms=s.first_token_overhead_ms + inj.first_token_add_ms,
                inter_token_ms=itl,
                fail_at_token=fail_at,
                cancel_at_token=cancel_at,
            )
        )
    return plans


# =================================================================== outcomes


@dataclass(slots=True)
class RequestOutcome:
    index: int
    status: str
    start_s: float
    end_s: float
    queue_ms: float
    prefill_ms: float
    ttft_ms: float | None
    generation_ms: float
    e2e_ms: float
    prompt_tokens: int
    generated_tokens: int
    cache_hit: bool
    cache_evictions: int
    cache_lookup_ms: float
    inter_token_ms: list[float] = field(default_factory=list)
    memory_pressure_slowdown: bool = False


@dataclass(slots=True)
class SimulationRun:
    scenario: ScenarioConfig
    clock: str
    outcomes: list[RequestOutcome]
    duration_s: float
    simulated_memory_peak_bytes: int
    simulated_memory_capacity_bytes: int
    process_rss: dict[str, int] = field(default_factory=dict)
    process_rss_source: str = "psutil"


# ============================================================ virtual clock


def simulate_virtual(s: ScenarioConfig, plan: list[RequestPlan] | None = None) -> SimulationRun:
    """Discrete-event simulation; deterministic for a given scenario."""
    plan = plan if plan is not None else build_plan(s)
    slow = [False] * len(plan)
    outcomes = _virtual_pass(s, plan, slow)
    peak, pressure_at_start = _memory_sweep(s, plan, outcomes)
    flagged = [p > s.memory.pressure_threshold for p in pressure_at_start]
    if any(flagged) and s.memory.pressure_slowdown != 1.0:
        outcomes = _virtual_pass(s, plan, flagged)
        peak, _ = _memory_sweep(s, plan, outcomes)
    duration = max((o.end_s for o in outcomes), default=0.0) - min(
        (o.start_s - o.queue_ms / 1000.0 for o in outcomes), default=0.0
    )
    return SimulationRun(
        scenario=s,
        clock="virtual",
        outcomes=outcomes,
        duration_s=max(duration, 1e-9),
        simulated_memory_peak_bytes=peak,
        simulated_memory_capacity_bytes=s.memory.capacity_bytes,
    )


def _virtual_pass(
    s: ScenarioConfig, plan: list[RequestPlan], slow: list[bool]
) -> list[RequestOutcome]:
    slots = [0.0] * s.concurrency  # next-free time per execution slot / client
    heapq.heapify(slots)
    outcomes = []
    for p in plan:
        free = heapq.heappop(slots)
        arrival = p.arrival_s if s.arrival_rate_rps > 0 else free
        start = max(arrival, free) + p.scheduler_delay_ms / 1000.0
        queue_ms = (start - arrival) * 1000.0
        lookup_ms = s.cache.lookup_ms if s.cache.enabled else 0.0
        factor = s.memory.pressure_slowdown if slow[p.index] else 1.0
        produced, itl, status = _generation_tokens(p, factor)
        ttft_ms = queue_ms + lookup_ms + p.prefill_ms + p.first_token_overhead_ms
        gen_ms = p.first_token_overhead_ms + sum(itl)
        tail_ms = 0.0
        if status == "ok" and s.cache.enabled and not p.cache_hit:
            tail_ms += s.cache.insert_ms + (s.cache.evict_ms if p.cache_evictions else 0.0)
        service_ms = lookup_ms + p.prefill_ms + gen_ms + tail_ms
        end = start + service_ms / 1000.0
        heapq.heappush(slots, end)
        outcomes.append(
            RequestOutcome(
                index=p.index,
                status=status,
                start_s=start,
                end_s=end,
                queue_ms=queue_ms,
                prefill_ms=p.prefill_ms,
                ttft_ms=ttft_ms if produced > 0 else None,
                generation_ms=gen_ms,
                e2e_ms=queue_ms + service_ms,
                prompt_tokens=p.prompt_tokens,
                generated_tokens=produced,
                cache_hit=p.cache_hit,
                cache_evictions=p.cache_evictions,
                cache_lookup_ms=lookup_ms,
                inter_token_ms=itl,
                memory_pressure_slowdown=slow[p.index],
            )
        )
    return outcomes


def _generation_tokens(p: RequestPlan, factor: float) -> tuple[int, list[float], str]:
    """Tokens actually produced, their inter-token latencies, and the final status."""
    stop = None
    status = "ok"
    if p.fail_at_token is not None:
        stop, status = p.fail_at_token, "error"
    elif p.cancel_at_token is not None:
        stop, status = p.cancel_at_token, "cancelled"
    produced = p.generated_tokens if stop is None else stop
    itl = [v * factor for v in p.inter_token_ms[: max(0, produced - 1)]]
    return produced, itl, status


def _memory_sweep(
    s: ScenarioConfig, plan: list[RequestPlan], outcomes: list[RequestOutcome]
) -> tuple[int, list[float]]:
    """Peak simulated KV bytes and pressure observed at each request's start."""
    events: list[tuple[float, int, int, int]] = []  # (time, order, delta, index)
    for p, o in zip(plan, outcomes, strict=True):
        resident = (p.prompt_tokens + o.generated_tokens) * s.memory.bytes_per_token
        events.append((o.start_s, 1, resident, p.index))
        events.append((o.end_s, 0, -resident, p.index))  # releases sort before starts
    events.sort()
    used = peak = 0
    at_start = [0.0] * len(plan)
    for _t, kind, delta, idx in events:
        used += delta
        peak = max(peak, used)
        if kind == 1:
            at_start[idx] = used / s.memory.capacity_bytes
    return peak, at_start


# ============================================================== wall clock


async def run_wall_clock(
    s: ScenarioConfig, profiler: Profiler, plan: list[RequestPlan] | None = None
) -> SimulationRun:
    """Execute the plan with real sleeps through ``profiler`` (emits real telemetry)."""
    from inference_profiler.memory import MemorySampler

    plan = plan if plan is not None else build_plan(s)
    scale = s.time_scale
    mem = SimulatedMemoryModel(s.memory.bytes_per_token, s.memory.capacity_bytes)
    sem = asyncio.Semaphore(s.concurrency)
    outcomes: list[RequestOutcome | None] = [None] * len(plan)
    sampler = MemorySampler(interval_ms=50)
    sampler.start()
    rss_start = sampler.latest_bytes
    ballast = bounded_allocation(s.memory.real_allocation_mb * 1024 * 1024)
    t0 = time.perf_counter()

    async def sleep_ms(ms: float) -> None:
        if ms > 0:
            await asyncio.sleep(ms * scale / 1000.0)

    async def one(p: RequestPlan) -> None:
        arrival = time.perf_counter()
        queue_ms = prefill_ms = gen_ms = lookup_ms = 0.0
        ttft_ms: float | None = None
        itl_measured: list[float] = []
        produced = 0
        status = "ok"
        slowed = False
        acquired = False
        resident = 0
        req = profiler.inference_request(
            model=s.model,
            request_id=p.request_id,
            prompt_tokens=p.prompt_tokens,
            runtime_version=s.runtime_version,
            attributes={"inference.simulated": True, "inference.scenario": s.name},
        )
        try:
            async with req:
                q0 = time.perf_counter()
                with req.queue():
                    await sem.acquire()
                    acquired = True
                    await sleep_ms(p.scheduler_delay_ms)
                queue_ms = (time.perf_counter() - q0) * 1000.0
                if s.cache.enabled:
                    profiler.set_cache_utilization(
                        p.cache_entries / s.cache.capacity_entries, model=s.model
                    )
                    c0 = time.perf_counter()
                    with req.cache_lookup() as lookup:
                        await sleep_ms(s.cache.lookup_ms)
                        lookup.hit = p.cache_hit
                    lookup_ms = (time.perf_counter() - c0) * 1000.0
                p0 = time.perf_counter()
                with req.prefill():
                    await sleep_ms(p.prefill_ms)
                prefill_ms = (time.perf_counter() - p0) * 1000.0
                resident = p.prompt_tokens
                mem.allocate(p.prompt_tokens)
                profiler.set_simulated_memory(mem.used_bytes, mem.pressure, model=s.model)
                g0 = time.perf_counter()
                try:
                    with req.generation():
                        await sleep_ms(p.first_token_overhead_ms)
                        stop = p.fail_at_token if p.fail_at_token is not None else p.cancel_at_token
                        if stop == 0:
                            _raise_injected(p)
                        req.mark_first_token()
                        ttft_ms = (time.perf_counter() - arrival) * 1000.0
                        produced = 1
                        last = time.perf_counter()
                        for delay in p.inter_token_ms:
                            if stop is not None and produced >= stop:
                                _raise_injected(p)
                            factor = 1.0
                            if mem.pressure > s.memory.pressure_threshold:
                                factor, slowed = s.memory.pressure_slowdown, True
                            await sleep_ms(delay * factor)
                            req.record_token()
                            now = time.perf_counter()
                            itl_measured.append((now - last) * 1000.0)
                            last = now
                            produced += 1
                            resident += 1
                            mem.allocate(1)
                finally:
                    gen_ms = (time.perf_counter() - g0) * 1000.0
                if s.cache.enabled and not p.cache_hit:
                    with req.cache_insert():
                        await sleep_ms(s.cache.insert_ms)
                    if p.cache_evictions:
                        with req.cache_evict(entries=p.cache_evictions):
                            await sleep_ms(s.cache.evict_ms)
                req.record_generation(produced)
        except SimulatedInferenceError:
            status = "error"
        except RequestCancelled:
            status = "cancelled"
        finally:
            if acquired:
                sem.release()
            mem.release(resident)
        end = time.perf_counter()
        outcomes[p.index] = RequestOutcome(
            index=p.index,
            status=status,
            start_s=arrival - t0,
            end_s=end - t0,
            queue_ms=queue_ms,
            prefill_ms=prefill_ms,
            ttft_ms=ttft_ms,
            generation_ms=gen_ms,
            e2e_ms=(end - arrival) * 1000.0,
            prompt_tokens=p.prompt_tokens,
            generated_tokens=produced,
            cache_hit=p.cache_hit,
            cache_evictions=p.cache_evictions,
            cache_lookup_ms=lookup_ms,
            inter_token_ms=itl_measured,
            memory_pressure_slowdown=slowed,
        )

    try:
        if s.arrival_rate_rps > 0:

            async def arrive(p: RequestPlan) -> None:
                await asyncio.sleep(p.arrival_s * scale)
                await one(p)

            await asyncio.gather(*(arrive(p) for p in plan))
        else:
            pending = iter(plan)

            async def client() -> None:
                for p in pending:
                    await one(p)

            await asyncio.gather(*(client() for _ in range(s.concurrency)))
    finally:
        duration = time.perf_counter() - t0
        del ballast
        sampler.stop()
    return SimulationRun(
        scenario=s,
        clock="wall",
        outcomes=[o for o in outcomes if o is not None],
        duration_s=duration,
        simulated_memory_peak_bytes=mem.peak_bytes,
        simulated_memory_capacity_bytes=s.memory.capacity_bytes,
        process_rss={
            "start_bytes": rss_start,
            "peak_bytes": sampler.peak_bytes,
            "end_bytes": sampler.latest_bytes,
            "samples": sampler.samples,
        },
    )


def _raise_injected(p: RequestPlan) -> None:
    if p.fail_at_token is not None:
        raise SimulatedInferenceError(f"injected failure in {p.request_id}")
    raise RequestCancelled(f"injected cancellation in {p.request_id}")


# ==================================================================== helpers


def _reject_unknown(cls: type, data: Mapping[str, Any], where: str) -> None:
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ScenarioError(
            f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}"
        )


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and math.isinf(value):
        return None
    return value

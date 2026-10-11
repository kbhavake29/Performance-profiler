"""Heuristic bottleneck analysis over benchmark results or live Prometheus data.

Every rule reports the observed value, the configured threshold, a severity,
and *what to investigate*. Findings are hypotheses for an engineer to verify,
not proven root causes; when data is missing the rule says so instead of
guessing.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import yaml

Severity = Literal["info", "warning", "critical"]
_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


@dataclass(frozen=True, slots=True)
class AnalysisThresholds:
    """All thresholds used by the rules. Override via YAML (``--thresholds``)."""

    min_samples: int = 20
    queue_share_warning: float = 0.25
    queue_share_critical: float = 0.5
    prefill_ms_per_token_warning: float = 0.5
    min_tokens_per_second: float = 20.0
    inter_token_p95_ms_warning: float = 75.0
    inter_token_p95_ms_critical: float = 150.0
    tail_ratio_warning: float = 3.0
    ttft_p95_ms_warning: float = 1_000.0
    error_rate_warning: float = 0.01
    error_rate_critical: float = 0.05
    memory_pressure_warning: float = 0.85
    memory_pressure_critical: float = 0.95
    rss_bytes_warning: int | None = None
    cache_hit_ratio_warning: float = 0.5
    eviction_ratio_warning: float = 0.5
    #: Saturation: throughput gain per concurrency doubling below this fraction...
    saturation_min_gain: float = 0.15
    #: ...while p95 end-to-end latency grows by more than this fraction.
    saturation_latency_growth: float = 0.25

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AnalysisThresholds:
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown threshold(s): {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_file(cls, path: str | Path) -> AnalysisThresholds:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        section = data.get("analysis", data)
        if not isinstance(section, Mapping):
            raise ValueError(f"{path}: expected a mapping of thresholds")
        return cls.from_mapping(section)


@dataclass(slots=True)
class Finding:
    rule: str
    severity: Severity
    title: str
    scenario: str | None
    observed: str
    threshold: str
    explanation: str
    recommendation: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnalysisResult:
    findings: list[Finding]
    limitations: list[str]
    scenarios_analyzed: int

    @property
    def max_severity(self) -> Severity | None:
        if not self.findings:
            return None
        return max(self.findings, key=lambda f: _SEVERITY_ORDER[f.severity]).severity

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenarios_analyzed": self.scenarios_analyzed,
            "max_severity": self.max_severity,
            "findings": [f.to_dict() for f in self.findings],
            "limitations": self.limitations,
        }


def _lat(scn: Mapping[str, Any], key: str, stat: str) -> float | None:
    dist = (scn.get("latency_ms") or {}).get(key)
    if not dist:
        return None
    value = dist.get(stat)
    return float(value) if value is not None else None


def analyze_report(
    report: Mapping[str, Any], thresholds: AnalysisThresholds | None = None
) -> AnalysisResult:
    t = thresholds or AnalysisThresholds()
    findings: list[Finding] = []
    limitations: list[str] = []
    if report.get("measurement", {}).get("clock") == "virtual":
        limitations.append(
            "Report uses the virtual clock: latencies come from the simulator model, so findings "
            "describe the configured scenario rather than measured behaviour."
        )
    scenarios = list(report.get("scenarios", []))
    for scn in scenarios:
        findings.extend(_scenario_rules(scn, t, limitations))
    findings.extend(_saturation_rule(scenarios, t, limitations))
    findings.sort(key=lambda f: (-_SEVERITY_ORDER[f.severity], f.scenario or "", f.rule))
    return AnalysisResult(
        findings=findings, limitations=limitations, scenarios_analyzed=len(scenarios)
    )


def _scenario_rules(
    scn: Mapping[str, Any], t: AnalysisThresholds, limitations: list[str]
) -> list[Finding]:
    name = scn.get("name")
    out: list[Finding] = []
    req = scn.get("requests", {})
    total = int(req.get("total", 0))
    if total < t.min_samples:
        limitations.append(
            f"{name}: only {total} requests (< {t.min_samples}); p95/p99 are unreliable and "
            "percentile-based rules may be noisy."
        )

    # ---- queueing
    q50, e50 = _lat(scn, "queue", "p50"), _lat(scn, "end_to_end", "p50")
    if q50 is not None and e50:
        share = q50 / e50
        if share >= t.queue_share_warning:
            sev: Severity = "critical" if share >= t.queue_share_critical else "warning"
            out.append(
                Finding(
                    rule="high_queue_latency",
                    severity=sev,
                    scenario=name,
                    title="Requests spend a large share of latency waiting in the queue",
                    observed=f"queue p50 {q50:.1f} ms = {share:.0%} of end-to-end p50 {e50:.1f} ms",
                    threshold=f"warning >= {t.queue_share_warning:.0%}, critical >= "
                    f"{t.queue_share_critical:.0%}",
                    explanation="Time is lost before execution starts, which points at admission or "
                    "capacity rather than model compute.",
                    recommendation="Compare arrival rate with sustainable throughput; check scheduler "
                    "batch limits, worker count and queue-depth metrics "
                    "(inference_queue_depth).",
                    evidence={
                        "queue_p50_ms": q50,
                        "end_to_end_p50_ms": e50,
                        "share": round(share, 4),
                    },
                )
            )
    elif q50 is None:
        limitations.append(f"{name}: no queue latency recorded; queueing rules skipped.")

    # ---- prefill
    pf50 = _lat(scn, "prefill", "p50")
    prompt_mean = (scn.get("tokens") or {}).get("prompt_mean")
    if pf50 is not None and prompt_mean:
        per_tok = pf50 / float(prompt_mean)
        if per_tok >= t.prefill_ms_per_token_warning:
            out.append(
                Finding(
                    rule="excessive_prefill",
                    severity="warning",
                    scenario=name,
                    title="Prefill is slow per prompt token",
                    observed=f"prefill p50 {pf50:.1f} ms for ~{prompt_mean:.0f} prompt tokens "
                    f"= {per_tok:.3f} ms/token",
                    threshold=f">= {t.prefill_ms_per_token_warning} ms/token",
                    explanation="Prompt processing cost per token is high for this configuration.",
                    recommendation="Check prefill batching/chunking, attention kernel selection and "
                    "prefix-cache effectiveness; compare against a known-good runtime.",
                    evidence={"prefill_p50_ms": pf50, "ms_per_prompt_token": round(per_tok, 5)},
                )
            )
    ttft95 = _lat(scn, "time_to_first_token", "p95")
    if ttft95 is not None and ttft95 >= t.ttft_p95_ms_warning:
        out.append(
            Finding(
                rule="slow_ttft",
                severity="warning",
                scenario=name,
                title="Time to first token p95 is high",
                observed=f"TTFT p95 {ttft95:.1f} ms",
                threshold=f">= {t.ttft_p95_ms_warning:.0f} ms",
                explanation="Users wait long before streaming starts.",
                recommendation="Break TTFT into queue + prefill using the Latency Breakdown dashboard.",
                evidence={"ttft_p95_ms": ttft95},
            )
        )

    # ---- decode
    tps = (scn.get("throughput") or {}).get("per_request_tokens_per_second") or {}
    tps50 = tps.get("p50")
    if tps50 is not None and tps50 < t.min_tokens_per_second:
        out.append(
            Finding(
                rule="slow_generation",
                severity="warning",
                scenario=name,
                title="Per-request generation throughput is low",
                observed=f"per-request decode p50 {tps50:.1f} tokens/s",
                threshold=f"< {t.min_tokens_per_second} tokens/s",
                explanation="Each request decodes slowly once generation starts.",
                recommendation="Inspect inter-token latency, batch size and memory-bandwidth limits.",
                evidence={"tokens_per_second_p50": tps50},
            )
        )
    itl95 = _lat(scn, "inter_token", "p95")
    if itl95 is None:
        limitations.append(f"{name}: no inter-token samples; decode rules partially skipped.")
    elif itl95 >= t.inter_token_p95_ms_warning:
        sev = "critical" if itl95 >= t.inter_token_p95_ms_critical else "warning"
        out.append(
            Finding(
                rule="high_inter_token_latency",
                severity=sev,
                scenario=name,
                title="Inter-token latency p95 is high",
                observed=f"inter-token p95 {itl95:.1f} ms",
                threshold=f"warning >= {t.inter_token_p95_ms_warning} ms, critical >= "
                f"{t.inter_token_p95_ms_critical} ms",
                explanation="Streaming output stutters; decode steps are slow or delayed.",
                recommendation="Correlate with active requests (batch contention) and memory "
                "pressure; check for preemption/swapping.",
                evidence={"inter_token_p95_ms": itl95},
            )
        )

    # ---- tail latency
    e99 = _lat(scn, "end_to_end", "p99")
    if e99 is not None and e50:
        ratio = e99 / e50
        if ratio >= t.tail_ratio_warning:
            out.append(
                Finding(
                    rule="long_tail_latency",
                    severity="warning",
                    scenario=name,
                    title="End-to-end latency has a long tail",
                    observed=f"p99/p50 = {ratio:.2f} (p99 {e99:.1f} ms, p50 {e50:.1f} ms)",
                    threshold=f">= {t.tail_ratio_warning}",
                    explanation="A minority of requests are much slower than typical.",
                    recommendation="Open slow traces in Tempo (Distributed Traces dashboard) and "
                    "compare their phase breakdown with typical requests.",
                    evidence={"p99_ms": e99, "p50_ms": e50},
                )
            )

    # ---- errors
    err = float(req.get("error_rate", 0.0))
    if total and err >= t.error_rate_warning:
        sev = "critical" if err >= t.error_rate_critical else "warning"
        out.append(
            Finding(
                rule="error_rate",
                severity=sev,
                scenario=name,
                title="Requests are failing",
                observed=f"error rate {err:.2%} ({req.get('failed')} of {total})",
                threshold=f"warning >= {t.error_rate_warning:.0%}, critical >= "
                f"{t.error_rate_critical:.0%}",
                explanation="Failed requests also distort latency percentiles (only successful "
                "requests are included in end-to-end latency).",
                recommendation="Inspect error spans (status=error) and their error.type attribute.",
                evidence={"error_rate": err},
            )
        )

    # ---- memory
    mem = scn.get("memory") or {}
    sim = mem.get("simulated_kv_cache") or {}
    pressure = sim.get("peak_pressure")
    if pressure is not None and pressure >= t.memory_pressure_warning:
        sev = "critical" if pressure >= t.memory_pressure_critical else "warning"
        out.append(
            Finding(
                rule="memory_pressure",
                severity=sev,
                scenario=name,
                title="Simulated KV-cache memory pressure is high",
                observed=f"peak simulated pressure {pressure:.0%} "
                f"({sim.get('requests_slowed_by_pressure', 0)} requests slowed)",
                threshold=f"warning >= {t.memory_pressure_warning:.0%}, critical >= "
                f"{t.memory_pressure_critical:.0%}",
                explanation="This value comes from the simulator's memory model, not from a real "
                "allocator. In a real engine the same pattern leads to preemption or "
                "swapping.",
                recommendation="Reduce max concurrency or max sequence length, or add KV capacity; "
                "verify with real memory metrics from the engine.",
                evidence={"peak_pressure": pressure, "simulated": True},
            )
        )
    rss = mem.get("process_rss") or {}
    if t.rss_bytes_warning is not None and rss.get("peak_bytes", 0) >= t.rss_bytes_warning:
        out.append(
            Finding(
                rule="process_rss",
                severity="warning",
                scenario=name,
                title="Process resident memory exceeded the configured limit",
                observed=f"peak RSS {rss['peak_bytes'] / 2**20:.1f} MiB",
                threshold=f">= {t.rss_bytes_warning / 2**20:.1f} MiB",
                explanation="Real process memory measured with psutil.",
                recommendation="Profile allocations; check for growth across requests (leaks).",
                evidence={"peak_rss_bytes": rss["peak_bytes"]},
            )
        )

    # ---- cache
    cache = scn.get("cache") or {}
    hit_ratio = cache.get("hit_ratio")
    if cache.get("enabled") and hit_ratio is not None and hit_ratio < t.cache_hit_ratio_warning:
        out.append(
            Finding(
                rule="low_cache_hit_ratio",
                severity="warning",
                scenario=name,
                title="Cache hit ratio is low",
                observed=f"hit ratio {hit_ratio:.1%} ({cache.get('hits')} hits / "
                f"{cache.get('lookups')} lookups)",
                threshold=f"< {t.cache_hit_ratio_warning:.0%}",
                explanation="Most lookups miss, so the cache contributes little to prefill savings.",
                recommendation="Check cache capacity versus working set and key design "
                "(prefix hashing).",
                evidence={"hit_ratio": hit_ratio},
            )
        )
    ev_ratio = cache.get("eviction_ratio")
    if ev_ratio is not None and ev_ratio >= t.eviction_ratio_warning:
        out.append(
            Finding(
                rule="cache_eviction_churn",
                severity="warning",
                scenario=name,
                title="Cache is churning (evictions per insertion are high)",
                observed=f"{ev_ratio:.2f} evictions per insertion ({cache.get('evictions')} total)",
                threshold=f">= {t.eviction_ratio_warning}",
                explanation="Entries are evicted almost as fast as they are inserted, suggesting the "
                "working set exceeds capacity.",
                recommendation="Increase capacity or improve admission policy; watch "
                "inference_cache_evictions_total.",
                evidence={"eviction_ratio": ev_ratio},
            )
        )
    return out


def _saturation_rule(
    scenarios: list[Mapping[str, Any]], t: AnalysisThresholds, limitations: list[str]
) -> list[Finding]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for s in scenarios:
        if s.get("group"):
            groups.setdefault(str(s["group"]), []).append(s)
    out = []
    for group, unsorted in groups.items():
        members = sorted(unsorted, key=lambda s: int(s["concurrency"]))
        if len(members) < 2:
            limitations.append(
                f"group {group}: needs >= 2 concurrency levels for saturation analysis."
            )
            continue
        for prev, cur in pairwise(members):
            c0, c1 = int(prev["concurrency"]), int(cur["concurrency"])
            th0 = float(prev["throughput"]["generated_tokens_per_second"])
            th1 = float(cur["throughput"]["generated_tokens_per_second"])
            l0, l1 = _lat(prev, "end_to_end", "p95"), _lat(cur, "end_to_end", "p95")
            if c1 <= c0 or th0 <= 0 or not l0 or l1 is None:
                continue
            # normalise to "per doubling of concurrency"
            doublings = math.log2(c1 / c0)
            gain = (th1 / th0) ** (1 / doublings) - 1 if doublings > 0 else 0.0
            growth = (l1 / l0) ** (1 / doublings) - 1 if doublings > 0 else 0.0
            if gain < t.saturation_min_gain and growth > t.saturation_latency_growth:
                out.append(
                    Finding(
                        rule="throughput_saturation",
                        severity="warning",
                        scenario=str(cur["name"]),
                        title=f"Throughput saturates between concurrency {c0} and {c1}",
                        observed=f"token throughput {th0:.0f} -> {th1:.0f} tok/s "
                        f"({gain:+.0%} per doubling) while p95 latency "
                        f"{l0:.0f} -> {l1:.0f} ms ({growth:+.0%} per doubling)",
                        threshold=f"gain < {t.saturation_min_gain:.0%} and latency growth > "
                        f"{t.saturation_latency_growth:.0%} per doubling",
                        explanation="Adding concurrency mostly adds latency, not throughput: the "
                        "service is near capacity at this level.",
                        recommendation=f"Cap concurrency near {c0} for this configuration or add "
                        "capacity; confirm with a finer-grained sweep.",
                        evidence={
                            "group": group,
                            "concurrency": [c0, c1],
                            "tokens_per_second": [th0, th1],
                            "p95_ms": [l0, l1],
                        },
                    )
                )
    return out


# =========================================================== Prometheus source

PromQuery = Callable[[str], float | None]


def prometheus_query_fn(base_url: str, timeout_s: float = 5.0) -> PromQuery:
    """Return a function evaluating an instant PromQL query to a single float."""

    def query(expr: str) -> float | None:
        url = base_url.rstrip("/") + "/api/v1/query?" + urllib.parse.urlencode({"query": expr})
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            payload = json.loads(resp.read())
        if payload.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {payload}")
        result = payload["data"]["result"]
        if not result:
            return None
        value = float(result[0]["value"][1])
        return None if math.isnan(value) else value  # NaN -> None

    return query


def report_from_prometheus(
    query: PromQuery, window: str = "15m", service: str | None = None
) -> dict[str, Any]:
    """Build a report-shaped summary from live metrics so the same rules can run.

    Percentiles use ``histogram_quantile`` over bucket rates aggregated with
    ``sum by (le)`` - the statistically valid way to combine histograms.
    """
    sel = f'job="{service}"' if service else ""

    def q(name: str, quant: float) -> float | None:
        v = query(
            f"histogram_quantile({quant}, sum by (le) (rate({name}_bucket{{{sel}}}[{window}])))"
        )
        return None if v is None else v * 1000.0  # seconds -> ms

    def dist(name: str) -> dict[str, Any] | None:
        count = query(f"sum(increase({name}_count{{{sel}}}[{window}]))")
        if not count:
            return None
        total = query(f"sum(increase({name}_sum{{{sel}}}[{window}]))") or 0.0
        values = {f"p{p}": q(name, p / 100) for p in (50, 90, 95, 99)}
        if any(v is None for v in values.values()):
            return None
        mean = total / count * 1000.0
        return {
            "count": max(1, round(count)),
            "min": 0.0,
            "mean": mean,
            **values,
            "max": values["p99"],
            "approximate": True,
        }

    def scalar(expr: str) -> float:
        return query(expr) or 0.0

    ok = scalar(
        f'sum(increase(inference_requests_total{{{sel}{"," if sel else ""}inference_status="ok"}}[{window}]))'
    )
    err = scalar(
        f'sum(increase(inference_requests_total{{{sel}{"," if sel else ""}inference_status="error"}}[{window}]))'
    )
    canc = scalar(
        f'sum(increase(inference_requests_total{{{sel}{"," if sel else ""}inference_status="cancelled"}}[{window}]))'
    )
    total = ok + err + canc
    hits = scalar(f"sum(increase(inference_cache_hits_total{{{sel}}}[{window}]))")
    misses = scalar(f"sum(increase(inference_cache_misses_total{{{sel}}}[{window}]))")
    ins = scalar(f"sum(increase(inference_cache_insertions_total{{{sel}}}[{window}]))")
    evs = scalar(f"sum(increase(inference_cache_evictions_total{{{sel}}}[{window}]))")
    tps50 = query(
        f"histogram_quantile(0.5, sum by (le) (rate(inference_generation_throughput_per_second_bucket{{{sel}}}[{window}])))"
    )
    prompt = scalar(
        f'sum(increase(inference_tokens_total{{{sel}{"," if sel else ""}gen_ai_token_type="input"}}[{window}]))'
    )
    pressure = query(f"max(max_over_time(inference_simulated_memory_pressure{{{sel}}}[{window}]))")
    rss = query(f"max(max_over_time(process_memory_usage_bytes{{{sel}}}[{window}]))")
    lookups = hits + misses
    return {
        "schema_version": "1.0",
        "measurement": {"clock": "wall", "workload": "unknown", "latency_source": "prometheus"},
        "scenarios": [
            {
                "name": f"prometheus[{window}]",
                "group": None,
                "concurrency": 1,
                "requests": {
                    "total": round(total),
                    "completed": round(ok),
                    "failed": round(err),
                    "cancelled": round(canc),
                    "error_rate": (err / total) if total else 0.0,
                },
                "tokens": {"prompt_mean": (prompt / total) if total else None},
                "throughput": {
                    "generated_tokens_per_second": 0.0,
                    "per_request_tokens_per_second": {"p50": tps50} if tps50 else None,
                },
                "latency_ms": {
                    "end_to_end": dist("inference_request_duration_seconds"),
                    "queue": dist("inference_queue_duration_seconds"),
                    "prefill": dist("inference_prefill_duration_seconds"),
                    "time_to_first_token": dist("inference_time_to_first_token_seconds"),
                    "generation": dist("inference_generation_duration_seconds"),
                    "inter_token": dist("inference_inter_token_latency_seconds"),
                },
                "cache": {
                    "enabled": lookups > 0,
                    "lookups": lookups,
                    "hits": hits,
                    "misses": misses,
                    "insertions": ins,
                    "evictions": evs,
                    "hit_ratio": hits / lookups if lookups else None,
                    "eviction_ratio": evs / ins if ins else None,
                },
                "memory": {
                    "process_rss": {"peak_bytes": rss} if rss else None,
                    "simulated_kv_cache": {"peak_pressure": pressure}
                    if pressure is not None
                    else {},
                },
            }
        ],
    }

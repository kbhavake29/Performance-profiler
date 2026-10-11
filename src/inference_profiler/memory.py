"""Process memory measurement and the *simulated* KV-cache memory model.

Two kinds of memory numbers exist in this project and they are never mixed:

* **Real measurements** - resident set size (RSS) of the current process, read
  with ``psutil``. Reported as ``process.memory.usage``.
* **Simulated values** - KV-cache bytes computed by :class:`SimulatedMemoryModel`
  from token counts. Reported as ``inference.simulated.memory.*`` and always
  flagged ``simulated: true`` in benchmark reports.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

import psutil

_log = logging.getLogger(__name__)

#: Upper bound for real allocations made by memory-pressure scenarios.
MAX_REAL_ALLOCATION_BYTES = 64 * 1024 * 1024


def process_rss_bytes() -> int:
    """Resident set size of the current process in bytes."""
    return int(psutil.Process().memory_info().rss)


class MemorySampler:
    """Samples process RSS on a daemon thread and tracks latest and peak values."""

    def __init__(
        self,
        interval_ms: int,
        on_sample: Callable[[int], None] | None = None,
        reader: Callable[[], int] = process_rss_bytes,
    ) -> None:
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self._interval_s = interval_ms / 1000.0
        self._on_sample = on_sample
        self._reader = reader
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.latest_bytes = 0
        self.peak_bytes = 0
        self.samples = 0

    def sample_once(self) -> int:
        value = self._reader()
        with self._lock:
            self.latest_bytes = value
            self.peak_bytes = max(self.peak_bytes, value)
            self.samples += 1
        if self._on_sample is not None:
            self._on_sample(value)
        return value

    def start(self) -> None:
        if self._thread is not None:
            return
        self.sample_once()
        self._thread = threading.Thread(
            target=self._run, name="inference-profiler-memory", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self.sample_once()
            except Exception:  # pragma: no cover - psutil failures are environmental
                _log.debug("memory sample failed", exc_info=True)

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            self._thread = None


@dataclass
class SimulatedMemoryModel:
    """Tracks simulated KV-cache usage: ``tokens_resident * bytes_per_token``.

    Thread-safe. Values are synthetic and must be labelled as such.
    """

    bytes_per_token: int
    capacity_bytes: int
    used_bytes: int = 0
    peak_bytes: int = 0

    def __post_init__(self) -> None:
        if self.bytes_per_token < 0 or self.capacity_bytes <= 0:
            raise ValueError("bytes_per_token must be >= 0 and capacity_bytes > 0")
        self._lock = threading.Lock()

    def allocate(self, tokens: int) -> int:
        delta = tokens * self.bytes_per_token
        with self._lock:
            self.used_bytes += delta
            self.peak_bytes = max(self.peak_bytes, self.used_bytes)
            return self.used_bytes

    def release(self, tokens: int) -> int:
        delta = tokens * self.bytes_per_token
        with self._lock:
            self.used_bytes = max(0, self.used_bytes - delta)
            return self.used_bytes

    @property
    def pressure(self) -> float:
        return self.used_bytes / self.capacity_bytes

    @property
    def peak_pressure(self) -> float:
        return self.peak_bytes / self.capacity_bytes


def bounded_allocation(num_bytes: int) -> bytearray:
    """Allocate and touch ``num_bytes`` (capped at 64 MiB) to create real, bounded RSS growth."""
    size = max(0, min(num_bytes, MAX_REAL_ALLOCATION_BYTES))
    buf = bytearray(size)
    for offset in range(0, size, 4096):  # touch every page so RSS actually grows
        buf[offset] = 1
    return buf

"""Lightweight pipeline metrics for performance instrumentation.

Provides structured timing, throughput, and slow-item tracking across
pipeline stages. Designed for near-zero overhead when not consumed.
"""

from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def read_current_rss_mb() -> float | None:
    """Return the current RSS in MiB when procfs is available."""
    status_path = Path("/proc/self/status")
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return round(int(parts[1]) / 1024, 1)
    except OSError:
        return None
    return None


def _read_rusage_peak_rss_mb(scope: int) -> float | None:
    try:
        peak_rss = resource.getrusage(scope).ru_maxrss
    except OSError:
        return None
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(float(peak_rss) / divisor, 1)


def read_peak_rss_self_mb() -> float | None:
    """Return the peak RSS for the current process in MiB."""
    return _read_rusage_peak_rss_mb(resource.RUSAGE_SELF)


def read_peak_rss_children_mb() -> float | None:
    """Return the peak RSS reported for child processes in MiB."""
    return _read_rusage_peak_rss_mb(resource.RUSAGE_CHILDREN)


@dataclass
class StageMetrics:
    """Per-stage timing and throughput metrics."""

    name: str
    items_processed: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    sub_timings: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    rss_start_mb: float | None = None
    rss_end_mb: float | None = None
    peak_rss_self_mb: float | None = None
    peak_rss_children_mb: float | None = None

    @property
    def elapsed_s(self) -> float:
        end = self.end_time or time.perf_counter()
        return end - self.start_time

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000

    @property
    def throughput(self) -> float:
        """Items per second."""
        elapsed = self.elapsed_s
        return self.items_processed / elapsed if elapsed > 0 else 0.0

    def stop(
        self,
        items: int | None = None,
        *,
        rss_end_mb: float | None = None,
        peak_rss_self_mb: float | None = None,
        peak_rss_children_mb: float | None = None,
    ) -> StageMetrics:
        """Mark stage as complete."""
        self.end_time = time.perf_counter()
        if items is not None:
            self.items_processed = items
        self.rss_end_mb = read_current_rss_mb() if rss_end_mb is None else rss_end_mb
        self.peak_rss_self_mb = (
            read_peak_rss_self_mb() if peak_rss_self_mb is None else peak_rss_self_mb
        )
        self.peak_rss_children_mb = (
            read_peak_rss_children_mb()
            if peak_rss_children_mb is None
            else peak_rss_children_mb
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        result = {
            "stage": self.name,
            "items": self.items_processed,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "throughput_per_sec": round(self.throughput, 1),
            **({"sub_timings_ms": {k: round(v * 1000, 1) for k, v in self.sub_timings.items()}} if self.sub_timings else {}),
        }
        if self.details:
            result["details"] = self.details
        if self.rss_start_mb is not None:
            result["rss_start_mb"] = self.rss_start_mb
        if self.rss_end_mb is not None:
            result["rss_end_mb"] = self.rss_end_mb
        if self.rss_start_mb is not None and self.rss_end_mb is not None:
            result["rss_delta_mb"] = round(self.rss_end_mb - self.rss_start_mb, 1)
        if self.peak_rss_self_mb is not None:
            result["peak_rss_self_mb"] = self.peak_rss_self_mb
        if self.peak_rss_children_mb is not None:
            result["peak_rss_children_mb"] = self.peak_rss_children_mb
        return result


class SlowItemTracker:
    """Track individual items exceeding a time threshold."""

    def __init__(self, threshold_s: float = 5.0, max_items: int = 50) -> None:
        self.threshold_s = threshold_s
        self.max_items = max_items
        self.items: list[dict[str, Any]] = []

    def record(self, item_id: str, stage: str, elapsed_s: float, **extra: Any) -> None:
        if elapsed_s >= self.threshold_s and len(self.items) < self.max_items:
            self.items.append({
                "id": item_id,
                "stage": stage,
                "elapsed_s": round(elapsed_s, 2),
                **extra,
            })

    def to_list(self) -> list[dict[str, Any]]:
        return sorted(self.items, key=lambda x: x["elapsed_s"], reverse=True)


class PipelineMetrics:
    """Aggregates metrics across a full pipeline execution."""

    def __init__(self) -> None:
        self.stages: dict[str, StageMetrics] = {}
        self.slow_items = SlowItemTracker(threshold_s=5.0)
        self._pipeline_start = time.perf_counter()

    def start_stage(self, name: str) -> StageMetrics:
        """Start timing a pipeline stage."""
        m = StageMetrics(name=name, rss_start_mb=read_current_rss_mb())
        self.stages[name] = m
        return m

    @property
    def total_elapsed_s(self) -> float:
        return time.perf_counter() - self._pipeline_start

    def to_summary(self) -> dict[str, Any]:
        """Export full metrics for logging/persistence."""
        return {
            "total_duration_ms": round(self.total_elapsed_s * 1000, 1),
            "current_rss_mb": read_current_rss_mb(),
            "peak_rss_self_mb": read_peak_rss_self_mb(),
            "peak_rss_children_mb": read_peak_rss_children_mb(),
            "stages": {name: m.to_dict() for name, m in self.stages.items()},
            "slow_items": self.slow_items.to_list(),
        }


__all__ = [
    "PipelineMetrics",
    "SlowItemTracker",
    "StageMetrics",
    "read_current_rss_mb",
    "read_peak_rss_children_mb",
    "read_peak_rss_self_mb",
]

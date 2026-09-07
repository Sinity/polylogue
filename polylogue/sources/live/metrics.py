"""Live-ingest metric payloads shared by daemon events and benchmarks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

#: Typed reasons a batch offered a file and ingested none of it. Every byte
#: of ``input_bytes`` that is neither ingested nor failed lands under exactly
#: one of these, so the three buckets reconcile with no unexplained remainder.
REFUSED_DAEMON_DEGRADED = "daemon_degraded"
REFUSED_DEFERRED_PENDING_AUTHORITY = "deferred_pending_authority"
REFUSED_UNATTEMPTED_TIME_BUDGET = "unattempted_time_budget"
REFUSED_UNATTEMPTED = "unattempted"


class LiveFullIngestMetricKwargs(TypedDict):
    ingested_session_count: int
    ingested_message_count: int
    changed_session_count: int


def split_offered_bytes(
    path_sizes: Mapping[Path, int],
    *,
    succeeded: Iterable[Path],
    failed: Iterable[Path],
    excluded: Mapping[Path, str],
    deferred: Iterable[Path],
    unattempted_reason: str,
) -> tuple[int, int, dict[str, int]]:
    """Partition offered bytes into ingested, failed, and refused-by-reason.

    Every offered path lands in exactly one bucket, in the precedence order
    of the arguments, so the three totals reconcile to ``sum(path_sizes)``
    with no remainder. A path that reached no terminal outcome at all -- the
    time budget broke out before it was attempted -- is refused under
    ``unattempted_reason`` rather than left to a residual subtraction, which
    is how a declined 3.5 GB file was previously reported as ingested work.
    """
    remaining = dict(path_sizes)
    ingested_bytes = 0
    for path in succeeded:
        ingested_bytes += remaining.pop(path, 0)
    failed_bytes = 0
    for path in failed:
        failed_bytes += remaining.pop(path, 0)
    refused: dict[str, int] = {}
    for path, reason in excluded.items():
        size = remaining.pop(path, None)
        if size is not None:
            refused[reason] = refused.get(reason, 0) + size
    for path in deferred:
        size = remaining.pop(path, None)
        if size is not None:
            refused[REFUSED_DEFERRED_PENDING_AUTHORITY] = refused.get(REFUSED_DEFERRED_PENDING_AUTHORITY, 0) + size
    if remaining:
        refused[unattempted_reason] = refused.get(unattempted_reason, 0) + sum(remaining.values())
    return ingested_bytes, failed_bytes, refused


@dataclass(frozen=True, slots=True)
class LiveBatchMetrics:
    """Observable counters and timings for one live ingest batch."""

    queued_file_count: int
    needed_file_count: int
    skipped_file_count: int
    succeeded_file_count: int
    failed_file_count: int
    source_group_count: int
    #: Bytes this batch was OFFERED. A file the batch planned and then
    #: declined -- unsupported source class, a halted source, a whale the
    #: time budget left unattempted -- counts here, so a throughput figure
    #: computed from this is inflated by whatever was not done. Throughput
    #: comes from ``ingested_bytes``.
    input_bytes: int
    source_payload_read_bytes: int
    cursor_fingerprint_read_bytes: int
    ingest_worker_count_max: int
    append_file_count: int
    full_file_count: int
    archive_bytes_before: int
    archive_bytes_after: int
    archive_write_bytes_delta: int
    parse_time_s: float
    convergence_time_s: float
    total_time_s: float
    #: The exhaustive split of ``input_bytes`` by terminal per-file outcome:
    #: ``ingested_bytes + failed_bytes + sum(refused_bytes_by_reason.values())``
    #: equals ``input_bytes``. Each offered path lands in exactly one bucket.
    ingested_bytes: int = 0
    failed_bytes: int = 0
    refused_bytes_by_reason: dict[str, int] = field(default_factory=dict)
    ingested_session_count: int = 0
    ingested_message_count: int = 0
    changed_session_count: int = 0
    rss_current_mb: float | None = None
    rss_peak_self_mb: float | None = None
    rss_peak_children_mb: float | None = None
    cgroup_path: str | None = None
    cgroup_memory_current_mb: float | None = None
    cgroup_memory_peak_mb: float | None = None
    cgroup_memory_swap_current_mb: float | None = None
    stale_cursor_write_count: int = 0
    stage_timings_s: dict[str, float] = field(default_factory=dict)
    #: Planned paths this batch deliberately admitted nothing for, mapped
    #: to the typed reason. Counted separately from succeeded/failed so a
    #: pass that admits nothing is never reported as an idle one.
    excluded_file_count: int = 0
    excluded_reasons: dict[str, int] = field(default_factory=dict)
    failed_paths: list[str] = field(default_factory=list)
    # Identity-scoped session touches for this batch (polylogue-20d.13):
    # ``new_sessions`` are session ids materialized for the first time via
    # the full-ingest route; ``updated_sessions`` are session ids that grew
    # via the append route (a cursor-tracked file that already had a
    # session). Both are (source_name, session_id) pairs, bounded by the
    # batch's own file-count caps -- never a proxy for the full archive.
    new_sessions: tuple[tuple[str, str], ...] = ()
    updated_sessions: tuple[tuple[str, str], ...] = ()
    # polylogue-11cg9: True when a declared ``max_pass_seconds`` budget cut
    # this batch short -- some queued full-ingest paths were left entirely
    # unattempted (not succeeded, not failed) so the single writer hold this
    # batch took could not grow unbounded. They remain ordinary backlog for
    # the next catch-up scan or watch tick.
    time_budget_exceeded: bool = False

    @property
    def refused_bytes(self) -> int:
        """Offered bytes this batch admitted nothing for, all reasons summed."""
        return sum(self.refused_bytes_by_reason.values())

    @property
    def unaccounted_bytes(self) -> int:
        """Offered bytes no bucket claimed. Zero whenever accounting is whole."""
        return self.input_bytes - self.ingested_bytes - self.failed_bytes - self.refused_bytes

    def to_payload(self) -> dict[str, object]:
        read_amplification = (
            round(self.source_payload_read_bytes / self.input_bytes, 6) if self.input_bytes > 0 else 0.0
        )
        files_per_second = round(self.succeeded_file_count / self.total_time_s, 6) if self.total_time_s > 0 else 0.0
        source_mb_per_second = (
            round((self.source_payload_read_bytes / 1_000_000) / self.total_time_s, 6) if self.total_time_s > 0 else 0.0
        )
        ingested_mb_per_second = (
            round((self.ingested_bytes / 1_000_000) / self.total_time_s, 6) if self.total_time_s > 0 else 0.0
        )
        return {
            "queued_file_count": self.queued_file_count,
            "needed_file_count": self.needed_file_count,
            "skipped_file_count": self.skipped_file_count,
            "succeeded_file_count": self.succeeded_file_count,
            "failed_file_count": self.failed_file_count,
            "source_group_count": self.source_group_count,
            "input_bytes": self.input_bytes,
            "ingested_bytes": self.ingested_bytes,
            "failed_bytes": self.failed_bytes,
            "refused_bytes": self.refused_bytes,
            "refused_bytes_by_reason": dict(self.refused_bytes_by_reason),
            "unaccounted_bytes": self.unaccounted_bytes,
            "source_payload_read_bytes": self.source_payload_read_bytes,
            "cursor_fingerprint_read_bytes": self.cursor_fingerprint_read_bytes,
            "read_amplification": read_amplification,
            "files_per_second": files_per_second,
            "source_mb_per_second": source_mb_per_second,
            "ingested_mb_per_second": ingested_mb_per_second,
            "ingest_worker_count_max": self.ingest_worker_count_max,
            "append_file_count": self.append_file_count,
            "full_file_count": self.full_file_count,
            "archive_bytes_before": self.archive_bytes_before,
            "archive_bytes_after": self.archive_bytes_after,
            "archive_write_bytes_delta": self.archive_write_bytes_delta,
            "ingested_session_count": self.ingested_session_count,
            "ingested_message_count": self.ingested_message_count,
            "changed_session_count": self.changed_session_count,
            "parse_time_s": self.parse_time_s,
            "convergence_time_s": self.convergence_time_s,
            "total_time_s": self.total_time_s,
            "rss_current_mb": self.rss_current_mb,
            "rss_peak_self_mb": self.rss_peak_self_mb,
            "rss_peak_children_mb": self.rss_peak_children_mb,
            "cgroup_path": self.cgroup_path,
            "cgroup_memory_current_mb": self.cgroup_memory_current_mb,
            "cgroup_memory_peak_mb": self.cgroup_memory_peak_mb,
            "cgroup_memory_swap_current_mb": self.cgroup_memory_swap_current_mb,
            "stale_cursor_write_count": self.stale_cursor_write_count,
            "stage_timings_s": self.stage_timings_s,
            "failed_paths": self.failed_paths,
            "new_sessions": [{"source_name": source_name, "session_id": sid} for source_name, sid in self.new_sessions],
            "updated_sessions": [
                {"source_name": source_name, "session_id": sid} for source_name, sid in self.updated_sessions
            ],
            "time_budget_exceeded": self.time_budget_exceeded,
        }


@dataclass(slots=True)
class LiveFullIngestAggregate:
    """Aggregated full-ingest counters folded into one live batch."""

    ingested_session_count: int = 0
    ingested_message_count: int = 0
    changed_session_count: int = 0

    def add(self, result: object) -> None:
        self.ingested_session_count += int(getattr(result, "ingested_session_count", 0))
        self.ingested_message_count += int(getattr(result, "ingested_message_count", 0))
        self.changed_session_count += int(getattr(result, "changed_session_count", 0))

    def to_metric_kwargs(self) -> LiveFullIngestMetricKwargs:
        return {
            "ingested_session_count": self.ingested_session_count,
            "ingested_message_count": self.ingested_message_count,
            "changed_session_count": self.changed_session_count,
        }


__all__ = [
    "REFUSED_DAEMON_DEGRADED",
    "REFUSED_DEFERRED_PENDING_AUTHORITY",
    "REFUSED_UNATTEMPTED",
    "REFUSED_UNATTEMPTED_TIME_BUDGET",
    "LiveBatchMetrics",
    "LiveFullIngestAggregate",
    "split_offered_bytes",
]

"""Live-ingest metric payloads shared by daemon events and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


class LiveFullIngestMetricKwargs(TypedDict):
    ingested_session_count: int
    ingested_message_count: int
    changed_session_count: int
    wal_bytes_before_checkpoint_max: int
    wal_bytes_after_checkpoint_max: int
    wal_checkpointed_pages_total: int
    wal_busy_pages_total: int
    wal_checkpoint_elapsed_s: float
    wal_checkpoint_modes: dict[str, int]
    wal_checkpoint_errors: list[str]


@dataclass(frozen=True, slots=True)
class LiveBatchMetrics:
    """Observable counters and timings for one live ingest batch."""

    queued_file_count: int
    needed_file_count: int
    skipped_file_count: int
    succeeded_file_count: int
    failed_file_count: int
    source_group_count: int
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
    ingested_session_count: int = 0
    ingested_message_count: int = 0
    changed_session_count: int = 0
    wal_bytes_before_checkpoint_max: int = 0
    wal_bytes_after_checkpoint_max: int = 0
    wal_checkpointed_pages_total: int = 0
    wal_busy_pages_total: int = 0
    wal_checkpoint_elapsed_s: float = 0.0
    wal_checkpoint_modes: dict[str, int] = field(default_factory=dict)
    wal_checkpoint_errors: list[str] = field(default_factory=list)
    rss_current_mb: float | None = None
    rss_peak_self_mb: float | None = None
    rss_peak_children_mb: float | None = None
    cgroup_path: str | None = None
    cgroup_memory_current_mb: float | None = None
    cgroup_memory_peak_mb: float | None = None
    cgroup_memory_swap_current_mb: float | None = None
    stale_cursor_write_count: int = 0
    stage_timings_s: dict[str, float] = field(default_factory=dict)
    failed_paths: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, object]:
        read_amplification = (
            round(self.source_payload_read_bytes / self.input_bytes, 6) if self.input_bytes > 0 else 0.0
        )
        files_per_second = round(self.succeeded_file_count / self.total_time_s, 6) if self.total_time_s > 0 else 0.0
        source_mb_per_second = (
            round((self.source_payload_read_bytes / 1_000_000) / self.total_time_s, 6) if self.total_time_s > 0 else 0.0
        )
        return {
            "queued_file_count": self.queued_file_count,
            "needed_file_count": self.needed_file_count,
            "skipped_file_count": self.skipped_file_count,
            "succeeded_file_count": self.succeeded_file_count,
            "failed_file_count": self.failed_file_count,
            "source_group_count": self.source_group_count,
            "input_bytes": self.input_bytes,
            "source_payload_read_bytes": self.source_payload_read_bytes,
            "cursor_fingerprint_read_bytes": self.cursor_fingerprint_read_bytes,
            "read_amplification": read_amplification,
            "files_per_second": files_per_second,
            "source_mb_per_second": source_mb_per_second,
            "ingest_worker_count_max": self.ingest_worker_count_max,
            "append_file_count": self.append_file_count,
            "full_file_count": self.full_file_count,
            "archive_bytes_before": self.archive_bytes_before,
            "archive_bytes_after": self.archive_bytes_after,
            "archive_write_bytes_delta": self.archive_write_bytes_delta,
            "ingested_session_count": self.ingested_session_count,
            "ingested_message_count": self.ingested_message_count,
            "changed_session_count": self.changed_session_count,
            "wal_bytes_before_checkpoint_max": self.wal_bytes_before_checkpoint_max,
            "wal_bytes_after_checkpoint_max": self.wal_bytes_after_checkpoint_max,
            "wal_checkpointed_pages_total": self.wal_checkpointed_pages_total,
            "wal_busy_pages_total": self.wal_busy_pages_total,
            "wal_checkpoint_elapsed_s": self.wal_checkpoint_elapsed_s,
            "wal_checkpoint_modes": self.wal_checkpoint_modes,
            "wal_checkpoint_errors": self.wal_checkpoint_errors,
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
        }


@dataclass(slots=True)
class LiveFullIngestAggregate:
    """Aggregated full-ingest counters folded into one live batch."""

    ingested_session_count: int = 0
    ingested_message_count: int = 0
    changed_session_count: int = 0
    wal_bytes_before_checkpoint_max: int = 0
    wal_bytes_after_checkpoint_max: int = 0
    wal_checkpointed_pages_total: int = 0
    wal_busy_pages_total: int = 0
    wal_checkpoint_elapsed_s: float = 0.0
    wal_checkpoint_modes: dict[str, int] = field(default_factory=dict)
    wal_checkpoint_errors: list[str] = field(default_factory=list)

    def add(self, result: object) -> None:
        self.ingested_session_count += int(getattr(result, "ingested_session_count", 0))
        self.ingested_message_count += int(getattr(result, "ingested_message_count", 0))
        self.changed_session_count += int(getattr(result, "changed_session_count", 0))
        self.wal_bytes_before_checkpoint_max = max(
            self.wal_bytes_before_checkpoint_max,
            int(getattr(result, "wal_bytes_before_checkpoint", 0)),
        )
        self.wal_bytes_after_checkpoint_max = max(
            self.wal_bytes_after_checkpoint_max,
            int(getattr(result, "wal_bytes_after_checkpoint", 0)),
        )
        self.wal_checkpointed_pages_total += int(getattr(result, "wal_checkpointed_pages", 0))
        self.wal_busy_pages_total += int(getattr(result, "wal_busy_pages", 0))
        self.wal_checkpoint_elapsed_s += float(getattr(result, "wal_checkpoint_elapsed_s", 0.0))
        mode = str(getattr(result, "wal_checkpoint_mode", "none"))
        self.wal_checkpoint_modes[mode] = self.wal_checkpoint_modes.get(mode, 0) + 1
        error = getattr(result, "wal_checkpoint_error", None)
        if error is not None:
            self.wal_checkpoint_errors.append(str(error))

    def to_metric_kwargs(self) -> LiveFullIngestMetricKwargs:
        return {
            "ingested_session_count": self.ingested_session_count,
            "ingested_message_count": self.ingested_message_count,
            "changed_session_count": self.changed_session_count,
            "wal_bytes_before_checkpoint_max": self.wal_bytes_before_checkpoint_max,
            "wal_bytes_after_checkpoint_max": self.wal_bytes_after_checkpoint_max,
            "wal_checkpointed_pages_total": self.wal_checkpointed_pages_total,
            "wal_busy_pages_total": self.wal_busy_pages_total,
            "wal_checkpoint_elapsed_s": round(self.wal_checkpoint_elapsed_s, 6),
            "wal_checkpoint_modes": self.wal_checkpoint_modes,
            "wal_checkpoint_errors": self.wal_checkpoint_errors,
        }


__all__ = ["LiveBatchMetrics", "LiveFullIngestAggregate"]

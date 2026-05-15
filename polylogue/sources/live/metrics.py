"""Live-ingest metric payloads shared by daemon events and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    rss_current_mb: float | None = None
    rss_peak_self_mb: float | None = None
    rss_peak_children_mb: float | None = None
    cgroup_path: str | None = None
    cgroup_memory_current_mb: float | None = None
    cgroup_memory_peak_mb: float | None = None
    cgroup_memory_swap_current_mb: float | None = None
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
            "stage_timings_s": self.stage_timings_s,
            "failed_paths": self.failed_paths,
        }


__all__ = ["LiveBatchMetrics"]

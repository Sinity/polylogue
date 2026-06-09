"""Typed daemon status models for live-ingest attempts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from polylogue.daemon.live_ingest_attempt_progress import ProgressClassification


class LiveIngestAttemptState(BaseModel):
    attempt_id: str
    started_at: str
    updated_at: str
    status: str
    phase: str
    queued_file_count: int = 0
    needed_file_count: int = 0
    succeeded_file_count: int = 0
    failed_file_count: int = 0
    input_bytes: int = 0
    source_payload_read_bytes: int = 0
    cursor_fingerprint_read_bytes: int = 0
    total_read_bytes: int = 0
    read_amplification: float = 0.0
    files_per_second: float = 0.0
    source_mb_per_second: float = 0.0
    archive_write_bytes_delta: int = 0
    parse_time_s: float = 0.0
    convergence_time_s: float = 0.0
    total_time_s: float = 0.0
    stage_timings_s: dict[str, float] = Field(default_factory=dict)
    current_source: str | None = None
    current_path: str | None = None
    storage_route: str | None = None
    storage_tiers: str | None = None
    payload_available_file_count: int | None = None
    payload_unavailable_file_count: int | None = None
    payload_replayed_from_blob_file_count: int | None = None
    written_raw_count: int | None = None
    error: str | None = None
    rss_current_mb: float | None = None
    rss_peak_self_mb: float | None = None
    rss_peak_children_mb: float | None = None
    cgroup_path: str | None = None
    cgroup_memory_current_mb: float | None = None
    cgroup_memory_peak_mb: float | None = None
    cgroup_memory_swap_current_mb: float | None = None
    cgroup_memory_anon_mb: float | None = None
    cgroup_memory_file_mb: float | None = None
    cgroup_memory_inactive_file_mb: float | None = None
    worker_in_flight_count: int | None = None
    worker_completed_count: int | None = None
    worker_total_count: int | None = None
    stale_cursor_write_count: int = 0
    updated_age_s: float | None = None
    stale: bool = False
    progress_classification: ProgressClassification = "healthy"
    slow_threshold_s: float | None = None
    completed_at: str | None = None


class LiveIngestAttemptSummary(BaseModel):
    running_count: int = 0
    stale_running_count: int = 0
    slow_running_count: int = 0
    stuck_running_count: int = 0
    slow_threshold_s: float | None = None
    recent: list[LiveIngestAttemptState] = Field(default_factory=list)

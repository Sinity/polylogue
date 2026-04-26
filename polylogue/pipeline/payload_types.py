"""Typed payload contracts for run/acquisition telemetry surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from polylogue.lib.json import JSONDocument
from polylogue.storage.session_product_runtime import SessionProductRefreshChunkPayload
from polylogue.types import SearchProvider


class SiteBuildOptions(TypedDict, total=False):
    output: Path
    title: str
    search: bool
    search_provider: str | SearchProvider
    dashboard: bool


class AcquireSplitPayloadSummary(TypedDict):
    count: int
    total_blob_mb: float
    max_blob_mb: float
    total_detect_provider_ms: float
    total_classify_ms: float
    total_serialize_ms: float
    max_detect_provider_ms: float
    max_classify_ms: float
    max_serialize_ms: float


class AcquireDiagnostics(TypedDict, total=False):
    peak_observation: JSONDocument
    observation_count: int
    split_payload_summary: AcquireSplitPayloadSummary


class ParseBatchObservation(TypedDict, total=False):
    records: int
    blob_mb: float
    result_mb: float
    max_result_mb: float
    conversations: int
    messages: int
    changed_conversations: int
    workers: int
    failed_raw_count: int
    skipped_raw_count: int
    elapsed_ms: float
    sync_ingest_elapsed_ms: float
    sync_setup_elapsed_ms: float
    result_wait_elapsed_ms: float
    drain_elapsed_ms: float
    write_elapsed_ms: float
    max_write_elapsed_ms: float
    flush_elapsed_ms: float
    commit_elapsed_ms: float
    executor_teardown_elapsed_ms: float
    raw_state_update_elapsed_ms: float
    unattributed_elapsed_ms: float
    rss_start_mb: float
    rss_end_mb: float
    rss_delta_mb: float
    process_peak_rss_self_mb: float
    peak_rss_growth_mb: float
    peak_rss_children_mb: float
    max_current_rss_mb: float
    max_result_raw_id: str
    batch: int
    processed_raw: int


class ParseBatchObservationSummary(TypedDict, total=False):
    batch_count: int
    slow_batch_count: int
    max_elapsed_ms: float
    max_blob_mb: float
    max_result_mb: float
    max_current_rss_mb: float
    max_rss_end_mb: float
    max_rss_delta_mb: float
    max_peak_rss_growth_mb: float
    batches: list[ParseBatchObservation]


class IngestDiagnostics(TypedDict, total=False):
    acquisition: AcquireDiagnostics
    batch_observations: ParseBatchObservationSummary


class RenderStageObservation(TypedDict, total=False):
    workers: int
    rss_start_mb: float
    rss_end_mb: float
    rss_delta_mb: float
    max_current_rss_mb: float


class MaterializeStageObservation(TypedDict, total=False):
    mode: str
    profiles: int
    work_events: int
    phases: int
    threads: int
    tag_rollups: int
    day_summaries: int
    conversations: int
    unique_thread_roots: int
    unique_provider_days: int
    elapsed_ms: float
    update_ms: float
    thread_refresh_ms: float
    aggregate_refresh_ms: float
    update_chunk_count: int
    update_slow_chunk_count: int
    update_max_chunk_ms: float
    update_max_chunk_load_ms: float
    update_max_chunk_hydrate_ms: float
    update_max_chunk_build_ms: float
    update_max_chunk_write_ms: float
    update_chunks: list[SessionProductRefreshChunkPayload]
    failed: bool
    error: str


__all__ = [
    "AcquireDiagnostics",
    "AcquireSplitPayloadSummary",
    "IngestDiagnostics",
    "MaterializeStageObservation",
    "ParseBatchObservation",
    "ParseBatchObservationSummary",
    "RenderStageObservation",
    "SiteBuildOptions",
]

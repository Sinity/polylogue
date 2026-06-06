"""Observation payload builders for ingest batches."""

from __future__ import annotations

from polylogue.pipeline.payload_types import ParseBatchObservation
from polylogue.pipeline.services.ingest_batch._models import _IngestBatchSummary


def _unattributed_batch_elapsed_s(
    *,
    elapsed_s: float,
    batch_summary: _IngestBatchSummary,
    raw_state_update_elapsed_s: float,
) -> float:
    accounted_elapsed_s = (
        batch_summary.setup_elapsed_s
        + batch_summary.result_wait_s
        + batch_summary.drain_elapsed_s
        + batch_summary.flush_elapsed_s
        + batch_summary.commit_elapsed_s
        + batch_summary.teardown_elapsed_s
        + raw_state_update_elapsed_s
    )
    return max(elapsed_s - accounted_elapsed_s, 0.0)


def _build_batch_memory_observation(
    *,
    rss_start_mb: float | None,
    rss_end_mb: float | None,
    peak_rss_self_start_mb: float | None,
    peak_rss_self_end_mb: float | None,
    peak_rss_children_mb: float | None,
    max_current_rss_mb: float | None,
) -> ParseBatchObservation:
    observation: ParseBatchObservation = {}
    if rss_start_mb is not None:
        observation["rss_start_mb"] = rss_start_mb
    if rss_end_mb is not None:
        observation["rss_end_mb"] = rss_end_mb
    if rss_start_mb is not None and rss_end_mb is not None:
        observation["rss_delta_mb"] = round(rss_end_mb - rss_start_mb, 1)
    if peak_rss_self_end_mb is not None:
        observation["process_peak_rss_self_mb"] = peak_rss_self_end_mb
    if peak_rss_self_start_mb is not None and peak_rss_self_end_mb is not None:
        observation["peak_rss_growth_mb"] = round(max(peak_rss_self_end_mb - peak_rss_self_start_mb, 0.0), 1)
    if peak_rss_children_mb is not None:
        observation["peak_rss_children_mb"] = peak_rss_children_mb
    if max_current_rss_mb is not None:
        observation["max_current_rss_mb"] = max_current_rss_mb
    return observation


def _build_parse_batch_observation(
    *,
    batch_summary: _IngestBatchSummary,
    elapsed_s: float,
    raw_state_update_elapsed_s: float,
    rss_start_mb: float | None,
    rss_end_mb: float | None,
    peak_rss_self_start_mb: float | None,
    peak_rss_self_end_mb: float | None,
    peak_rss_children_mb: float | None,
) -> ParseBatchObservation:
    observation: ParseBatchObservation = {
        "primary_ingest_store": "archive_file_set",
        "archive_primary_write": False,
        "archive_write_mode": "unsupported",
        "records": batch_summary.raw_record_count,
        "blob_mb": round(batch_summary.total_blob_mb, 1),
        "result_mb": round(batch_summary.total_result_bytes / (1024 * 1024), 3),
        "max_result_mb": round(batch_summary.max_result_bytes / (1024 * 1024), 3),
        "sessions": batch_summary.total_convos,
        "messages": batch_summary.total_msgs,
        "changed_sessions": len(batch_summary.changed_session_ids),
        "workers": batch_summary.worker_count,
        "failed_raw_count": len(batch_summary.failed_raw_ids),
        "skipped_raw_count": len(batch_summary.skipped_raw_ids),
        "elapsed_ms": round(elapsed_s * 1000, 1),
        "sync_ingest_elapsed_ms": round(batch_summary.elapsed_s * 1000, 1),
        "sync_setup_elapsed_ms": round(batch_summary.setup_elapsed_s * 1000, 1),
        "result_wait_elapsed_ms": round(batch_summary.result_wait_s * 1000, 1),
        "drain_elapsed_ms": round(batch_summary.drain_elapsed_s * 1000, 1),
        "write_elapsed_ms": round(batch_summary.write_elapsed_s * 1000, 1),
        "max_write_elapsed_ms": round(batch_summary.max_write_elapsed_s * 1000, 1),
        "flush_elapsed_ms": round(batch_summary.flush_elapsed_s * 1000, 1),
        "commit_elapsed_ms": round(batch_summary.commit_elapsed_s * 1000, 1),
        "wal_checkpoint_mode": batch_summary.wal_checkpoint_mode,
        "wal_bytes_before_checkpoint": batch_summary.wal_bytes_before_checkpoint,
        "wal_bytes_after_checkpoint": batch_summary.wal_bytes_after_checkpoint,
        "wal_checkpointed_pages": batch_summary.wal_checkpointed_pages,
        "wal_busy_pages": batch_summary.wal_busy_pages,
        "wal_checkpoint_elapsed_ms": round(batch_summary.wal_checkpoint_elapsed_s * 1000, 1),
        "executor_teardown_elapsed_ms": round(batch_summary.teardown_elapsed_s * 1000, 1),
        "raw_state_update_elapsed_ms": round(raw_state_update_elapsed_s * 1000, 1),
    }
    if batch_summary.wal_checkpoint_error is not None:
        observation["wal_checkpoint_error"] = batch_summary.wal_checkpoint_error
    residual_elapsed_s = _unattributed_batch_elapsed_s(
        elapsed_s=elapsed_s,
        batch_summary=batch_summary,
        raw_state_update_elapsed_s=raw_state_update_elapsed_s,
    )
    observation["unattributed_elapsed_ms"] = round(residual_elapsed_s * 1000, 1)
    observation.update(
        _build_batch_memory_observation(
            rss_start_mb=rss_start_mb,
            rss_end_mb=rss_end_mb,
            peak_rss_self_start_mb=peak_rss_self_start_mb,
            peak_rss_self_end_mb=peak_rss_self_end_mb,
            peak_rss_children_mb=peak_rss_children_mb,
            max_current_rss_mb=batch_summary.max_current_rss_mb,
        )
    )
    if batch_summary.max_result_raw_id is not None:
        observation["max_result_raw_id"] = batch_summary.max_result_raw_id
    return observation

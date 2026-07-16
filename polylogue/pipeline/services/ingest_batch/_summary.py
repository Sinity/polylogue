"""Batch summary projection helpers."""

from __future__ import annotations

from polylogue.pipeline.services.ingest_batch._models import _IngestBatchSummary
from polylogue.pipeline.services.parsing_models import ParseResult


def apply_ingest_batch_summary(result: ParseResult, batch_summary: _IngestBatchSummary) -> None:
    result.parse_failures += batch_summary.parse_failures
    result.processed_ids.update(batch_summary.processed_ids)
    result._changed_session_ids.extend(batch_summary.changed_session_ids)
    for key, value in batch_summary.counts.items():
        if key in result.counts:
            result.counts[key] += value
    for key, value in batch_summary.changed_counts.items():
        if key in result.changed_counts:
            result.changed_counts[key] += value


def progressed_raw_count(batch_summary: _IngestBatchSummary) -> int:
    return sum(
        1
        for raw_id, outcome in batch_summary.outcomes.items()
        if outcome.had_sessions and outcome.error is None and raw_id not in batch_summary.publication_deferred_raw_ids
    )


def successful_raw_ids(batch_summary: _IngestBatchSummary) -> set[str]:
    return {
        raw_id
        for raw_id, outcome in batch_summary.outcomes.items()
        if outcome.had_sessions
        and raw_id not in batch_summary.failed_raw_ids
        and raw_id not in batch_summary.publication_deferred_raw_ids
    }


__all__ = ["apply_ingest_batch_summary", "progressed_raw_count", "successful_raw_ids"]

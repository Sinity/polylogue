"""Row mappers for durable session-profile records."""

from __future__ import annotations

import sqlite3

from polylogue.core.types import SessionId
from polylogue.insights.archive_models import (
    SessionEnrichmentPayload,
    SessionEvidencePayload,
    SessionInferencePayload,
)
from polylogue.storage.runtime import SessionProfileRecord
from polylogue.storage.sqlite.queries.mappers_insight_fallback import (
    parse_fallback_payload_dict,
    parse_payload_model,
    session_profile_enrichment_from_fallback,
    session_profile_evidence_from_fallback,
    session_profile_inference_from_fallback,
)
from polylogue.storage.sqlite.queries.mappers_support import (
    _json_text_tuple,
    _parse_json,
    _row_float,
    _row_get,
    _row_int,
    _row_text,
)


def _row_to_session_profile_record(row: sqlite3.Row) -> SessionProfileRecord:
    search_text = row["search_text"]
    evidence_search_text = (_row_get(row, "evidence_search_text", "") or "").strip() or search_text
    inference_search_text = (_row_get(row, "inference_search_text", "") or "").strip() or search_text
    enrichment_search_text = (_row_get(row, "enrichment_search_text", "") or "").strip() or inference_search_text
    fallback_payload = parse_fallback_payload_dict(
        row,
        record_id=row["session_id"],
    )
    evidence_payload = parse_payload_model(
        row,
        "evidence_payload_json",
        record_id=row["session_id"],
        model=SessionEvidencePayload,
    )
    if evidence_payload is None:
        evidence_payload = session_profile_evidence_from_fallback(row, fallback_payload)
    inference_payload = parse_payload_model(
        row,
        "inference_payload_json",
        record_id=row["session_id"],
        model=SessionInferencePayload,
    )
    if inference_payload is None:
        inference_payload = session_profile_inference_from_fallback(row, fallback_payload)
    enrichment_payload = parse_payload_model(
        row,
        "enrichment_payload_json",
        record_id=row["session_id"],
        model=SessionEnrichmentPayload,
    )
    if enrichment_payload is None:
        enrichment_payload = session_profile_enrichment_from_fallback(
            row,
            fallback_payload,
            inference_payload=inference_payload,
        )
    return SessionProfileRecord(
        session_id=SessionId(row["session_id"]),
        logical_session_id=SessionId(_row_text(row, "logical_session_id") or row["session_id"]),
        materializer_version=int(_row_int(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        source_updated_at=_row_text(row, "source_updated_at"),
        source_sort_key=_row_float(row, "source_sort_key"),
        input_high_water_mark=_row_text(row, "input_high_water_mark"),
        input_high_water_mark_source=_row_text(row, "input_high_water_mark_source"),
        input_row_count=int(_row_int(row, "input_row_count", 0) or 0),
        source_name=row["source_name"],
        title=_row_text(row, "title"),
        first_message_at=_row_text(row, "first_message_at"),
        last_message_at=_row_text(row, "last_message_at"),
        canonical_session_date=_row_text(row, "canonical_session_date"),
        repo_paths=_json_text_tuple(_parse_json(_row_get(row, "repo_paths_json"))),
        repo_names=_json_text_tuple(_parse_json(_row_get(row, "repo_names_json"))),
        tags=_json_text_tuple(_parse_json(_row_get(row, "tags_json"))),
        auto_tags=_json_text_tuple(_parse_json(_row_get(row, "auto_tags_json"))),
        message_count=int(_row_int(row, "message_count", 0) or 0),
        substantive_count=int(_row_int(row, "substantive_count", 0) or 0),
        attachment_count=int(_row_int(row, "attachment_count", 0) or 0),
        work_event_count=int(_row_int(row, "work_event_count", 0) or 0),
        phase_count=int(_row_int(row, "phase_count", 0) or 0),
        word_count=int(_row_int(row, "word_count", 0) or 0),
        tool_use_count=int(_row_int(row, "tool_use_count", 0) or 0),
        thinking_count=int(_row_int(row, "thinking_count", 0) or 0),
        total_cost_usd=float(_row_float(row, "total_cost_usd", 0.0) or 0.0),
        total_duration_ms=int(_row_int(row, "total_duration_ms", 0) or 0),
        engaged_duration_ms=int(_row_int(row, "engaged_duration_ms", 0) or 0),
        tool_active_duration_ms=int(_row_int(row, "tool_active_duration_ms", 0) or 0),
        wall_duration_ms=int(_row_int(row, "wall_duration_ms", 0) or 0),
        workflow_shape=_row_text(row, "workflow_shape") or "unknown",
        workflow_shape_confidence=float(_row_float(row, "workflow_shape_confidence", 0.0) or 0.0),
        workflow_shape_features_json=_row_text(row, "workflow_shape_features_json") or "{}",
        terminal_state=_row_text(row, "terminal_state") or "unknown",
        terminal_state_confidence=float(_row_float(row, "terminal_state_confidence", 0.0) or 0.0),
        terminal_state_evidence_json=_row_text(row, "terminal_state_evidence_json") or "{}",
        cost_is_estimated=bool(int(_row_int(row, "cost_is_estimated", 0) or 0)),
        thinking_duration_ms=int(_row_int(row, "thinking_duration_ms", 0) or 0),
        output_duration_ms=int(_row_int(row, "output_duration_ms", 0) or 0),
        tool_duration_ms=int(_row_int(row, "tool_duration_ms", 0) or 0),
        latency_percentiles_ms_json=_row_text(row, "latency_percentiles_ms_json") or "{}",
        tool_calls_per_minute=float(_row_float(row, "tool_calls_per_minute", 0.0) or 0.0),
        timing_provenance=_row_text(row, "timing_provenance") or "sort_key_estimated",
        total_input_tokens=int(_row_int(row, "total_input_tokens", 0) or 0),
        total_output_tokens=int(_row_int(row, "total_output_tokens", 0) or 0),
        total_cache_read_tokens=int(_row_int(row, "total_cache_read_tokens", 0) or 0),
        total_cache_write_tokens=int(_row_int(row, "total_cache_write_tokens", 0) or 0),
        total_credit_cost=float(_row_float(row, "total_credit_cost", 0.0) or 0.0),
        cost_provenance=_row_text(row, "cost_provenance") or "unknown",
        per_model_cost_json=_row_text(row, "per_model_cost_json") or "{}",
        primary_model_name=_row_text(row, "primary_model_name"),
        primary_model_family=_row_text(row, "primary_model_family"),
        evidence_payload=evidence_payload,
        inference_payload=inference_payload,
        enrichment_payload=enrichment_payload,
        search_text=search_text,
        evidence_search_text=evidence_search_text,
        inference_search_text=inference_search_text,
        enrichment_search_text=enrichment_search_text,
        enrichment_version=int(_row_int(row, "enrichment_version", 1) or 1),
        enrichment_family=_row_text(row, "enrichment_family") or "scored_session_enrichment",
        inference_version=int(_row_int(row, "inference_version", 1) or 1),
        inference_family=_row_text(row, "inference_family") or "heuristic_session_semantics",
    )


__all__ = ["_row_to_session_profile_record"]

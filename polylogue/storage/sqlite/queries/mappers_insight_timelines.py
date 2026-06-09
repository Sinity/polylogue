"""Row mappers for timeline-oriented session-insight records."""

from __future__ import annotations

import sqlite3

from polylogue.insights.archive_models import (
    SessionPhaseEvidencePayload,
    SessionPhaseInferencePayload,
    ThreadPayload,
    WorkEventEvidencePayload,
    WorkEventInferencePayload,
)
from polylogue.storage.runtime import SessionPhaseRecord, SessionWorkEventRecord, ThreadRecord
from polylogue.storage.sqlite.queries.mappers_insight_fallback import (
    parse_fallback_payload_dict,
    parse_payload_model,
    session_phase_evidence_from_fallback,
    session_phase_inference_from_fallback,
    session_work_event_evidence_from_fallback,
    session_work_event_inference_from_fallback,
)
from polylogue.storage.sqlite.queries.mappers_support import (
    _json_int_dict,
    _json_text_tuple,
    _parse_json,
    _row_float,
    _row_get,
    _row_int,
    _row_text,
)
from polylogue.types import SessionId


def _row_to_session_work_event_record(row: sqlite3.Row) -> SessionWorkEventRecord:
    fallback_payload = parse_fallback_payload_dict(
        row,
        record_id=row["event_id"],
    )
    evidence_payload = parse_payload_model(
        row,
        "evidence_payload_json",
        record_id=row["event_id"],
        model=WorkEventEvidencePayload,
    )
    if evidence_payload is None:
        evidence_payload = session_work_event_evidence_from_fallback(row, fallback_payload)
    inference_payload = parse_payload_model(
        row,
        "inference_payload_json",
        record_id=row["event_id"],
        model=WorkEventInferencePayload,
    )
    if inference_payload is None:
        inference_payload = session_work_event_inference_from_fallback(row, fallback_payload)
    return SessionWorkEventRecord(
        event_id=row["event_id"],
        session_id=SessionId(row["session_id"]),
        materializer_version=int(_row_int(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        source_updated_at=_row_text(row, "source_updated_at"),
        source_sort_key=_row_float(row, "source_sort_key"),
        input_high_water_mark=_row_text(row, "input_high_water_mark"),
        input_high_water_mark_source=_row_text(row, "input_high_water_mark_source"),
        input_row_count=int(_row_int(row, "input_row_count", 0) or 0),
        source_name=row["source_name"],
        event_index=int(_row_int(row, "event_index", 0) or 0),
        heuristic_label=row["heuristic_label"],
        confidence=float(_row_float(row, "confidence", 0.0) or 0.0),
        start_index=int(_row_int(row, "start_index", 0) or 0),
        end_index=int(_row_int(row, "end_index", 0) or 0),
        start_time=_row_text(row, "start_time"),
        end_time=_row_text(row, "end_time"),
        duration_ms=int(_row_int(row, "duration_ms", 0) or 0),
        canonical_session_date=_row_text(row, "canonical_session_date"),
        summary=row["summary"],
        file_paths=_json_text_tuple(_parse_json(_row_get(row, "file_paths_json"))),
        tools_used=_json_text_tuple(_parse_json(_row_get(row, "tools_used_json"))),
        evidence_payload=evidence_payload,
        inference_payload=inference_payload,
        search_text=row["search_text"],
        inference_version=int(_row_int(row, "inference_version", 1) or 1),
        inference_family=_row_text(row, "inference_family") or "heuristic_session_semantics",
    )


def _row_to_session_phase_record(row: sqlite3.Row) -> SessionPhaseRecord:
    fallback_payload = parse_fallback_payload_dict(
        row,
        record_id=row["phase_id"],
    )
    evidence_payload = parse_payload_model(
        row,
        "evidence_payload_json",
        record_id=row["phase_id"],
        model=SessionPhaseEvidencePayload,
    )
    if evidence_payload is None:
        evidence_payload = session_phase_evidence_from_fallback(row, fallback_payload)
    inference_payload = parse_payload_model(
        row,
        "inference_payload_json",
        record_id=row["phase_id"],
        model=SessionPhaseInferencePayload,
    )
    if inference_payload is None:
        inference_payload = session_phase_inference_from_fallback(row, fallback_payload)
    return SessionPhaseRecord(
        phase_id=row["phase_id"],
        session_id=SessionId(row["session_id"]),
        materializer_version=int(_row_int(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        source_updated_at=_row_text(row, "source_updated_at"),
        source_sort_key=_row_float(row, "source_sort_key"),
        input_high_water_mark=_row_text(row, "input_high_water_mark"),
        input_high_water_mark_source=_row_text(row, "input_high_water_mark_source"),
        input_row_count=int(_row_int(row, "input_row_count", 0) or 0),
        source_name=row["source_name"],
        phase_index=int(_row_int(row, "phase_index", 0) or 0),
        kind=row["kind"],
        start_index=int(_row_int(row, "start_index", 0) or 0),
        end_index=int(_row_int(row, "end_index", 0) or 0),
        start_time=_row_text(row, "start_time"),
        end_time=_row_text(row, "end_time"),
        duration_ms=int(_row_int(row, "duration_ms", 0) or 0),
        canonical_session_date=_row_text(row, "canonical_session_date"),
        confidence=float(_row_float(row, "confidence", 0.0) or 0.0),
        evidence_reasons=_json_text_tuple(_parse_json(_row_get(row, "evidence_reasons_json"))),
        tool_counts=_json_int_dict(
            _parse_json(
                _row_get(row, "tool_counts_json"),
                field="tool_counts_json",
                record_id=row["phase_id"],
            )
        ),
        word_count=int(_row_int(row, "word_count", 0) or 0),
        evidence_payload=evidence_payload,
        inference_payload=inference_payload,
        search_text=row["search_text"],
        inference_version=int(_row_int(row, "inference_version", 1) or 1),
        inference_family=_row_text(row, "inference_family") or "heuristic_session_semantics",
    )


def _row_to_thread_record(row: sqlite3.Row) -> ThreadRecord:
    return ThreadRecord(
        thread_id=row["thread_id"],
        root_id=SessionId(row["root_id"]),
        materializer_version=int(_row_int(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        source_updated_at=_row_text(row, "source_updated_at"),
        input_high_water_mark=_row_text(row, "input_high_water_mark"),
        input_high_water_mark_source=_row_text(row, "input_high_water_mark_source"),
        input_row_count=int(_row_int(row, "input_row_count", 0) or 0),
        start_time=_row_text(row, "start_time"),
        end_time=_row_text(row, "end_time"),
        dominant_repo=_row_text(row, "dominant_repo"),
        session_ids=_json_text_tuple(_parse_json(_row_get(row, "session_ids_json"))),
        session_count=int(_row_int(row, "session_count", 0) or 0),
        depth=int(_row_int(row, "depth", 0) or 0),
        branch_count=int(_row_int(row, "branch_count", 0) or 0),
        total_messages=int(_row_int(row, "total_messages", 0) or 0),
        total_cost_usd=float(_row_float(row, "total_cost_usd", 0.0) or 0.0),
        wall_duration_ms=int(_row_int(row, "wall_duration_ms", 0) or 0),
        work_event_breakdown=_json_int_dict(_parse_json(_row_get(row, "work_event_breakdown_json"))),
        payload=ThreadPayload.model_validate(
            _parse_json(row["payload_json"], field="payload_json", record_id=row["thread_id"]) or {}
        ),
        search_text=row["search_text"],
    )


__all__ = [
    "_row_to_session_phase_record",
    "_row_to_session_work_event_record",
    "_row_to_thread_record",
]

"""Row mappers for timeline-oriented session-product records."""

from __future__ import annotations

import sqlite3

from polylogue.storage.backends.queries.mappers import _parse_json, _row_get
from polylogue.storage.store import SessionPhaseRecord, SessionWorkEventRecord, WorkThreadRecord
from polylogue.types import ConversationId


def _row_to_session_work_event_record(row: sqlite3.Row) -> SessionWorkEventRecord:
    legacy_payload = (
        _parse_json(
            _row_get(row, "payload_json"),
            field="payload_json",
            record_id=row["event_id"],
        )
        or {}
    )
    evidence_payload = (
        _parse_json(
            _row_get(row, "evidence_payload_json"),
            field="evidence_payload_json",
            record_id=row["event_id"],
        )
        or {}
    )
    if not evidence_payload:
        evidence_payload = {
            "start_index": int(_row_get(row, "start_index", 0) or legacy_payload.get("start_index") or 0),
            "end_index": int(_row_get(row, "end_index", 0) or legacy_payload.get("end_index") or 0),
            "start_time": _row_get(row, "start_time") or legacy_payload.get("start_time"),
            "end_time": _row_get(row, "end_time") or legacy_payload.get("end_time"),
            "canonical_session_date": _row_get(row, "canonical_session_date")
            or legacy_payload.get("canonical_session_date"),
            "duration_ms": int(_row_get(row, "duration_ms", 0) or legacy_payload.get("duration_ms") or 0),
            "file_paths": tuple(
                _parse_json(_row_get(row, "file_paths_json")) or legacy_payload.get("file_paths") or []
            ),
            "tools_used": tuple(
                _parse_json(_row_get(row, "tools_used_json")) or legacy_payload.get("tools_used") or []
            ),
        }
    inference_payload = (
        _parse_json(
            _row_get(row, "inference_payload_json"),
            field="inference_payload_json",
            record_id=row["event_id"],
        )
        or {}
    )
    if not inference_payload:
        inference_payload = {
            "kind": row["kind"],
            "summary": row["summary"],
            "confidence": float(_row_get(row, "confidence", 0.0) or legacy_payload.get("confidence") or 0.0),
            "evidence": tuple(legacy_payload.get("evidence") or ()),
        }
    return SessionWorkEventRecord(
        event_id=row["event_id"],
        conversation_id=ConversationId(row["conversation_id"]),
        materializer_version=int(_row_get(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        source_updated_at=_row_get(row, "source_updated_at"),
        source_sort_key=_row_get(row, "source_sort_key"),
        provider_name=row["provider_name"],
        event_index=int(_row_get(row, "event_index", 0) or 0),
        kind=row["kind"],
        confidence=float(_row_get(row, "confidence", 0.0) or 0.0),
        start_index=int(_row_get(row, "start_index", 0) or 0),
        end_index=int(_row_get(row, "end_index", 0) or 0),
        start_time=_row_get(row, "start_time"),
        end_time=_row_get(row, "end_time"),
        duration_ms=int(_row_get(row, "duration_ms", 0) or 0),
        canonical_session_date=_row_get(row, "canonical_session_date"),
        summary=row["summary"],
        file_paths=tuple(_parse_json(_row_get(row, "file_paths_json")) or []),
        tools_used=tuple(_parse_json(_row_get(row, "tools_used_json")) or []),
        evidence_payload=evidence_payload,
        inference_payload=inference_payload,
        search_text=row["search_text"],
        inference_version=int(_row_get(row, "inference_version", 1) or 1),
        inference_family=_row_get(row, "inference_family", "heuristic_session_semantics"),
    )


def _row_to_session_phase_record(row: sqlite3.Row) -> SessionPhaseRecord:
    legacy_payload = (
        _parse_json(
            _row_get(row, "payload_json"),
            field="payload_json",
            record_id=row["phase_id"],
        )
        or {}
    )
    evidence_payload = (
        _parse_json(
            _row_get(row, "evidence_payload_json"),
            field="evidence_payload_json",
            record_id=row["phase_id"],
        )
        or {}
    )
    if not evidence_payload:
        evidence_payload = {
            "start_time": _row_get(row, "start_time") or legacy_payload.get("start_time"),
            "end_time": _row_get(row, "end_time") or legacy_payload.get("end_time"),
            "canonical_session_date": _row_get(row, "canonical_session_date")
            or legacy_payload.get("canonical_session_date"),
            "message_range": (
                int(_row_get(row, "start_index", 0) or legacy_payload.get("start_index") or 0),
                int(_row_get(row, "end_index", 0) or legacy_payload.get("end_index") or 0),
            ),
            "duration_ms": int(_row_get(row, "duration_ms", 0) or legacy_payload.get("duration_ms") or 0),
            "tool_counts": _parse_json(_row_get(row, "tool_counts_json")) or legacy_payload.get("tool_counts") or {},
            "word_count": int(_row_get(row, "word_count", 0) or legacy_payload.get("word_count") or 0),
        }
    inference_payload = (
        _parse_json(
            _row_get(row, "inference_payload_json"),
            field="inference_payload_json",
            record_id=row["phase_id"],
        )
        or {}
    )
    if not inference_payload:
        inference_payload = {
            "confidence": float(_row_get(row, "confidence", 0.0) or legacy_payload.get("confidence") or 0.0),
            "evidence": tuple(
                _parse_json(_row_get(row, "evidence_reasons_json")) or legacy_payload.get("evidence") or []
            ),
        }
    return SessionPhaseRecord(
        phase_id=row["phase_id"],
        conversation_id=ConversationId(row["conversation_id"]),
        materializer_version=int(_row_get(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        source_updated_at=_row_get(row, "source_updated_at"),
        source_sort_key=_row_get(row, "source_sort_key"),
        provider_name=row["provider_name"],
        phase_index=int(_row_get(row, "phase_index", 0) or 0),
        kind=row["kind"],
        start_index=int(_row_get(row, "start_index", 0) or 0),
        end_index=int(_row_get(row, "end_index", 0) or 0),
        start_time=_row_get(row, "start_time"),
        end_time=_row_get(row, "end_time"),
        duration_ms=int(_row_get(row, "duration_ms", 0) or 0),
        canonical_session_date=_row_get(row, "canonical_session_date"),
        confidence=float(_row_get(row, "confidence", 0.0) or 0.0),
        evidence_reasons=tuple(_parse_json(_row_get(row, "evidence_reasons_json")) or []),
        tool_counts=_parse_json(
            _row_get(row, "tool_counts_json"),
            field="tool_counts_json",
            record_id=row["phase_id"],
        )
        or {},
        word_count=int(_row_get(row, "word_count", 0) or 0),
        evidence_payload=evidence_payload,
        inference_payload=inference_payload,
        search_text=row["search_text"],
        inference_version=int(_row_get(row, "inference_version", 1) or 1),
        inference_family=_row_get(row, "inference_family", "heuristic_session_semantics"),
    )


def _row_to_work_thread_record(row: sqlite3.Row) -> WorkThreadRecord:
    return WorkThreadRecord(
        thread_id=row["thread_id"],
        root_id=ConversationId(row["root_id"]),
        materializer_version=int(_row_get(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        start_time=_row_get(row, "start_time"),
        end_time=_row_get(row, "end_time"),
        dominant_project=_row_get(row, "dominant_project"),
        session_ids=tuple(_parse_json(_row_get(row, "session_ids_json")) or []),
        session_count=int(_row_get(row, "session_count", 0) or 0),
        depth=int(_row_get(row, "depth", 0) or 0),
        branch_count=int(_row_get(row, "branch_count", 0) or 0),
        total_messages=int(_row_get(row, "total_messages", 0) or 0),
        total_cost_usd=float(_row_get(row, "total_cost_usd", 0.0) or 0.0),
        wall_duration_ms=int(_row_get(row, "wall_duration_ms", 0) or 0),
        work_event_breakdown=_parse_json(_row_get(row, "work_event_breakdown_json")) or {},
        payload=_parse_json(row["payload_json"], field="payload_json", record_id=row["thread_id"]) or {},
        search_text=row["search_text"],
    )


__all__ = [
    "_row_to_session_phase_record",
    "_row_to_session_work_event_record",
    "_row_to_work_thread_record",
]

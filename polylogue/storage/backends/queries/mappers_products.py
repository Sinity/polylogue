"""Row mappers for durable session-product and maintenance records."""

from __future__ import annotations

import sqlite3

from polylogue.storage.backends.queries.mappers_support import _parse_json, _row_get
from polylogue.storage.store import (
    DaySessionSummaryRecord,
    MaintenanceRunRecord,
    SessionPhaseRecord,
    SessionProfileRecord,
    SessionTagRollupRecord,
    SessionWorkEventRecord,
    WorkThreadRecord,
)
from polylogue.types import ConversationId


def _row_to_session_profile_record(row: sqlite3.Row) -> SessionProfileRecord:
    search_text = row["search_text"]
    evidence_search_text = (_row_get(row, "evidence_search_text", "") or "").strip() or search_text
    inference_search_text = (_row_get(row, "inference_search_text", "") or "").strip() or search_text
    enrichment_search_text = (_row_get(row, "enrichment_search_text", "") or "").strip() or inference_search_text
    legacy_payload = _parse_json(
        _row_get(row, "payload_json"),
        field="payload_json",
        record_id=row["conversation_id"],
    ) or {}
    evidence_payload = (
        _parse_json(
            _row_get(row, "evidence_payload_json"),
            field="evidence_payload_json",
            record_id=row["conversation_id"],
        )
        or {}
    )
    if not evidence_payload:
        evidence_payload = {
            "created_at": legacy_payload.get("created_at"),
            "updated_at": legacy_payload.get("updated_at") or _row_get(row, "source_updated_at"),
            "first_message_at": _row_get(row, "first_message_at") or legacy_payload.get("first_message_at"),
            "last_message_at": _row_get(row, "last_message_at") or legacy_payload.get("last_message_at"),
            "canonical_session_date": _row_get(row, "canonical_session_date") or legacy_payload.get("canonical_session_date"),
            "message_count": int(_row_get(row, "message_count", 0) or legacy_payload.get("message_count") or 0),
            "substantive_count": int(_row_get(row, "substantive_count", 0) or legacy_payload.get("substantive_count") or 0),
            "attachment_count": int(_row_get(row, "attachment_count", 0) or legacy_payload.get("attachment_count") or 0),
            "tool_use_count": int(_row_get(row, "tool_use_count", 0) or legacy_payload.get("tool_use_count") or 0),
            "thinking_count": int(_row_get(row, "thinking_count", 0) or legacy_payload.get("thinking_count") or 0),
            "word_count": int(_row_get(row, "word_count", 0) or legacy_payload.get("word_count") or 0),
            "total_cost_usd": float(_row_get(row, "total_cost_usd", 0.0) or legacy_payload.get("total_cost_usd") or 0.0),
            "total_duration_ms": int(_row_get(row, "total_duration_ms", 0) or legacy_payload.get("total_duration_ms") or 0),
            "wall_duration_ms": int(_row_get(row, "wall_duration_ms", 0) or legacy_payload.get("wall_duration_ms") or 0),
            "cost_is_estimated": bool(int(_row_get(row, "cost_is_estimated", 0) or 0) or legacy_payload.get("cost_is_estimated")),
            "tool_categories": legacy_payload.get("tool_categories") or {},
            "repo_paths": tuple(_parse_json(_row_get(row, "repo_paths_json")) or legacy_payload.get("repo_paths") or []),
            "cwd_paths": tuple(legacy_payload.get("cwd_paths") or ()),
            "branch_names": tuple(legacy_payload.get("branch_names") or ()),
            "file_paths_touched": tuple(legacy_payload.get("file_paths_touched") or ()),
            "languages_detected": tuple(legacy_payload.get("languages_detected") or ()),
            "tags": tuple(_parse_json(_row_get(row, "tags_json")) or legacy_payload.get("tags") or []),
            "is_continuation": bool(legacy_payload.get("is_continuation", False)),
            "parent_id": legacy_payload.get("parent_id"),
        }
    inference_payload = (
        _parse_json(
            _row_get(row, "inference_payload_json"),
            field="inference_payload_json",
            record_id=row["conversation_id"],
        )
        or {}
    )
    if not inference_payload:
        inference_payload = {
            "primary_work_kind": _row_get(row, "primary_work_kind") or legacy_payload.get("primary_work_kind"),
            "canonical_projects": tuple(_parse_json(_row_get(row, "canonical_projects_json")) or legacy_payload.get("canonical_projects") or []),
            "work_event_count": int(_row_get(row, "work_event_count", 0) or legacy_payload.get("work_event_count") or 0),
            "phase_count": int(_row_get(row, "phase_count", 0) or legacy_payload.get("phase_count") or 0),
            "engaged_duration_ms": int(_row_get(row, "engaged_duration_ms", 0) or legacy_payload.get("engaged_duration_ms") or 0),
            "engaged_minutes": float(legacy_payload.get("engaged_minutes") or 0.0),
            "support_level": str(legacy_payload.get("support_level") or "weak"),
            "support_signals": tuple(legacy_payload.get("support_signals") or ()),
            "engaged_duration_source": str(legacy_payload.get("engaged_duration_source") or "session_total_fallback"),
            "project_inference_strength": str(legacy_payload.get("project_inference_strength") or "weak"),
            "decision_signal_strength": str(legacy_payload.get("decision_signal_strength") or "weak"),
            "auto_tags": tuple(_parse_json(_row_get(row, "auto_tags_json")) or legacy_payload.get("auto_tags") or []),
            "work_events": tuple(legacy_payload.get("work_events") or ()),
            "phases": tuple(legacy_payload.get("phases") or ()),
            "decisions": tuple(legacy_payload.get("decisions") or ()),
        }
    enrichment_payload = (
        _parse_json(
            _row_get(row, "enrichment_payload_json"),
            field="enrichment_payload_json",
            record_id=row["conversation_id"],
        )
        or {}
    )
    if not enrichment_payload:
        decisions = tuple(inference_payload.get("decisions") or ())
        enrichment_payload = {
            "intent_summary": row["title"] or legacy_payload.get("title"),
            "outcome_summary": decisions[-1].get("summary") if decisions else None,
            "blockers": (),
            "refined_work_kind": inference_payload.get("primary_work_kind") or _row_get(row, "primary_work_kind"),
            "confidence": 0.35 if (inference_payload.get("primary_work_kind") or _row_get(row, "primary_work_kind")) else 0.0,
            "support_level": "weak",
            "support_signals": tuple(inference_payload.get("support_signals") or ()),
            "input_band_summary": {
                "user_turns": 0,
                "assistant_turns": 0,
                "action_events": 0,
                "touched_paths": len(_parse_json(_row_get(row, "repo_paths_json")) or []),
                "canonical_projects": len(_parse_json(_row_get(row, "canonical_projects_json")) or []),
                "decisions": len(decisions),
            },
        }
    return SessionProfileRecord(
        conversation_id=ConversationId(row["conversation_id"]),
        materializer_version=int(_row_get(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        source_updated_at=_row_get(row, "source_updated_at"),
        source_sort_key=_row_get(row, "source_sort_key"),
        provider_name=row["provider_name"],
        title=_row_get(row, "title"),
        first_message_at=_row_get(row, "first_message_at"),
        last_message_at=_row_get(row, "last_message_at"),
        canonical_session_date=_row_get(row, "canonical_session_date"),
        primary_work_kind=_row_get(row, "primary_work_kind"),
        repo_paths=tuple(_parse_json(_row_get(row, "repo_paths_json")) or []),
        canonical_projects=tuple(_parse_json(_row_get(row, "canonical_projects_json")) or []),
        tags=tuple(_parse_json(_row_get(row, "tags_json")) or []),
        auto_tags=tuple(_parse_json(_row_get(row, "auto_tags_json")) or []),
        message_count=int(_row_get(row, "message_count", 0) or 0),
        substantive_count=int(_row_get(row, "substantive_count", 0) or 0),
        attachment_count=int(_row_get(row, "attachment_count", 0) or 0),
        work_event_count=int(_row_get(row, "work_event_count", 0) or 0),
        phase_count=int(_row_get(row, "phase_count", 0) or 0),
        word_count=int(_row_get(row, "word_count", 0) or 0),
        tool_use_count=int(_row_get(row, "tool_use_count", 0) or 0),
        thinking_count=int(_row_get(row, "thinking_count", 0) or 0),
        total_cost_usd=float(_row_get(row, "total_cost_usd", 0.0) or 0.0),
        total_duration_ms=int(_row_get(row, "total_duration_ms", 0) or 0),
        engaged_duration_ms=int(_row_get(row, "engaged_duration_ms", 0) or 0),
        wall_duration_ms=int(_row_get(row, "wall_duration_ms", 0) or 0),
        cost_is_estimated=bool(int(_row_get(row, "cost_is_estimated", 0) or 0)),
        evidence_payload=evidence_payload,
        inference_payload=inference_payload,
        enrichment_payload=enrichment_payload,
        search_text=search_text,
        evidence_search_text=evidence_search_text,
        inference_search_text=inference_search_text,
        enrichment_search_text=enrichment_search_text,
        enrichment_version=int(_row_get(row, "enrichment_version", 1) or 1),
        enrichment_family=_row_get(row, "enrichment_family", "scored_session_enrichment"),
        inference_version=int(_row_get(row, "inference_version", 1) or 1),
        inference_family=_row_get(row, "inference_family", "heuristic_session_semantics"),
    )


def _row_to_session_work_event_record(row: sqlite3.Row) -> SessionWorkEventRecord:
    legacy_payload = _parse_json(
        _row_get(row, "payload_json"),
        field="payload_json",
        record_id=row["event_id"],
    ) or {}
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
            "canonical_session_date": _row_get(row, "canonical_session_date") or legacy_payload.get("canonical_session_date"),
            "duration_ms": int(_row_get(row, "duration_ms", 0) or legacy_payload.get("duration_ms") or 0),
            "file_paths": tuple(_parse_json(_row_get(row, "file_paths_json")) or legacy_payload.get("file_paths") or []),
            "tools_used": tuple(_parse_json(_row_get(row, "tools_used_json")) or legacy_payload.get("tools_used") or []),
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
    legacy_payload = _parse_json(
        _row_get(row, "payload_json"),
        field="payload_json",
        record_id=row["phase_id"],
    ) or {}
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
            "canonical_session_date": _row_get(row, "canonical_session_date") or legacy_payload.get("canonical_session_date"),
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
            "kind": row["kind"],
            "confidence": float(_row_get(row, "confidence", 0.0) or legacy_payload.get("confidence") or 0.0),
            "evidence": tuple(_parse_json(_row_get(row, "evidence_reasons_json")) or legacy_payload.get("evidence") or []),
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


def _row_to_session_tag_rollup_record(row: sqlite3.Row) -> SessionTagRollupRecord:
    return SessionTagRollupRecord(
        tag=row["tag"],
        bucket_day=row["bucket_day"],
        provider_name=row["provider_name"],
        materializer_version=int(_row_get(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        source_updated_at=_row_get(row, "source_updated_at"),
        source_sort_key=_row_get(row, "source_sort_key"),
        conversation_count=int(_row_get(row, "conversation_count", 0) or 0),
        explicit_count=int(_row_get(row, "explicit_count", 0) or 0),
        auto_count=int(_row_get(row, "auto_count", 0) or 0),
        project_breakdown=_parse_json(
            row["project_breakdown_json"],
            field="project_breakdown_json",
            record_id=f'{row["provider_name"]}:{row["bucket_day"]}:{row["tag"]}',
        )
        or {},
        search_text=row["search_text"],
    )


def _row_to_day_session_summary_record(row: sqlite3.Row) -> DaySessionSummaryRecord:
    return DaySessionSummaryRecord(
        day=row["day"],
        provider_name=row["provider_name"],
        materializer_version=int(_row_get(row, "materializer_version", 1) or 1),
        materialized_at=row["materialized_at"],
        source_updated_at=_row_get(row, "source_updated_at"),
        source_sort_key=_row_get(row, "source_sort_key"),
        conversation_count=int(_row_get(row, "conversation_count", 0) or 0),
        total_cost_usd=float(_row_get(row, "total_cost_usd", 0.0) or 0.0),
        total_duration_ms=int(_row_get(row, "total_duration_ms", 0) or 0),
        total_wall_duration_ms=int(_row_get(row, "total_wall_duration_ms", 0) or 0),
        total_messages=int(_row_get(row, "total_messages", 0) or 0),
        total_words=int(_row_get(row, "total_words", 0) or 0),
        work_event_breakdown=_parse_json(
            row["work_event_breakdown_json"],
            field="work_event_breakdown_json",
            record_id=f'{row["provider_name"]}:{row["day"]}',
        )
        or {},
        projects_active=tuple(_parse_json(_row_get(row, "projects_active_json")) or []),
        payload=_parse_json(
            row["payload_json"],
            field="payload_json",
            record_id=f'{row["provider_name"]}:{row["day"]}',
        )
        or {},
        search_text=row["search_text"],
    )


def _row_to_maintenance_run_record(row: sqlite3.Row) -> MaintenanceRunRecord:
    return MaintenanceRunRecord(
        maintenance_run_id=row["maintenance_run_id"],
        schema_version=int(_row_get(row, "schema_version", 1) or 1),
        executed_at=row["executed_at"],
        mode=row["mode"],
        preview=bool(_row_get(row, "preview", 0)),
        repair_selected=bool(_row_get(row, "repair_selected", 0)),
        cleanup_selected=bool(_row_get(row, "cleanup_selected", 0)),
        vacuum_requested=bool(_row_get(row, "vacuum_requested", 0)),
        target_names=tuple(_parse_json(_row_get(row, "target_names_json")) or []),
        success=bool(_row_get(row, "success", 0)),
        manifest=_parse_json(row["manifest_json"], field="manifest_json", record_id=row["maintenance_run_id"]) or {},
    )


__all__ = [
    "_row_to_day_session_summary_record",
    "_row_to_maintenance_run_record",
    "_row_to_session_phase_record",
    "_row_to_session_profile_record",
    "_row_to_session_tag_rollup_record",
    "_row_to_session_work_event_record",
    "_row_to_work_thread_record",
]

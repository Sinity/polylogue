"""Typed legacy-payload hydrators for session-insight row mappers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from typing import TypeAlias, TypeVar

from pydantic import BaseModel

from polylogue.archive.session.documents import SessionPhaseDocument, WorkEventDocument
from polylogue.core.json import JSONDocument, JSONValue, json_document
from polylogue.insights.archive_models import (
    SessionEnrichmentPayload,
    SessionEvidencePayload,
    SessionInferencePayload,
    SessionPhaseEvidencePayload,
    SessionPhaseInferencePayload,
    WorkEventEvidencePayload,
    WorkEventInferencePayload,
)
from polylogue.storage.backends.queries.mappers_support import _parse_json, _row_get

PayloadModel = TypeVar("PayloadModel", bound=BaseModel)
LegacyPayload: TypeAlias = JSONDocument
LegacyPayloadContainer: TypeAlias = Mapping[str, JSONValue]


def parse_payload_model(
    row: sqlite3.Row,
    column: str,
    *,
    record_id: str,
    model: type[PayloadModel],
) -> PayloadModel | None:
    raw_payload = _parse_json(
        _row_get(row, column),
        field=column,
        record_id=record_id,
    )
    if not raw_payload:
        return None
    return model.model_validate(raw_payload)


def parse_legacy_payload_dict(row: sqlite3.Row, *, record_id: str) -> LegacyPayload:
    legacy_payload = _parse_json(
        _row_get(row, "payload_json"),
        field="payload_json",
        record_id=record_id,
    )
    if isinstance(legacy_payload, dict):
        return json_document(legacy_payload)
    return {}


def session_profile_evidence_from_legacy(
    row: sqlite3.Row,
    legacy_payload: LegacyPayload,
) -> SessionEvidencePayload:
    return SessionEvidencePayload.model_validate(
        {
            "created_at": _legacy_text(legacy_payload.get("created_at")),
            "updated_at": _row_text(row, "source_updated_at") or _legacy_text(legacy_payload.get("updated_at")),
            "first_message_at": _row_text(row, "first_message_at")
            or _legacy_text(legacy_payload.get("first_message_at")),
            "last_message_at": _row_text(row, "last_message_at") or _legacy_text(legacy_payload.get("last_message_at")),
            "canonical_session_date": _row_text(row, "canonical_session_date")
            or _legacy_text(legacy_payload.get("canonical_session_date")),
            "message_count": _row_int(row, "message_count", legacy_payload, legacy_key="message_count"),
            "substantive_count": _row_int(row, "substantive_count", legacy_payload, legacy_key="substantive_count"),
            "attachment_count": _row_int(row, "attachment_count", legacy_payload, legacy_key="attachment_count"),
            "tool_use_count": _row_int(row, "tool_use_count", legacy_payload, legacy_key="tool_use_count"),
            "thinking_count": _row_int(row, "thinking_count", legacy_payload, legacy_key="thinking_count"),
            "word_count": _row_int(row, "word_count", legacy_payload, legacy_key="word_count"),
            "total_cost_usd": _row_float(row, "total_cost_usd", legacy_payload, legacy_key="total_cost_usd"),
            "total_duration_ms": _row_int(row, "total_duration_ms", legacy_payload, legacy_key="total_duration_ms"),
            "wall_duration_ms": _row_int(row, "wall_duration_ms", legacy_payload, legacy_key="wall_duration_ms"),
            "cost_is_estimated": _row_bool(row, "cost_is_estimated", legacy_payload, legacy_key="cost_is_estimated"),
            "tool_categories": _legacy_int_dict(legacy_payload.get("tool_categories")),
            "repo_paths": _row_text_tuple(row, "repo_paths_json", legacy_payload, legacy_key="repo_paths"),
            "cwd_paths": _legacy_text_tuple(legacy_payload.get("cwd_paths")),
            "branch_names": _legacy_text_tuple(legacy_payload.get("branch_names")),
            "file_paths_touched": _legacy_text_tuple(legacy_payload.get("file_paths_touched")),
            "languages_detected": _legacy_text_tuple(legacy_payload.get("languages_detected")),
            "tags": _row_text_tuple(row, "tags_json", legacy_payload, legacy_key="tags"),
            "is_continuation": bool(legacy_payload.get("is_continuation", False)),
            "parent_id": _legacy_text(legacy_payload.get("parent_id")),
        }
    )


def session_profile_inference_from_legacy(
    row: sqlite3.Row,
    legacy_payload: LegacyPayload,
) -> SessionInferencePayload:
    return SessionInferencePayload.model_validate(
        {
            "repo_names": _row_text_tuple(row, "repo_names_json", legacy_payload, legacy_key="repo_names"),
            "work_event_count": _row_int(row, "work_event_count", legacy_payload, legacy_key="work_event_count"),
            "phase_count": _row_int(row, "phase_count", legacy_payload, legacy_key="phase_count"),
            "engaged_duration_ms": _row_int(
                row,
                "engaged_duration_ms",
                legacy_payload,
                legacy_key="engaged_duration_ms",
            ),
            "engaged_minutes": _legacy_float(legacy_payload.get("engaged_minutes")),
            "support_level": _legacy_text(legacy_payload.get("support_level")) or "weak",
            "support_signals": _legacy_text_tuple(legacy_payload.get("support_signals")),
            "engaged_duration_source": _legacy_text(legacy_payload.get("engaged_duration_source"))
            or "session_total_fallback",
            "repo_inference_strength": _legacy_text(legacy_payload.get("repo_inference_strength")) or "weak",
            "auto_tags": _row_text_tuple(row, "auto_tags_json", legacy_payload, legacy_key="auto_tags"),
            "work_events": _legacy_work_event_documents(legacy_payload.get("work_events")),
            "phases": _legacy_phase_documents(legacy_payload.get("phases")),
        }
    )


def session_profile_enrichment_from_legacy(
    row: sqlite3.Row,
    legacy_payload: LegacyPayload,
    *,
    inference_payload: SessionInferencePayload,
) -> SessionEnrichmentPayload:
    repo_paths = _row_text_tuple(row, "repo_paths_json", legacy_payload, legacy_key="repo_paths")
    repo_names = _row_text_tuple(row, "repo_names_json", legacy_payload, legacy_key="repo_names")
    return SessionEnrichmentPayload.model_validate(
        {
            "intent_summary": _row_text(row, "title") or _legacy_text(legacy_payload.get("title")),
            "outcome_summary": None,
            "blockers": (),
            "confidence": 0.0,
            "support_level": "weak",
            "support_signals": inference_payload.support_signals,
            "input_band_summary": {
                "user_turns": 0,
                "assistant_turns": 0,
                "action_events": 0,
                "touched_paths": len(repo_paths),
                "repo_names": len(repo_names),
            },
        }
    )


def session_work_event_evidence_from_legacy(
    row: sqlite3.Row,
    legacy_payload: LegacyPayload,
) -> WorkEventEvidencePayload:
    return WorkEventEvidencePayload.model_validate(
        {
            "start_index": _row_int(row, "start_index", legacy_payload, legacy_key="start_index"),
            "end_index": _row_int(row, "end_index", legacy_payload, legacy_key="end_index"),
            "start_time": _row_text(row, "start_time") or _legacy_text(legacy_payload.get("start_time")),
            "end_time": _row_text(row, "end_time") or _legacy_text(legacy_payload.get("end_time")),
            "canonical_session_date": _row_text(row, "canonical_session_date")
            or _legacy_text(legacy_payload.get("canonical_session_date")),
            "duration_ms": _row_int(row, "duration_ms", legacy_payload, legacy_key="duration_ms"),
            "file_paths": _row_text_tuple(row, "file_paths_json", legacy_payload, legacy_key="file_paths"),
            "tools_used": _row_text_tuple(row, "tools_used_json", legacy_payload, legacy_key="tools_used"),
        }
    )


def session_work_event_inference_from_legacy(
    row: sqlite3.Row,
    legacy_payload: LegacyPayload,
) -> WorkEventInferencePayload:
    return WorkEventInferencePayload.model_validate(
        {
            "kind": _row_text(row, "kind") or "",
            "summary": _row_text(row, "summary") or "",
            "confidence": _row_float(row, "confidence", legacy_payload, legacy_key="confidence"),
            "evidence": _legacy_text_tuple(legacy_payload.get("evidence")),
        }
    )


def session_phase_evidence_from_legacy(
    row: sqlite3.Row,
    legacy_payload: LegacyPayload,
) -> SessionPhaseEvidencePayload:
    return SessionPhaseEvidencePayload.model_validate(
        {
            "start_time": _row_text(row, "start_time") or _legacy_text(legacy_payload.get("start_time")),
            "end_time": _row_text(row, "end_time") or _legacy_text(legacy_payload.get("end_time")),
            "canonical_session_date": _row_text(row, "canonical_session_date")
            or _legacy_text(legacy_payload.get("canonical_session_date")),
            "message_range": (
                _row_int(row, "start_index", legacy_payload, legacy_key="start_index"),
                _row_int(row, "end_index", legacy_payload, legacy_key="end_index"),
            ),
            "duration_ms": _row_int(row, "duration_ms", legacy_payload, legacy_key="duration_ms"),
            "tool_counts": _row_int_dict(row, "tool_counts_json", legacy_payload, legacy_key="tool_counts"),
            "word_count": _row_int(row, "word_count", legacy_payload, legacy_key="word_count"),
        }
    )


def session_phase_inference_from_legacy(
    row: sqlite3.Row,
    legacy_payload: LegacyPayload,
) -> SessionPhaseInferencePayload:
    return SessionPhaseInferencePayload.model_validate(
        {
            "confidence": _row_float(row, "confidence", legacy_payload, legacy_key="confidence"),
            "evidence": _row_text_tuple(
                row,
                "evidence_reasons_json",
                legacy_payload,
                legacy_key="evidence",
            ),
        }
    )


def _row_text(row: sqlite3.Row, column: str) -> str | None:
    value = _row_get(row, column)
    return None if value is None else str(value)


def _row_int(
    row: sqlite3.Row,
    column: str,
    legacy_payload: LegacyPayloadContainer,
    *,
    legacy_key: str,
    default: int = 0,
) -> int:
    raw_value: JSONValue | bytes | bytearray | None = _row_get(row, column, None)
    if raw_value is None:
        raw_value = legacy_payload.get(legacy_key, default)
    return _legacy_int(raw_value, default)


def _row_float(
    row: sqlite3.Row,
    column: str,
    legacy_payload: LegacyPayloadContainer,
    *,
    legacy_key: str,
    default: float = 0.0,
) -> float:
    raw_value: JSONValue | bytes | bytearray | None = _row_get(row, column, None)
    if raw_value is None:
        raw_value = legacy_payload.get(legacy_key, default)
    return _legacy_float(raw_value, default)


def _row_bool(
    row: sqlite3.Row,
    column: str,
    legacy_payload: LegacyPayloadContainer,
    *,
    legacy_key: str,
) -> bool:
    raw_value: JSONValue | bytes | bytearray | None = _row_get(row, column, None)
    if raw_value is None:
        raw_value = legacy_payload.get(legacy_key, False)
    return _legacy_bool(raw_value)


def _row_text_tuple(
    row: sqlite3.Row,
    column: str,
    legacy_payload: LegacyPayloadContainer,
    *,
    legacy_key: str,
) -> tuple[str, ...]:
    parsed = _parse_json(_row_get(row, column))
    if parsed is not None:
        return _legacy_text_tuple(parsed)
    return _legacy_text_tuple(legacy_payload.get(legacy_key))


def _row_int_dict(
    row: sqlite3.Row,
    column: str,
    legacy_payload: LegacyPayloadContainer,
    *,
    legacy_key: str,
) -> dict[str, int]:
    parsed = _parse_json(_row_get(row, column))
    if parsed is not None:
        return _legacy_int_dict(parsed)
    return _legacy_int_dict(legacy_payload.get(legacy_key))


def _legacy_text(value: JSONValue | bytes | bytearray) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _legacy_float(value: JSONValue | bytes | bytearray, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, str | int | float):
        return float(value or default)
    return default


def _legacy_int(value: JSONValue | bytes | bytearray, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str | int | float):
        return int(value or default)
    return default


def _legacy_text_tuple(
    value: JSONValue | bytes | bytearray | list[JSONValue] | tuple[JSONValue, ...],
) -> tuple[str, ...]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes | Mapping):
        return ()
    return tuple(str(item) for item in value if item is not None)


def _legacy_dict_tuple(
    value: JSONValue | list[JSONValue] | tuple[JSONValue, ...] | bytes | bytearray,
) -> tuple[LegacyPayload, ...]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes | Mapping):
        return ()
    items: list[LegacyPayload] = []
    for item in value:
        if isinstance(item, Mapping):
            items.append(json_document(dict(item)))
    return tuple(items)


def _legacy_int_dict(value: JSONValue | bytes | bytearray | Mapping[str, JSONValue]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _legacy_int(item) for key, item in value.items() if item is not None}


def _legacy_bool(value: JSONValue | bytes | bytearray) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str | int | float):
        return bool(int(value or 0))
    return bool(value)


def _legacy_work_event_documents(
    value: JSONValue | list[JSONValue] | tuple[JSONValue, ...],
) -> tuple[WorkEventDocument, ...]:
    documents: list[WorkEventDocument] = []
    for item in _legacy_dict_tuple(value):
        documents.append(
            {
                "kind": _legacy_text(item.get("kind")) or "conversation",
                "start_index": _legacy_int(item.get("start_index")),
                "end_index": _legacy_int(item.get("end_index")),
                "start_time": _legacy_text(item.get("start_time")),
                "end_time": _legacy_text(item.get("end_time")),
                "canonical_session_date": _legacy_text(item.get("canonical_session_date")),
                "duration_ms": _legacy_int(item.get("duration_ms")),
                "confidence": _legacy_float(item.get("confidence")),
                "evidence": list(_legacy_text_tuple(item.get("evidence"))),
                "file_paths": list(_legacy_text_tuple(item.get("file_paths"))),
                "tools_used": list(_legacy_text_tuple(item.get("tools_used"))),
                "summary": _legacy_text(item.get("summary")) or "",
            }
        )
    return tuple(documents)


def _legacy_phase_documents(
    value: JSONValue | list[JSONValue] | tuple[JSONValue, ...],
) -> tuple[SessionPhaseDocument, ...]:
    documents: list[SessionPhaseDocument] = []
    for item in _legacy_dict_tuple(value):
        start_index = _legacy_int(item.get("start_index"))
        end_index = _legacy_int(item.get("end_index"))
        documents.append(
            {
                "start_time": _legacy_text(item.get("start_time")),
                "end_time": _legacy_text(item.get("end_time")),
                "canonical_session_date": _legacy_text(item.get("canonical_session_date")),
                "message_range": [start_index, end_index],
                "duration_ms": _legacy_int(item.get("duration_ms")),
                "tool_counts": _legacy_int_dict(item.get("tool_counts")),
                "word_count": _legacy_int(item.get("word_count")),
                "confidence": _legacy_float(item.get("confidence")),
                "evidence": list(_legacy_text_tuple(item.get("evidence"))),
            }
        )
    return tuple(documents)


__all__ = [
    "parse_legacy_payload_dict",
    "parse_payload_model",
    "session_phase_evidence_from_legacy",
    "session_phase_inference_from_legacy",
    "session_profile_enrichment_from_legacy",
    "session_profile_evidence_from_legacy",
    "session_profile_inference_from_legacy",
    "session_work_event_evidence_from_legacy",
    "session_work_event_inference_from_legacy",
]

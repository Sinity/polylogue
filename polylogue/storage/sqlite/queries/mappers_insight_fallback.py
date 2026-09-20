"""Typed fallback-payload hydrators for session-insight row mappers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from typing import TypeAlias, TypeVar

from pydantic import BaseModel

from polylogue.analysis.archive_models import (
    SessionEnrichmentPayload,
    SessionEvidencePayload,
    SessionInferencePayload,
)
from polylogue.core.json import JSONDocument, JSONValue, json_document
from polylogue.storage.sqlite.queries.mappers_support import _parse_json, _row_get

PayloadModel = TypeVar("PayloadModel", bound=BaseModel)
FallbackPayload: TypeAlias = JSONDocument
FallbackPayloadContainer: TypeAlias = Mapping[str, JSONValue]


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


def parse_fallback_payload_dict(row: sqlite3.Row, *, record_id: str) -> FallbackPayload:
    fallback_payload = _parse_json(
        _row_get(row, "payload_json"),
        field="payload_json",
        record_id=record_id,
    )
    if isinstance(fallback_payload, dict):
        return json_document(fallback_payload)
    return {}


def session_profile_evidence_from_fallback(
    row: sqlite3.Row,
    fallback_payload: FallbackPayload,
) -> SessionEvidencePayload:
    return SessionEvidencePayload.model_validate(
        {
            "created_at": _fallback_text(fallback_payload.get("created_at")),
            "updated_at": _row_text(row, "source_updated_at") or _fallback_text(fallback_payload.get("updated_at")),
            "first_message_at": _row_text(row, "first_message_at")
            or _fallback_text(fallback_payload.get("first_message_at")),
            "last_message_at": _row_text(row, "last_message_at")
            or _fallback_text(fallback_payload.get("last_message_at")),
            "canonical_session_date": _row_text(row, "canonical_session_date")
            or _fallback_text(fallback_payload.get("canonical_session_date")),
            "message_count": _row_int(row, "message_count", fallback_payload, fallback_key="message_count"),
            "substantive_count": _row_int(row, "substantive_count", fallback_payload, fallback_key="substantive_count"),
            "attachment_count": _row_int(row, "attachment_count", fallback_payload, fallback_key="attachment_count"),
            "tool_use_count": _row_int(row, "tool_use_count", fallback_payload, fallback_key="tool_use_count"),
            "thinking_count": _row_int(row, "thinking_count", fallback_payload, fallback_key="thinking_count"),
            "word_count": _row_int(row, "word_count", fallback_payload, fallback_key="word_count"),
            "total_cost_usd": _row_float(row, "total_cost_usd", fallback_payload, fallback_key="total_cost_usd"),
            "total_duration_ms": _row_int(row, "total_duration_ms", fallback_payload, fallback_key="total_duration_ms"),
            "wall_duration_ms": _row_int(row, "wall_duration_ms", fallback_payload, fallback_key="wall_duration_ms"),
            "cost_is_estimated": _row_bool(
                row, "cost_is_estimated", fallback_payload, fallback_key="cost_is_estimated"
            ),
            "tool_categories": _fallback_int_dict(fallback_payload.get("tool_categories")),
            "repo_paths": _row_text_tuple(row, "repo_paths_json", fallback_payload, fallback_key="repo_paths"),
            "cwd_paths": _fallback_text_tuple(fallback_payload.get("cwd_paths")),
            "branch_names": _fallback_text_tuple(fallback_payload.get("branch_names")),
            "file_paths_touched": _fallback_text_tuple(fallback_payload.get("file_paths_touched")),
            "languages_detected": _fallback_text_tuple(fallback_payload.get("languages_detected")),
            "tags": _row_text_tuple(row, "tags_json", fallback_payload, fallback_key="tags"),
            "is_continuation": bool(fallback_payload.get("is_continuation", False)),
            "parent_id": _fallback_text(fallback_payload.get("parent_id")),
        }
    )


def session_profile_inference_from_fallback(
    row: sqlite3.Row,
    fallback_payload: FallbackPayload,
) -> SessionInferencePayload:
    return SessionInferencePayload.model_validate(
        {
            "repo_names": _row_text_tuple(row, "repo_names_json", fallback_payload, fallback_key="repo_names"),
            "engaged_duration_ms": _row_int(
                row,
                "engaged_duration_ms",
                fallback_payload,
                fallback_key="engaged_duration_ms",
            ),
            "engaged_minutes": _fallback_float(fallback_payload.get("engaged_minutes")),
            "support_level": _fallback_text(fallback_payload.get("support_level")) or "weak",
            "support_signals": _fallback_text_tuple(fallback_payload.get("support_signals")),
            "engaged_duration_source": _fallback_text(fallback_payload.get("engaged_duration_source"))
            or "session_total_fallback",
            "repo_inference_strength": _fallback_text(fallback_payload.get("repo_inference_strength")) or "weak",
            "auto_tags": _row_text_tuple(row, "auto_tags_json", fallback_payload, fallback_key="auto_tags"),
        }
    )


def session_profile_enrichment_from_fallback(
    row: sqlite3.Row,
    fallback_payload: FallbackPayload,
    *,
    inference_payload: SessionInferencePayload,
) -> SessionEnrichmentPayload:
    repo_paths = _row_text_tuple(row, "repo_paths_json", fallback_payload, fallback_key="repo_paths")
    repo_names = _row_text_tuple(row, "repo_names_json", fallback_payload, fallback_key="repo_names")
    return SessionEnrichmentPayload.model_validate(
        {
            "intent_summary": _row_text(row, "title") or _fallback_text(fallback_payload.get("title")),
            "outcome_summary": None,
            "blockers": (),
            "confidence": 0.0,
            "support_level": "weak",
            "support_signals": inference_payload.support_signals,
            "input_band_summary": {
                "user_turns": 0,
                "assistant_turns": 0,
                "actions": 0,
                "touched_paths": len(repo_paths),
                "repo_names": len(repo_names),
            },
        }
    )


def _row_text(row: sqlite3.Row, column: str) -> str | None:
    value = _row_get(row, column)
    return None if value is None else str(value)


def _row_int(
    row: sqlite3.Row,
    column: str,
    fallback_payload: FallbackPayloadContainer,
    *,
    fallback_key: str,
    default: int = 0,
) -> int:
    raw_value: JSONValue | bytes | bytearray | None = _row_get(row, column, None)
    if raw_value is None:
        raw_value = fallback_payload.get(fallback_key, default)
    return _fallback_int(raw_value, default)


def _row_float(
    row: sqlite3.Row,
    column: str,
    fallback_payload: FallbackPayloadContainer,
    *,
    fallback_key: str,
    default: float = 0.0,
) -> float:
    raw_value: JSONValue | bytes | bytearray | None = _row_get(row, column, None)
    if raw_value is None:
        raw_value = fallback_payload.get(fallback_key, default)
    return _fallback_float(raw_value, default)


def _row_bool(
    row: sqlite3.Row,
    column: str,
    fallback_payload: FallbackPayloadContainer,
    *,
    fallback_key: str,
) -> bool:
    raw_value: JSONValue | bytes | bytearray | None = _row_get(row, column, None)
    if raw_value is None:
        raw_value = fallback_payload.get(fallback_key, False)
    return _fallback_bool(raw_value)


def _row_text_tuple(
    row: sqlite3.Row,
    column: str,
    fallback_payload: FallbackPayloadContainer,
    *,
    fallback_key: str,
) -> tuple[str, ...]:
    parsed = _parse_json(_row_get(row, column))
    if parsed is not None:
        return _fallback_text_tuple(parsed)
    return _fallback_text_tuple(fallback_payload.get(fallback_key))


def _row_int_dict(
    row: sqlite3.Row,
    column: str,
    fallback_payload: FallbackPayloadContainer,
    *,
    fallback_key: str,
) -> dict[str, int]:
    parsed = _parse_json(_row_get(row, column))
    if parsed is not None:
        return _fallback_int_dict(parsed)
    return _fallback_int_dict(fallback_payload.get(fallback_key))


def _fallback_text(value: JSONValue | bytes | bytearray) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _fallback_float(value: JSONValue | bytes | bytearray, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, str | int | float):
        return float(value or default)
    return default


def _fallback_int(value: JSONValue | bytes | bytearray, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str | int | float):
        return int(value or default)
    return default


def _fallback_text_tuple(
    value: JSONValue | bytes | bytearray | list[JSONValue] | tuple[JSONValue, ...],
) -> tuple[str, ...]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes | Mapping):
        return ()
    return tuple(str(item) for item in value if item is not None)


def _fallback_dict_tuple(
    value: JSONValue | list[JSONValue] | tuple[JSONValue, ...] | bytes | bytearray,
) -> tuple[FallbackPayload, ...]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes | Mapping):
        return ()
    items: list[FallbackPayload] = []
    for item in value:
        if isinstance(item, Mapping):
            items.append(json_document(dict(item)))
    return tuple(items)


def _fallback_int_dict(value: JSONValue | bytes | bytearray | Mapping[str, JSONValue]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _fallback_int(item) for key, item in value.items() if item is not None}


def _fallback_bool(value: JSONValue | bytes | bytearray) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str | int | float):
        return bool(int(value or 0))
    return bool(value)


__all__ = [
    "parse_fallback_payload_dict",
    "parse_payload_model",
    "session_profile_enrichment_from_fallback",
    "session_profile_evidence_from_fallback",
    "session_profile_inference_from_fallback",
]

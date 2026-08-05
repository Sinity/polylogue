"""Concrete column specifications for archive_tiers tables.

Defines the single source of truth for:
  - messages table structure (29 writable columns + 1 GENERATED message_id)
  - blocks table structure (16 writable columns + 1 GENERATED block_id)
  - Other key tables (sessions, session_events, etc.)

Each spec drives INSERT/SELECT generation and typed row extraction.

Extractors are typically defined at module level or passed as callables at spec
construction time. This approach consolidates the column triplicates (column list,
placeholder string, tuple order) into a single spec that drives INSERT/SELECT.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from operator import itemgetter

from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import (
    BlockType,
    MaterialOrigin,
    PasteBoundary,
    StopReason,
    ToolResultUnknownReason,
)
from polylogue.storage.sqlite.archive_tiers.column_spec import ColumnSpec, TableColumnSpec
from polylogue.storage.sqlite.archive_tiers.common import CONTENT_HASH_CHECK, check, json_object_check, nullable_check


def _value(name: str) -> Callable[[dict[str, object]], object]:
    return itemgetter(name)


def _none_to_zero(value: object) -> int:
    if isinstance(value, (int, float, str)):
        return int(value)
    return 0


def _epoch_seconds_to_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float, str)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        return None
    except (TypeError, ValueError, OSError):
        return None


def _bool_value(value: object) -> bool:
    return bool(value)


def _optional_bool_value(value: object) -> bool | None:
    return None if value is None else bool(value)


def _ddl(name: str, sql: str) -> ColumnSpec:
    return ColumnSpec(name=name, is_generated="GENERATED ALWAYS" in sql, ddl_sql=f"{name} {sql}")


def _make_messages_spec() -> TableColumnSpec:
    """Create the messages table column specification.

    The messages table structure (from schema):
      session_id, native_id, parent_message_id, position, role, message_type,
      material_origin, model_name, model_effort, sender_name, recipient,
      delivery_status, end_turn, user_context_text, has_tool_use, has_thinking,
      has_paste, paste_boundary, variant_index, is_active_path, is_active_leaf,
      word_count, input_tokens, output_tokens, cache_read_tokens,
      cache_write_tokens, duration_ms, content_hash, occurred_at_ms

    GENERATED (not writable): message_id

    Special handling: parent_message_id is always NULL on INSERT (no tuple value).
    """
    all_columns: tuple[ColumnSpec, ...] = (
        _ddl(
            "message_id",
            "TEXT GENERATED ALWAYS AS (session_id || ':' || CASE WHEN native_id IS NULL THEN 'p:' || position || '.' || variant_index ELSE 'n:' || native_id END) STORED UNIQUE",
        ),
        _ddl("session_id", "TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE"),
        _ddl("native_id", "TEXT"),
        ColumnSpec(
            "parent_message_id",
            "TEXT",
            extract_placeholder="NULL",
            ddl_sql="parent_message_id TEXT REFERENCES messages(message_id) ON DELETE SET NULL",
        ),
        _ddl("position", "INTEGER NOT NULL CHECK(position >= 0)"),
        ColumnSpec("role", "TEXT", ddl_sql=f"role TEXT NOT NULL CHECK ({check('role', Role)})"),
        ColumnSpec(
            "message_type",
            "TEXT",
            ddl_sql=f"message_type TEXT NOT NULL DEFAULT 'message' CHECK ({check('message_type', MessageType)})",
        ),
        ColumnSpec(
            "material_origin",
            "TEXT",
            ddl_sql=f"material_origin TEXT NOT NULL DEFAULT 'unknown' CHECK ({check('material_origin', MaterialOrigin)})",
        ),
        _ddl("model_name", "TEXT"),
        _ddl("model_effort", "TEXT"),
        _ddl("sender_name", "TEXT"),
        _ddl("recipient", "TEXT"),
        _ddl("delivery_status", "TEXT"),
        _ddl("end_turn", "INTEGER CHECK(end_turn IN (0, 1) OR end_turn IS NULL)"),
        _ddl("user_context_text", "TEXT"),
        _ddl("has_tool_use", "INTEGER NOT NULL DEFAULT 0 CHECK(has_tool_use IN (0, 1))"),
        _ddl("has_thinking", "INTEGER NOT NULL DEFAULT 0 CHECK(has_thinking IN (0, 1))"),
        _ddl("has_paste", "INTEGER NOT NULL DEFAULT 0 CHECK(has_paste IN (0, 1))"),
        ColumnSpec(
            "paste_boundary",
            "TEXT",
            ddl_sql=f"paste_boundary TEXT CHECK ({nullable_check('paste_boundary', PasteBoundary)})",
        ),
        _ddl("variant_index", "INTEGER NOT NULL DEFAULT 0 CHECK(variant_index >= 0)"),
        _ddl("is_active_path", "INTEGER NOT NULL DEFAULT 1 CHECK(is_active_path IN (0, 1))"),
        _ddl("is_active_leaf", "INTEGER NOT NULL DEFAULT 0 CHECK(is_active_leaf IN (0, 1))"),
        _ddl("word_count", "INTEGER NOT NULL DEFAULT 0 CHECK(word_count >= 0)"),
        _ddl("input_tokens", "INTEGER NOT NULL DEFAULT 0 CHECK(input_tokens >= 0)"),
        _ddl("output_tokens", "INTEGER NOT NULL DEFAULT 0 CHECK(output_tokens >= 0)"),
        _ddl("cache_read_tokens", "INTEGER NOT NULL DEFAULT 0 CHECK(cache_read_tokens >= 0)"),
        _ddl("cache_write_tokens", "INTEGER NOT NULL DEFAULT 0 CHECK(cache_write_tokens >= 0)"),
        _ddl("duration_ms", "INTEGER CHECK(duration_ms IS NULL OR duration_ms >= 0)"),
        _ddl("content_hash", f"BLOB NOT NULL {CONTENT_HASH_CHECK}"),
        _ddl("occurred_at_ms", "INTEGER"),
        ColumnSpec(
            "stop_reason", "TEXT", ddl_sql=f"stop_reason TEXT CHECK ({nullable_check('stop_reason', StopReason)})"
        ),
    )

    record_columns = (
        ColumnSpec(
            "text",
            record_name="text",
            domain_name="text",
            select_expression="COALESCE((SELECT group_concat(b.text, char(10)) FROM blocks b WHERE b.message_id = {alias}.message_id AND b.text IS NOT NULL), '')",
        ),
        ColumnSpec("source_name", record_name="source_name", select_expression="s.origin"),
        ColumnSpec("version", record_name="version", select_expression="1"),
    )

    def record(
        name: str,
        *,
        record_name: str | None = None,
        domain_name: str | None = None,
        select_expression: str | None = None,
        record_transform: Callable[[object], object] | None = None,
        domain_transform: Callable[[object], object] | None = None,
    ) -> ColumnSpec:
        return ColumnSpec(
            name,
            record_name=record_name or name,
            domain_name=domain_name,
            select_expression=select_expression,
            record_transform=record_transform,
            domain_transform=domain_transform,
        )

    mappings = {
        "message_id": record("message_id", domain_name="id"),
        "session_id": record("session_id"),
        "native_id": record("native_id", record_name="provider_message_id"),
        "parent_message_id": record("parent_message_id", domain_name="parent_id"),
        "position": record("position", domain_name="position", record_transform=_none_to_zero),
        "role": record("role", domain_name="role"),
        "message_type": record("message_type", domain_name="message_type"),
        "material_origin": record("material_origin", domain_name="material_origin"),
        "model_name": record("model_name", domain_name="model_name"),
        "has_tool_use": record(
            "has_tool_use", domain_name="has_tool_use", record_transform=_none_to_zero, domain_transform=_bool_value
        ),
        "has_thinking": record(
            "has_thinking", domain_name="has_thinking", record_transform=_none_to_zero, domain_transform=_bool_value
        ),
        "has_paste": record(
            "has_paste", domain_name="has_paste", record_transform=_none_to_zero, domain_transform=_bool_value
        ),
        "paste_boundary": record(
            "paste_boundary", record_name="paste_boundary_state", domain_name="paste_boundary_state"
        ),
        "variant_index": record(
            "variant_index", record_name="branch_index", domain_name="branch_index", record_transform=_none_to_zero
        ),
        "is_active_path": record(
            "is_active_path",
            domain_name="is_active_path",
            record_transform=_optional_bool_value,
            domain_transform=lambda value: value,
        ),
        "is_active_leaf": record(
            "is_active_leaf", domain_name="is_active_leaf", record_transform=_bool_value, domain_transform=_bool_value
        ),
        "word_count": record("word_count", record_transform=_none_to_zero),
        "input_tokens": record("input_tokens", domain_name="input_tokens", record_transform=_none_to_zero),
        "output_tokens": record("output_tokens", domain_name="output_tokens", record_transform=_none_to_zero),
        "cache_read_tokens": record(
            "cache_read_tokens", domain_name="cache_read_tokens", record_transform=_none_to_zero
        ),
        "cache_write_tokens": record(
            "cache_write_tokens", domain_name="cache_write_tokens", record_transform=_none_to_zero
        ),
        "duration_ms": record("duration_ms", domain_name="duration_ms", domain_transform=_none_to_zero),
        "content_hash": record("content_hash", select_expression="lower(hex({alias}.content_hash))"),
        "occurred_at_ms": record(
            "occurred_at_ms",
            record_name="sort_key",
            domain_name="timestamp",
            select_expression="{alias}.occurred_at_ms / 1000.0",
            domain_transform=_epoch_seconds_to_datetime,
        ),
        "stop_reason": record("stop_reason", domain_name="stop_reason"),
    }
    all_columns = tuple(
        replace(
            col,
            extract=_value(col.name) if col.extract_placeholder == "?" else None,
            record_name=mappings[col.name].record_name,
            select_expression=mappings[col.name].select_expression,
            record_transform=mappings[col.name].record_transform,
            domain_name=mappings[col.name].domain_name,
            domain_transform=mappings[col.name].domain_transform,
        )
        if col.name in mappings
        else replace(col, extract=_value(col.name) if col.extract_placeholder == "?" else None)
        for col in all_columns
    )

    writable_columns = tuple(col for col in all_columns if not col.is_generated)

    return TableColumnSpec(
        table_name="messages",
        all_columns=all_columns,
        writable_columns=writable_columns,
        record_only_columns=record_columns,
    )


def _make_blocks_spec() -> TableColumnSpec:
    """Create the blocks table column specification.

    The blocks table structure (from schema):
      session_id, message_id, position, block_type, text, tool_name, tool_id,
      tool_input, semantic_type, media_type, language, tool_result_is_error,
      tool_result_exit_code, tool_result_outcome_unknown_reason, signature,
      content_hash

    GENERATED (not writable):
      block_id, tool_command, tool_path, search_text, tool_detail_text
    """
    all_columns: tuple[ColumnSpec, ...] = (
        _ddl("block_id", "TEXT GENERATED ALWAYS AS (message_id || ':' || position) STORED UNIQUE"),
        _ddl("message_id", "TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE"),
        _ddl("session_id", "TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE"),
        _ddl("position", "INTEGER NOT NULL CHECK(position >= 0)"),
        ColumnSpec("block_type", "TEXT", ddl_sql=f"block_type TEXT NOT NULL CHECK ({check('block_type', BlockType)})"),
        _ddl("text", "TEXT"),
        _ddl("tool_name", "TEXT"),
        _ddl("tool_id", "TEXT"),
        ColumnSpec(
            "tool_input", "TEXT", ddl_sql=f"tool_input TEXT CHECK ({json_object_check('tool_input', nullable=True)})"
        ),
        _ddl("semantic_type", "TEXT"),
        _ddl("media_type", "TEXT"),
        _ddl("language", "TEXT"),
        _ddl("tool_result_is_error", "INTEGER CHECK (tool_result_is_error IN (0, 1))"),
        _ddl("tool_result_exit_code", "INTEGER"),
        ColumnSpec(
            "tool_result_outcome_unknown_reason",
            "TEXT",
            ddl_sql=f"tool_result_outcome_unknown_reason TEXT CHECK ({nullable_check('tool_result_outcome_unknown_reason', ToolResultUnknownReason)})",
        ),
        _ddl("signature", "TEXT"),
        _ddl("content_hash", "BLOB CHECK(content_hash IS NULL OR length(content_hash) = 32)"),
        _ddl("tool_command", "TEXT GENERATED ALWAYS AS (json_extract(tool_input, '$.command')) VIRTUAL"),
        _ddl(
            "tool_path",
            "TEXT GENERATED ALWAYS AS (COALESCE(json_extract(tool_input, '$.file_path'), json_extract(tool_input, '$.path'))) VIRTUAL",
        ),
        _ddl(
            "search_text",
            "TEXT GENERATED ALWAYS AS (trim(COALESCE(text, '') || ' ' || COALESCE(tool_name, '') || ' ' || COALESCE(json_extract(tool_input, '$.command'), '') || ' ' || COALESCE(json_extract(tool_input, '$.file_path'), '') || ' ' || COALESCE(json_extract(tool_input, '$.path'), ''))) VIRTUAL",
        ),
        _ddl(
            "tool_detail_text",
            "TEXT GENERATED ALWAYS AS (lower(COALESCE(tool_command, '') || ' ' || COALESCE(tool_path, ''))) VIRTUAL",
        ),
    )

    record_columns = (ColumnSpec("metadata", record_name="metadata", select_expression="NULL"),)
    mappings = {
        "block_id": ("block_id", None),
        "message_id": ("message_id", None),
        "session_id": ("session_id", None),
        "position": ("block_index", None),
        "block_type": ("type", None),
        "text": ("text", None),
        "tool_name": ("tool_name", None),
        "tool_id": ("tool_id", None),
        "tool_input": ("tool_input", None),
        "semantic_type": ("semantic_type", None),
        "tool_result_is_error": ("tool_result_is_error", None),
        "tool_result_exit_code": ("tool_result_exit_code", None),
        "tool_result_outcome_unknown_reason": ("tool_result_outcome_unknown_reason", None),
        "signature": ("signature", None),
    }
    all_columns = tuple(
        replace(
            col,
            extract=_value(col.name) if col.extract_placeholder == "?" else None,
            record_name=mappings[col.name][0],
        )
        if col.name in mappings
        else replace(col, extract=_value(col.name) if col.extract_placeholder == "?" else None)
        for col in all_columns
    )

    writable_columns = tuple(col for col in all_columns if not col.is_generated)

    return TableColumnSpec(
        table_name="blocks",
        all_columns=all_columns,
        writable_columns=writable_columns,
        record_only_columns=record_columns,
    )


# Global registry of table specs
MESSAGES_SPEC = _make_messages_spec()
BLOCKS_SPEC = _make_blocks_spec()

TABLE_SPECS = {
    "messages": MESSAGES_SPEC,
    "blocks": BLOCKS_SPEC,
}

"""Row-mapping functions from SQLite rows to storage record models."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from polylogue.errors import DatabaseError
from polylogue.storage.store import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    RawConversationRecord,
)
from polylogue.types import (
    ContentBlockType,
    ConversationId,
    MessageId,
    Provider,
    SemanticBlockType,
    ValidationMode,
    ValidationStatus,
)


def _parse_json(raw: str | None, *, field: str = "", record_id: str = "") -> Any:
    """Parse a JSON string with diagnostic context on failure."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatabaseError(
            f"Corrupt JSON in {field} for {record_id}: {exc} (value starts: {raw[:80]!r})"
        ) from exc


def _row_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    """Get a column value, returning default if the column doesn't exist.

    Handles schema version differences where optional columns may be absent.
    """
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


def _row_to_conversation(row: sqlite3.Row) -> ConversationRecord:
    """Map a SQLite row to a ConversationRecord."""
    return ConversationRecord(
        conversation_id=row["conversation_id"],
        provider_name=row["provider_name"],
        provider_conversation_id=row["provider_conversation_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        sort_key=_row_get(row, "sort_key"),
        content_hash=row["content_hash"],
        provider_meta=_parse_json(row["provider_meta"], field="provider_meta", record_id=row["conversation_id"]),
        metadata=_parse_json(row["metadata"], field="metadata", record_id=row["conversation_id"]),
        version=row["version"],
        parent_conversation_id=_row_get(row, "parent_conversation_id"),
        branch_type=_row_get(row, "branch_type"),
        raw_id=_row_get(row, "raw_id"),
    )


def _row_to_message(row: sqlite3.Row) -> MessageRecord:
    """Map a SQLite row to a MessageRecord."""
    return MessageRecord(
        message_id=row["message_id"],
        conversation_id=row["conversation_id"],
        provider_message_id=_row_get(row, "provider_message_id"),
        role=_row_get(row, "role"),
        text=_row_get(row, "text"),
        sort_key=_row_get(row, "sort_key"),
        content_hash=row["content_hash"],
        version=row["version"],
        parent_message_id=_row_get(row, "parent_message_id"),
        branch_index=_row_get(row, "branch_index", 0) or 0,
        provider_name=_row_get(row, "provider_name", '') or '',
        word_count=_row_get(row, "word_count", 0) or 0,
        has_tool_use=_row_get(row, "has_tool_use", 0) or 0,
        has_thinking=_row_get(row, "has_thinking", 0) or 0,
    )


def _row_to_content_block(row: sqlite3.Row) -> ContentBlockRecord:
    """Map a SQLite row to a ContentBlockRecord."""
    return ContentBlockRecord(
        block_id=row["block_id"],
        message_id=MessageId(row["message_id"]),
        conversation_id=ConversationId(row["conversation_id"]),
        block_index=row["block_index"],
        type=ContentBlockType.from_string(row["type"]),
        text=_row_get(row, "text"),
        tool_name=_row_get(row, "tool_name"),
        tool_id=_row_get(row, "tool_id"),
        tool_input=_row_get(row, "tool_input"),
        media_type=_row_get(row, "media_type"),
        metadata=_row_get(row, "metadata"),
        semantic_type=(
            SemanticBlockType.from_string(_row_get(row, "semantic_type"))
            if _row_get(row, "semantic_type") is not None
            else None
        ),
    )


def _row_to_raw_conversation(row: sqlite3.Row) -> RawConversationRecord:
    """Map a SQLite row to a RawConversationRecord."""
    return RawConversationRecord(
        raw_id=row["raw_id"],
        provider_name=row["provider_name"],
        payload_provider=(
            Provider.from_string(_row_get(row, "payload_provider"))
            if _row_get(row, "payload_provider") is not None
            else None
        ),
        source_name=row["source_name"],
        source_path=row["source_path"],
        source_index=row["source_index"],
        raw_content=row["raw_content"],
        acquired_at=row["acquired_at"],
        file_mtime=row["file_mtime"],
        parsed_at=_row_get(row, "parsed_at"),
        parse_error=_row_get(row, "parse_error"),
        validated_at=_row_get(row, "validated_at"),
        validation_status=(
            ValidationStatus.from_string(_row_get(row, "validation_status"))
            if _row_get(row, "validation_status") is not None
            else None
        ),
        validation_error=_row_get(row, "validation_error"),
        validation_drift_count=_row_get(row, "validation_drift_count"),
        validation_provider=(
            Provider.from_string(_row_get(row, "validation_provider"))
            if _row_get(row, "validation_provider") is not None
            else None
        ),
        validation_mode=(
            ValidationMode.from_string(_row_get(row, "validation_mode"))
            if _row_get(row, "validation_mode") is not None
            else None
        ),
    )

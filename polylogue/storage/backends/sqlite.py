"""SQLite storage backend implementation."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from polylogue.errors import PolylogueError
from polylogue.lib.json import dumps as json_dumps
from polylogue.lib.log import get_logger
from polylogue.storage.backends.connection import (
    _load_sqlite_vec,
    connection_context,
    create_default_backend,
    default_db_path,
    open_connection,
)

# Re-export from submodules for backward compatibility
from polylogue.storage.backends.schema import (
    _MIGRATIONS,
    _VEC0_DDL,
    SCHEMA_DDL,
    SCHEMA_VERSION,
    _apply_schema,
    _ensure_schema,
    _ensure_vec0_table,
    _run_migrations,
)
from polylogue.storage.store import (
    AttachmentRecord,
    ConversationRecord,
    MessageRecord,
    RawConversationRecord,
    RunRecord,
    _json_or_none,
)
from polylogue.types import ConversationId

LOGGER = get_logger(__name__)


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
    """Map a SQLite row to a ConversationRecord.

    Handles missing optional columns (parent_conversation_id, branch_type,
    raw_id) for compatibility with older schema versions.
    """
    return ConversationRecord(
        conversation_id=row["conversation_id"],
        provider_name=row["provider_name"],
        provider_conversation_id=row["provider_conversation_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        content_hash=row["content_hash"],
        provider_meta=_parse_json(row["provider_meta"], field="provider_meta", record_id=row["conversation_id"]),
        metadata=_parse_json(row["metadata"], field="metadata", record_id=row["conversation_id"]),
        version=row["version"],
        parent_conversation_id=_row_get(row, "parent_conversation_id"),
        branch_type=_row_get(row, "branch_type"),
        raw_id=_row_get(row, "raw_id"),
    )


def _row_to_message(row: sqlite3.Row) -> MessageRecord:
    """Map a SQLite row to a MessageRecord.

    Handles missing optional columns (parent_message_id, branch_index)
    for compatibility with older schema versions.
    """
    return MessageRecord(
        message_id=row["message_id"],
        conversation_id=row["conversation_id"],
        provider_message_id=row["provider_message_id"],
        role=row["role"],
        text=row["text"],
        timestamp=row["timestamp"],
        content_hash=row["content_hash"],
        provider_meta=_parse_json(row["provider_meta"], field="provider_meta", record_id=row["message_id"]),
        version=row["version"],
        parent_message_id=_row_get(row, "parent_message_id"),
        branch_index=_row_get(row, "branch_index", 0) or 0,
    )


def _row_to_raw_conversation(row: sqlite3.Row) -> RawConversationRecord:
    """Map a SQLite row to a RawConversationRecord."""
    return RawConversationRecord(
        raw_id=row["raw_id"],
        provider_name=row["provider_name"],
        source_name=row["source_name"],
        source_path=row["source_path"],
        source_index=row["source_index"],
        raw_content=row["raw_content"],
        acquired_at=row["acquired_at"],
        file_mtime=row["file_mtime"],
    )


def _iso_to_epoch(iso_str: str) -> float:
    """Convert an ISO date string to epoch seconds for SQL comparison."""
    from datetime import datetime, timezone

    try:
        return datetime.fromisoformat(iso_str).timestamp()
    except (ValueError, TypeError):
        # Fallback: try parsing as epoch directly
        try:
            return float(iso_str)
        except (ValueError, TypeError):
            return 0.0


def _build_conversation_filters(
    *,
    source: str | None = None,
    provider: str | None = None,
    providers: list[str] | None = None,
    parent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    title_contains: str | None = None,
) -> tuple[str, list[str | int | float]]:
    """Build WHERE clause and params for conversation queries."""
    where_clauses: list[str] = []
    params: list[str | int | float] = []

    if source is not None:
        where_clauses.append("source_name = ?")
        params.append(source)
    if provider is not None:
        where_clauses.append("provider_name = ?")
        params.append(provider)
    if providers:
        placeholders = ",".join("?" for _ in providers)
        where_clauses.append(
            f"(provider_name IN ({placeholders}) OR source_name IN ({placeholders}))"
        )
        params.extend(providers)
        params.extend(providers)
    if parent_id is not None:
        where_clauses.append("parent_conversation_id = ?")
        params.append(parent_id)
    if since is not None:
        # Handle mixed updated_at formats: some providers store epoch seconds
        # (e.g. "1706000120.0"), others store ISO strings (e.g. "2024-01-23T10:45:00Z").
        # Use GLOB to detect numeric values and compare accordingly.
        where_clauses.append(
            "((updated_at GLOB '[0-9]*' AND CAST(updated_at AS REAL) >= ?)"
            " OR (updated_at NOT GLOB '[0-9]*' AND updated_at >= ?))"
        )
        params.append(_iso_to_epoch(since))
        params.append(since)
    if until is not None:
        where_clauses.append(
            "((updated_at GLOB '[0-9]*' AND CAST(updated_at AS REAL) <= ?)"
            " OR (updated_at NOT GLOB '[0-9]*' AND updated_at <= ?))"
        )
        params.append(_iso_to_epoch(until))
        params.append(until)
    if title_contains is not None:
        escaped = title_contains.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        where_clauses.append("title LIKE ? ESCAPE '\\'")
        params.append(f"%{escaped}%")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    return where_sql, params


class DatabaseError(PolylogueError):
    """Base class for database errors."""



def __getattr__(name: str) -> object:
    """Lazy re-export of SQLiteBackend from async_sqlite to break circular import."""
    if name == "SQLiteBackend":
        from polylogue.storage.backends.async_sqlite import SQLiteBackend

        return SQLiteBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SQLiteBackend",
    "DatabaseError",
    "default_db_path",
    "connection_context",
    "open_connection",
    "create_default_backend",
    "SCHEMA_VERSION",
    "SCHEMA_DDL",
    "_VEC0_DDL",
    "_apply_schema",
    "_MIGRATIONS",
    "_run_migrations",
    "_ensure_schema",
    "_ensure_vec0_table",
    "_load_sqlite_vec",
]

"""Shared helpers for optional embedding-related archive statistics."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import aiosqlite

_EMBEDDED_CONVERSATIONS_SQL = "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 0"
_PENDING_CONVERSATIONS_SQL = "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 1"
_EMBEDDED_MESSAGES_SQL = "SELECT COUNT(*) FROM message_embeddings"
_MISSING_META_MESSAGES_SQL = """
    SELECT COUNT(*)
    FROM message_embeddings me
    LEFT JOIN embeddings_meta em
      ON em.target_id = me.message_id
     AND em.target_type = 'message'
    WHERE em.target_id IS NULL
"""
_STALE_MESSAGES_SQL = """
    SELECT COUNT(*)
    FROM message_embeddings me
    JOIN messages m ON m.message_id = me.message_id
    LEFT JOIN embeddings_meta em
      ON em.target_id = me.message_id
     AND em.target_type = 'message'
    WHERE em.target_id IS NULL
       OR (em.content_hash IS NOT NULL AND em.content_hash != m.content_hash)
"""
_EMBEDDED_AT_BOUNDS_SQL = """
    SELECT MIN(embedded_at) AS oldest_embedded_at, MAX(embedded_at) AS newest_embedded_at
    FROM embeddings_meta
    WHERE target_type = 'message'
"""
_MODEL_COUNTS_SQL = """
    SELECT model, COUNT(*) AS count
    FROM embeddings_meta
    WHERE target_type = 'message'
    GROUP BY model
    ORDER BY count DESC, model ASC
"""
_DIMENSION_COUNTS_SQL = """
    SELECT dimension, COUNT(*) AS count
    FROM embeddings_meta
    WHERE target_type = 'message'
    GROUP BY dimension
    ORDER BY count DESC, dimension ASC
"""


@dataclass(frozen=True)
class EmbeddingStatsSnapshot:
    embedded_conversations: int = 0
    embedded_messages: int = 0
    pending_conversations: int = 0
    stale_messages: int = 0
    messages_missing_provenance: int = 0
    oldest_embedded_at: str | None = None
    newest_embedded_at: str | None = None
    model_counts: dict[str, int] = field(default_factory=dict)
    dimension_counts: dict[int, int] = field(default_factory=dict)


def _is_missing_table_error(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return (
        "no such table" in message
        or "does not exist" in message
        or "table not found" in message
        or "no such module: vec0" in message
    )


def _optional_count_sync(conn: sqlite3.Connection, sql: str) -> int:
    try:
        row = conn.execute(sql).fetchone()
    except sqlite3.OperationalError as exc:
        if _is_missing_table_error(exc):
            return 0
        raise
    return int(row[0]) if row is not None else 0


def _optional_row_sync(conn: sqlite3.Connection, sql: str) -> sqlite3.Row | tuple | None:
    try:
        return conn.execute(sql).fetchone()
    except sqlite3.OperationalError as exc:
        if _is_missing_table_error(exc):
            return None
        raise


def _optional_rows_sync(conn: sqlite3.Connection, sql: str) -> list[sqlite3.Row]:
    try:
        return conn.execute(sql).fetchall()
    except sqlite3.OperationalError as exc:
        if _is_missing_table_error(exc):
            return []
        raise


async def _optional_count_async(conn: aiosqlite.Connection, sql: str) -> int:
    try:
        cursor = await conn.execute(sql)
        row = await cursor.fetchone()
    except sqlite3.OperationalError as exc:
        if _is_missing_table_error(exc):
            return 0
        raise
    return int(row[0]) if row is not None else 0


async def _optional_row_async(conn: aiosqlite.Connection, sql: str) -> sqlite3.Row | tuple | None:
    try:
        cursor = await conn.execute(sql)
        return await cursor.fetchone()
    except sqlite3.OperationalError as exc:
        if _is_missing_table_error(exc):
            return None
        raise


async def _optional_rows_async(conn: aiosqlite.Connection, sql: str) -> list[sqlite3.Row]:
    try:
        cursor = await conn.execute(sql)
        return await cursor.fetchall()
    except sqlite3.OperationalError as exc:
        if _is_missing_table_error(exc):
            return []
        raise


def read_embedding_stats_sync(conn: sqlite3.Connection) -> EmbeddingStatsSnapshot:
    """Read embedding stats from a sync SQLite connection."""
    bounds = _optional_row_sync(conn, _EMBEDDED_AT_BOUNDS_SQL)
    model_rows = _optional_rows_sync(conn, _MODEL_COUNTS_SQL)
    dimension_rows = _optional_rows_sync(conn, _DIMENSION_COUNTS_SQL)
    return EmbeddingStatsSnapshot(
        embedded_conversations=_optional_count_sync(conn, _EMBEDDED_CONVERSATIONS_SQL),
        embedded_messages=_optional_count_sync(conn, _EMBEDDED_MESSAGES_SQL),
        pending_conversations=_optional_count_sync(conn, _PENDING_CONVERSATIONS_SQL),
        stale_messages=_optional_count_sync(conn, _STALE_MESSAGES_SQL),
        messages_missing_provenance=_optional_count_sync(conn, _MISSING_META_MESSAGES_SQL),
        oldest_embedded_at=(bounds["oldest_embedded_at"] if bounds is not None else None),
        newest_embedded_at=(bounds["newest_embedded_at"] if bounds is not None else None),
        model_counts={str(row["model"]): int(row["count"]) for row in model_rows if row["model"]},
        dimension_counts={
            int(row["dimension"]): int(row["count"])
            for row in dimension_rows
            if row["dimension"] is not None
        },
    )


async def read_embedding_stats_async(conn: aiosqlite.Connection) -> EmbeddingStatsSnapshot:
    """Read embedding stats from an async SQLite connection."""
    bounds = await _optional_row_async(conn, _EMBEDDED_AT_BOUNDS_SQL)
    model_rows = await _optional_rows_async(conn, _MODEL_COUNTS_SQL)
    dimension_rows = await _optional_rows_async(conn, _DIMENSION_COUNTS_SQL)
    return EmbeddingStatsSnapshot(
        embedded_conversations=await _optional_count_async(conn, _EMBEDDED_CONVERSATIONS_SQL),
        embedded_messages=await _optional_count_async(conn, _EMBEDDED_MESSAGES_SQL),
        pending_conversations=await _optional_count_async(conn, _PENDING_CONVERSATIONS_SQL),
        stale_messages=await _optional_count_async(conn, _STALE_MESSAGES_SQL),
        messages_missing_provenance=await _optional_count_async(conn, _MISSING_META_MESSAGES_SQL),
        oldest_embedded_at=(bounds["oldest_embedded_at"] if bounds is not None else None),
        newest_embedded_at=(bounds["newest_embedded_at"] if bounds is not None else None),
        model_counts={str(row["model"]): int(row["count"]) for row in model_rows if row["model"]},
        dimension_counts={
            int(row["dimension"]): int(row["count"])
            for row in dimension_rows
            if row["dimension"] is not None
        },
    )


__all__ = [
    "EmbeddingStatsSnapshot",
    "read_embedding_stats_async",
    "read_embedding_stats_sync",
]

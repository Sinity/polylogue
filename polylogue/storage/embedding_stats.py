"""Shared helpers for optional embedding-related archive statistics."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import aiosqlite

_EMBEDDED_CONVERSATIONS_SQL = "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 0"
_PENDING_CONVERSATIONS_SQL = "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 1"
_EMBEDDED_MESSAGES_SQL = "SELECT COUNT(*) FROM message_embeddings"


@dataclass(frozen=True)
class EmbeddingStatsSnapshot:
    embedded_conversations: int = 0
    embedded_messages: int = 0
    pending_conversations: int = 0


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


async def _optional_count_async(conn: aiosqlite.Connection, sql: str) -> int:
    try:
        cursor = await conn.execute(sql)
        row = await cursor.fetchone()
    except sqlite3.OperationalError as exc:
        if _is_missing_table_error(exc):
            return 0
        raise
    return int(row[0]) if row is not None else 0


def read_embedding_stats_sync(conn: sqlite3.Connection) -> EmbeddingStatsSnapshot:
    """Read embedding stats from a sync SQLite connection."""
    return EmbeddingStatsSnapshot(
        embedded_conversations=_optional_count_sync(conn, _EMBEDDED_CONVERSATIONS_SQL),
        embedded_messages=_optional_count_sync(conn, _EMBEDDED_MESSAGES_SQL),
        pending_conversations=_optional_count_sync(conn, _PENDING_CONVERSATIONS_SQL),
    )


async def read_embedding_stats_async(conn: aiosqlite.Connection) -> EmbeddingStatsSnapshot:
    """Read embedding stats from an async SQLite connection."""
    return EmbeddingStatsSnapshot(
        embedded_conversations=await _optional_count_async(conn, _EMBEDDED_CONVERSATIONS_SQL),
        embedded_messages=await _optional_count_async(conn, _EMBEDDED_MESSAGES_SQL),
        pending_conversations=await _optional_count_async(conn, _PENDING_CONVERSATIONS_SQL),
    )


__all__ = [
    "EmbeddingStatsSnapshot",
    "read_embedding_stats_async",
    "read_embedding_stats_sync",
]

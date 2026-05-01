"""Canonical embedding-statistics readers shared across operator surfaces."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace

import aiosqlite

from polylogue.storage.action_events.status import (
    action_event_read_model_status_async,
    action_event_read_model_status_sync,
)
from polylogue.storage.embeddings.models import EmbeddingStatsSnapshot
from polylogue.storage.embeddings.sql import (
    CONVERSATIONS_EXISTS_SQL,
    DIMENSION_COUNTS_SQL,
    EMBEDDED_AT_BOUNDS_SQL,
    EMBEDDED_CONVERSATIONS_SQL,
    EMBEDDED_MESSAGES_SQL,
    MISSING_META_MESSAGES_SQL,
    MODEL_COUNTS_SQL,
    PENDING_CONVERSATIONS_SQL,
    STALE_MESSAGES_SQL,
)
from polylogue.storage.embeddings.support import (
    StatsRow,
    build_retrieval_bands_from_status,
    optional_count_async,
    optional_count_sync,
    optional_row_async,
    optional_row_sync,
    optional_rows_async,
    optional_rows_sync,
)
from polylogue.storage.insights.session.status import (
    session_product_status_async,
    session_product_status_sync,
)


@dataclass(frozen=True, slots=True)
class _EmbeddingStatsParts:
    bounds: StatsRow | None
    model_rows: list[sqlite3.Row]
    dimension_rows: list[sqlite3.Row]
    embedded_conversations: int
    embedded_messages: int
    pending_conversations: int
    stale_messages: int
    missing_provenance: int
    conversations_exist: bool
    total_conversations: int = 0


def _row_count(row: object) -> int:
    if row is None or not isinstance(row, (sqlite3.Row, tuple)):
        return 0
    value = row[0]
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _bounds_value(bounds: StatsRow | None, *, index: int, key: str) -> str | None:
    if bounds is None:
        return None
    if isinstance(bounds, tuple):
        value = bounds[index] if index < len(bounds) else None
    else:
        try:
            value = bounds[key]
        except (IndexError, KeyError):
            return None
    if value is None:
        return None
    return str(value)


def _model_counts(rows: list[sqlite3.Row]) -> dict[str, int]:
    return {str(row["model"]): int(row["count"]) for row in rows if row["model"]}


def _dimension_counts(rows: list[sqlite3.Row]) -> dict[int, int]:
    return {int(row["dimension"]): int(row["count"]) for row in rows if row["dimension"] is not None}


def _base_parts_sync(conn: sqlite3.Connection) -> _EmbeddingStatsParts:
    return _EmbeddingStatsParts(
        bounds=optional_row_sync(conn, EMBEDDED_AT_BOUNDS_SQL),
        model_rows=optional_rows_sync(conn, MODEL_COUNTS_SQL),
        dimension_rows=optional_rows_sync(conn, DIMENSION_COUNTS_SQL),
        embedded_conversations=optional_count_sync(conn, EMBEDDED_CONVERSATIONS_SQL),
        embedded_messages=optional_count_sync(conn, EMBEDDED_MESSAGES_SQL),
        pending_conversations=optional_count_sync(conn, PENDING_CONVERSATIONS_SQL),
        stale_messages=optional_count_sync(conn, STALE_MESSAGES_SQL),
        missing_provenance=optional_count_sync(conn, MISSING_META_MESSAGES_SQL),
        conversations_exist=bool(optional_row_sync(conn, CONVERSATIONS_EXISTS_SQL)),
    )


async def _base_parts_async(conn: aiosqlite.Connection) -> _EmbeddingStatsParts:
    return _EmbeddingStatsParts(
        bounds=await optional_row_async(conn, EMBEDDED_AT_BOUNDS_SQL),
        model_rows=await optional_rows_async(conn, MODEL_COUNTS_SQL),
        dimension_rows=await optional_rows_async(conn, DIMENSION_COUNTS_SQL),
        embedded_conversations=await optional_count_async(conn, EMBEDDED_CONVERSATIONS_SQL),
        embedded_messages=await optional_count_async(conn, EMBEDDED_MESSAGES_SQL),
        pending_conversations=await optional_count_async(conn, PENDING_CONVERSATIONS_SQL),
        stale_messages=await optional_count_async(conn, STALE_MESSAGES_SQL),
        missing_provenance=await optional_count_async(conn, MISSING_META_MESSAGES_SQL),
        conversations_exist=bool(await optional_row_async(conn, CONVERSATIONS_EXISTS_SQL)),
    )


def _with_total_conversations(parts: _EmbeddingStatsParts, total_conversations: int) -> _EmbeddingStatsParts:
    return replace(
        parts,
        total_conversations=total_conversations,
        pending_conversations=max(
            parts.pending_conversations,
            total_conversations - parts.embedded_conversations,
        ),
    )


def _total_conversations_sync(conn: sqlite3.Connection) -> int:
    return _row_count(conn.execute("SELECT COUNT(*) FROM conversations").fetchone())


async def _total_conversations_async(conn: aiosqlite.Connection) -> int:
    return _row_count(await (await conn.execute("SELECT COUNT(*) FROM conversations")).fetchone())


def _snapshot(
    parts: _EmbeddingStatsParts,
    *,
    retrieval_bands: dict[str, dict[str, object]],
) -> EmbeddingStatsSnapshot:
    return EmbeddingStatsSnapshot(
        embedded_conversations=parts.embedded_conversations,
        embedded_messages=parts.embedded_messages,
        pending_conversations=parts.pending_conversations,
        stale_messages=parts.stale_messages,
        messages_missing_provenance=parts.missing_provenance,
        oldest_embedded_at=_bounds_value(parts.bounds, index=0, key="oldest_embedded_at"),
        newest_embedded_at=_bounds_value(parts.bounds, index=1, key="newest_embedded_at"),
        model_counts=_model_counts(parts.model_rows),
        dimension_counts=_dimension_counts(parts.dimension_rows),
        retrieval_bands=retrieval_bands,
    )


def _retrieval_bands_sync(
    conn: sqlite3.Connection,
    parts: _EmbeddingStatsParts,
    *,
    include_retrieval_bands: bool,
) -> dict[str, dict[str, object]]:
    if not parts.conversations_exist or not include_retrieval_bands:
        return {}
    action_status = action_event_read_model_status_sync(conn)
    session_status = session_product_status_sync(conn)
    return build_retrieval_bands_from_status(
        total_conversations=parts.total_conversations,
        embedded_conversations=parts.embedded_conversations,
        embedded_messages=parts.embedded_messages,
        pending_conversations=parts.pending_conversations,
        stale_messages=parts.stale_messages,
        missing_provenance=parts.missing_provenance,
        action_status=action_status,
        session_status=session_status,
    )


async def _retrieval_bands_async(
    conn: aiosqlite.Connection,
    parts: _EmbeddingStatsParts,
    *,
    include_retrieval_bands: bool,
) -> dict[str, dict[str, object]]:
    if not parts.conversations_exist or not include_retrieval_bands:
        return {}
    action_status = await action_event_read_model_status_async(conn)
    session_status = await session_product_status_async(conn)
    return build_retrieval_bands_from_status(
        total_conversations=parts.total_conversations,
        embedded_conversations=parts.embedded_conversations,
        embedded_messages=parts.embedded_messages,
        pending_conversations=parts.pending_conversations,
        stale_messages=parts.stale_messages,
        missing_provenance=parts.missing_provenance,
        action_status=action_status,
        session_status=session_status,
    )


def read_embedding_stats_sync(
    conn: sqlite3.Connection,
    *,
    include_retrieval_bands: bool = True,
) -> EmbeddingStatsSnapshot:
    """Read embedding stats from a sync SQLite connection."""
    parts = _base_parts_sync(conn)
    if parts.conversations_exist:
        parts = _with_total_conversations(parts, _total_conversations_sync(conn))
    return _snapshot(
        parts,
        retrieval_bands=_retrieval_bands_sync(
            conn,
            parts,
            include_retrieval_bands=include_retrieval_bands,
        ),
    )


async def read_embedding_stats_async(
    conn: aiosqlite.Connection,
    *,
    include_retrieval_bands: bool = True,
) -> EmbeddingStatsSnapshot:
    """Read embedding stats from an async SQLite connection."""
    parts = await _base_parts_async(conn)
    if parts.conversations_exist:
        parts = _with_total_conversations(parts, await _total_conversations_async(conn))
    return _snapshot(
        parts,
        retrieval_bands=await _retrieval_bands_async(
            conn,
            parts,
            include_retrieval_bands=include_retrieval_bands,
        ),
    )


__all__ = [
    "action_event_read_model_status_async",
    "action_event_read_model_status_sync",
    "read_embedding_stats_async",
    "read_embedding_stats_sync",
    "session_product_status_async",
    "session_product_status_sync",
]

"""Canonical embedding-statistics readers shared across operator surfaces."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace

import aiosqlite

from polylogue.storage.embeddings.models import EmbeddingStatsSnapshot
from polylogue.storage.embeddings.sql import (
    DIMENSION_COUNTS_SQL,
    EMBEDDED_AT_BOUNDS_SQL,
    EMBEDDED_SESSIONS_SQL,
    EMBEDDING_FAILURE_COUNT_SQL,
    MISSING_META_MESSAGES_SQL,
    MODEL_COUNTS_SQL,
    PENDING_MESSAGES_SQL,
    PENDING_SESSIONS_SQL,
    STALE_MESSAGES_SQL,
    TOTAL_MESSAGES_SQL,
)
from polylogue.storage.embeddings.support import (
    StatsRow,
    build_retrieval_bands_from_status,
    embedded_message_count_async,
    embedded_message_count_sync,
    optional_count_async,
    optional_count_sync,
    optional_row_async,
    optional_row_sync,
    optional_rows_async,
    optional_rows_sync,
    table_exists_async,
    table_exists_sync,
)
from polylogue.storage.insights.session.status import (
    session_insight_status_async,
    session_insight_status_sync,
)
from polylogue.storage.search_providers.sqlite_vec_support import (
    ESTIMATED_TOKENS_PER_MESSAGE,
    VOYAGE_4_COST_PER_1M_TOKENS,
)


@dataclass(frozen=True, slots=True)
class _EmbeddingStatsParts:
    bounds: StatsRow | None
    model_rows: list[sqlite3.Row]
    dimension_rows: list[sqlite3.Row]
    embedded_sessions: int
    embedded_messages: int
    pending_sessions: int
    pending_messages: int
    stale_messages: int
    missing_provenance: int
    sessions_exist: bool
    failure_count: int = 0
    total_message_count: int = 0
    total_sessions: int = 0


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


def _row_value(row: sqlite3.Row | tuple[object, ...], key: str, index: int) -> object:
    if isinstance(row, tuple):
        return row[index] if index < len(row) else None
    return row[key]


def _int_value(value: object) -> int | None:
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
            return None
    return None


def _model_counts(rows: list[sqlite3.Row]) -> dict[str, int]:
    return {
        str(model): count_value
        for row in rows
        if (model := _row_value(row, "model", 0))
        for count in (_row_value(row, "count", 1),)
        for count_value in (_int_value(count),)
        if count_value is not None
    }


def _dimension_counts(rows: list[sqlite3.Row]) -> dict[int, int]:
    return {
        dimension_value: count_value
        for row in rows
        for dimension_value in (_int_value(_row_value(row, "dimension", 0)),)
        if dimension_value is not None
        for count in (_row_value(row, "count", 1),)
        for count_value in (_int_value(count),)
        if count_value is not None
    }


def _base_parts_sync(conn: sqlite3.Connection, *, detail: bool) -> _EmbeddingStatsParts:
    sessions_exist = table_exists_sync(conn, "sessions")
    if not detail:
        return _EmbeddingStatsParts(
            bounds=None,
            model_rows=[],
            dimension_rows=[],
            embedded_sessions=optional_count_sync(conn, EMBEDDED_SESSIONS_SQL),
            embedded_messages=embedded_message_count_sync(conn),
            pending_sessions=optional_count_sync(conn, PENDING_SESSIONS_SQL),
            pending_messages=0,
            stale_messages=0,
            missing_provenance=0,
            sessions_exist=sessions_exist,
            failure_count=optional_count_sync(conn, EMBEDDING_FAILURE_COUNT_SQL),
            total_message_count=0,
        )
    return _EmbeddingStatsParts(
        bounds=optional_row_sync(conn, EMBEDDED_AT_BOUNDS_SQL),
        model_rows=optional_rows_sync(conn, MODEL_COUNTS_SQL),
        dimension_rows=optional_rows_sync(conn, DIMENSION_COUNTS_SQL),
        embedded_sessions=optional_count_sync(conn, EMBEDDED_SESSIONS_SQL),
        embedded_messages=embedded_message_count_sync(conn),
        pending_sessions=optional_count_sync(conn, PENDING_SESSIONS_SQL),
        pending_messages=optional_count_sync(conn, PENDING_MESSAGES_SQL),
        stale_messages=optional_count_sync(conn, STALE_MESSAGES_SQL),
        missing_provenance=optional_count_sync(conn, MISSING_META_MESSAGES_SQL),
        sessions_exist=sessions_exist,
        failure_count=optional_count_sync(conn, EMBEDDING_FAILURE_COUNT_SQL),
        total_message_count=optional_count_sync(conn, TOTAL_MESSAGES_SQL),
    )


async def _base_parts_async(conn: aiosqlite.Connection, *, detail: bool) -> _EmbeddingStatsParts:
    sessions_exist = await table_exists_async(conn, "sessions")
    if not detail:
        return _EmbeddingStatsParts(
            bounds=None,
            model_rows=[],
            dimension_rows=[],
            embedded_sessions=await optional_count_async(conn, EMBEDDED_SESSIONS_SQL),
            embedded_messages=await embedded_message_count_async(conn),
            pending_sessions=await optional_count_async(conn, PENDING_SESSIONS_SQL),
            pending_messages=0,
            stale_messages=0,
            missing_provenance=0,
            sessions_exist=sessions_exist,
            failure_count=await optional_count_async(conn, EMBEDDING_FAILURE_COUNT_SQL),
            total_message_count=0,
        )
    return _EmbeddingStatsParts(
        bounds=await optional_row_async(conn, EMBEDDED_AT_BOUNDS_SQL),
        model_rows=await optional_rows_async(conn, MODEL_COUNTS_SQL),
        dimension_rows=await optional_rows_async(conn, DIMENSION_COUNTS_SQL),
        embedded_sessions=await optional_count_async(conn, EMBEDDED_SESSIONS_SQL),
        embedded_messages=await embedded_message_count_async(conn),
        pending_sessions=await optional_count_async(conn, PENDING_SESSIONS_SQL),
        pending_messages=await optional_count_async(conn, PENDING_MESSAGES_SQL),
        stale_messages=await optional_count_async(conn, STALE_MESSAGES_SQL),
        missing_provenance=await optional_count_async(conn, MISSING_META_MESSAGES_SQL),
        sessions_exist=sessions_exist,
        failure_count=await optional_count_async(conn, EMBEDDING_FAILURE_COUNT_SQL),
        total_message_count=await optional_count_async(conn, TOTAL_MESSAGES_SQL),
    )


def _with_total_sessions(parts: _EmbeddingStatsParts, total_sessions: int) -> _EmbeddingStatsParts:
    pending_sessions = max(
        parts.pending_sessions,
        total_sessions - parts.embedded_sessions,
    )
    pending_messages = parts.pending_messages
    if parts.total_message_count > 0:
        pending_messages = max(
            parts.pending_messages,
            parts.total_message_count - parts.embedded_messages if pending_sessions > 0 else 0,
        )
    return replace(
        parts,
        total_sessions=total_sessions,
        pending_sessions=pending_sessions,
        pending_messages=pending_messages,
    )


def _total_sessions_sync(conn: sqlite3.Connection) -> int:
    return _row_count(conn.execute("SELECT COUNT(*) FROM sessions").fetchone())


async def _total_sessions_async(conn: aiosqlite.Connection) -> int:
    return _row_count(await (await conn.execute("SELECT COUNT(*) FROM sessions")).fetchone())


def _estimated_cost(message_count: int) -> float:
    """Estimate Voyage API cost from message counts.

    Uses rough per-message token estimate (500 tokens/msg average)
    and voyage-4 pricing ($0.10 / 1M tokens).
    """
    estimated_tokens = message_count * ESTIMATED_TOKENS_PER_MESSAGE
    return estimated_tokens * VOYAGE_4_COST_PER_1M_TOKENS / 1_000_000


def _snapshot(
    parts: _EmbeddingStatsParts,
    *,
    retrieval_bands: dict[str, dict[str, object]],
) -> EmbeddingStatsSnapshot:
    return EmbeddingStatsSnapshot(
        embedded_sessions=parts.embedded_sessions,
        embedded_messages=parts.embedded_messages,
        pending_sessions=parts.pending_sessions,
        pending_messages=parts.pending_messages,
        stale_messages=parts.stale_messages,
        messages_missing_provenance=parts.missing_provenance,
        oldest_embedded_at=_bounds_value(parts.bounds, index=0, key="oldest_embedded_at"),
        newest_embedded_at=_bounds_value(parts.bounds, index=1, key="newest_embedded_at"),
        model_counts=_model_counts(parts.model_rows),
        dimension_counts=_dimension_counts(parts.dimension_rows),
        retrieval_bands=retrieval_bands,
        failure_count=parts.failure_count,
        total_estimated_cost_usd=round(
            _estimated_cost(parts.total_message_count),
            2,
        ),
    )


def _retrieval_bands_sync(
    conn: sqlite3.Connection,
    parts: _EmbeddingStatsParts,
    *,
    include_retrieval_bands: bool,
) -> dict[str, dict[str, object]]:
    if not parts.sessions_exist or not include_retrieval_bands:
        return {}
    session_status = session_insight_status_sync(conn)
    return build_retrieval_bands_from_status(
        total_sessions=parts.total_sessions,
        embedded_sessions=parts.embedded_sessions,
        embedded_messages=parts.embedded_messages,
        pending_sessions=parts.pending_sessions,
        stale_messages=parts.stale_messages,
        missing_provenance=parts.missing_provenance,
        session_status=session_status,
    )


async def _retrieval_bands_async(
    conn: aiosqlite.Connection,
    parts: _EmbeddingStatsParts,
    *,
    include_retrieval_bands: bool,
) -> dict[str, dict[str, object]]:
    if not parts.sessions_exist or not include_retrieval_bands:
        return {}
    session_status = await session_insight_status_async(conn)
    return build_retrieval_bands_from_status(
        total_sessions=parts.total_sessions,
        embedded_sessions=parts.embedded_sessions,
        embedded_messages=parts.embedded_messages,
        pending_sessions=parts.pending_sessions,
        stale_messages=parts.stale_messages,
        missing_provenance=parts.missing_provenance,
        session_status=session_status,
    )


def read_embedding_stats_sync(
    conn: sqlite3.Connection,
    *,
    include_retrieval_bands: bool = True,
    detail: bool = True,
) -> EmbeddingStatsSnapshot:
    """Read embedding stats from a sync SQLite connection."""
    parts = _base_parts_sync(conn, detail=detail)
    if parts.sessions_exist:
        parts = _with_total_sessions(parts, _total_sessions_sync(conn))
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
    detail: bool = True,
) -> EmbeddingStatsSnapshot:
    """Read embedding stats from an async SQLite connection."""
    parts = await _base_parts_async(conn, detail=detail)
    if parts.sessions_exist:
        parts = _with_total_sessions(parts, await _total_sessions_async(conn))
    return _snapshot(
        parts,
        retrieval_bands=await _retrieval_bands_async(
            conn,
            parts,
            include_retrieval_bands=include_retrieval_bands,
        ),
    )


__all__ = [
    "read_embedding_stats_async",
    "read_embedding_stats_sync",
    "session_insight_status_async",
    "session_insight_status_sync",
]

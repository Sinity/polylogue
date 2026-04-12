"""Canonical embedding-statistics readers shared across operator surfaces."""

from __future__ import annotations

import sqlite3

import aiosqlite

from polylogue.storage.action_event_status import (
    action_event_read_model_status_async,
    action_event_read_model_status_sync,
)
from polylogue.storage.embedding_stats_models import EmbeddingStatsSnapshot
from polylogue.storage.embedding_stats_sql import (
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
from polylogue.storage.embedding_stats_support import (
    build_retrieval_bands_from_status,
    optional_count_async,
    optional_count_sync,
    optional_row_async,
    optional_row_sync,
    optional_rows_async,
    optional_rows_sync,
)
from polylogue.storage.session_product_status import (
    session_product_status_async,
    session_product_status_sync,
)


def read_embedding_stats_sync(
    conn: sqlite3.Connection,
    *,
    include_retrieval_bands: bool = True,
) -> EmbeddingStatsSnapshot:
    """Read embedding stats from a sync SQLite connection."""
    bounds = optional_row_sync(conn, EMBEDDED_AT_BOUNDS_SQL)
    model_rows = optional_rows_sync(conn, MODEL_COUNTS_SQL)
    dimension_rows = optional_rows_sync(conn, DIMENSION_COUNTS_SQL)
    embedded_conversations = optional_count_sync(conn, EMBEDDED_CONVERSATIONS_SQL)
    embedded_messages = optional_count_sync(conn, EMBEDDED_MESSAGES_SQL)
    pending_conversations = optional_count_sync(conn, PENDING_CONVERSATIONS_SQL)
    stale_messages = optional_count_sync(conn, STALE_MESSAGES_SQL)
    missing_provenance = optional_count_sync(conn, MISSING_META_MESSAGES_SQL)
    conversations_exist = bool(optional_row_sync(conn, CONVERSATIONS_EXISTS_SQL))

    total_conversations = 0
    retrieval_bands: dict[str, dict[str, object]] = {}
    if conversations_exist:
        total_conversations_row = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
        total_conversations = int(total_conversations_row[0]) if total_conversations_row is not None else 0
        pending_conversations = max(pending_conversations, total_conversations - embedded_conversations)
        if include_retrieval_bands:
            action_status = action_event_read_model_status_sync(conn)
            session_status = session_product_status_sync(conn)
            retrieval_bands = build_retrieval_bands_from_status(
                total_conversations=total_conversations,
                embedded_conversations=embedded_conversations,
                embedded_messages=embedded_messages,
                pending_conversations=pending_conversations,
                stale_messages=stale_messages,
                missing_provenance=missing_provenance,
                action_status=action_status,
                session_status=session_status,
            )

    return EmbeddingStatsSnapshot(
        embedded_conversations=embedded_conversations,
        embedded_messages=embedded_messages,
        pending_conversations=pending_conversations,
        stale_messages=stale_messages,
        messages_missing_provenance=missing_provenance,
        oldest_embedded_at=(bounds["oldest_embedded_at"] if bounds is not None else None),
        newest_embedded_at=(bounds["newest_embedded_at"] if bounds is not None else None),
        model_counts={str(row["model"]): int(row["count"]) for row in model_rows if row["model"]},
        dimension_counts={
            int(row["dimension"]): int(row["count"]) for row in dimension_rows if row["dimension"] is not None
        },
        retrieval_bands=retrieval_bands,
    )


async def read_embedding_stats_async(
    conn: aiosqlite.Connection,
    *,
    include_retrieval_bands: bool = True,
) -> EmbeddingStatsSnapshot:
    """Read embedding stats from an async SQLite connection."""
    bounds = await optional_row_async(conn, EMBEDDED_AT_BOUNDS_SQL)
    model_rows = await optional_rows_async(conn, MODEL_COUNTS_SQL)
    dimension_rows = await optional_rows_async(conn, DIMENSION_COUNTS_SQL)
    embedded_conversations = await optional_count_async(conn, EMBEDDED_CONVERSATIONS_SQL)
    embedded_messages = await optional_count_async(conn, EMBEDDED_MESSAGES_SQL)
    pending_conversations = await optional_count_async(conn, PENDING_CONVERSATIONS_SQL)
    stale_messages = await optional_count_async(conn, STALE_MESSAGES_SQL)
    missing_provenance = await optional_count_async(conn, MISSING_META_MESSAGES_SQL)
    conversations_exist = bool(await optional_row_async(conn, CONVERSATIONS_EXISTS_SQL))

    total_conversations = 0
    retrieval_bands: dict[str, dict[str, object]] = {}
    if conversations_exist:
        total_conversations_row = await (await conn.execute("SELECT COUNT(*) FROM conversations")).fetchone()
        total_conversations = int(total_conversations_row[0]) if total_conversations_row is not None else 0
        pending_conversations = max(pending_conversations, total_conversations - embedded_conversations)
        if include_retrieval_bands:
            action_status = await action_event_read_model_status_async(conn)
            session_status = await session_product_status_async(conn)
            retrieval_bands = build_retrieval_bands_from_status(
                total_conversations=total_conversations,
                embedded_conversations=embedded_conversations,
                embedded_messages=embedded_messages,
                pending_conversations=pending_conversations,
                stale_messages=stale_messages,
                missing_provenance=missing_provenance,
                action_status=action_status,
                session_status=session_status,
            )

    return EmbeddingStatsSnapshot(
        embedded_conversations=embedded_conversations,
        embedded_messages=embedded_messages,
        pending_conversations=pending_conversations,
        stale_messages=stale_messages,
        messages_missing_provenance=missing_provenance,
        oldest_embedded_at=(bounds["oldest_embedded_at"] if bounds is not None else None),
        newest_embedded_at=(bounds["newest_embedded_at"] if bounds is not None else None),
        model_counts={str(row["model"]): int(row["count"]) for row in model_rows if row["model"]},
        dimension_counts={
            int(row["dimension"]): int(row["count"]) for row in dimension_rows if row["dimension"] is not None
        },
        retrieval_bands=retrieval_bands,
    )


__all__ = [
    "action_event_read_model_status_async",
    "action_event_read_model_status_sync",
    "read_embedding_stats_async",
    "read_embedding_stats_sync",
    "session_product_status_async",
    "session_product_status_sync",
]

"""Shared helpers for optional embedding-related archive statistics."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import aiosqlite

from polylogue.storage.action_event_lifecycle import (
    action_event_read_model_status_async,
    action_event_read_model_status_sync,
)
from polylogue.storage.session_product_lifecycle import (
    session_product_status_async,
    session_product_status_sync,
)

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
_CONVERSATIONS_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"


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
    retrieval_bands: dict[str, dict[str, object]] = field(default_factory=dict)


def _build_retrieval_bands_from_status(
    *,
    total_conversations: int,
    embedded_conversations: int,
    embedded_messages: int,
    pending_conversations: int,
    stale_messages: int,
    missing_provenance: int,
    action_status: dict[str, object],
    session_status: dict[str, int | bool],
) -> dict[str, dict[str, object]]:
    transcript_ready = (
        total_conversations == 0
        or (
            embedded_conversations == total_conversations
            and pending_conversations == 0
            and stale_messages == 0
            and missing_provenance == 0
        )
    )
    transcript_status = "empty" if total_conversations == 0 else ("ready" if transcript_ready else "pending")

    evidence_source_rows = (
        int(action_status["count"])
        + int(session_status["profile_row_count"])
    )
    evidence_materialized_rows = (
        int(action_status["action_fts_count"])
        + int(session_status["profile_evidence_fts_count"])
    )
    evidence_ready = bool(action_status["action_fts_ready"]) and bool(session_status["profile_evidence_fts_ready"])

    inference_source_rows = (
        int(session_status["profile_row_count"])
        + int(session_status["work_event_inference_count"])
        + int(session_status["phase_inference_count"])
    )
    inference_materialized_rows = (
        int(session_status["profile_inference_fts_count"])
        + int(session_status["work_event_inference_fts_count"])
        + int(session_status["phase_inference_count"])
    )
    inference_ready = (
        bool(session_status["profile_inference_fts_ready"])
        and bool(session_status["work_event_inference_fts_ready"])
        and bool(session_status["phase_inference_rows_ready"])
    )
    enrichment_source_rows = int(session_status["profile_row_count"])
    enrichment_materialized_rows = int(session_status["profile_enrichment_fts_count"])
    enrichment_ready = bool(session_status["profile_enrichment_fts_ready"])

    return {
        "transcript_embeddings": {
            "status": transcript_status,
            "ready": transcript_ready,
            "source_documents": total_conversations,
            "materialized_documents": embedded_conversations,
            "materialized_rows": embedded_messages,
            "pending_documents": pending_conversations,
            "stale_rows": stale_messages,
            "missing_provenance_rows": missing_provenance,
            "detail": (
                f"Transcript embeddings ready ({embedded_conversations:,}/{total_conversations:,} conversations, {embedded_messages:,} messages)"
                if transcript_ready
                else (
                    f"Transcript embeddings pending ({embedded_conversations:,}/{total_conversations:,} conversations, "
                    f"pending {pending_conversations:,}, stale {stale_messages:,}, missing provenance {missing_provenance:,})"
                )
            ),
        },
        "evidence_retrieval": {
            "status": "ready" if evidence_ready else "pending",
            "ready": evidence_ready,
            "source_rows": evidence_source_rows,
            "materialized_rows": evidence_materialized_rows,
            "pending_rows": max(0, evidence_source_rows - evidence_materialized_rows),
            "stale_rows": int(session_status["profile_evidence_fts_duplicate_count"]) + int(action_status["stale_count"]),
            "detail": (
                f"Evidence retrieval ready ({evidence_materialized_rows:,}/{evidence_source_rows:,} supporting rows)"
                if evidence_ready
                else (
                    f"Evidence retrieval pending ({evidence_materialized_rows:,}/{evidence_source_rows:,} supporting rows; "
                    f"profile_evidence_fts={int(session_status['profile_evidence_fts_count']):,}/{int(session_status['profile_row_count']):,}, "
                    f"action_event_fts={int(action_status['action_fts_count']):,}/{int(action_status['count']):,})"
                )
            ),
        },
        "inference_retrieval": {
            "status": "ready" if inference_ready else "pending",
            "ready": inference_ready,
            "source_rows": inference_source_rows,
            "materialized_rows": inference_materialized_rows,
            "pending_rows": max(0, inference_source_rows - inference_materialized_rows),
            "stale_rows": (
                int(session_status["profile_inference_fts_duplicate_count"])
                + int(session_status["work_event_inference_fts_duplicate_count"])
                + int(session_status["stale_work_event_inference_count"])
                + int(session_status["stale_phase_inference_count"])
            ),
            "detail": (
                f"Inference retrieval ready ({inference_materialized_rows:,}/{inference_source_rows:,} supporting rows)"
                if inference_ready
                else (
                    f"Inference retrieval pending ({inference_materialized_rows:,}/{inference_source_rows:,} supporting rows; "
                    f"profile_inference_fts={int(session_status['profile_inference_fts_count']):,}/{int(session_status['profile_row_count']):,}, "
                    f"work_event_inference_fts={int(session_status['work_event_inference_fts_count']):,}/{int(session_status['work_event_inference_count']):,}, "
                    f"phase_inference={int(session_status['phase_inference_count']):,}/{int(session_status['expected_phase_inference_count']):,})"
                )
            ),
        },
        "enrichment_retrieval": {
            "status": "ready" if enrichment_ready else "pending",
            "ready": enrichment_ready,
            "source_rows": enrichment_source_rows,
            "materialized_rows": enrichment_materialized_rows,
            "pending_rows": max(0, enrichment_source_rows - enrichment_materialized_rows),
            "stale_rows": int(session_status["profile_enrichment_fts_duplicate_count"]),
            "detail": (
                f"Enrichment retrieval ready ({enrichment_materialized_rows:,}/{enrichment_source_rows:,} supporting rows)"
                if enrichment_ready
                else (
                    f"Enrichment retrieval pending ({enrichment_materialized_rows:,}/{enrichment_source_rows:,} supporting rows; "
                    f"profile_enrichment_fts={int(session_status['profile_enrichment_fts_count']):,}/{int(session_status['profile_row_count']):,})"
                )
            ),
        },
    }


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
    embedded_conversations = _optional_count_sync(conn, _EMBEDDED_CONVERSATIONS_SQL)
    embedded_messages = _optional_count_sync(conn, _EMBEDDED_MESSAGES_SQL)
    pending_conversations = _optional_count_sync(conn, _PENDING_CONVERSATIONS_SQL)
    stale_messages = _optional_count_sync(conn, _STALE_MESSAGES_SQL)
    missing_provenance = _optional_count_sync(conn, _MISSING_META_MESSAGES_SQL)
    conversations_exist = bool(_optional_row_sync(conn, _CONVERSATIONS_EXISTS_SQL))
    total_conversations = 0
    retrieval_bands: dict[str, dict[str, object]] = {}
    if conversations_exist:
        total_conversations_row = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
        total_conversations = int(total_conversations_row[0]) if total_conversations_row is not None else 0
        action_status = action_event_read_model_status_sync(conn)
        session_status = session_product_status_sync(conn)
        retrieval_bands = _build_retrieval_bands_from_status(
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
            int(row["dimension"]): int(row["count"])
            for row in dimension_rows
            if row["dimension"] is not None
        },
        retrieval_bands=retrieval_bands,
    )


async def read_embedding_stats_async(conn: aiosqlite.Connection) -> EmbeddingStatsSnapshot:
    """Read embedding stats from an async SQLite connection."""
    bounds = await _optional_row_async(conn, _EMBEDDED_AT_BOUNDS_SQL)
    model_rows = await _optional_rows_async(conn, _MODEL_COUNTS_SQL)
    dimension_rows = await _optional_rows_async(conn, _DIMENSION_COUNTS_SQL)
    embedded_conversations = await _optional_count_async(conn, _EMBEDDED_CONVERSATIONS_SQL)
    embedded_messages = await _optional_count_async(conn, _EMBEDDED_MESSAGES_SQL)
    pending_conversations = await _optional_count_async(conn, _PENDING_CONVERSATIONS_SQL)
    stale_messages = await _optional_count_async(conn, _STALE_MESSAGES_SQL)
    missing_provenance = await _optional_count_async(conn, _MISSING_META_MESSAGES_SQL)
    conversations_exist = bool(await _optional_row_async(conn, _CONVERSATIONS_EXISTS_SQL))
    total_conversations = 0
    retrieval_bands: dict[str, dict[str, object]] = {}
    if conversations_exist:
        total_conversations_row = await (await conn.execute("SELECT COUNT(*) FROM conversations")).fetchone()
        total_conversations = int(total_conversations_row[0]) if total_conversations_row is not None else 0
        action_status = await action_event_read_model_status_async(conn)
        session_status = await session_product_status_async(conn)
        retrieval_bands = _build_retrieval_bands_from_status(
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
            int(row["dimension"]): int(row["count"])
            for row in dimension_rows
            if row["dimension"] is not None
        },
        retrieval_bands=retrieval_bands,
    )


__all__ = [
    "EmbeddingStatsSnapshot",
    "read_embedding_stats_async",
    "read_embedding_stats_sync",
]

"""Shared helpers for optional embedding-related archive statistics."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

import aiosqlite

from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot

StatsRow = sqlite3.Row | tuple[object, ...]


def _stats_row(row: object) -> StatsRow | None:
    if row is None:
        return None
    if isinstance(row, (sqlite3.Row, tuple)):
        return row
    return None


def _sqlite_rows(rows: Iterable[object]) -> list[sqlite3.Row]:
    return [row for row in rows if isinstance(row, sqlite3.Row)]


def _coerce_int(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
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
            return default
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def build_retrieval_bands_from_status(
    *,
    total_sessions: int,
    embedded_sessions: int,
    embedded_messages: int,
    pending_sessions: int,
    stale_messages: int,
    missing_provenance: int,
    session_status: SessionInsightStatusSnapshot,
) -> dict[str, dict[str, object]]:
    transcript_ready = total_sessions == 0 or (
        embedded_sessions == total_sessions
        and pending_sessions == 0
        and stale_messages == 0
        and missing_provenance == 0
    )
    transcript_status = "empty" if total_sessions == 0 else ("ready" if transcript_ready else "pending")

    evidence_source_rows = session_status.profile_row_count
    evidence_materialized_rows = session_status.profile_row_count
    evidence_ready = True

    inference_source_rows = session_status.profile_row_count + session_status.work_event_inference_count
    inference_materialized_rows = session_status.profile_row_count + session_status.work_event_inference_fts_count
    inference_ready = session_status.work_event_inference_fts_ready
    enrichment_source_rows = session_status.profile_row_count
    enrichment_materialized_rows = session_status.profile_row_count
    enrichment_ready = True

    return {
        "transcript_embeddings": {
            "status": transcript_status,
            "ready": transcript_ready,
            "source_documents": total_sessions,
            "materialized_documents": embedded_sessions,
            "materialized_rows": embedded_messages,
            "pending_documents": pending_sessions,
            "stale_rows": stale_messages,
            "missing_provenance_rows": missing_provenance,
            "detail": (
                f"Transcript embeddings ready ({embedded_sessions:,}/{total_sessions:,} sessions, {embedded_messages:,} messages)"
                if transcript_ready
                else (
                    f"Transcript embeddings pending ({embedded_sessions:,}/{total_sessions:,} sessions, "
                    f"pending {pending_sessions:,}, stale {stale_messages:,}, missing provenance {missing_provenance:,})"
                )
            ),
        },
        "evidence_retrieval": {
            "status": "ready" if evidence_ready else "pending",
            "ready": evidence_ready,
            "source_rows": evidence_source_rows,
            "materialized_rows": evidence_materialized_rows,
            "pending_rows": max(0, evidence_source_rows - evidence_materialized_rows),
            "stale_rows": session_status.profile_evidence_fts_duplicate_count,
            "detail": (
                f"Evidence retrieval ready ({evidence_materialized_rows:,}/{evidence_source_rows:,} supporting rows)"
                if evidence_ready
                else (
                    f"Evidence retrieval pending ({evidence_materialized_rows:,}/{evidence_source_rows:,} supporting rows; "
                    "profile evidence FTS pending)"
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
                session_status.profile_inference_fts_duplicate_count
                + session_status.work_event_inference_fts_duplicate_count
                + session_status.stale_work_event_inference_count
            ),
            "detail": (
                f"Inference retrieval ready ({inference_materialized_rows:,}/{inference_source_rows:,} supporting rows)"
                if inference_ready
                else (
                    f"Inference retrieval pending ({inference_materialized_rows:,}/{inference_source_rows:,} supporting rows; "
                    f"work_event_inference_fts={session_status.work_event_inference_fts_count:,}/{session_status.work_event_inference_count:,})"
                )
            ),
        },
        "enrichment_retrieval": {
            "status": "ready" if enrichment_ready else "pending",
            "ready": enrichment_ready,
            "source_rows": enrichment_source_rows,
            "materialized_rows": enrichment_materialized_rows,
            "pending_rows": max(0, enrichment_source_rows - enrichment_materialized_rows),
            "stale_rows": session_status.profile_enrichment_fts_duplicate_count,
            "detail": (
                f"Enrichment retrieval ready ({enrichment_materialized_rows:,}/{enrichment_source_rows:,} supporting rows)"
                if enrichment_ready
                else (
                    f"Enrichment retrieval pending ({enrichment_materialized_rows:,}/{enrichment_source_rows:,} supporting rows)"
                )
            ),
        },
    }


def is_missing_table_error(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return (
        "no such table" in message
        or "no such column" in message
        or "does not exist" in message
        or "table not found" in message
        or "no such module: vec0" in message
    )


def table_exists_sync(conn: sqlite3.Connection, table: str) -> bool:
    table_name = table.replace("'", "''")
    try:
        row = conn.execute(
            f"SELECT 1 FROM sqlite_master WHERE type='table' AND name='{table_name}' LIMIT 1",
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if is_missing_table_error(exc):
            return False
        raise
    return row is not None


async def table_exists_async(conn: aiosqlite.Connection, table: str) -> bool:
    table_name = table.replace("'", "''")
    try:
        cursor = await conn.execute(
            f"SELECT 1 FROM sqlite_master WHERE type='table' AND name='{table_name}' LIMIT 1",
        )
    except sqlite3.OperationalError as exc:
        if is_missing_table_error(exc):
            return False
        raise
    return await cursor.fetchone() is not None


def optional_count_sync(conn: sqlite3.Connection, sql: str) -> int:
    try:
        row = conn.execute(sql).fetchone()
    except sqlite3.OperationalError as exc:
        if is_missing_table_error(exc):
            return 0
        raise
    return int(row[0]) if row is not None else 0


def embedded_message_count_sync(conn: sqlite3.Connection) -> int:
    """Count embedded vectors without scanning the sqlite-vec virtual table."""
    if table_exists_sync(conn, "message_embeddings_rowids"):
        return optional_count_sync(conn, "SELECT COUNT(*) FROM message_embeddings_rowids")
    return optional_count_sync(conn, "SELECT COUNT(*) FROM message_embeddings")


def optional_row_sync(conn: sqlite3.Connection, sql: str) -> StatsRow | None:
    try:
        return _stats_row(conn.execute(sql).fetchone())
    except sqlite3.OperationalError as exc:
        if is_missing_table_error(exc):
            return None
        raise


def optional_rows_sync(conn: sqlite3.Connection, sql: str) -> list[sqlite3.Row]:
    try:
        return conn.execute(sql).fetchall()
    except sqlite3.OperationalError as exc:
        if is_missing_table_error(exc):
            return []
        raise


async def optional_count_async(conn: aiosqlite.Connection, sql: str) -> int:
    try:
        cursor = await conn.execute(sql)
        row = await cursor.fetchone()
    except sqlite3.OperationalError as exc:
        if is_missing_table_error(exc):
            return 0
        raise
    return _coerce_int(row[0]) if row is not None else 0


async def embedded_message_count_async(conn: aiosqlite.Connection) -> int:
    """Count embedded vectors without scanning the sqlite-vec virtual table."""
    if await table_exists_async(conn, "message_embeddings_rowids"):
        return await optional_count_async(conn, "SELECT COUNT(*) FROM message_embeddings_rowids")
    return await optional_count_async(conn, "SELECT COUNT(*) FROM message_embeddings")


async def optional_row_async(conn: aiosqlite.Connection, sql: str) -> StatsRow | None:
    try:
        cursor = await conn.execute(sql)
        return _stats_row(await cursor.fetchone())
    except sqlite3.OperationalError as exc:
        if is_missing_table_error(exc):
            return None
        raise


async def optional_rows_async(conn: aiosqlite.Connection, sql: str) -> list[sqlite3.Row]:
    try:
        cursor = await conn.execute(sql)
        return _sqlite_rows(await cursor.fetchall())
    except sqlite3.OperationalError as exc:
        if is_missing_table_error(exc):
            return []
        raise

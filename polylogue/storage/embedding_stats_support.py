"""Shared helpers for optional embedding-related archive statistics."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import cast

import aiosqlite

from polylogue.storage.session_product_runtime import SessionProductStatusSnapshot

StatsRow = sqlite3.Row | tuple[object, ...]


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
    total_conversations: int,
    embedded_conversations: int,
    embedded_messages: int,
    pending_conversations: int,
    stale_messages: int,
    missing_provenance: int,
    action_status: Mapping[str, object],
    session_status: SessionProductStatusSnapshot,
) -> dict[str, dict[str, object]]:
    transcript_ready = total_conversations == 0 or (
        embedded_conversations == total_conversations
        and pending_conversations == 0
        and stale_messages == 0
        and missing_provenance == 0
    )
    transcript_status = "empty" if total_conversations == 0 else ("ready" if transcript_ready else "pending")

    evidence_source_rows = _coerce_int(action_status["count"]) + session_status.profile_row_count
    evidence_materialized_rows = (
        _coerce_int(action_status["action_fts_count"]) + session_status.profile_evidence_fts_count
    )
    evidence_ready = bool(action_status["action_fts_ready"]) and session_status.profile_evidence_fts_ready

    inference_source_rows = (
        session_status.profile_row_count
        + session_status.work_event_inference_count
        + session_status.phase_inference_count
    )
    inference_materialized_rows = (
        session_status.profile_inference_fts_count
        + session_status.work_event_inference_fts_count
        + session_status.phase_inference_count
    )
    inference_ready = (
        session_status.profile_inference_fts_ready
        and session_status.work_event_inference_fts_ready
        and session_status.phase_inference_rows_ready
    )
    enrichment_source_rows = session_status.profile_row_count
    enrichment_materialized_rows = session_status.profile_enrichment_fts_count
    enrichment_ready = session_status.profile_enrichment_fts_ready

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
            "stale_rows": session_status.profile_evidence_fts_duplicate_count
            + _coerce_int(action_status["stale_count"])
            + _coerce_int(action_status.get("action_fts_stale_rows", 0)),
            "detail": (
                f"Evidence retrieval ready ({evidence_materialized_rows:,}/{evidence_source_rows:,} supporting rows)"
                if evidence_ready
                else (
                    f"Evidence retrieval pending ({evidence_materialized_rows:,}/{evidence_source_rows:,} supporting rows; "
                    f"profile_evidence_fts={session_status.profile_evidence_fts_count:,}/{session_status.profile_row_count:,}, "
                    f"action_event_fts={_coerce_int(action_status['action_fts_count']):,}/{_coerce_int(action_status['count']):,})"
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
                + session_status.stale_phase_inference_count
            ),
            "detail": (
                f"Inference retrieval ready ({inference_materialized_rows:,}/{inference_source_rows:,} supporting rows)"
                if inference_ready
                else (
                    f"Inference retrieval pending ({inference_materialized_rows:,}/{inference_source_rows:,} supporting rows; "
                    f"profile_inference_fts={session_status.profile_inference_fts_count:,}/{session_status.profile_row_count:,}, "
                    f"work_event_inference_fts={session_status.work_event_inference_fts_count:,}/{session_status.work_event_inference_count:,}, "
                    f"phase_inference={session_status.phase_inference_count:,}/{session_status.expected_phase_inference_count:,})"
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
                    f"Enrichment retrieval pending ({enrichment_materialized_rows:,}/{enrichment_source_rows:,} supporting rows; "
                    f"profile_enrichment_fts={session_status.profile_enrichment_fts_count:,}/{session_status.profile_row_count:,})"
                )
            ),
        },
    }


def is_missing_table_error(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return (
        "no such table" in message
        or "does not exist" in message
        or "table not found" in message
        or "no such module: vec0" in message
    )


def optional_count_sync(conn: sqlite3.Connection, sql: str) -> int:
    try:
        row = conn.execute(sql).fetchone()
    except sqlite3.OperationalError as exc:
        if is_missing_table_error(exc):
            return 0
        raise
    return int(row[0]) if row is not None else 0


def optional_row_sync(conn: sqlite3.Connection, sql: str) -> StatsRow | None:
    try:
        return cast(StatsRow | None, conn.execute(sql).fetchone())
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


async def optional_row_async(conn: aiosqlite.Connection, sql: str) -> StatsRow | None:
    try:
        cursor = await conn.execute(sql)
        return cast(StatsRow | None, await cursor.fetchone())
    except sqlite3.OperationalError as exc:
        if is_missing_table_error(exc):
            return None
        raise


async def optional_rows_async(conn: aiosqlite.Connection, sql: str) -> list[sqlite3.Row]:
    try:
        cursor = await conn.execute(sql)
        return cast(list[sqlite3.Row], await cursor.fetchall())
    except sqlite3.OperationalError as exc:
        if is_missing_table_error(exc):
            return []
        raise

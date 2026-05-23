"""Durable FTS freshness state for fast retrieval readiness checks."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

import aiosqlite

FRESHNESS_TABLE = "fts_freshness_state"
READY = "ready"
STALE = "stale"
UNKNOWN = "unknown"
MESSAGE_SURFACE = "messages_fts"

_CREATE_TABLE_SQL = f"""
    CREATE TABLE IF NOT EXISTS {FRESHNESS_TABLE} (
        surface TEXT PRIMARY KEY,
        state TEXT NOT NULL CHECK (state IN ('ready', 'stale', 'unknown')),
        checked_at TEXT NOT NULL,
        source_rows INTEGER NOT NULL DEFAULT 0,
        indexed_rows INTEGER NOT NULL DEFAULT 0,
        missing_rows INTEGER NOT NULL DEFAULT 0,
        excess_rows INTEGER NOT NULL DEFAULT 0,
        duplicate_rows INTEGER NOT NULL DEFAULT 0,
        detail TEXT
    )
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _table_exists_sync(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (FRESHNESS_TABLE,),
    ).fetchone()
    return row is not None


async def _table_exists_async(conn: aiosqlite.Connection) -> bool:
    row = await (
        await conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (FRESHNESS_TABLE,),
        )
    ).fetchone()
    return row is not None


def ensure_fts_freshness_table_sync(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_TABLE_SQL)


async def ensure_fts_freshness_table_async(conn: aiosqlite.Connection) -> None:
    await conn.execute(_CREATE_TABLE_SQL)


def record_fts_surface_state_sync(
    conn: sqlite3.Connection,
    *,
    surface: str,
    state: str,
    source_rows: int = 0,
    indexed_rows: int = 0,
    missing_rows: int = 0,
    excess_rows: int = 0,
    duplicate_rows: int = 0,
    detail: str | None = None,
) -> None:
    ensure_fts_freshness_table_sync(conn)
    conn.execute(
        """
        INSERT INTO fts_freshness_state (
            surface, state, checked_at, source_rows, indexed_rows,
            missing_rows, excess_rows, duplicate_rows, detail
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(surface) DO UPDATE SET
            state=excluded.state,
            checked_at=excluded.checked_at,
            source_rows=excluded.source_rows,
            indexed_rows=excluded.indexed_rows,
            missing_rows=excluded.missing_rows,
            excess_rows=excluded.excess_rows,
            duplicate_rows=excluded.duplicate_rows,
            detail=excluded.detail
        """,
        (
            surface,
            state,
            _now_iso(),
            int(source_rows),
            int(indexed_rows),
            int(missing_rows),
            int(excess_rows),
            int(duplicate_rows),
            detail,
        ),
    )


async def record_fts_surface_state_async(
    conn: aiosqlite.Connection,
    *,
    surface: str,
    state: str,
    source_rows: int = 0,
    indexed_rows: int = 0,
    missing_rows: int = 0,
    excess_rows: int = 0,
    duplicate_rows: int = 0,
    detail: str | None = None,
) -> None:
    await ensure_fts_freshness_table_async(conn)
    await conn.execute(
        """
        INSERT INTO fts_freshness_state (
            surface, state, checked_at, source_rows, indexed_rows,
            missing_rows, excess_rows, duplicate_rows, detail
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(surface) DO UPDATE SET
            state=excluded.state,
            checked_at=excluded.checked_at,
            source_rows=excluded.source_rows,
            indexed_rows=excluded.indexed_rows,
            missing_rows=excluded.missing_rows,
            excess_rows=excluded.excess_rows,
            duplicate_rows=excluded.duplicate_rows,
            detail=excluded.detail
        """,
        (
            surface,
            state,
            _now_iso(),
            int(source_rows),
            int(indexed_rows),
            int(missing_rows),
            int(excess_rows),
            int(duplicate_rows),
            detail,
        ),
    )


def record_fts_invariant_snapshot_sync(conn: sqlite3.Connection, snapshot: Any) -> None:
    for surface in snapshot.surfaces:
        record_fts_surface_state_sync(
            conn,
            surface=str(surface.name),
            state=READY if bool(surface.ready) else STALE,
            source_rows=int(surface.source_rows),
            indexed_rows=int(surface.indexed_rows),
            missing_rows=int(surface.missing_rows),
            excess_rows=int(surface.excess_rows),
            duplicate_rows=int(surface.duplicate_rows),
            detail=None if bool(surface.ready) else "exact invariant failed",
        )


def mark_all_fts_stale_sync(conn: sqlite3.Connection, *, detail: str) -> None:
    ensure_fts_freshness_table_sync(conn)
    for surface in ("messages_fts", "action_events_fts", "session_work_events_fts", "work_threads_fts"):
        record_fts_surface_state_sync(conn, surface=surface, state=STALE, detail=detail)


async def mark_all_fts_stale_async(conn: aiosqlite.Connection, *, detail: str) -> None:
    await ensure_fts_freshness_table_async(conn)
    for surface in ("messages_fts", "action_events_fts", "session_work_events_fts", "work_threads_fts"):
        await record_fts_surface_state_async(conn, surface=surface, state=STALE, detail=detail)


def message_fts_marked_ready_sync(conn: sqlite3.Connection) -> bool:
    return message_fts_recorded_state_sync(conn) == READY


async def message_fts_marked_ready_async(conn: aiosqlite.Connection) -> bool:
    return await message_fts_recorded_state_async(conn) == READY


def message_fts_recorded_state_sync(conn: sqlite3.Connection) -> str | None:
    if not _table_exists_sync(conn):
        return None
    row = conn.execute(
        "SELECT state FROM fts_freshness_state WHERE surface=?",
        (MESSAGE_SURFACE,),
    ).fetchone()
    return None if row is None else str(row[0])


async def message_fts_recorded_state_async(conn: aiosqlite.Connection) -> str | None:
    if not await _table_exists_async(conn):
        return None
    row = await (
        await conn.execute(
            "SELECT state FROM fts_freshness_state WHERE surface=?",
            (MESSAGE_SURFACE,),
        )
    ).fetchone()
    return None if row is None else str(row[0])


__all__ = [
    "FRESHNESS_TABLE",
    "MESSAGE_SURFACE",
    "READY",
    "STALE",
    "UNKNOWN",
    "ensure_fts_freshness_table_async",
    "ensure_fts_freshness_table_sync",
    "mark_all_fts_stale_async",
    "mark_all_fts_stale_sync",
    "message_fts_marked_ready_async",
    "message_fts_marked_ready_sync",
    "message_fts_recorded_state_async",
    "message_fts_recorded_state_sync",
    "record_fts_invariant_snapshot_sync",
    "record_fts_surface_state_async",
    "record_fts_surface_state_sync",
]

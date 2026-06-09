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

_COLUMN_UPGRADES: tuple[tuple[str, str], ...] = (
    ("source_rows", "INTEGER NOT NULL DEFAULT 0"),
    ("indexed_rows", "INTEGER NOT NULL DEFAULT 0"),
    ("missing_rows", "INTEGER NOT NULL DEFAULT 0"),
    ("excess_rows", "INTEGER NOT NULL DEFAULT 0"),
    ("duplicate_rows", "INTEGER NOT NULL DEFAULT 0"),
    ("detail", "TEXT"),
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _int_or_zero(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        try:
            return int(value)
        except ValueError:
            return 0
    if value is None:
        return 0
    try:
        return int(str(value))
    except ValueError:
        return 0


def freshness_ready_record_trusted(
    *,
    state: str | None,
    source_rows: int,
    indexed_rows: int,
    missing_rows: int,
    excess_rows: int,
    duplicate_rows: int,
    source_has_rows: bool | None,
) -> bool:
    """Return whether a durable freshness row is safe to use as ready.

    A ``ready`` row must be internally clean. The historical poisoned shape
    ``source_rows=0`` and ``indexed_rows=0`` is trusted only after proving the
    source table has no rows; otherwise readiness is unknown and must be
    recomputed by repair/search paths.
    """
    if state != READY:
        return False
    counters = (source_rows, indexed_rows, missing_rows, excess_rows, duplicate_rows)
    if any(counter < 0 for counter in counters):
        return False
    if source_rows != indexed_rows:
        return False
    if missing_rows != 0 or excess_rows != 0 or duplicate_rows != 0:
        return False
    return not (source_rows == 0 and indexed_rows == 0 and source_has_rows is not False)


def _table_exists_sync(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (FRESHNESS_TABLE,),
    ).fetchone()
    return row is not None


def _named_table_exists_sync(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
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


async def _named_table_exists_async(conn: aiosqlite.Connection, table_name: str) -> bool:
    row = await (
        await conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table_name,),
        )
    ).fetchone()
    return row is not None


def _source_table_has_rows_sync(conn: sqlite3.Connection, table_name: str) -> bool | None:
    if not _named_table_exists_sync(conn, table_name):
        return None
    row = conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1").fetchone()
    return row is not None


async def _source_table_has_rows_async(conn: aiosqlite.Connection, table_name: str) -> bool | None:
    if not await _named_table_exists_async(conn, table_name):
        return None
    row = await (await conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1")).fetchone()
    return row is not None


def ensure_fts_freshness_table_sync(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_TABLE_SQL)
    columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({FRESHNESS_TABLE})").fetchall()}
    for name, definition in _COLUMN_UPGRADES:
        if name not in columns:
            conn.execute(f"ALTER TABLE {FRESHNESS_TABLE} ADD COLUMN {name} {definition}")


async def ensure_fts_freshness_table_async(conn: aiosqlite.Connection) -> None:
    await conn.execute(_CREATE_TABLE_SQL)
    rows = await (await conn.execute(f"PRAGMA table_info({FRESHNESS_TABLE})")).fetchall()
    columns = {str(row[1]) for row in rows}
    for name, definition in _COLUMN_UPGRADES:
        if name not in columns:
            await conn.execute(f"ALTER TABLE {FRESHNESS_TABLE} ADD COLUMN {name} {definition}")


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
    for surface in ("messages_fts", "session_work_events_fts", "threads_fts"):
        record_fts_surface_state_sync(conn, surface=surface, state=STALE, detail=detail)


async def mark_all_fts_stale_async(conn: aiosqlite.Connection, *, detail: str) -> None:
    await ensure_fts_freshness_table_async(conn)
    for surface in ("messages_fts", "session_work_events_fts", "threads_fts"):
        await record_fts_surface_state_async(conn, surface=surface, state=STALE, detail=detail)


def message_fts_marked_ready_sync(conn: sqlite3.Connection) -> bool:
    return message_fts_recorded_ready_trusted_sync(conn)


async def message_fts_marked_ready_async(conn: aiosqlite.Connection) -> bool:
    return await message_fts_recorded_ready_trusted_async(conn)


def message_fts_recorded_state_sync(conn: sqlite3.Connection) -> str | None:
    if not _table_exists_sync(conn):
        return None
    row = conn.execute(
        "SELECT state FROM fts_freshness_state WHERE surface=?",
        (MESSAGE_SURFACE,),
    ).fetchone()
    if row is None:
        return None
    state = str(row[0])
    if state != READY:
        return state
    return READY if message_fts_recorded_ready_trusted_sync(conn) else UNKNOWN


async def message_fts_recorded_state_async(conn: aiosqlite.Connection) -> str | None:
    if not await _table_exists_async(conn):
        return None
    row = await (
        await conn.execute(
            "SELECT state FROM fts_freshness_state WHERE surface=?",
            (MESSAGE_SURFACE,),
        )
    ).fetchone()
    if row is None:
        return None
    state = str(row[0])
    if state != READY:
        return state
    return READY if await message_fts_recorded_ready_trusted_async(conn) else UNKNOWN


def message_fts_recorded_ready_trusted_sync(conn: sqlite3.Connection) -> bool:
    if not _table_exists_sync(conn):
        return False
    columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({FRESHNESS_TABLE})").fetchall()}
    selected = ["state"]
    for name in ("source_rows", "indexed_rows", "missing_rows", "excess_rows", "duplicate_rows"):
        if name in columns:
            selected.append(name)
    row = conn.execute(
        f"SELECT {', '.join(selected)} FROM {FRESHNESS_TABLE} WHERE surface=?",
        (MESSAGE_SURFACE,),
    ).fetchone()
    if row is None:
        return False
    record = dict(zip(selected, row, strict=True))
    return freshness_ready_record_trusted(
        state=str(record["state"]),
        source_rows=_int_or_zero(record.get("source_rows")),
        indexed_rows=_int_or_zero(record.get("indexed_rows")),
        missing_rows=_int_or_zero(record.get("missing_rows")),
        excess_rows=_int_or_zero(record.get("excess_rows")),
        duplicate_rows=_int_or_zero(record.get("duplicate_rows")),
        source_has_rows=_source_table_has_rows_sync(conn, "messages")
        if _int_or_zero(record.get("source_rows")) == 0 and _int_or_zero(record.get("indexed_rows")) == 0
        else False,
    )


async def message_fts_recorded_ready_trusted_async(conn: aiosqlite.Connection) -> bool:
    if not await _table_exists_async(conn):
        return False
    rows = await (await conn.execute(f"PRAGMA table_info({FRESHNESS_TABLE})")).fetchall()
    columns = {str(row[1]) for row in rows}
    selected = ["state"]
    for name in ("source_rows", "indexed_rows", "missing_rows", "excess_rows", "duplicate_rows"):
        if name in columns:
            selected.append(name)
    row = await (
        await conn.execute(
            f"SELECT {', '.join(selected)} FROM {FRESHNESS_TABLE} WHERE surface=?",
            (MESSAGE_SURFACE,),
        )
    ).fetchone()
    if row is None:
        return False
    record = dict(zip(selected, row, strict=True))
    source_rows = _int_or_zero(record.get("source_rows"))
    indexed_rows = _int_or_zero(record.get("indexed_rows"))
    return freshness_ready_record_trusted(
        state=str(record["state"]),
        source_rows=source_rows,
        indexed_rows=indexed_rows,
        missing_rows=_int_or_zero(record.get("missing_rows")),
        excess_rows=_int_or_zero(record.get("excess_rows")),
        duplicate_rows=_int_or_zero(record.get("duplicate_rows")),
        source_has_rows=await _source_table_has_rows_async(conn, "messages")
        if source_rows == 0 and indexed_rows == 0
        else False,
    )


__all__ = [
    "FRESHNESS_TABLE",
    "MESSAGE_SURFACE",
    "READY",
    "STALE",
    "UNKNOWN",
    "ensure_fts_freshness_table_async",
    "ensure_fts_freshness_table_sync",
    "freshness_ready_record_trusted",
    "mark_all_fts_stale_async",
    "mark_all_fts_stale_sync",
    "message_fts_marked_ready_async",
    "message_fts_marked_ready_sync",
    "message_fts_recorded_ready_trusted_async",
    "message_fts_recorded_ready_trusted_sync",
    "message_fts_recorded_state_async",
    "message_fts_recorded_state_sync",
    "record_fts_invariant_snapshot_sync",
    "record_fts_surface_state_async",
    "record_fts_surface_state_sync",
]

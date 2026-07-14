"""Durable FTS freshness state for fast retrieval readiness checks."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from polylogue.storage.sqlite.archive_tiers.index import FTS_FRESHNESS_STATE_DDL
from polylogue.storage.sqlite.introspection import (
    table_exists_async,
)

FRESHNESS_TABLE = "fts_freshness_state"
READY = "ready"
STALE = "stale"
UNKNOWN = "unknown"
MESSAGE_SURFACE = "messages_fts"
_MESSAGE_FTS_TRIGGER_NAMES = ("messages_fts_ai", "messages_fts_ad", "messages_fts_au")

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


def _named_table_exists_sync(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
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


def _message_fts_source_has_rows_sync(conn: sqlite3.Connection) -> bool | None:
    if not _named_table_exists_sync(conn, "blocks"):
        return None
    row = conn.execute("SELECT 1 FROM blocks WHERE search_text != '' LIMIT 1").fetchone()
    return row is not None


def _message_fts_triggers_present_sync(conn: sqlite3.Connection) -> bool:
    placeholders = ", ".join("?" for _ in _MESSAGE_FTS_TRIGGER_NAMES)
    row = conn.execute(
        f"SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
        _MESSAGE_FTS_TRIGGER_NAMES,
    ).fetchone()
    return row is not None and int(row[0] or 0) == len(_MESSAGE_FTS_TRIGGER_NAMES)


async def _message_fts_source_has_rows_async(conn: aiosqlite.Connection) -> bool | None:
    if not await _named_table_exists_async(conn, "blocks"):
        return None
    row = await (await conn.execute("SELECT 1 FROM blocks WHERE search_text != '' LIMIT 1")).fetchone()
    return row is not None


async def _message_fts_triggers_present_async(conn: aiosqlite.Connection) -> bool:
    placeholders = ", ".join("?" for _ in _MESSAGE_FTS_TRIGGER_NAMES)
    row = await (
        await conn.execute(
            f"SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
            _MESSAGE_FTS_TRIGGER_NAMES,
        )
    ).fetchone()
    return row is not None and int(row[0] or 0) == len(_MESSAGE_FTS_TRIGGER_NAMES)


def ensure_fts_freshness_table_sync(conn: sqlite3.Connection) -> None:
    conn.execute(FTS_FRESHNESS_STATE_DDL)
    columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({FRESHNESS_TABLE})").fetchall()}
    for name, definition in _COLUMN_UPGRADES:
        if name not in columns:
            conn.execute(f"ALTER TABLE {FRESHNESS_TABLE} ADD COLUMN {name} {definition}")


async def ensure_fts_freshness_table_async(conn: aiosqlite.Connection) -> None:
    await conn.execute(FTS_FRESHNESS_STATE_DDL)
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


def _message_fts_record_sync(conn: sqlite3.Connection) -> dict[str, object] | None:
    if not _table_exists_sync(conn):
        return None
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
        return None
    return dict(zip(selected, row, strict=True))


async def _message_fts_record_async(conn: aiosqlite.Connection) -> dict[str, object] | None:
    if not await _table_exists_async(conn):
        return None
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
        return None
    return dict(zip(selected, row, strict=True))


def _recorded_counter(record: dict[str, object], name: str) -> int:
    return _int_or_zero(record.get(name))


def _recorded_ready_state_sync(conn: sqlite3.Connection, record: dict[str, object]) -> bool:
    source_rows = _recorded_counter(record, "source_rows")
    indexed_rows = _recorded_counter(record, "indexed_rows")
    return freshness_ready_record_trusted(
        state=str(record["state"]),
        source_rows=source_rows,
        indexed_rows=indexed_rows,
        missing_rows=_recorded_counter(record, "missing_rows"),
        excess_rows=_recorded_counter(record, "excess_rows"),
        duplicate_rows=_recorded_counter(record, "duplicate_rows"),
        source_has_rows=_message_fts_source_has_rows_sync(conn) if source_rows == 0 and indexed_rows == 0 else False,
    )


async def _recorded_ready_state_async(conn: aiosqlite.Connection, record: dict[str, object]) -> bool:
    source_rows = _recorded_counter(record, "source_rows")
    indexed_rows = _recorded_counter(record, "indexed_rows")
    return freshness_ready_record_trusted(
        state=str(record["state"]),
        source_rows=source_rows,
        indexed_rows=indexed_rows,
        missing_rows=_recorded_counter(record, "missing_rows"),
        excess_rows=_recorded_counter(record, "excess_rows"),
        duplicate_rows=_recorded_counter(record, "duplicate_rows"),
        source_has_rows=await _message_fts_source_has_rows_async(conn)
        if source_rows == 0 and indexed_rows == 0
        else False,
    )


def message_fts_recorded_readiness_sync(conn: sqlite3.Connection) -> dict[str, int | bool] | None:
    """Return a trusted recorded search-readiness verdict when one exists.

    ``ready`` rows are trusted only after the full consistency check. ``stale``
    rows are also useful: they are already a durable negative verdict with
    recorded counters, so the hot query path can fail fast instead of repeating
    an exact archive-scale recount. ``unknown``/poisoned rows intentionally
    return ``None`` so callers recompute and repair the ledger once.
    """
    record = _message_fts_record_sync(conn)
    if record is None:
        return None
    state = str(record["state"])
    exists = _named_table_exists_sync(conn, MESSAGE_SURFACE)
    triggers_present = exists and _message_fts_triggers_present_sync(conn)
    if state == READY:
        if not triggers_present or not _recorded_ready_state_sync(conn, record):
            return None
        return {
            "exists": True,
            "indexed_rows": _recorded_counter(record, "indexed_rows"),
            "total_rows": _recorded_counter(record, "source_rows"),
            "ready": True,
            "triggers_present": True,
        }
    if state == STALE:
        return {
            "exists": exists,
            "indexed_rows": _recorded_counter(record, "indexed_rows"),
            "total_rows": _recorded_counter(record, "source_rows"),
            "ready": False,
            "triggers_present": triggers_present,
        }
    return None


async def message_fts_recorded_readiness_async(conn: aiosqlite.Connection) -> dict[str, int | bool] | None:
    """Async counterpart to :func:`message_fts_recorded_readiness_sync`."""
    record = await _message_fts_record_async(conn)
    if record is None:
        return None
    state = str(record["state"])
    exists = await _named_table_exists_async(conn, MESSAGE_SURFACE)
    triggers_present = exists and await _message_fts_triggers_present_async(conn)
    if state == READY:
        if not triggers_present or not await _recorded_ready_state_async(conn, record):
            return None
        return {
            "exists": True,
            "indexed_rows": _recorded_counter(record, "indexed_rows"),
            "total_rows": _recorded_counter(record, "source_rows"),
            "ready": True,
            "triggers_present": True,
        }
    if state == STALE:
        return {
            "exists": exists,
            "indexed_rows": _recorded_counter(record, "indexed_rows"),
            "total_rows": _recorded_counter(record, "source_rows"),
            "ready": False,
            "triggers_present": triggers_present,
        }
    return None


def message_fts_recorded_ready_trusted_sync(conn: sqlite3.Connection) -> bool:
    record = _message_fts_record_sync(conn)
    if record is None:
        return False
    if not _message_fts_triggers_present_sync(conn):
        return False
    return _recorded_ready_state_sync(conn, record)


async def message_fts_recorded_ready_trusted_async(conn: aiosqlite.Connection) -> bool:
    record = await _message_fts_record_async(conn)
    if record is None:
        return False
    if not await _message_fts_triggers_present_async(conn):
        return False
    return await _recorded_ready_state_async(conn, record)


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
    "message_fts_recorded_readiness_async",
    "message_fts_recorded_readiness_sync",
    "message_fts_recorded_state_async",
    "message_fts_recorded_state_sync",
    "record_fts_invariant_snapshot_sync",
    "record_fts_surface_state_async",
    "record_fts_surface_state_sync",
]

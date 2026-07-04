"""Canonical FTS lifecycle operations shared across sync and async callers."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import TypeAlias

import aiosqlite

from polylogue.storage.fts.sql import (
    FTS_INDEX_DOC_COUNT_SQL,
    FTS_INDEX_EXISTS_SQL,
    FTS_INDEXABLE_MESSAGE_COUNT_SQL,
    FTS_MESSAGES_TABLE_SQL,
    FTS_REBUILD_SQL,
    IndexedMessage,
    chunked,
    delete_session_rows_sql,
    insert_all_message_rows_sql,
    insert_missing_message_rows_range_sql,
    insert_session_rows_sql,
)

_chunked = chunked
IndexedMessageLike: TypeAlias = tuple[str, str, str | None] | IndexedMessage


def _indexed_message_parts(message: IndexedMessageLike) -> tuple[str, str, str | None]:
    if isinstance(message, tuple):
        return message
    return message.message_id, message.session_id, message.text


def _row_int(row: sqlite3.Row | None, key: int | str) -> int:
    if row is None:
        return 0
    try:
        return int(row[key])
    except (TypeError, ValueError):
        return 0


def _status_int(status: dict[str, object], key: str) -> int:
    value = status.get(key, 0)
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


def _passive_wal_checkpoint_sync(conn: sqlite3.Connection) -> None:
    with suppress(sqlite3.Error):
        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")


def _message_trigger_names_for_sync(conn: sqlite3.Connection) -> tuple[str, ...]:
    del conn
    return _BLOCKS_FTS_TRIGGER_NAMES


async def _message_trigger_names_for_async(conn: aiosqlite.Connection) -> tuple[str, ...]:
    del conn
    return _BLOCKS_FTS_TRIGGER_NAMES


_SESSION_WORK_EVENT_FTS_TRIGGER_NAMES = (
    "session_work_events_fts_ai",
    "session_work_events_fts_ad",
    "session_work_events_fts_au",
)

_THREAD_FTS_TRIGGER_NAMES = (
    "threads_fts_ai",
    "threads_fts_ad",
    "threads_fts_au",
)

_BLOCKS_FTS_TRIGGER_NAMES = (
    "messages_fts_ai",
    "messages_fts_ad",
    "messages_fts_au",
)

_FTS_TRIGGER_NAMES = _BLOCKS_FTS_TRIGGER_NAMES + _SESSION_WORK_EVENT_FTS_TRIGGER_NAMES + _THREAD_FTS_TRIGGER_NAMES

FTS_TRIGGER_NAMES = _FTS_TRIGGER_NAMES
"""Canonical FTS trigger set for all archive and insight search surfaces."""

DEFAULT_MISSING_MESSAGE_FTS_BATCH_ROWS = 50_000
"""Rowid window size for archive-wide missing message FTS repair."""


@dataclass(frozen=True, slots=True)
class FtsSurfaceInvariant:
    """Exact freshness status for one FTS-backed surface."""

    name: str
    source_exists: bool
    exists: bool
    source_rows: int
    indexed_rows: int
    triggers_present: bool
    missing_rows: int = 0
    excess_rows: int = 0
    duplicate_rows: int = 0

    @property
    def ready(self) -> bool:
        if not self.source_exists:
            return not self.exists
        return (
            self.exists
            and self.triggers_present
            and self.missing_rows == 0
            and self.excess_rows == 0
            and self.duplicate_rows == 0
        )


@dataclass(frozen=True, slots=True)
class FtsInvariantSnapshot:
    """Exact freshness status for every active FTS-backed search surface."""

    messages: FtsSurfaceInvariant
    retired_action_surface: FtsSurfaceInvariant
    session_work_events: FtsSurfaceInvariant
    threads: FtsSurfaceInvariant

    @property
    def ready(self) -> bool:
        return all(surface.ready for surface in self.surfaces)

    @property
    def surfaces(self) -> tuple[FtsSurfaceInvariant, ...]:
        return (self.messages, self.retired_action_surface, self.session_work_events, self.threads)


def _triggers_present_sync(conn: sqlite3.Connection, names: tuple[str, ...]) -> bool:
    """Check whether every named trigger exists in sqlite_master."""
    placeholders = ", ".join("?" for _ in names)
    row = conn.execute(
        f"SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
        names,
    ).fetchone()
    return row is not None and row[0] == len(names)


async def _triggers_present_async(conn: aiosqlite.Connection, names: tuple[str, ...]) -> bool:
    """Check whether every named trigger exists in sqlite_master."""
    placeholders = ", ".join("?" for _ in names)
    cursor = await conn.execute(
        f"SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
        names,
    )
    row = await cursor.fetchone()
    return row is not None and row[0] == len(names)


async def suspend_fts_triggers_async(conn: aiosqlite.Connection, *, mark_stale: bool = True) -> None:
    """Drop FTS triggers to avoid per-row overhead during bulk inserts.

    Call rebuild_fts_index_async() after to repopulate the FTS index.
    """
    if mark_stale:
        from polylogue.storage.fts.freshness import mark_all_fts_stale_async

        await mark_all_fts_stale_async(conn, detail="FTS triggers suspended for bulk write")
    for name in _FTS_TRIGGER_NAMES:
        await conn.execute(f"DROP TRIGGER IF EXISTS {name}")


async def restore_fts_triggers_async(conn: aiosqlite.Connection) -> None:
    """Re-create FTS triggers after bulk insert."""
    await suspend_fts_triggers_async(conn)
    for ddl in await _fts_trigger_ddl_for_existing_surfaces_async(conn):
        if ";" in ddl:
            await conn.executescript(ddl)
        else:
            await conn.execute(ddl)


_BLOCKS_FTS_TRIGGER_DDL = [
    """CREATE TRIGGER IF NOT EXISTS messages_fts_ai
       AFTER INSERT ON blocks WHEN new.search_text != '' BEGIN
           INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
           VALUES (new.rowid, new.block_id, new.message_id, new.session_id, new.block_type, new.search_text);
       END""",
    """CREATE TRIGGER IF NOT EXISTS messages_fts_ad
       AFTER DELETE ON blocks WHEN old.search_text != '' BEGIN
           DELETE FROM messages_fts WHERE rowid = old.rowid;
       END""",
    """CREATE TRIGGER IF NOT EXISTS messages_fts_au
       AFTER UPDATE ON blocks BEGIN
           DELETE FROM messages_fts WHERE rowid = old.rowid;
           INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
           SELECT new.rowid, new.block_id, new.message_id, new.session_id, new.block_type, new.search_text
           WHERE new.search_text != '';
       END""",
]


_SESSION_WORK_EVENT_FTS_TRIGGER_DDL = [
    """CREATE TRIGGER IF NOT EXISTS session_work_events_fts_ai
       AFTER INSERT ON session_work_events BEGIN
           INSERT INTO session_work_events_fts (event_id, session_id, work_event_type, text)
           VALUES (new.event_id, new.session_id, new.work_event_type, new.search_text);
       END""",
    """CREATE TRIGGER IF NOT EXISTS session_work_events_fts_ad
       AFTER DELETE ON session_work_events BEGIN
           DELETE FROM session_work_events_fts WHERE event_id = old.event_id;
       END""",
    """CREATE TRIGGER IF NOT EXISTS session_work_events_fts_au
       AFTER UPDATE ON session_work_events BEGIN
           DELETE FROM session_work_events_fts WHERE event_id = old.event_id;
           INSERT INTO session_work_events_fts (event_id, session_id, work_event_type, text)
           VALUES (new.event_id, new.session_id, new.work_event_type, new.search_text);
       END""",
]

_THREAD_FTS_TRIGGER_DDL = [
    """CREATE TRIGGER IF NOT EXISTS threads_fts_ai
       AFTER INSERT ON threads BEGIN
           INSERT INTO threads_fts (thread_id, root_id, text)
           VALUES (new.thread_id, new.thread_id, new.search_text);
       END""",
    """CREATE TRIGGER IF NOT EXISTS threads_fts_ad
       AFTER DELETE ON threads BEGIN
           DELETE FROM threads_fts WHERE thread_id = old.thread_id;
       END""",
    """CREATE TRIGGER IF NOT EXISTS threads_fts_au
       AFTER UPDATE ON threads BEGIN
           DELETE FROM threads_fts WHERE thread_id = old.thread_id;
           INSERT INTO threads_fts (thread_id, root_id, text)
           VALUES (new.thread_id, new.thread_id, new.search_text);
       END""",
]

_FTS_TRIGGER_DDL = _BLOCKS_FTS_TRIGGER_DDL + _SESSION_WORK_EVENT_FTS_TRIGGER_DDL + _THREAD_FTS_TRIGGER_DDL


def suspend_fts_triggers_sync(conn: sqlite3.Connection, *, mark_stale: bool = True) -> None:
    """Drop FTS triggers for bulk sync operations."""
    if mark_stale:
        from polylogue.storage.fts.freshness import mark_all_fts_stale_sync

        mark_all_fts_stale_sync(conn, detail="FTS triggers suspended for bulk write")
    for name in _FTS_TRIGGER_NAMES:
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")


def message_fts_triggers_present_sync(conn: sqlite3.Connection) -> bool:
    """Return true when the block-backed message FTS triggers are present."""
    return _triggers_present_sync(conn, _BLOCKS_FTS_TRIGGER_NAMES)


def suspend_message_fts_triggers_sync(conn: sqlite3.Connection) -> None:
    """Drop only block-backed message FTS triggers inside the caller's transaction."""
    for name in _BLOCKS_FTS_TRIGGER_NAMES:
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")


def restore_message_fts_triggers_sync(conn: sqlite3.Connection) -> None:
    """Restore only block-backed message FTS triggers inside the caller's transaction."""
    if not _table_exists_sync(conn, "blocks") or not _table_exists_sync(conn, "messages_fts"):
        return
    for ddl in _BLOCKS_FTS_TRIGGER_DDL:
        conn.execute(ddl)


def restore_fts_triggers_sync(conn: sqlite3.Connection) -> None:
    """Re-create FTS triggers after bulk insert."""
    suspend_fts_triggers_sync(conn)
    for ddl in _fts_trigger_ddl_for_existing_surfaces_sync(conn):
        conn.executescript(ddl) if ";" in ddl else conn.execute(ddl)


def ensure_fts_triggers_sync(conn: sqlite3.Connection) -> None:
    """Create missing FTS triggers without dropping existing triggers.

    Steady-state archive writes must not create a dropped-trigger window.
    ``restore_fts_triggers_sync`` remains the explicit recovery/rebuild path
    for replacing trigger definitions and repairing global FTS state.

    Fast-path: when all expected triggers are already present, return
    immediately without issuing any ``executescript()`` calls.  Each
    ``executescript()`` call issues an implicit COMMIT that fragments the
    caller's WAL transaction into multiple smaller ones; avoiding it in
    steady state preserves the intended one-transaction boundary for
    ``commit_archive_write_effects`` (#1851).
    """
    if _triggers_present_sync(conn, _FTS_TRIGGER_NAMES):
        return
    for ddl in _fts_trigger_ddl_for_existing_surfaces_sync(conn):
        conn.executescript(ddl) if ";" in ddl else conn.execute(ddl)


def ensure_fts_index_sync(conn: sqlite3.Connection) -> None:
    """Ensure the FTS5 tables and triggers exist on a sync SQLite connection."""
    conn.execute(FTS_MESSAGES_TABLE_SQL)
    ensure_fts_triggers_sync(conn)


async def ensure_fts_index_async(conn: aiosqlite.Connection) -> None:
    """Ensure the FTS5 tables and triggers exist on an async SQLite connection."""
    await conn.execute(FTS_MESSAGES_TABLE_SQL)
    for ddl in await _fts_trigger_ddl_for_existing_surfaces_async(conn):
        if ";" in ddl:
            await conn.executescript(ddl)
        else:
            await conn.execute(ddl)


def _fts_trigger_ddl_for_existing_surfaces_sync(conn: sqlite3.Connection) -> tuple[str, ...]:
    ddl: list[str] = []
    if _table_exists_sync(conn, "blocks") and _table_exists_sync(conn, "messages_fts"):
        ddl.extend(_BLOCKS_FTS_TRIGGER_DDL)
    if _table_exists_sync(conn, "session_work_events") and _table_exists_sync(conn, "session_work_events_fts"):
        ddl.extend(_SESSION_WORK_EVENT_FTS_TRIGGER_DDL)
    if _table_exists_sync(conn, "threads") and _table_exists_sync(conn, "threads_fts"):
        ddl.extend(_THREAD_FTS_TRIGGER_DDL)
    return tuple(ddl)


async def _table_exists_async(conn: aiosqlite.Connection, table_name: str) -> bool:
    cursor = await conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    )
    row = await cursor.fetchone()
    return row is not None


async def _fts_trigger_ddl_for_existing_surfaces_async(conn: aiosqlite.Connection) -> tuple[str, ...]:
    ddl: list[str] = []
    if await _table_exists_async(conn, "blocks") and await _table_exists_async(conn, "messages_fts"):
        ddl.extend(_BLOCKS_FTS_TRIGGER_DDL)
    if await _table_exists_async(conn, "session_work_events") and await _table_exists_async(
        conn, "session_work_events_fts"
    ):
        ddl.extend(_SESSION_WORK_EVENT_FTS_TRIGGER_DDL)
    if await _table_exists_async(conn, "threads") and await _table_exists_async(conn, "threads_fts"):
        ddl.extend(_THREAD_FTS_TRIGGER_DDL)
    return tuple(ddl)


def rebuild_fts_index_sync(conn: sqlite3.Connection) -> None:
    """Rebuild the full FTS index from persisted archive rows.

    ``messages_fts`` is contentless, so it must be cleared with
    ``delete-all`` and repopulated from the canonical message/content-block
    projection.
    """
    ensure_fts_index_sync(conn)
    conn.execute(FTS_REBUILD_SQL)
    conn.execute(insert_all_message_rows_sql())
    _rebuild_session_work_events_fts_sync(conn)
    _rebuild_threads_fts_sync(conn)
    from polylogue.storage.fts.freshness import record_fts_invariant_snapshot_sync

    record_fts_invariant_snapshot_sync(conn, fts_invariant_snapshot_sync(conn))


def reset_message_fts_index_sync(conn: sqlite3.Connection) -> None:
    """Drop and recreate the block-backed message FTS surface.

    ``messages_fts`` is a contentless-delete FTS5 table.  Clearing it with
    delete-all is correct but can be catastrophically slow on a large archive
    because SQLite walks the virtual table and shadow tables row by row.  Global
    recovery owns the whole message surface, so it can reset the virtual table
    structurally, recreate the block triggers, and repopulate from ``blocks``.
    """
    for name in _BLOCKS_FTS_TRIGGER_NAMES:
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    conn.execute("DROP TABLE IF EXISTS messages_fts")
    conn.execute(FTS_MESSAGES_TABLE_SQL)
    if _table_exists_sync(conn, "blocks"):
        for ddl in _BLOCKS_FTS_TRIGGER_DDL:
            conn.execute(ddl)
        insert_missing_message_rows_batched_sync(conn)
        source_rows = _row_int(conn.execute(FTS_INDEXABLE_MESSAGE_COUNT_SQL).fetchone(), 0)
    else:
        source_rows = 0
    indexed_rows = _row_int(conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone(), 0)
    from polylogue.storage.fts.freshness import READY, STALE, record_fts_surface_state_sync

    ready = source_rows == indexed_rows and _triggers_present_sync(conn, _BLOCKS_FTS_TRIGGER_NAMES)
    record_fts_surface_state_sync(
        conn,
        surface="messages_fts",
        state=READY if ready else STALE,
        source_rows=source_rows,
        indexed_rows=indexed_rows,
        missing_rows=max(source_rows - indexed_rows, 0),
        excess_rows=max(indexed_rows - source_rows, 0),
        detail=None if ready else "message FTS reset did not converge",
    )


def insert_missing_message_rows_batched_sync(
    conn: sqlite3.Connection,
    *,
    batch_rows: int = DEFAULT_MISSING_MESSAGE_FTS_BATCH_ROWS,
    measure_counts: bool = True,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> int:
    """Insert missing block-backed FTS rows in committed rowid windows."""
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")

    ensure_fts_index_sync(conn)
    before = _row_int(conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone(), 0) if measure_counts else 0
    max_rowid = _row_int(
        conn.execute(
            """
            SELECT COALESCE(MAX(rowid), 0)
            FROM blocks
            WHERE search_text != ''
            """
        ).fetchone(),
        0,
    )
    sql = insert_missing_message_rows_range_sql()
    lower = 0
    while lower < max_rowid:
        upper = min(lower + batch_rows, max_rowid)
        changes_before = conn.total_changes
        conn.execute(sql, (lower, upper))
        inserted = conn.total_changes - changes_before
        if inserted:
            conn.commit()
            _passive_wal_checkpoint_sync(conn)
        if progress_callback is not None:
            progress_callback(lower, upper, max(0, inserted))
        lower = upper

    if not measure_counts:
        return 0
    after = _row_int(conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone(), 0)
    return max(0, after - before)


def rebuild_session_insight_fts_sync(conn: sqlite3.Connection) -> None:
    """Rebuild only the durable session-insight FTS projections."""
    restore_fts_triggers_sync(conn)
    _rebuild_session_work_events_fts_sync(conn)
    _rebuild_threads_fts_sync(conn)
    from polylogue.storage.fts.freshness import record_fts_invariant_snapshot_sync

    record_fts_invariant_snapshot_sync(conn, fts_invariant_snapshot_sync(conn))


def _rebuild_session_work_events_fts_sync(conn: sqlite3.Connection) -> None:
    if not (_table_exists_sync(conn, "session_work_events") and _table_exists_sync(conn, "session_work_events_fts")):
        return
    conn.execute("DELETE FROM session_work_events_fts")
    conn.execute(
        """
        INSERT INTO session_work_events_fts (event_id, session_id, work_event_type, text)
        SELECT event_id, session_id, work_event_type, search_text
        FROM session_work_events
        """
    )


def _rebuild_threads_fts_sync(conn: sqlite3.Connection) -> None:
    if not (_table_exists_sync(conn, "threads") and _table_exists_sync(conn, "threads_fts")):
        return
    conn.execute("DELETE FROM threads_fts")
    conn.execute(
        """
        INSERT INTO threads_fts (thread_id, root_id, text)
        SELECT thread_id, thread_id AS root_id, search_text
        FROM threads
        """
    )


async def rebuild_fts_index_async(
    conn: aiosqlite.Connection,
    *,
    session_ids: Sequence[str] | None = None,
    progress_callback: Callable[[int, str | None], None] | None = None,
    progress_desc: Callable[[int, int], str] | None = None,
) -> None:
    """Rebuild the full FTS index from persisted archive rows."""
    await ensure_fts_index_async(conn)
    if session_ids is not None:
        await repair_fts_index_async(
            conn,
            session_ids,
            progress_callback=progress_callback,
            progress_desc=progress_desc,
        )
        return
    await conn.execute(FTS_REBUILD_SQL)
    await conn.execute(insert_all_message_rows_sql())
    readiness = await message_fts_readiness_async(conn, verify_total_rows=True)
    from polylogue.storage.fts.freshness import READY, STALE, record_fts_surface_state_async

    await record_fts_surface_state_async(
        conn,
        surface="messages_fts",
        state=READY if bool(readiness["ready"]) else STALE,
        source_rows=int(readiness["total_rows"]),
        indexed_rows=int(readiness["indexed_rows"]),
        detail=None if bool(readiness["ready"]) else "exact message invariant failed after rebuild",
    )


def repair_message_fts_index_sync(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
    *,
    record_exact_snapshot: bool = True,
) -> None:
    """Repair message FTS rows for the supplied sessions.

    ``record_exact_snapshot`` intentionally defaults to the historical exact
    diagnostic behavior for explicit repair callers. Archive writes that have
    already scoped the changed sessions pass ``False`` so a small ingest does
    not scan the whole block/FTS surface just to refresh global freshness
    counters.
    """
    if not session_ids:
        return
    for chunk in chunked(list(session_ids), size=500):
        params = tuple(chunk)
        conn.execute(delete_session_rows_sql(len(chunk)), params)
        conn.execute(insert_session_rows_sql(len(chunk)), params)
    if not record_exact_snapshot:
        return
    from polylogue.storage.fts.freshness import record_fts_invariant_snapshot_sync

    record_fts_invariant_snapshot_sync(conn, fts_invariant_snapshot_sync(conn))


def repair_fts_index_sync(conn: sqlite3.Connection, session_ids: Sequence[str]) -> None:
    """Repair FTS rows for the supplied sessions from persisted rows."""
    ensure_fts_index_sync(conn)
    repair_message_fts_index_sync(conn, session_ids)


async def repair_fts_index_async(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
    *,
    progress_callback: Callable[[int, str | None], None] | None = None,
    progress_desc: Callable[[int, int], str] | None = None,
) -> None:
    """Repair FTS rows for the supplied sessions from persisted rows."""
    await ensure_fts_index_async(conn)
    if not session_ids:
        return
    total = len(session_ids)
    processed = 0
    for chunk in chunked(list(session_ids), size=500):
        params = tuple(chunk)
        await conn.execute(delete_session_rows_sql(len(chunk)), params)
        await conn.execute(insert_session_rows_sql(len(chunk)), params)
        processed += len(chunk)
        if progress_callback is not None:
            desc = progress_desc(processed, total) if progress_desc is not None else None
            progress_callback(len(chunk), desc)
    await _record_message_fts_exact_state_async(conn)


def replace_fts_rows_for_messages_sync(
    conn: sqlite3.Connection,
    messages: Sequence[IndexedMessageLike],
) -> None:
    """Replace FTS rows for the supplied persisted message sessions."""
    ensure_fts_index_sync(conn)
    if not messages:
        return

    session_ids = sorted({_indexed_message_parts(message)[1] for message in messages})
    for chunk in chunked(session_ids, size=500):
        conn.execute(delete_session_rows_sql(len(chunk)), tuple(chunk))
        conn.execute(insert_session_rows_sql(len(chunk)), tuple(chunk))
    from polylogue.storage.fts.freshness import record_fts_invariant_snapshot_sync

    record_fts_invariant_snapshot_sync(conn, fts_invariant_snapshot_sync(conn))


async def _record_message_fts_exact_state_async(conn: aiosqlite.Connection) -> None:
    """Record exact message FTS readiness after async targeted rewrites."""
    from polylogue.storage.fts.freshness import READY, STALE, record_fts_surface_state_async

    source_exists = await _table_exists_async(conn, "blocks")
    exists = await _table_exists_async(conn, "messages_fts")
    source_rows = 0
    indexed_rows = 0
    missing_rows = 0
    excess_rows = 0
    if source_exists:
        source_rows = _row_int(
            await (await conn.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''")).fetchone(),
            0,
        )
    if exists:
        indexed_rows = _row_int(await (await conn.execute("SELECT COUNT(*) FROM messages_fts_docsize")).fetchone(), 0)
    if source_exists and exists:
        missing_rows = _row_int(
            await (
                await conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM blocks AS b
                    LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
                    WHERE b.search_text != '' AND d.id IS NULL
                    """
                )
            ).fetchone(),
            0,
        )
        excess_rows = _row_int(
            await (
                await conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM messages_fts_docsize AS d
                    LEFT JOIN blocks AS b ON b.rowid = d.id AND b.search_text != ''
                    WHERE b.rowid IS NULL
                    """
                )
            ).fetchone(),
            0,
        )
    triggers_present = exists and await _triggers_present_async(conn, await _message_trigger_names_for_async(conn))
    ready = exists and triggers_present and missing_rows == 0 and excess_rows == 0
    await record_fts_surface_state_async(
        conn,
        surface="messages_fts",
        state=READY if ready else STALE,
        source_rows=source_rows,
        indexed_rows=indexed_rows,
        missing_rows=missing_rows,
        excess_rows=excess_rows,
        detail=None if ready else "exact message invariant failed after targeted repair",
    )


def fts_index_status_sync(conn: sqlite3.Connection) -> dict[str, object]:
    """Return existence and document counts for the sync FTS index."""
    row = conn.execute(FTS_INDEX_EXISTS_SQL).fetchone()
    exists = bool(row)
    count = 0
    if exists:
        count = _row_int(conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone(), 0)
    return {"exists": exists, "count": int(count), "action_exists": False, "action_count": 0}


async def fts_index_status_async(conn: aiosqlite.Connection) -> dict[str, object]:
    """Return existence and document counts for the async FTS index."""
    row = await (await conn.execute(FTS_INDEX_EXISTS_SQL)).fetchone()
    exists = bool(row)
    count = 0
    if exists:
        count_row = await (await conn.execute(FTS_INDEX_DOC_COUNT_SQL)).fetchone()
        count = count_row[0] if count_row else 0
    return {"exists": exists, "count": int(count), "action_exists": False, "action_count": 0}


def message_fts_readiness_sync(
    conn: sqlite3.Connection,
    *,
    verify_total_rows: bool = True,
) -> dict[str, int | bool]:
    """Return whether the message FTS index is present and fully populated."""
    if verify_total_rows:
        status = fts_index_status_sync(conn)
        indexed_rows = _status_int(status, "count")
        exists = bool(status.get("exists", False))
        total_rows = _row_int(conn.execute(FTS_INDEXABLE_MESSAGE_COUNT_SQL).fetchone(), 0)
        triggers_present = exists and _triggers_present_sync(conn, _message_trigger_names_for_sync(conn))
        ready = exists and triggers_present and indexed_rows == total_rows
    else:
        exists = bool(conn.execute(FTS_INDEX_EXISTS_SQL).fetchone())
        has_indexed_rows = exists and bool(conn.execute("SELECT 1 FROM messages_fts_docsize LIMIT 1").fetchone())
        has_indexable_rows = bool(conn.execute("SELECT 1 FROM blocks WHERE search_text != '' LIMIT 1").fetchone())
        triggers_present = exists and _triggers_present_sync(conn, _message_trigger_names_for_sync(conn))
        indexed_rows = 0
        total_rows = 0
        ready = exists and triggers_present and (has_indexed_rows or not has_indexable_rows)
    return {
        "exists": exists,
        "indexed_rows": indexed_rows,
        "total_rows": total_rows,
        "ready": ready,
        "triggers_present": triggers_present,
    }


def message_fts_search_readiness_sync(conn: sqlite3.Connection) -> dict[str, int | bool]:
    """Return retrieval readiness, using the daemon-maintained freshness row when available."""
    from polylogue.storage.fts.freshness import (
        READY,
        STALE,
        message_fts_recorded_readiness_sync,
        record_fts_surface_state_sync,
    )

    recorded_readiness = message_fts_recorded_readiness_sync(conn)
    if recorded_readiness is not None:
        return recorded_readiness
    readiness = message_fts_readiness_sync(conn, verify_total_rows=True)
    if bool(readiness["ready"]):
        with suppress(sqlite3.Error):
            record_fts_surface_state_sync(
                conn,
                surface="messages_fts",
                state=READY,
                source_rows=int(readiness["total_rows"]),
                indexed_rows=int(readiness["indexed_rows"]),
            )
    elif bool(readiness["exists"]):
        with suppress(sqlite3.Error):
            record_fts_surface_state_sync(
                conn,
                surface="messages_fts",
                state=STALE,
                source_rows=int(readiness["total_rows"]),
                indexed_rows=int(readiness["indexed_rows"]),
                detail="exact message readiness failed",
            )
    return readiness


async def message_fts_readiness_async(
    conn: aiosqlite.Connection,
    *,
    verify_total_rows: bool = True,
) -> dict[str, int | bool]:
    """Return whether the message FTS index is present and fully populated."""
    if verify_total_rows:
        status = await fts_index_status_async(conn)
        indexed_rows = _status_int(status, "count")
        exists = bool(status.get("exists", False))
        row = await (await conn.execute(FTS_INDEXABLE_MESSAGE_COUNT_SQL)).fetchone()
        total_rows = _row_int(row, 0)
        triggers_present = exists and await _triggers_present_async(conn, await _message_trigger_names_for_async(conn))
        ready = exists and triggers_present and indexed_rows == total_rows
    else:
        exists = bool(await (await conn.execute(FTS_INDEX_EXISTS_SQL)).fetchone())
        has_indexed_rows = exists and bool(
            await (await conn.execute("SELECT 1 FROM messages_fts_docsize LIMIT 1")).fetchone()
        )
        has_indexable_rows = bool(
            await (await conn.execute("SELECT 1 FROM blocks WHERE search_text != '' LIMIT 1")).fetchone()
        )
        triggers_present = exists and await _triggers_present_async(conn, await _message_trigger_names_for_async(conn))
        indexed_rows = 0
        total_rows = 0
        ready = exists and triggers_present and (has_indexed_rows or not has_indexable_rows)
    return {
        "exists": exists,
        "indexed_rows": indexed_rows,
        "total_rows": total_rows,
        "ready": ready,
        "triggers_present": triggers_present,
    }


async def message_fts_search_readiness_async(conn: aiosqlite.Connection) -> dict[str, int | bool]:
    """Async retrieval readiness with durable-freshness fast path."""
    from polylogue.storage.fts.freshness import (
        READY,
        STALE,
        message_fts_recorded_readiness_async,
        record_fts_surface_state_async,
    )

    recorded_readiness = await message_fts_recorded_readiness_async(conn)
    if recorded_readiness is not None:
        return recorded_readiness
    readiness = await message_fts_readiness_async(conn, verify_total_rows=True)
    if bool(readiness["ready"]):
        with suppress(sqlite3.Error):
            await record_fts_surface_state_async(
                conn,
                surface="messages_fts",
                state=READY,
                source_rows=int(readiness["total_rows"]),
                indexed_rows=int(readiness["indexed_rows"]),
            )
    elif bool(readiness["exists"]):
        with suppress(sqlite3.Error):
            await record_fts_surface_state_async(
                conn,
                surface="messages_fts",
                state=STALE,
                source_rows=int(readiness["total_rows"]),
                indexed_rows=int(readiness["indexed_rows"]),
                detail="exact message readiness failed",
            )
    return readiness


def check_fts_readiness(readiness: Mapping[str, object], repair_hint: str = "") -> None:
    """Raise DatabaseError unless the FTS index is exactly ready."""
    from polylogue.errors import DatabaseError

    if not bool(readiness["exists"]):
        raise DatabaseError(f"Search index not built. {repair_hint}")
    if bool(readiness["ready"]):
        return
    raise DatabaseError(f"Search index is incomplete. {repair_hint}")


def _table_exists_sync(conn: sqlite3.Connection, table_name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
            (table_name,),
        ).fetchone()
        is not None
    )


def _trigger_invariant_sync(
    conn: sqlite3.Connection,
    *,
    name: str,
    source_table_name: str,
    table_name: str,
    source_sql: str,
    indexed_sql: str,
    trigger_names: tuple[str, ...],
    missing_sql: str | None = None,
    excess_sql: str | None = None,
    duplicate_sql: str | None = None,
) -> FtsSurfaceInvariant:
    source_exists = _table_exists_sync(conn, source_table_name)
    exists = _table_exists_sync(conn, table_name)
    source_rows = _row_int(conn.execute(source_sql).fetchone(), 0) if source_exists else 0
    indexed_rows = _row_int(conn.execute(indexed_sql).fetchone(), 0) if exists else 0
    missing_rows = _row_int(conn.execute(missing_sql).fetchone(), 0) if source_exists and exists and missing_sql else 0
    excess_rows = _row_int(conn.execute(excess_sql).fetchone(), 0) if source_exists and exists and excess_sql else 0
    duplicate_rows = _row_int(conn.execute(duplicate_sql).fetchone(), 0) if exists and duplicate_sql else 0
    return FtsSurfaceInvariant(
        name=name,
        source_exists=source_exists,
        exists=exists,
        source_rows=source_rows,
        indexed_rows=indexed_rows,
        triggers_present=exists and _triggers_present_sync(conn, trigger_names),
        missing_rows=missing_rows,
        excess_rows=excess_rows,
        duplicate_rows=duplicate_rows,
    )


def fts_invariant_snapshot_sync(conn: sqlite3.Connection) -> FtsInvariantSnapshot:
    """Return exact freshness status for every active FTS search surface.

    The snapshot spans multiple aggregate queries. Start an explicit read
    transaction when the caller has not already opened one so live ingest
    commits cannot make source counts and FTS shadow counts describe
    different moments in time.
    """
    if conn.in_transaction:
        return _fts_invariant_snapshot_sync(conn)
    conn.execute("BEGIN")
    try:
        snapshot = _fts_invariant_snapshot_sync(conn)
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return snapshot


def _fts_invariant_snapshot_sync(conn: sqlite3.Connection) -> FtsInvariantSnapshot:
    """Return exact freshness status for every active FTS search surface."""
    if _table_exists_sync(conn, "blocks") or _table_exists_sync(conn, "messages_fts"):
        message_surface = _trigger_invariant_sync(
            conn,
            name="messages_fts",
            source_table_name="blocks",
            table_name="messages_fts",
            source_sql="SELECT COUNT(*) FROM blocks WHERE search_text != ''",
            indexed_sql="SELECT COUNT(*) FROM messages_fts_docsize",
            trigger_names=_BLOCKS_FTS_TRIGGER_NAMES,
            missing_sql="""
                SELECT COUNT(*)
                FROM blocks AS b
                LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
                WHERE b.search_text != '' AND d.id IS NULL
            """,
            excess_sql="""
                SELECT COUNT(*)
                FROM messages_fts_docsize AS d
                LEFT JOIN blocks AS b ON b.rowid = d.id AND b.search_text != ''
                WHERE b.rowid IS NULL
            """,
        )
    else:
        message_surface = _messages_fts_invariant_sync(conn)
    return FtsInvariantSnapshot(
        messages=message_surface,
        retired_action_surface=_absent_optional_surface("retired_action_surface"),
        session_work_events=_optional_session_work_events_fts_invariant_sync(conn),
        threads=_optional_threads_fts_invariant_sync(conn),
    )


def _absent_optional_surface(name: str) -> FtsSurfaceInvariant:
    return FtsSurfaceInvariant(
        name=name,
        source_exists=False,
        exists=False,
        source_rows=0,
        indexed_rows=0,
        triggers_present=False,
    )


def _optional_session_work_events_fts_invariant_sync(conn: sqlite3.Connection) -> FtsSurfaceInvariant:
    if not _table_exists_sync(conn, "session_work_events_fts"):
        return _absent_optional_surface("session_work_events_fts")
    return _trigger_invariant_sync(
        conn,
        name="session_work_events_fts",
        source_table_name="session_work_events",
        table_name="session_work_events_fts",
        source_sql="SELECT COUNT(*) FROM session_work_events",
        indexed_sql="SELECT COUNT(DISTINCT event_id) FROM session_work_events_fts",
        trigger_names=_SESSION_WORK_EVENT_FTS_TRIGGER_NAMES,
        missing_sql="""
            SELECT COUNT(*)
            FROM session_work_events AS swe
            LEFT JOIN session_work_events_fts AS f ON f.event_id = swe.event_id
            WHERE f.event_id IS NULL
        """,
        excess_sql="""
            SELECT COUNT(DISTINCT f.event_id)
            FROM session_work_events_fts AS f
            LEFT JOIN session_work_events AS swe ON swe.event_id = f.event_id
            WHERE swe.event_id IS NULL
        """,
        duplicate_sql="SELECT COUNT(*) - COUNT(DISTINCT event_id) FROM session_work_events_fts",
    )


def _optional_threads_fts_invariant_sync(conn: sqlite3.Connection) -> FtsSurfaceInvariant:
    if not _table_exists_sync(conn, "threads_fts"):
        return _absent_optional_surface("threads_fts")
    return _trigger_invariant_sync(
        conn,
        name="threads_fts",
        source_table_name="threads",
        table_name="threads_fts",
        source_sql="SELECT COUNT(*) FROM threads",
        indexed_sql="SELECT COUNT(DISTINCT thread_id) FROM threads_fts",
        trigger_names=_THREAD_FTS_TRIGGER_NAMES,
        missing_sql="""
            SELECT COUNT(*)
            FROM threads AS wt
            LEFT JOIN threads_fts AS f ON f.thread_id = wt.thread_id
            WHERE f.thread_id IS NULL
        """,
        excess_sql="""
            SELECT COUNT(DISTINCT f.thread_id)
            FROM threads_fts AS f
            LEFT JOIN threads AS wt ON wt.thread_id = f.thread_id
            WHERE wt.thread_id IS NULL
        """,
        duplicate_sql="SELECT COUNT(*) - COUNT(DISTINCT thread_id) FROM threads_fts",
    )


def _messages_fts_invariant_sync(conn: sqlite3.Connection) -> FtsSurfaceInvariant:
    """Return the block-backed message FTS invariant."""
    return _trigger_invariant_sync(
        conn,
        name="messages_fts",
        source_table_name="blocks",
        table_name="messages_fts",
        source_sql=FTS_INDEXABLE_MESSAGE_COUNT_SQL,
        indexed_sql=FTS_INDEX_DOC_COUNT_SQL,
        trigger_names=_BLOCKS_FTS_TRIGGER_NAMES,
        missing_sql="""
            SELECT COUNT(*)
            FROM blocks AS b
            LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
            WHERE d.id IS NULL AND b.search_text != ''
        """,
        excess_sql="""
            SELECT COUNT(*)
            FROM messages_fts_docsize AS d
            LEFT JOIN blocks AS b ON b.rowid = d.id AND b.search_text != ''
            WHERE b.rowid IS NULL
        """,
    )


__all__ = [
    "FtsInvariantSnapshot",
    "FtsSurfaceInvariant",
    "FTS_TRIGGER_NAMES",
    "_BLOCKS_FTS_TRIGGER_DDL",
    "_chunked",
    "check_fts_readiness",
    "ensure_fts_index_async",
    "ensure_fts_index_sync",
    "ensure_fts_triggers_sync",
    "fts_index_status_async",
    "fts_index_status_sync",
    "fts_invariant_snapshot_sync",
    "message_fts_readiness_async",
    "message_fts_readiness_sync",
    "message_fts_search_readiness_async",
    "message_fts_search_readiness_sync",
    "message_fts_triggers_present_sync",
    "insert_missing_message_rows_batched_sync",
    "rebuild_fts_index_async",
    "rebuild_fts_index_sync",
    "rebuild_session_insight_fts_sync",
    "repair_fts_index_async",
    "repair_fts_index_sync",
    "repair_message_fts_index_sync",
    "reset_message_fts_index_sync",
    "replace_fts_rows_for_messages_sync",
    "restore_message_fts_triggers_sync",
    "restore_fts_triggers_sync",
    "suspend_message_fts_triggers_sync",
    "suspend_fts_triggers_sync",
]

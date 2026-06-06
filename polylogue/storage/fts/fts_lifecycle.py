"""Canonical FTS lifecycle operations shared across sync and async callers."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import TypeAlias

import aiosqlite

from polylogue.storage.fts.sql import (
    ACTION_FTS_INDEX_DOC_COUNT_SQL,
    ACTION_FTS_INDEX_EXISTS_SQL,
    ACTION_FTS_REBUILD_SQL,
    FTS_ACTIONS_TABLE_SQL,
    FTS_INDEX_DOC_COUNT_SQL,
    FTS_INDEX_EXISTS_SQL,
    FTS_INDEXABLE_MESSAGE_COUNT_SQL,
    FTS_MESSAGES_TABLE_SQL,
    FTS_REBUILD_SQL,
    IndexedMessage,
    chunked,
    delete_action_rows_sql,
    delete_session_rows_sql,
    insert_action_rows_sql,
    insert_all_message_rows_sql,
    insert_missing_action_rows_sql,
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


def _message_trigger_names_for_sync(conn: sqlite3.Connection) -> tuple[str, ...]:
    if _table_exists_sync(conn, "content_blocks"):
        return _MESSAGE_FTS_TRIGGER_NAMES
    return ("messages_fts_ai", "messages_fts_ad", "messages_fts_au")


async def _message_trigger_names_for_async(conn: aiosqlite.Connection) -> tuple[str, ...]:
    if await _table_exists_async(conn, "content_blocks"):
        return _MESSAGE_FTS_TRIGGER_NAMES
    return ("messages_fts_ai", "messages_fts_ad", "messages_fts_au")


_MESSAGE_FTS_TRIGGER_NAMES = (
    "messages_fts_ai",
    "messages_fts_ad",
    "messages_fts_au",
    "content_blocks_fts_ai",
    "content_blocks_fts_ad",
    "content_blocks_fts_au",
)

_ACTION_EVENT_FTS_TRIGGER_NAMES = (
    "action_events_fts_ai",
    "action_events_fts_ad",
    "action_events_fts_au",
)

_SESSION_WORK_EVENT_FTS_TRIGGER_NAMES = (
    "session_work_events_fts_ai",
    "session_work_events_fts_ad",
    "session_work_events_fts_au",
)

_WORK_THREAD_FTS_TRIGGER_NAMES = (
    "work_threads_fts_ai",
    "work_threads_fts_ad",
    "work_threads_fts_au",
)

_FTS_TRIGGER_NAMES = (
    _MESSAGE_FTS_TRIGGER_NAMES
    + _ACTION_EVENT_FTS_TRIGGER_NAMES
    + _SESSION_WORK_EVENT_FTS_TRIGGER_NAMES
    + _WORK_THREAD_FTS_TRIGGER_NAMES
)

FTS_TRIGGER_NAMES = _FTS_TRIGGER_NAMES
"""Canonical FTS trigger set for all archive and insight search surfaces."""


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
    action_events: FtsSurfaceInvariant
    session_work_events: FtsSurfaceInvariant
    work_threads: FtsSurfaceInvariant

    @property
    def ready(self) -> bool:
        return all(surface.ready for surface in self.surfaces)

    @property
    def surfaces(self) -> tuple[FtsSurfaceInvariant, ...]:
        return (self.messages, self.action_events, self.session_work_events, self.work_threads)


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


# Message FTS trigger DDL is imported lazily from schema_ddl_archive to avoid
# a storage package import cycle while keeping the fresh schema and lifecycle
# repair path byte-for-byte aligned.
_MESSAGE_FTS_TRIGGER_DDL: list[str] = []


def _message_fts_trigger_ddl(*, include_content_blocks: bool) -> tuple[str, ...]:
    if include_content_blocks:
        from polylogue.storage.sqlite.schema_ddl_archive import MESSAGE_FTS_DDL

        return (MESSAGE_FTS_DDL,)
    return (
        """
        CREATE TRIGGER IF NOT EXISTS messages_fts_ai
        AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, message_id, session_id, text)
            SELECT new.rowid, new.message_id, new.session_id, new.text
            WHERE new.text IS NOT NULL AND new.text != '';
        END;

        CREATE TRIGGER IF NOT EXISTS messages_fts_ad
        AFTER DELETE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.rowid;
        END;

        CREATE TRIGGER IF NOT EXISTS messages_fts_au
        AFTER UPDATE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.rowid;
            INSERT INTO messages_fts(rowid, message_id, session_id, text)
            SELECT new.rowid, new.message_id, new.session_id, new.text
            WHERE new.text IS NOT NULL AND new.text != '';
        END;
        """,
    )


_ACTION_FTS_TRIGGER_DDL = [
    """CREATE TRIGGER IF NOT EXISTS action_events_fts_ai
       AFTER INSERT ON action_events BEGIN
           INSERT INTO action_events_fts(rowid, event_id, message_id, session_id, action_kind, normalized_tool_name, search_text)
           VALUES (new.rowid, new.event_id, new.message_id, new.session_id, new.action_kind, new.normalized_tool_name, new.search_text);
       END""",
    """CREATE TRIGGER IF NOT EXISTS action_events_fts_ad
       AFTER DELETE ON action_events BEGIN
           INSERT INTO action_events_fts(action_events_fts, rowid, event_id, message_id, session_id, action_kind, normalized_tool_name, search_text)
           VALUES ('delete', old.rowid, old.event_id, old.message_id, old.session_id, old.action_kind, old.normalized_tool_name, old.search_text);
       END""",
    """CREATE TRIGGER IF NOT EXISTS action_events_fts_au
       AFTER UPDATE ON action_events BEGIN
           INSERT INTO action_events_fts(action_events_fts, rowid, event_id, message_id, session_id, action_kind, normalized_tool_name, search_text)
           VALUES ('delete', old.rowid, old.event_id, old.message_id, old.session_id, old.action_kind, old.normalized_tool_name, old.search_text);
           INSERT INTO action_events_fts(rowid, event_id, message_id, session_id, action_kind, normalized_tool_name, search_text)
           VALUES (new.rowid, new.event_id, new.message_id, new.session_id, new.action_kind, new.normalized_tool_name, new.search_text);
       END""",
]

_SESSION_WORK_EVENT_FTS_TRIGGER_DDL = [
    """CREATE TRIGGER IF NOT EXISTS session_work_events_fts_ai
       AFTER INSERT ON session_work_events BEGIN
           INSERT INTO session_work_events_fts (event_id, session_id, source_name, heuristic_label, text)
           VALUES (new.event_id, new.session_id, new.source_name, new.heuristic_label, new.search_text);
       END""",
    """CREATE TRIGGER IF NOT EXISTS session_work_events_fts_ad
       AFTER DELETE ON session_work_events BEGIN
           DELETE FROM session_work_events_fts WHERE event_id = old.event_id;
       END""",
    """CREATE TRIGGER IF NOT EXISTS session_work_events_fts_au
       AFTER UPDATE ON session_work_events BEGIN
           DELETE FROM session_work_events_fts WHERE event_id = old.event_id;
           INSERT INTO session_work_events_fts (event_id, session_id, source_name, heuristic_label, text)
           VALUES (new.event_id, new.session_id, new.source_name, new.heuristic_label, new.search_text);
       END""",
]

_WORK_THREAD_FTS_TRIGGER_DDL = [
    """CREATE TRIGGER IF NOT EXISTS work_threads_fts_ai
       AFTER INSERT ON work_threads BEGIN
           INSERT INTO work_threads_fts (thread_id, root_id, text)
           VALUES (new.thread_id, new.root_id, new.search_text);
       END""",
    """CREATE TRIGGER IF NOT EXISTS work_threads_fts_ad
       AFTER DELETE ON work_threads BEGIN
           DELETE FROM work_threads_fts WHERE thread_id = old.thread_id;
       END""",
    """CREATE TRIGGER IF NOT EXISTS work_threads_fts_au
       AFTER UPDATE ON work_threads BEGIN
           DELETE FROM work_threads_fts WHERE thread_id = old.thread_id;
           INSERT INTO work_threads_fts (thread_id, root_id, text)
           VALUES (new.thread_id, new.root_id, new.search_text);
       END""",
]

_FTS_TRIGGER_DDL = (
    _MESSAGE_FTS_TRIGGER_DDL
    + _ACTION_FTS_TRIGGER_DDL
    + _SESSION_WORK_EVENT_FTS_TRIGGER_DDL
    + _WORK_THREAD_FTS_TRIGGER_DDL
)


def suspend_fts_triggers_sync(conn: sqlite3.Connection, *, mark_stale: bool = True) -> None:
    """Drop FTS triggers for bulk sync operations."""
    if mark_stale:
        from polylogue.storage.fts.freshness import mark_all_fts_stale_sync

        mark_all_fts_stale_sync(conn, detail="FTS triggers suspended for bulk write")
    for name in _FTS_TRIGGER_NAMES:
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")


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
    """
    for ddl in _fts_trigger_ddl_for_existing_surfaces_sync(conn):
        conn.executescript(ddl) if ";" in ddl else conn.execute(ddl)


def ensure_fts_index_sync(conn: sqlite3.Connection) -> None:
    """Ensure the FTS5 tables and triggers exist on a sync SQLite connection."""
    conn.execute(FTS_MESSAGES_TABLE_SQL)
    conn.execute(FTS_ACTIONS_TABLE_SQL)
    ensure_fts_triggers_sync(conn)


async def ensure_fts_index_async(conn: aiosqlite.Connection) -> None:
    """Ensure the FTS5 tables and triggers exist on an async SQLite connection."""
    await conn.execute(FTS_MESSAGES_TABLE_SQL)
    await conn.execute(FTS_ACTIONS_TABLE_SQL)
    for ddl in await _fts_trigger_ddl_for_existing_surfaces_async(conn):
        if ";" in ddl:
            await conn.executescript(ddl)
        else:
            await conn.execute(ddl)


def _fts_trigger_ddl_for_existing_surfaces_sync(conn: sqlite3.Connection) -> tuple[str, ...]:
    ddl: list[str] = []
    if _table_exists_sync(conn, "messages") and _table_exists_sync(conn, "messages_fts"):
        ddl.extend(_message_fts_trigger_ddl(include_content_blocks=_table_exists_sync(conn, "content_blocks")))
    if _table_exists_sync(conn, "action_events") and _table_exists_sync(conn, "action_events_fts"):
        ddl.extend(_ACTION_FTS_TRIGGER_DDL)
    if _table_exists_sync(conn, "session_work_events") and _table_exists_sync(conn, "session_work_events_fts"):
        ddl.extend(_SESSION_WORK_EVENT_FTS_TRIGGER_DDL)
    if _table_exists_sync(conn, "work_threads") and _table_exists_sync(conn, "work_threads_fts"):
        ddl.extend(_WORK_THREAD_FTS_TRIGGER_DDL)
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
    if await _table_exists_async(conn, "messages") and await _table_exists_async(conn, "messages_fts"):
        ddl.extend(_message_fts_trigger_ddl(include_content_blocks=await _table_exists_async(conn, "content_blocks")))
    if await _table_exists_async(conn, "action_events") and await _table_exists_async(conn, "action_events_fts"):
        ddl.extend(_ACTION_FTS_TRIGGER_DDL)
    if await _table_exists_async(conn, "session_work_events") and await _table_exists_async(
        conn, "session_work_events_fts"
    ):
        ddl.extend(_SESSION_WORK_EVENT_FTS_TRIGGER_DDL)
    if await _table_exists_async(conn, "work_threads") and await _table_exists_async(conn, "work_threads_fts"):
        ddl.extend(_WORK_THREAD_FTS_TRIGGER_DDL)
    return tuple(ddl)


def rebuild_fts_index_sync(conn: sqlite3.Connection) -> None:
    """Rebuild the full FTS index from persisted archive rows.

    ``messages_fts`` is contentless, so it must be cleared with
    ``delete-all`` and repopulated from the canonical message/content-block
    projection.  ``action_events_fts`` remains external-content and uses the
    FTS5 ``rebuild`` command.
    """
    ensure_fts_index_sync(conn)
    conn.execute(FTS_REBUILD_SQL)
    if _table_exists_sync(conn, "content_blocks"):
        conn.execute(insert_all_message_rows_sql())
    else:
        conn.execute(
            """
            INSERT INTO messages_fts (rowid, message_id, session_id, text)
            SELECT rowid, message_id, session_id, text
            FROM messages
            WHERE NULLIF(text, '') IS NOT NULL
            """
        )
    conn.execute(ACTION_FTS_REBUILD_SQL)
    _rebuild_session_work_events_fts_sync(conn)
    _rebuild_work_threads_fts_sync(conn)
    from polylogue.storage.fts.freshness import record_fts_invariant_snapshot_sync

    record_fts_invariant_snapshot_sync(conn, fts_invariant_snapshot_sync(conn))


def rebuild_session_insight_fts_sync(conn: sqlite3.Connection) -> None:
    """Rebuild only the durable session-insight FTS projections."""
    restore_fts_triggers_sync(conn)
    _rebuild_session_work_events_fts_sync(conn)
    _rebuild_work_threads_fts_sync(conn)
    from polylogue.storage.fts.freshness import record_fts_invariant_snapshot_sync

    record_fts_invariant_snapshot_sync(conn, fts_invariant_snapshot_sync(conn))


def _rebuild_session_work_events_fts_sync(conn: sqlite3.Connection) -> None:
    if not (_table_exists_sync(conn, "session_work_events") and _table_exists_sync(conn, "session_work_events_fts")):
        return
    conn.execute("DELETE FROM session_work_events_fts")
    conn.execute(
        """
        INSERT INTO session_work_events_fts (event_id, session_id, source_name, heuristic_label, text)
        SELECT event_id, session_id, source_name, heuristic_label, search_text
        FROM session_work_events
        """
    )


def _rebuild_work_threads_fts_sync(conn: sqlite3.Connection) -> None:
    if not (_table_exists_sync(conn, "work_threads") and _table_exists_sync(conn, "work_threads_fts")):
        return
    conn.execute("DELETE FROM work_threads_fts")
    conn.execute(
        """
        INSERT INTO work_threads_fts (thread_id, root_id, text)
        SELECT thread_id, root_id, search_text
        FROM work_threads
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
    if await _table_exists_async(conn, "content_blocks"):
        await conn.execute(insert_all_message_rows_sql())
    else:
        await conn.execute(
            """
            INSERT INTO messages_fts (rowid, message_id, session_id, text)
            SELECT rowid, message_id, session_id, text
            FROM messages
            WHERE NULLIF(text, '') IS NOT NULL
            """
        )
    await conn.execute(ACTION_FTS_REBUILD_SQL)
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


def repair_message_fts_index_sync(conn: sqlite3.Connection, session_ids: Sequence[str]) -> None:
    """Repair message FTS rows for the supplied sessions."""
    if not session_ids:
        return
    for chunk in chunked(list(session_ids), size=500):
        params = tuple(chunk)
        conn.execute(delete_session_rows_sql(len(chunk)), params)
        conn.execute(insert_session_rows_sql(len(chunk)), params)


def repair_action_fts_index_sync(conn: sqlite3.Connection, session_ids: Sequence[str]) -> None:
    """Repair action-event FTS rows for the supplied sessions."""
    if not session_ids:
        return
    for chunk in chunked(list(session_ids), size=500):
        params = tuple(chunk)
        conn.execute(delete_action_rows_sql(len(chunk)), params)
        conn.execute(insert_action_rows_sql(len(chunk)), params)


def insert_missing_action_fts_index_sync(conn: sqlite3.Connection, session_ids: Sequence[str]) -> None:
    """Insert action-event FTS rows that do not already exist."""
    if not session_ids:
        return
    ensure_fts_index_sync(conn)
    for chunk in chunked(list(session_ids), size=500):
        params = tuple(chunk)
        conn.execute(insert_missing_action_rows_sql(len(chunk)), params)


def repair_fts_index_sync(conn: sqlite3.Connection, session_ids: Sequence[str]) -> None:
    """Repair FTS rows for the supplied sessions from persisted rows."""
    ensure_fts_index_sync(conn)
    repair_message_fts_index_sync(conn, session_ids)
    repair_action_fts_index_sync(conn, session_ids)


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
        await conn.execute(delete_action_rows_sql(len(chunk)), params)
        await conn.execute(insert_action_rows_sql(len(chunk)), params)
        processed += len(chunk)
        if progress_callback is not None:
            desc = progress_desc(processed, total) if progress_desc is not None else None
            progress_callback(len(chunk), desc)


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


def fts_index_status_sync(conn: sqlite3.Connection) -> dict[str, object]:
    """Return existence and document counts for the sync FTS index."""
    row = conn.execute(FTS_INDEX_EXISTS_SQL).fetchone()
    exists = bool(row)
    count = 0
    action_exists = bool(conn.execute(ACTION_FTS_INDEX_EXISTS_SQL).fetchone())
    action_count = 0
    if exists:
        count = _row_int(conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone(), 0)
    if action_exists:
        action_count = _row_int(conn.execute(ACTION_FTS_INDEX_DOC_COUNT_SQL).fetchone(), 0)
    return {"exists": exists, "count": int(count), "action_exists": action_exists, "action_count": int(action_count)}


async def fts_index_status_async(conn: aiosqlite.Connection) -> dict[str, object]:
    """Return existence and document counts for the async FTS index."""
    row = await (await conn.execute(FTS_INDEX_EXISTS_SQL)).fetchone()
    exists = bool(row)
    count = 0
    action_exists = bool(await (await conn.execute(ACTION_FTS_INDEX_EXISTS_SQL)).fetchone())
    action_count = 0
    if exists:
        count_row = await (await conn.execute(FTS_INDEX_DOC_COUNT_SQL)).fetchone()
        count = count_row[0] if count_row else 0
    if action_exists:
        action_count_row = await (await conn.execute(ACTION_FTS_INDEX_DOC_COUNT_SQL)).fetchone()
        action_count = action_count_row[0] if action_count_row else 0
    return {"exists": exists, "count": int(count), "action_exists": action_exists, "action_count": int(action_count)}


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
        total_messages = _row_int(conn.execute(FTS_INDEXABLE_MESSAGE_COUNT_SQL).fetchone(), 0)
        triggers_present = exists and _triggers_present_sync(conn, _message_trigger_names_for_sync(conn))
        ready = exists and triggers_present and indexed_rows == total_messages
    else:
        exists = bool(conn.execute(FTS_INDEX_EXISTS_SQL).fetchone())
        # messages_fts is a contentless FTS5 virtual table backed by
        # messages — DELETE FROM messages_fts does not actually drop
        # rows from the virtual surface, but it does drop the rows in
        # the messages_fts_docsize shadow table. Probe the shadow table
        # so the cheap path can still see the "FTS exists but empty"
        # state without paying for a full COUNT(*) (#1314).
        has_indexed_rows = exists and bool(conn.execute("SELECT 1 FROM messages_fts_docsize LIMIT 1").fetchone())
        has_indexable_messages = bool(
            conn.execute("SELECT 1 FROM messages WHERE NULLIF(text, '') IS NOT NULL LIMIT 1").fetchone()
        )
        triggers_present = exists and _triggers_present_sync(conn, _message_trigger_names_for_sync(conn))
        indexed_rows = 0
        total_messages = 0
        ready = exists and triggers_present and (has_indexed_rows or not has_indexable_messages)
    return {
        "exists": exists,
        "indexed_rows": indexed_rows,
        "total_rows": total_messages,
        "ready": ready,
        "triggers_present": triggers_present,
    }


def message_fts_search_readiness_sync(conn: sqlite3.Connection) -> dict[str, int | bool]:
    """Return retrieval readiness, using the daemon-maintained freshness row when available."""
    from polylogue.storage.fts.freshness import (
        READY,
        STALE,
        message_fts_recorded_state_sync,
        record_fts_surface_state_sync,
    )

    recorded_state = message_fts_recorded_state_sync(conn)
    if recorded_state == READY:
        return {
            "exists": True,
            "indexed_rows": 0,
            "total_rows": 0,
            "ready": True,
            "triggers_present": True,
        }
    if recorded_state is not None:
        readiness = message_fts_readiness_sync(conn, verify_total_rows=True)
        if bool(readiness["ready"]):
            with suppress(sqlite3.Error):
                record_fts_surface_state_sync(
                    conn,
                    surface="messages_fts",
                    state=READY,
                    source_rows=int(readiness["total_rows"]),
                    indexed_rows=int(readiness["indexed_rows"]),
                    detail="stale freshness marker repaired during search readiness",
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
        total_messages = _row_int(row, 0)
        triggers_present = exists and await _triggers_present_async(conn, await _message_trigger_names_for_async(conn))
        ready = exists and triggers_present and indexed_rows == total_messages
    else:
        exists = bool(await (await conn.execute(FTS_INDEX_EXISTS_SQL)).fetchone())
        # See message_fts_readiness_sync above for why we probe
        # messages_fts_docsize rather than messages_fts itself (#1314).
        has_indexed_rows = exists and bool(
            await (await conn.execute("SELECT 1 FROM messages_fts_docsize LIMIT 1")).fetchone()
        )
        has_indexable_messages = bool(
            await (await conn.execute("SELECT 1 FROM messages WHERE NULLIF(text, '') IS NOT NULL LIMIT 1")).fetchone()
        )
        triggers_present = exists and await _triggers_present_async(conn, await _message_trigger_names_for_async(conn))
        indexed_rows = 0
        total_messages = 0
        ready = exists and triggers_present and (has_indexed_rows or not has_indexable_messages)
    return {
        "exists": exists,
        "indexed_rows": indexed_rows,
        "total_rows": total_messages,
        "ready": ready,
        "triggers_present": triggers_present,
    }


async def message_fts_search_readiness_async(conn: aiosqlite.Connection) -> dict[str, int | bool]:
    """Async retrieval readiness with durable-freshness fast path."""
    from polylogue.storage.fts.freshness import (
        READY,
        STALE,
        message_fts_recorded_state_async,
        record_fts_surface_state_async,
    )

    recorded_state = await message_fts_recorded_state_async(conn)
    if recorded_state == READY:
        return {
            "exists": True,
            "indexed_rows": 0,
            "total_rows": 0,
            "ready": True,
            "triggers_present": True,
        }
    if recorded_state is not None:
        readiness = await message_fts_readiness_async(conn, verify_total_rows=True)
        if bool(readiness["ready"]):
            with suppress(sqlite3.Error):
                await record_fts_surface_state_async(
                    conn,
                    surface="messages_fts",
                    state=READY,
                    source_rows=int(readiness["total_rows"]),
                    indexed_rows=int(readiness["indexed_rows"]),
                    detail="stale freshness marker repaired during search readiness",
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
    has_content_blocks = _table_exists_sync(conn, "content_blocks")
    message_source_sql = (
        FTS_INDEXABLE_MESSAGE_COUNT_SQL
        if has_content_blocks
        else "SELECT COUNT(*) FROM messages WHERE NULLIF(text, '') IS NOT NULL"
    )
    message_trigger_names = (
        _MESSAGE_FTS_TRIGGER_NAMES if has_content_blocks else ("messages_fts_ai", "messages_fts_ad", "messages_fts_au")
    )
    message_missing_sql = (
        """
            SELECT COUNT(*)
            FROM messages AS m
            LEFT JOIN messages_fts_docsize AS d ON d.id = m.rowid
            WHERE d.id IS NULL
              AND (
                  NULLIF(m.text, '') IS NOT NULL
                  OR EXISTS (
                      SELECT 1
                      FROM content_blocks AS cb
                      WHERE cb.message_id = m.message_id
                        AND (
                            NULLIF(cb.text, '') IS NOT NULL
                            OR NULLIF(cb.tool_input, '') IS NOT NULL
                            OR NULLIF(cb.metadata, '') IS NOT NULL
                        )
                  )
              )
        """
        if has_content_blocks
        else """
            SELECT COUNT(*)
            FROM messages AS m
            LEFT JOIN messages_fts_docsize AS d ON d.id = m.rowid
            WHERE NULLIF(m.text, '') IS NOT NULL AND d.id IS NULL
        """
    )
    message_excess_sql = (
        """
            SELECT COUNT(*)
            FROM messages_fts_docsize AS d
            LEFT JOIN messages AS m ON m.rowid = d.id
            WHERE m.rowid IS NULL
               OR (
                   NULLIF(m.text, '') IS NULL
                   AND NOT EXISTS (
                       SELECT 1
                       FROM content_blocks AS cb
                       WHERE cb.message_id = m.message_id
                         AND (
                             NULLIF(cb.text, '') IS NOT NULL
                             OR NULLIF(cb.tool_input, '') IS NOT NULL
                             OR NULLIF(cb.metadata, '') IS NOT NULL
                         )
                   )
               )
        """
        if has_content_blocks
        else """
            SELECT COUNT(*)
            FROM messages_fts_docsize AS d
            LEFT JOIN messages AS m ON m.rowid = d.id AND NULLIF(m.text, '') IS NOT NULL
            WHERE m.rowid IS NULL
        """
    )
    return FtsInvariantSnapshot(
        messages=_trigger_invariant_sync(
            conn,
            name="messages_fts",
            source_table_name="messages",
            table_name="messages_fts",
            source_sql=message_source_sql,
            indexed_sql=FTS_INDEX_DOC_COUNT_SQL,
            trigger_names=message_trigger_names,
            missing_sql=message_missing_sql,
            excess_sql=message_excess_sql,
        ),
        action_events=_trigger_invariant_sync(
            conn,
            name="action_events_fts",
            source_table_name="action_events",
            table_name="action_events_fts",
            source_sql="SELECT COUNT(*) FROM action_events",
            indexed_sql=ACTION_FTS_INDEX_DOC_COUNT_SQL,
            trigger_names=_ACTION_EVENT_FTS_TRIGGER_NAMES,
            missing_sql="""
                SELECT COUNT(*)
                FROM action_events AS ae
                LEFT JOIN action_events_fts_docsize AS d ON d.id = ae.rowid
                WHERE d.id IS NULL
            """,
            excess_sql="""
                SELECT COUNT(*)
                FROM action_events_fts_docsize AS d
                LEFT JOIN action_events AS ae ON ae.rowid = d.id
                WHERE ae.rowid IS NULL
            """,
        ),
        session_work_events=_trigger_invariant_sync(
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
        ),
        work_threads=_trigger_invariant_sync(
            conn,
            name="work_threads_fts",
            source_table_name="work_threads",
            table_name="work_threads_fts",
            source_sql="SELECT COUNT(*) FROM work_threads",
            indexed_sql="SELECT COUNT(DISTINCT thread_id) FROM work_threads_fts",
            trigger_names=_WORK_THREAD_FTS_TRIGGER_NAMES,
            missing_sql="""
                SELECT COUNT(*)
                FROM work_threads AS wt
                LEFT JOIN work_threads_fts AS f ON f.thread_id = wt.thread_id
                WHERE f.thread_id IS NULL
            """,
            excess_sql="""
                SELECT COUNT(DISTINCT f.thread_id)
                FROM work_threads_fts AS f
                LEFT JOIN work_threads AS wt ON wt.thread_id = f.thread_id
                WHERE wt.thread_id IS NULL
            """,
            duplicate_sql="SELECT COUNT(*) - COUNT(DISTINCT thread_id) FROM work_threads_fts",
        ),
    )


__all__ = [
    "FtsInvariantSnapshot",
    "FtsSurfaceInvariant",
    "FTS_TRIGGER_NAMES",
    "_MESSAGE_FTS_TRIGGER_DDL",
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
    "rebuild_fts_index_async",
    "rebuild_fts_index_sync",
    "rebuild_session_insight_fts_sync",
    "repair_action_fts_index_sync",
    "repair_fts_index_async",
    "repair_fts_index_sync",
    "repair_message_fts_index_sync",
    "insert_missing_action_fts_index_sync",
    "replace_fts_rows_for_messages_sync",
    "restore_fts_triggers_sync",
    "suspend_fts_triggers_sync",
]

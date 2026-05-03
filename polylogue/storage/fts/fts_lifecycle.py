"""Canonical FTS lifecycle operations shared across sync and async callers."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias, cast

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
    delete_conversation_rows_sql,
    insert_action_rows_sql,
    insert_conversation_rows_sql,
)

_chunked = chunked
IndexedMessageLike: TypeAlias = tuple[str, str, str | None] | IndexedMessage


def _indexed_message_parts(message: IndexedMessageLike) -> tuple[str, str, str | None]:
    if isinstance(message, tuple):
        return message
    return message.message_id, message.conversation_id, message.text


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


# ---------------------------------------------------------------------------
# Trigger suspension for bulk writes
# ---------------------------------------------------------------------------

_MESSAGE_FTS_TRIGGER_NAMES = (
    "messages_fts_ai",
    "messages_fts_ad",
    "messages_fts_au",
)

_ACTION_EVENT_FTS_TRIGGER_NAMES = (
    "action_events_fts_ai",
    "action_events_fts_ad",
    "action_events_fts_au",
)

_FTS_TRIGGER_NAMES = _MESSAGE_FTS_TRIGGER_NAMES + _ACTION_EVENT_FTS_TRIGGER_NAMES


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


async def suspend_fts_triggers_async(conn: aiosqlite.Connection) -> None:
    """Drop FTS triggers to avoid per-row overhead during bulk inserts.

    Call rebuild_fts_index_async() after to repopulate the FTS index.
    """
    for name in _FTS_TRIGGER_NAMES:
        await conn.execute(f"DROP TRIGGER IF EXISTS {name}")


async def restore_fts_triggers_async(conn: aiosqlite.Connection) -> None:
    """Re-create FTS triggers after bulk insert."""
    await suspend_fts_triggers_async(conn)
    for ddl in _MESSAGE_FTS_TRIGGER_DDL + _ACTION_FTS_TRIGGER_DDL:
        await conn.execute(ddl)


# Trigger DDL — must match schema_ddl_archive.py and schema_ddl_actions.py
_MESSAGE_FTS_TRIGGER_DDL = [
    """CREATE TRIGGER IF NOT EXISTS messages_fts_ai
       AFTER INSERT ON messages BEGIN
           INSERT INTO messages_fts(rowid, message_id, conversation_id, text)
           SELECT new.rowid, new.message_id, new.conversation_id, new.text
           WHERE new.text IS NOT NULL;
       END""",
    """CREATE TRIGGER IF NOT EXISTS messages_fts_ad
       AFTER DELETE ON messages BEGIN
           DELETE FROM messages_fts WHERE rowid = old.rowid;
       END""",
    """CREATE TRIGGER IF NOT EXISTS messages_fts_au
       AFTER UPDATE ON messages BEGIN
           DELETE FROM messages_fts WHERE rowid = old.rowid;
           INSERT INTO messages_fts(rowid, message_id, conversation_id, text)
           SELECT new.rowid, new.message_id, new.conversation_id, new.text
           WHERE new.text IS NOT NULL;
       END""",
]

_ACTION_FTS_TRIGGER_DDL = [
    """CREATE TRIGGER IF NOT EXISTS action_events_fts_ai
       AFTER INSERT ON action_events BEGIN
           INSERT INTO action_events_fts(rowid, event_id, message_id, conversation_id, action_kind, tool_name, text)
           VALUES (new.rowid, new.event_id, new.message_id, new.conversation_id, new.action_kind, new.normalized_tool_name, new.search_text);
       END""",
    """CREATE TRIGGER IF NOT EXISTS action_events_fts_ad
       AFTER DELETE ON action_events BEGIN
           DELETE FROM action_events_fts WHERE rowid = old.rowid;
       END""",
    """CREATE TRIGGER IF NOT EXISTS action_events_fts_au
       AFTER UPDATE ON action_events BEGIN
           DELETE FROM action_events_fts WHERE rowid = old.rowid;
           INSERT INTO action_events_fts(rowid, event_id, message_id, conversation_id, action_kind, tool_name, text)
           VALUES (new.rowid, new.event_id, new.message_id, new.conversation_id, new.action_kind, new.normalized_tool_name, new.search_text);
       END""",
]


def suspend_fts_triggers_sync(conn: sqlite3.Connection) -> None:
    """Drop FTS triggers for bulk sync operations."""
    for name in _FTS_TRIGGER_NAMES:
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")


def restore_fts_triggers_sync(conn: sqlite3.Connection) -> None:
    """Re-create FTS triggers after bulk insert."""
    suspend_fts_triggers_sync(conn)
    for ddl in _MESSAGE_FTS_TRIGGER_DDL + _ACTION_FTS_TRIGGER_DDL:
        conn.execute(ddl)


def ensure_fts_index_sync(conn: sqlite3.Connection) -> None:
    """Ensure the FTS5 tables and triggers exist on a sync SQLite connection."""
    conn.execute(FTS_MESSAGES_TABLE_SQL)
    conn.execute(FTS_ACTIONS_TABLE_SQL)
    restore_fts_triggers_sync(conn)


async def ensure_fts_index_async(conn: aiosqlite.Connection) -> None:
    """Ensure the FTS5 tables and triggers exist on an async SQLite connection."""
    await conn.execute(FTS_MESSAGES_TABLE_SQL)
    await conn.execute(FTS_ACTIONS_TABLE_SQL)
    await restore_fts_triggers_async(conn)


def rebuild_fts_index_sync(conn: sqlite3.Connection) -> None:
    """Rebuild the full FTS index from persisted archive rows."""
    ensure_fts_index_sync(conn)
    conn.execute("DELETE FROM messages_fts")
    conn.execute("DELETE FROM action_events_fts")
    conn.execute(FTS_REBUILD_SQL)
    conn.execute(ACTION_FTS_REBUILD_SQL)


async def rebuild_fts_index_async(
    conn: aiosqlite.Connection,
    *,
    conversation_ids: Sequence[str] | None = None,
    progress_callback: Callable[[int, str | None], None] | None = None,
    progress_desc: Callable[[int, int], str] | None = None,
) -> None:
    """Rebuild the full FTS index from persisted archive rows."""
    await ensure_fts_index_async(conn)
    await conn.execute("DELETE FROM messages_fts")
    await conn.execute("DELETE FROM action_events_fts")
    if conversation_ids is not None:
        await repair_fts_index_async(
            conn,
            conversation_ids,
            progress_callback=progress_callback,
            progress_desc=progress_desc,
        )
        return
    await conn.execute(FTS_REBUILD_SQL)
    await conn.execute(ACTION_FTS_REBUILD_SQL)


def repair_fts_index_sync(conn: sqlite3.Connection, conversation_ids: Sequence[str]) -> None:
    """Repair FTS rows for the supplied conversations from persisted rows."""
    ensure_fts_index_sync(conn)
    if not conversation_ids:
        return
    for chunk in chunked(list(conversation_ids), size=500):
        params = tuple(chunk)
        conn.execute(delete_conversation_rows_sql(len(chunk)), params)
        conn.execute(insert_conversation_rows_sql(len(chunk)), params)
        conn.execute(delete_action_rows_sql(len(chunk)), params)
        conn.execute(insert_action_rows_sql(len(chunk)), params)


async def repair_fts_index_async(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
    *,
    progress_callback: Callable[[int, str | None], None] | None = None,
    progress_desc: Callable[[int, int], str] | None = None,
) -> None:
    """Repair FTS rows for the supplied conversations from persisted rows."""
    await ensure_fts_index_async(conn)
    if not conversation_ids:
        return
    total = len(conversation_ids)
    processed = 0
    for chunk in chunked(list(conversation_ids), size=500):
        params = tuple(chunk)
        await conn.execute(delete_conversation_rows_sql(len(chunk)), params)
        await conn.execute(insert_conversation_rows_sql(len(chunk)), params)
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
    """Replace FTS rows for the supplied message payloads."""
    ensure_fts_index_sync(conn)
    if not messages:
        return

    conversation_ids = sorted({_indexed_message_parts(message)[1] for message in messages})
    for chunk in chunked(conversation_ids, size=500):
        conn.execute(delete_conversation_rows_sql(len(chunk)), tuple(chunk))

    message_ids = [_indexed_message_parts(message)[0] for message in messages]
    rowids_by_message_id: dict[str, int] = {}
    for chunk in chunked(message_ids, size=500):
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT rowid, message_id FROM messages WHERE message_id IN ({placeholders})",
            tuple(chunk),
        ).fetchall()
        rowids_by_message_id.update({row["message_id"]: row["rowid"] for row in rows})

    with_rowid: list[tuple[int, str, str, str]] = []
    without_rowid: list[tuple[str, str, str]] = []
    for message in messages:
        payload_message_id, payload_conversation_id, payload_text = _indexed_message_parts(message)
        if not payload_text:
            continue
        rowid = rowids_by_message_id.get(payload_message_id)
        if rowid is not None:
            with_rowid.append((rowid, payload_message_id, payload_conversation_id, payload_text))
        else:
            without_rowid.append((payload_message_id, payload_conversation_id, payload_text))

    if with_rowid:
        conn.executemany(
            """
            INSERT INTO messages_fts (rowid, message_id, conversation_id, text)
            VALUES (?, ?, ?, ?)
            """,
            with_rowid,
        )
    if without_rowid:
        conn.executemany(
            """
            INSERT INTO messages_fts (message_id, conversation_id, text)
            VALUES (?, ?, ?)
            """,
            without_rowid,
        )


def fts_index_status_sync(conn: sqlite3.Connection) -> dict[str, object]:
    """Return existence and document counts for the sync FTS index."""
    row = conn.execute(FTS_INDEX_EXISTS_SQL).fetchone()
    exists = bool(row)
    count = 0
    action_count = 0
    if exists:
        count = _row_int(conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone(), 0)
        action_row = conn.execute(ACTION_FTS_INDEX_EXISTS_SQL).fetchone()
        if action_row:
            action_count = _row_int(conn.execute(ACTION_FTS_INDEX_DOC_COUNT_SQL).fetchone(), 0)
    return {"exists": exists, "count": int(count), "action_count": int(action_count)}


async def fts_index_status_async(conn: aiosqlite.Connection) -> dict[str, object]:
    """Return existence and document counts for the async FTS index."""
    row = await (await conn.execute(FTS_INDEX_EXISTS_SQL)).fetchone()
    exists = bool(row)
    count = 0
    action_count = 0
    if exists:
        count_row = await (await conn.execute(FTS_INDEX_DOC_COUNT_SQL)).fetchone()
        count = count_row[0] if count_row else 0
        action_exists_row = await (await conn.execute(ACTION_FTS_INDEX_EXISTS_SQL)).fetchone()
        if action_exists_row:
            action_count_row = await (await conn.execute(ACTION_FTS_INDEX_DOC_COUNT_SQL)).fetchone()
            action_count = action_count_row[0] if action_count_row else 0
    return {"exists": exists, "count": int(count), "action_count": int(action_count)}


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
        triggers_present = exists and _triggers_present_sync(conn, _MESSAGE_FTS_TRIGGER_NAMES)
        ready = exists and triggers_present and indexed_rows == total_messages
    else:
        exists = bool(conn.execute(FTS_INDEX_EXISTS_SQL).fetchone())
        has_indexed_rows = exists and bool(conn.execute("SELECT 1 FROM messages_fts LIMIT 1").fetchone())
        has_indexable_messages = bool(conn.execute("SELECT 1 FROM messages WHERE text IS NOT NULL LIMIT 1").fetchone())
        triggers_present = exists and _triggers_present_sync(conn, _MESSAGE_FTS_TRIGGER_NAMES)
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
        triggers_present = exists and await _triggers_present_async(conn, _MESSAGE_FTS_TRIGGER_NAMES)
        ready = exists and triggers_present and indexed_rows == total_messages
    else:
        exists = bool(await (await conn.execute(FTS_INDEX_EXISTS_SQL)).fetchone())
        has_indexed_rows = exists and bool(await (await conn.execute("SELECT 1 FROM messages_fts LIMIT 1")).fetchone())
        has_indexable_messages = bool(
            await (await conn.execute("SELECT 1 FROM messages WHERE text IS NOT NULL LIMIT 1")).fetchone()
        )
        triggers_present = exists and await _triggers_present_async(conn, _MESSAGE_FTS_TRIGGER_NAMES)
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


FTS_GAP_THRESHOLD = 0.01


def check_fts_readiness(readiness: Mapping[str, object], repair_hint: str = "") -> None:
    """Raise DatabaseError if the FTS index doesn't exist.

    Degrade gracefully on small gaps: if the gap is ≤ FTS_GAP_THRESHOLD
    (default 1%), warn and return instead of raising.
    """
    from polylogue.errors import DatabaseError

    if not bool(readiness["exists"]):
        raise DatabaseError(f"Search index not built. {repair_hint}")
    if bool(readiness["ready"]):
        return
    indexed = int(cast(int, readiness.get("indexed_rows", 0)))
    total = int(cast(int, readiness.get("total_rows", 0)))
    if total > 0 and indexed > 0 and bool(readiness.get("triggers_present", False)):
        gap_ratio = (total - indexed) / total
        if 0 <= gap_ratio <= FTS_GAP_THRESHOLD:
            missing = total - indexed
            pct = gap_ratio * 100
            import logging

            logging.getLogger(__name__).warning(
                "Search index is %d/%d messages behind (%.3f%%). "
                "Results may be incomplete. Run `polylogue doctor --repair --target dangling_fts`.",
                missing,
                total,
                pct,
            )
            return
    raise DatabaseError(f"Search index is incomplete. {repair_hint}")


__all__ = [
    "FTS_GAP_THRESHOLD",
    "_MESSAGE_FTS_TRIGGER_DDL",
    "_chunked",
    "check_fts_readiness",
    "ensure_fts_index_async",
    "ensure_fts_index_sync",
    "fts_index_status_async",
    "fts_index_status_sync",
    "message_fts_readiness_async",
    "message_fts_readiness_sync",
    "rebuild_fts_index_async",
    "rebuild_fts_index_sync",
    "repair_fts_index_async",
    "repair_fts_index_sync",
    "replace_fts_rows_for_messages_sync",
    "restore_fts_triggers_sync",
    "suspend_fts_triggers_sync",
]

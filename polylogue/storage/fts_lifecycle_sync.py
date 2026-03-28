"""Synchronous FTS lifecycle operations."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from polylogue.storage.fts_lifecycle_sql import (
    ACTION_FTS_INDEX_DOC_COUNT_SQL,
    ACTION_FTS_INDEX_EXISTS_SQL,
    ACTION_FTS_REBUILD_SQL,
    FTS_ACTIONS_TABLE_SQL,
    FTS_INDEX_DOC_COUNT_SQL,
    FTS_INDEX_EXISTS_SQL,
    FTS_MESSAGES_TABLE_SQL,
    FTS_REBUILD_SQL,
    IndexedMessage,
    chunked,
    delete_action_rows_sql,
    delete_conversation_rows_sql,
    insert_action_rows_sql,
    insert_conversation_rows_sql,
)


def ensure_fts_index_sync(conn: sqlite3.Connection) -> None:
    """Ensure the FTS5 tables exist on a sync SQLite connection."""
    conn.execute(FTS_MESSAGES_TABLE_SQL)
    conn.execute(FTS_ACTIONS_TABLE_SQL)


def rebuild_fts_index_sync(conn: sqlite3.Connection) -> None:
    """Rebuild the full FTS index from persisted message rows."""
    ensure_fts_index_sync(conn)
    conn.execute("DELETE FROM messages_fts")
    conn.execute("DELETE FROM action_events_fts")
    conn.execute(FTS_REBUILD_SQL)
    conn.execute(ACTION_FTS_REBUILD_SQL)


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


def replace_fts_rows_for_messages_sync(
    conn: sqlite3.Connection,
    messages: Sequence[tuple[str, str, str | None]] | Sequence[IndexedMessage],
) -> None:
    """Replace FTS rows for the supplied message payloads."""
    ensure_fts_index_sync(conn)
    if not messages:
        return

    def message_id(message: IndexedMessage) -> str:
        return message.message_id

    def conversation_id(message: IndexedMessage) -> str:
        return message.conversation_id

    def text(message: IndexedMessage) -> str | None:
        return message.text

    conversation_ids = sorted({conversation_id(message) for message in messages})
    for chunk in chunked(conversation_ids, size=500):
        conn.execute(delete_conversation_rows_sql(len(chunk)), tuple(chunk))

    message_ids = [message_id(message) for message in messages]
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
        payload_text = text(message)
        if not payload_text:
            continue
        payload_message_id = message_id(message)
        payload_conversation_id = conversation_id(message)
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
        count = conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone()[0]
        action_row = conn.execute(ACTION_FTS_INDEX_EXISTS_SQL).fetchone()
        if action_row:
            action_count = conn.execute(ACTION_FTS_INDEX_DOC_COUNT_SQL).fetchone()[0]
    return {"exists": exists, "count": int(count), "action_count": int(action_count)}


__all__ = [
    "ensure_fts_index_sync",
    "fts_index_status_sync",
    "rebuild_fts_index_sync",
    "repair_fts_index_sync",
    "replace_fts_rows_for_messages_sync",
]

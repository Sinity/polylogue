"""Canonical FTS5 lifecycle helpers shared by sync and async surfaces."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence

import aiosqlite

FTS_MESSAGES_TABLE_SQL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        message_id UNINDEXED,
        conversation_id UNINDEXED,
        text,
        tokenize='unicode61'
    );
"""

FTS_INDEX_EXISTS_SQL = "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
FTS_INDEX_DOC_COUNT_SQL = "SELECT COUNT(*) FROM messages_fts_docsize"
FTS_REBUILD_SQL = """
    INSERT INTO messages_fts (rowid, message_id, conversation_id, text)
    SELECT messages.rowid, messages.message_id, messages.conversation_id, messages.text
    FROM messages
    WHERE messages.text IS NOT NULL
"""


def _chunked(items: Sequence[str], *, size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _delete_conversation_rows_sql(chunk_size: int) -> str:
    placeholders = ", ".join("?" for _ in range(chunk_size))
    return f"DELETE FROM messages_fts WHERE conversation_id IN ({placeholders})"


def _insert_conversation_rows_sql(chunk_size: int) -> str:
    placeholders = ", ".join("?" for _ in range(chunk_size))
    return f"""
        INSERT INTO messages_fts (rowid, message_id, conversation_id, text)
        SELECT messages.rowid, messages.message_id, messages.conversation_id, messages.text
        FROM messages
        WHERE messages.text IS NOT NULL AND messages.conversation_id IN ({placeholders})
    """


def ensure_fts_index_sync(conn: sqlite3.Connection) -> None:
    """Ensure the FTS5 table exists on a sync SQLite connection."""
    conn.execute(FTS_MESSAGES_TABLE_SQL)


async def ensure_fts_index_async(conn: aiosqlite.Connection) -> None:
    """Ensure the FTS5 table exists on an async SQLite connection."""
    await conn.execute(FTS_MESSAGES_TABLE_SQL)


def rebuild_fts_index_sync(conn: sqlite3.Connection) -> None:
    """Rebuild the full FTS index from persisted message rows."""
    ensure_fts_index_sync(conn)
    conn.execute("DELETE FROM messages_fts")
    conn.execute(FTS_REBUILD_SQL)


async def rebuild_fts_index_async(conn: aiosqlite.Connection) -> None:
    """Rebuild the full FTS index from persisted message rows."""
    await ensure_fts_index_async(conn)
    await conn.execute("DELETE FROM messages_fts")
    await conn.execute(FTS_REBUILD_SQL)


def repair_fts_index_sync(conn: sqlite3.Connection, conversation_ids: Sequence[str]) -> None:
    """Repair FTS rows for the supplied conversations from persisted message rows."""
    ensure_fts_index_sync(conn)
    if not conversation_ids:
        return

    for chunk in _chunked(list(conversation_ids), size=500):
        params = tuple(chunk)
        conn.execute(_delete_conversation_rows_sql(len(chunk)), params)
        conn.execute(_insert_conversation_rows_sql(len(chunk)), params)


def replace_fts_rows_for_messages_sync(
    conn: sqlite3.Connection,
    messages: Sequence[tuple[str, str, str | None]] | Sequence[object],
) -> None:
    """Replace FTS rows for the supplied message payloads.

    Each message object must expose ``message_id``, ``conversation_id``, and ``text``.
    When the message already exists in ``messages``, its SQLite rowid is preserved so
    future schema triggers continue to update the matching FTS row correctly.
    """
    ensure_fts_index_sync(conn)
    if not messages:
        return

    def _message_id(message: object) -> str:
        return getattr(message, "message_id")

    def _conversation_id(message: object) -> str:
        return getattr(message, "conversation_id")

    def _text(message: object) -> str | None:
        return getattr(message, "text")

    conversation_ids = sorted({_conversation_id(message) for message in messages})
    for chunk in _chunked(conversation_ids, size=500):
        conn.execute(_delete_conversation_rows_sql(len(chunk)), tuple(chunk))

    message_ids = [_message_id(message) for message in messages]
    rowids_by_message_id: dict[str, int] = {}
    for chunk in _chunked(message_ids, size=500):
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT rowid, message_id FROM messages WHERE message_id IN ({placeholders})",
            tuple(chunk),
        ).fetchall()
        rowids_by_message_id.update({row["message_id"]: row["rowid"] for row in rows})

    with_rowid: list[tuple[int, str, str, str]] = []
    without_rowid: list[tuple[str, str, str]] = []
    for message in messages:
        text = _text(message)
        if not text:
            continue
        message_id = _message_id(message)
        conversation_id = _conversation_id(message)
        rowid = rowids_by_message_id.get(message_id)
        if rowid is not None:
            with_rowid.append((rowid, message_id, conversation_id, text))
        else:
            without_rowid.append((message_id, conversation_id, text))

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


async def repair_fts_index_async(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
) -> None:
    """Repair FTS rows for the supplied conversations from persisted message rows."""
    await ensure_fts_index_async(conn)
    if not conversation_ids:
        return

    for chunk in _chunked(list(conversation_ids), size=500):
        params = tuple(chunk)
        await conn.execute(_delete_conversation_rows_sql(len(chunk)), params)
        await conn.execute(_insert_conversation_rows_sql(len(chunk)), params)


def fts_index_status_sync(conn: sqlite3.Connection) -> dict[str, object]:
    """Return existence and document count for the FTS index."""
    row = conn.execute(FTS_INDEX_EXISTS_SQL).fetchone()
    exists = bool(row)
    count = 0
    if exists:
        count = conn.execute(FTS_INDEX_DOC_COUNT_SQL).fetchone()[0]
    return {"exists": exists, "count": int(count)}


async def fts_index_status_async(conn: aiosqlite.Connection) -> dict[str, object]:
    """Return existence and document count for the FTS index."""
    row = await (await conn.execute(FTS_INDEX_EXISTS_SQL)).fetchone()
    exists = bool(row)
    count = 0
    if exists:
        count_row = await (await conn.execute(FTS_INDEX_DOC_COUNT_SQL)).fetchone()
        count = count_row[0] if count_row else 0
    return {"exists": exists, "count": int(count)}


__all__ = [
    "ensure_fts_index_sync",
    "ensure_fts_index_async",
    "rebuild_fts_index_sync",
    "rebuild_fts_index_async",
    "repair_fts_index_sync",
    "repair_fts_index_async",
    "replace_fts_rows_for_messages_sync",
    "fts_index_status_sync",
    "fts_index_status_async",
]

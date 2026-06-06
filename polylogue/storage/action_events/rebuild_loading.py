"""Chunking and record-loading helpers for action-event rebuilds."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Iterable, Sequence

import aiosqlite

# Re-export canonical chunked from polylogue.core.common.
from polylogue.core.common import chunked  # noqa: F401
from polylogue.storage.runtime import ContentBlockRecord, MessageRecord, SessionRecord
from polylogue.storage.sqlite.queries.mappers import (
    _row_to_content_block,
    _row_to_message,
)

_ALL_ACTION_EVENT_SESSION_IDS_SQL = "SELECT session_id FROM sessions ORDER BY COALESCE(sort_key, 0) DESC, session_id"
_ACTION_EVENT_SESSION_SQL_TEMPLATE = """
SELECT
    session_id,
    source_name,
    provider_session_id,
    content_hash
FROM sessions
WHERE session_id IN ({placeholders})
"""


def iter_session_id_pages_sync(
    conn: sqlite3.Connection,
    *,
    page_size: int,
) -> Iterable[list[str]]:
    cursor = conn.execute(_ALL_ACTION_EVENT_SESSION_IDS_SQL)
    while True:
        rows = cursor.fetchmany(page_size)
        if not rows:
            break
        yield [str(row["session_id"]) for row in rows]


async def iter_session_id_pages_async(
    conn: aiosqlite.Connection,
    *,
    page_size: int,
) -> AsyncIterator[list[str]]:
    cursor = await conn.execute(_ALL_ACTION_EVENT_SESSION_IDS_SQL)
    while True:
        rows = list(await cursor.fetchmany(page_size))
        if not rows:
            break
        yield [str(row["session_id"]) for row in rows]


def _row_to_action_event_session(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        session_id=row["session_id"],
        source_name=row["source_name"],
        provider_session_id=row["provider_session_id"],
        content_hash=row["content_hash"],
    )


def load_sync_batch(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
) -> tuple[list[SessionRecord], list[MessageRecord], list[ContentBlockRecord]]:
    placeholders = ", ".join("?" for _ in session_ids)
    sessions = [
        _row_to_action_event_session(row)
        for row in conn.execute(
            _ACTION_EVENT_SESSION_SQL_TEMPLATE.format(placeholders=placeholders),
            tuple(session_ids),
        ).fetchall()
    ]
    messages = [
        _row_to_message(row)
        for row in conn.execute(
            f"""
            SELECT *
            FROM messages
            WHERE session_id IN ({placeholders})
            ORDER BY session_id, sort_key, message_id
            """,
            tuple(session_ids),
        ).fetchall()
    ]
    blocks = [
        _row_to_content_block(row)
        for row in conn.execute(
            f"""
            SELECT *
            FROM content_blocks
            WHERE session_id IN ({placeholders})
            ORDER BY session_id, message_id, block_index
            """,
            tuple(session_ids),
        ).fetchall()
    ]
    return sessions, messages, blocks


async def load_async_batch(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
) -> tuple[list[SessionRecord], list[MessageRecord], list[ContentBlockRecord]]:
    placeholders = ", ".join("?" for _ in session_ids)
    sessions = [
        _row_to_action_event_session(row)
        for row in await (
            await conn.execute(
                _ACTION_EVENT_SESSION_SQL_TEMPLATE.format(placeholders=placeholders),
                tuple(session_ids),
            )
        ).fetchall()
    ]
    messages = [
        _row_to_message(row)
        for row in await (
            await conn.execute(
                f"""
                SELECT *
                FROM messages
                WHERE session_id IN ({placeholders})
                ORDER BY session_id, sort_key, message_id
                """,
                tuple(session_ids),
            )
        ).fetchall()
    ]
    blocks = [
        _row_to_content_block(row)
        for row in await (
            await conn.execute(
                f"""
                SELECT *
                FROM content_blocks
                WHERE session_id IN ({placeholders})
                ORDER BY session_id, message_id, block_index
                """,
                tuple(session_ids),
            )
        ).fetchall()
    ]
    return sessions, messages, blocks

"""Chunking and record-loading helpers for action-event rebuilds."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Iterable, Sequence

import aiosqlite

from polylogue.storage.backends.queries.mappers import (
    _row_to_content_block,
    _row_to_message,
)
from polylogue.storage.store import ContentBlockRecord, ConversationRecord, MessageRecord

_ALL_ACTION_EVENT_CONVERSATION_IDS_SQL = (
    "SELECT conversation_id FROM conversations ORDER BY COALESCE(sort_key, 0) DESC, conversation_id"
)
_ACTION_EVENT_CONVERSATION_SQL_TEMPLATE = """
SELECT
    conversation_id,
    provider_name,
    provider_conversation_id,
    content_hash
FROM conversations
WHERE conversation_id IN ({placeholders})
"""


def chunked(items: Sequence[str], *, size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def iter_conversation_id_pages_sync(
    conn: sqlite3.Connection,
    *,
    page_size: int,
) -> Iterable[list[str]]:
    cursor = conn.execute(_ALL_ACTION_EVENT_CONVERSATION_IDS_SQL)
    while True:
        rows = cursor.fetchmany(page_size)
        if not rows:
            break
        yield [str(row["conversation_id"]) for row in rows]


async def iter_conversation_id_pages_async(
    conn: aiosqlite.Connection,
    *,
    page_size: int,
) -> AsyncIterator[list[str]]:
    cursor = await conn.execute(_ALL_ACTION_EVENT_CONVERSATION_IDS_SQL)
    while True:
        rows = await cursor.fetchmany(page_size)
        if not rows:
            break
        yield [str(row["conversation_id"]) for row in rows]


def _row_to_action_event_conversation(row: sqlite3.Row) -> ConversationRecord:
    return ConversationRecord(
        conversation_id=row["conversation_id"],
        provider_name=row["provider_name"],
        provider_conversation_id=row["provider_conversation_id"],
        content_hash=row["content_hash"],
    )


def load_sync_batch(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
) -> tuple[list[ConversationRecord], list[MessageRecord], list[ContentBlockRecord]]:
    placeholders = ", ".join("?" for _ in conversation_ids)
    conversations = [
        _row_to_action_event_conversation(row)
        for row in conn.execute(
            _ACTION_EVENT_CONVERSATION_SQL_TEMPLATE.format(placeholders=placeholders),
            tuple(conversation_ids),
        ).fetchall()
    ]
    messages = [
        _row_to_message(row)
        for row in conn.execute(
            f"""
            SELECT *
            FROM messages
            WHERE conversation_id IN ({placeholders})
            ORDER BY conversation_id, sort_key, message_id
            """,
            tuple(conversation_ids),
        ).fetchall()
    ]
    blocks = [
        _row_to_content_block(row)
        for row in conn.execute(
            f"""
            SELECT *
            FROM content_blocks
            WHERE conversation_id IN ({placeholders})
            ORDER BY conversation_id, message_id, block_index
            """,
            tuple(conversation_ids),
        ).fetchall()
    ]
    return conversations, messages, blocks


async def load_async_batch(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
) -> tuple[list[ConversationRecord], list[MessageRecord], list[ContentBlockRecord]]:
    placeholders = ", ".join("?" for _ in conversation_ids)
    conversations = [
        _row_to_action_event_conversation(row)
        for row in await (
            await conn.execute(
                _ACTION_EVENT_CONVERSATION_SQL_TEMPLATE.format(placeholders=placeholders),
                tuple(conversation_ids),
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
                WHERE conversation_id IN ({placeholders})
                ORDER BY conversation_id, sort_key, message_id
                """,
                tuple(conversation_ids),
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
                WHERE conversation_id IN ({placeholders})
                ORDER BY conversation_id, message_id, block_index
                """,
                tuple(conversation_ids),
            )
        ).fetchall()
    ]
    return conversations, messages, blocks

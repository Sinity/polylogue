"""Chunking and record-loading helpers for action-event rebuilds."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence

import aiosqlite

from polylogue.storage.backends.queries.mappers import (
    _row_to_content_block,
    _row_to_conversation,
    _row_to_message,
)
from polylogue.storage.store import ContentBlockRecord, ConversationRecord, MessageRecord


def chunked(items: Sequence[str], *, size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def load_sync_batch(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
) -> tuple[list[ConversationRecord], list[MessageRecord], list[ContentBlockRecord]]:
    placeholders = ", ".join("?" for _ in conversation_ids)
    conversations = [
        _row_to_conversation(row)
        for row in conn.execute(
            f"SELECT * FROM conversations WHERE conversation_id IN ({placeholders})",
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
        _row_to_conversation(row)
        for row in await (
            await conn.execute(
                f"SELECT * FROM conversations WHERE conversation_id IN ({placeholders})",
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

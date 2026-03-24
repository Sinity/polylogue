"""Batch loading and hydration for session-product lifecycle flows."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence

import aiosqlite

from polylogue.storage.action_event_rows import attach_blocks_to_messages
from polylogue.storage.backends.queries.attachments import get_attachments_batch
from polylogue.storage.backends.queries.mappers import (
    _row_to_content_block,
    _row_to_conversation,
    _row_to_message,
)
from polylogue.storage.hydrators import conversation_from_records
from polylogue.storage.store import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
)

_ALL_CONVERSATION_IDS_SQL = "SELECT conversation_id FROM conversations ORDER BY COALESCE(sort_key, 0) DESC, conversation_id"


def chunked(items: Sequence[str], *, size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def sync_attachment_batch(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
) -> dict[str, list[AttachmentRecord]]:
    if not conversation_ids:
        return {}
    placeholders = ", ".join("?" for _ in conversation_ids)
    rows = conn.execute(
        f"""
        SELECT a.*, r.message_id, r.conversation_id
        FROM attachments a
        JOIN attachment_refs r ON a.attachment_id = r.attachment_id
        WHERE r.conversation_id IN ({placeholders})
        """,
        tuple(conversation_ids),
    ).fetchall()
    result: dict[str, list[AttachmentRecord]] = {conversation_id: [] for conversation_id in conversation_ids}
    for row in rows:
        conversation_id = str(row["conversation_id"])
        result.setdefault(conversation_id, []).append(
            AttachmentRecord(
                attachment_id=row["attachment_id"],
                conversation_id=conversation_id,
                message_id=row["message_id"],
                mime_type=row["mime_type"],
                size_bytes=row["size_bytes"],
                path=row["path"],
                provider_meta=None,
            )
        )
    return result


def load_sync_batch(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
) -> tuple[list[ConversationRecord], list[MessageRecord], dict[str, list[AttachmentRecord]], list[ContentBlockRecord]]:
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
    return conversations, messages, sync_attachment_batch(conn, conversation_ids), blocks


async def load_async_batch(
    conn: aiosqlite.Connection,
    conversation_ids: Sequence[str],
) -> tuple[list[ConversationRecord], list[MessageRecord], dict[str, list[AttachmentRecord]], list[ContentBlockRecord]]:
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
    attachments = await get_attachments_batch(conn, list(conversation_ids))
    return conversations, messages, attachments, blocks


def hydrate_conversations(
    conversations: list[ConversationRecord],
    messages: list[MessageRecord],
    attachments_by_conversation: dict[str, list[AttachmentRecord]],
    blocks: list[ContentBlockRecord],
) -> list[object]:
    messages_by_conversation: dict[str, list[MessageRecord]] = defaultdict(list)
    blocks_by_conversation: dict[str, list[ContentBlockRecord]] = defaultdict(list)
    for message in messages:
        messages_by_conversation[str(message.conversation_id)].append(message)
    for block in blocks:
        blocks_by_conversation[str(block.conversation_id)].append(block)

    hydrated: list[object] = []
    for conversation in conversations:
        conversation_id = str(conversation.conversation_id)
        attached_messages = attach_blocks_to_messages(
            messages_by_conversation.get(conversation_id, []),
            blocks_by_conversation.get(conversation_id, []),
        )
        hydrated.append(
            conversation_from_records(
                conversation,
                attached_messages,
                attachments_by_conversation.get(conversation_id, []),
            )
        )
    return hydrated


__all__ = [
    "_ALL_CONVERSATION_IDS_SQL",
    "chunked",
    "hydrate_conversations",
    "load_async_batch",
    "load_sync_batch",
    "sync_attachment_batch",
]

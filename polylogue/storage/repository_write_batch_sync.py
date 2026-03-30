"""Sync batch write path for pipeline bulk operations.

Bypasses aiosqlite entirely — each sqlite3 execute() call is ~0.01ms
vs aiosqlite's ~1.2ms thread-crossing overhead. For 10 calls per
conversation × 9,688 conversations = 96,880 calls, this saves ~116s.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.lib.json import dumps as json_dumps
from polylogue.logging import get_logger
from polylogue.storage.action_event_rows import attach_blocks_to_messages, build_action_event_records
from polylogue.storage.store import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    _json_array_or_none,
    _json_or_none,
)

logger = get_logger(__name__)


def save_conversation_sync(
    conn: sqlite3.Connection,
    conversation: ConversationRecord,
    messages: list[MessageRecord],
    attachments: list[AttachmentRecord],
    content_blocks: list[ContentBlockRecord] | None = None,
) -> dict[str, int]:
    """Save a conversation and all its records using sync SQLite.

    No aiosqlite thread-crossing — direct sqlite3 calls.
    """
    counts = {
        "conversations": 0,
        "messages": 0,
        "attachments": 0,
        "skipped_conversations": 0,
        "skipped_messages": 0,
        "skipped_attachments": 0,
    }

    # Hash check
    row = conn.execute(
        "SELECT content_hash FROM conversations WHERE conversation_id = ?",
        (conversation.conversation_id,),
    ).fetchone()
    existing_hash = row["content_hash"] if row else None
    content_unchanged = existing_hash is not None and existing_hash == conversation.content_hash

    # Save conversation record (INSERT OR UPDATE)
    conn.execute(
        """
        INSERT INTO conversations (
            conversation_id, provider_name, provider_conversation_id,
            title, created_at, updated_at, sort_key, content_hash,
            provider_meta, metadata, version, parent_conversation_id,
            branch_type, raw_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(conversation_id) DO UPDATE SET
            title = excluded.title,
            created_at = excluded.created_at,
            updated_at = excluded.updated_at,
            sort_key = excluded.sort_key,
            content_hash = excluded.content_hash,
            provider_meta = excluded.provider_meta,
            parent_conversation_id = excluded.parent_conversation_id,
            branch_type = excluded.branch_type,
            raw_id = COALESCE(excluded.raw_id, conversations.raw_id)
        WHERE
            content_hash != excluded.content_hash
            OR IFNULL(title, '') != IFNULL(excluded.title, '')
            OR IFNULL(created_at, '') != IFNULL(excluded.created_at, '')
            OR IFNULL(updated_at, '') != IFNULL(excluded.updated_at, '')
            OR IFNULL(provider_meta, '') != IFNULL(excluded.provider_meta, '')
            OR IFNULL(parent_conversation_id, '') != IFNULL(excluded.parent_conversation_id, '')
            OR IFNULL(branch_type, '') != IFNULL(excluded.branch_type, '')
            OR IFNULL(raw_id, '') != IFNULL(excluded.raw_id, '')
        """,
        (
            conversation.conversation_id,
            conversation.provider_name,
            conversation.provider_conversation_id,
            conversation.title,
            conversation.created_at,
            conversation.updated_at,
            conversation.sort_key,
            conversation.content_hash,
            _json_or_none(conversation.provider_meta),
            _json_or_none(conversation.metadata) or "{}",
            conversation.version,
            conversation.parent_conversation_id,
            conversation.branch_type,
            conversation.raw_id,
        ),
    )

    if content_unchanged:
        counts["skipped_conversations"] = 1
        counts["skipped_messages"] = len(messages)
        counts["skipped_attachments"] = len(attachments)
        return counts

    counts["conversations"] = 1

    if messages:
        # Save messages (executemany — fast)
        conn.executemany(
            """
            INSERT OR REPLACE INTO messages (
                message_id, conversation_id, provider_message_id,
                role, text, sort_key, content_hash, version,
                parent_message_id, branch_index, provider_name,
                word_count, has_tool_use, has_thinking
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    m.message_id, m.conversation_id, m.provider_message_id,
                    m.role, m.text, m.sort_key, m.content_hash, 1,
                    m.parent_message_id, m.branch_index, m.provider_name,
                    m.word_count, m.has_tool_use, m.has_thinking,
                )
                for m in messages
            ],
        )
        counts["messages"] = len(messages)

        # Upsert conversation stats
        pname = conversation.provider_name or ""
        msg_count = len(messages)
        word_count = sum(m.word_count for m in messages)
        tool_use_count = sum(1 for m in messages if m.has_tool_use)
        thinking_count = sum(1 for m in messages if m.has_thinking)
        conn.execute(
            """
            INSERT INTO conversation_stats (conversation_id, provider_name, message_count, word_count, tool_use_count, thinking_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                provider_name = excluded.provider_name,
                message_count = excluded.message_count,
                word_count = excluded.word_count,
                tool_use_count = excluded.tool_use_count,
                thinking_count = excluded.thinking_count
            """,
            (conversation.conversation_id, pname, msg_count, word_count, tool_use_count, thinking_count),
        )

    # Save content blocks
    all_blocks: list[ContentBlockRecord] = list(content_blocks or [])
    for message in messages:
        all_blocks.extend(message.content_blocks)
    if all_blocks:
        conn.executemany(
            """
            INSERT OR REPLACE INTO content_blocks (
                block_id, message_id, conversation_id, block_index,
                type, text, tool_name, tool_id, tool_input,
                media_type, metadata, semantic_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    b.block_id, b.message_id, b.conversation_id, b.block_index,
                    b.type, b.text, b.tool_name, b.tool_id, b.tool_input,
                    b.media_type, b.metadata, b.semantic_type,
                )
                for b in all_blocks
            ],
        )

    # Action events
    action_messages = attach_blocks_to_messages(messages, all_blocks)
    action_records = build_action_event_records(conversation, action_messages)
    conn.execute("DELETE FROM action_events WHERE conversation_id = ?", (conversation.conversation_id,))
    if action_records:
        conn.executemany(
            """
            INSERT INTO action_events (
                event_id, conversation_id, message_id, materializer_version,
                source_block_id, timestamp, sort_key, sequence_index,
                provider_name, action_kind, tool_name, normalized_tool_name,
                tool_id, affected_paths_json, cwd_path, branch_names_json,
                command, query_text, url, output_text, search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.event_id, r.conversation_id, r.message_id,
                    r.materializer_version, r.source_block_id,
                    r.timestamp, r.sort_key, r.sequence_index,
                    r.provider_name, r.action_kind, r.tool_name,
                    r.normalized_tool_name, r.tool_id,
                    _json_array_or_none(r.affected_paths),
                    r.cwd_path,
                    _json_array_or_none(r.branch_names),
                    r.command, r.query_text, r.url, r.output_text,
                    r.search_text,
                )
                for r in action_records
            ],
        )

    # Attachments
    conn.execute(
        "DELETE FROM attachment_refs WHERE conversation_id = ? AND attachment_id NOT IN ({})".format(
            ",".join("?" for _ in attachments)
        ) if attachments else "DELETE FROM attachment_refs WHERE conversation_id = ?",
        [conversation.conversation_id] + [str(a.attachment_id) for a in attachments]
        if attachments else [conversation.conversation_id],
    )
    # Simplified: just insert attachments
    for att in attachments:
        conn.execute(
            "INSERT OR REPLACE INTO attachments (attachment_id, mime_type, size_bytes, path, ref_count, provider_meta) VALUES (?, ?, ?, ?, 1, ?)",
            (str(att.attachment_id), att.mime_type, att.size_bytes, att.path, _json_or_none(att.provider_meta)),
        )
        counts["attachments"] += 1

    return counts


__all__ = ["save_conversation_sync"]

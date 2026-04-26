"""Conversation write/delete helpers."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.store import ConversationRecord, _json_or_none


async def conversation_exists_by_hash(conn: aiosqlite.Connection, content_hash: str) -> bool:
    cursor = await conn.execute(
        "SELECT 1 FROM conversations WHERE content_hash = ? LIMIT 1",
        (content_hash,),
    )
    row = await cursor.fetchone()
    return row is not None


async def save_conversation_record(
    conn: aiosqlite.Connection,
    record: ConversationRecord,
    transaction_depth: int,
) -> None:
    await conn.execute(
        """
        INSERT INTO conversations (
            conversation_id,
            provider_name,
            provider_conversation_id,
            title,
            created_at,
            updated_at,
            sort_key,
            content_hash,
            provider_meta,
            metadata,
            version,
            parent_conversation_id,
            branch_type,
            raw_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(conversation_id) DO UPDATE SET
            title = excluded.title,
            created_at = excluded.created_at,
            updated_at = excluded.updated_at,
            sort_key = excluded.sort_key,
            content_hash = excluded.content_hash,
            provider_meta = excluded.provider_meta,
            metadata = COALESCE(excluded.metadata, conversations.metadata),
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
            record.conversation_id,
            record.provider_name,
            record.provider_conversation_id,
            record.title,
            record.created_at,
            record.updated_at,
            record.sort_key,
            record.content_hash,
            _json_or_none(record.provider_meta),
            _json_or_none(record.metadata) or "{}",
            record.version,
            record.parent_conversation_id,
            record.branch_type,
            record.raw_id,
        ),
    )
    if transaction_depth == 0:
        await conn.commit()


async def delete_conversation_sql(
    conn: aiosqlite.Connection,
    conversation_id: str,
    transaction_depth: int,
) -> bool:
    cursor = await conn.execute(
        "SELECT parent_conversation_id FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return False

    parent_conversation_id = row[0]

    await conn.execute(
        """
        UPDATE conversations
        SET parent_conversation_id = ?
        WHERE parent_conversation_id = ?
        """,
        (parent_conversation_id, conversation_id),
    )

    cursor = await conn.execute(
        """SELECT DISTINCT ar.attachment_id FROM attachment_refs ar
           JOIN messages m ON ar.message_id = m.message_id
           WHERE m.conversation_id = ?""",
        (conversation_id,),
    )
    affected_attachments = [r[0] for r in await cursor.fetchall()]

    await conn.execute(
        "DELETE FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    )

    if affected_attachments:
        placeholders = ",".join("?" * len(affected_attachments))
        await conn.execute(
            f"""UPDATE attachments SET ref_count = (
                    SELECT COUNT(*) FROM attachment_refs
                    WHERE attachment_refs.attachment_id = attachments.attachment_id
                ) WHERE attachment_id IN ({placeholders})""",
            affected_attachments,
        )
        await conn.execute(
            f"DELETE FROM attachments WHERE attachment_id IN ({placeholders}) AND ref_count <= 0",
            affected_attachments,
        )

    if transaction_depth == 0:
        await conn.commit()
    return True


__all__ = [
    "conversation_exists_by_hash",
    "delete_conversation_sql",
    "save_conversation_record",
]

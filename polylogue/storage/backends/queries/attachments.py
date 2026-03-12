"""Attachment and content block queries."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.backends.queries.mappers import _parse_json, _row_to_content_block
from polylogue.storage.store import (
    AttachmentRecord,
    ContentBlockRecord,
    _json_or_none,
    _make_ref_id,
)
from polylogue.types import ConversationId

__all__ = [
    "get_content_blocks",
    "save_content_blocks",
    "get_attachments",
    "get_attachments_batch",
    "save_attachments",
    "prune_attachments",
]


async def get_content_blocks(
    conn: aiosqlite.Connection,
    message_ids: list[str],
) -> dict[str, list[ContentBlockRecord]]:
    """Get content blocks for a list of message IDs.

    Batches queries to stay under SQLite's 999-variable limit.
    """
    if not message_ids:
        return {}
    result: dict[str, list[ContentBlockRecord]] = {mid: [] for mid in message_ids}
    batch_size = 900  # stay well under SQLite's 999-variable limit
    for i in range(0, len(message_ids), batch_size):
        batch = message_ids[i : i + batch_size]
        placeholders = ",".join("?" for _ in batch)
        cursor = await conn.execute(
            f"SELECT * FROM content_blocks WHERE message_id IN ({placeholders}) ORDER BY message_id, block_index",
            batch,
        )
        rows = await cursor.fetchall()
        for row in rows:
            mid = row["message_id"]
            if mid in result:
                result[mid].append(_row_to_content_block(row))
    return result


async def save_content_blocks(
    conn: aiosqlite.Connection,
    records: list[ContentBlockRecord],
    transaction_depth: int,
) -> None:
    """Persist content block records using bulk insert."""
    if not records:
        return
    query = """
        INSERT INTO content_blocks (
            block_id,
            message_id,
            conversation_id,
            block_index,
            type,
            text,
            tool_name,
            tool_id,
            tool_input,
            media_type,
            metadata,
            semantic_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(message_id, block_index) DO UPDATE SET
            type = excluded.type,
            text = excluded.text,
            tool_name = excluded.tool_name,
            tool_id = excluded.tool_id,
            tool_input = excluded.tool_input,
            media_type = excluded.media_type,
            metadata = excluded.metadata,
            semantic_type = excluded.semantic_type
    """
    data = [
        (
            r.block_id,
            r.message_id,
            r.conversation_id,
            r.block_index,
            r.type,
            r.text,
            r.tool_name,
            r.tool_id,
            r.tool_input,
            r.media_type,
            r.metadata,
            r.semantic_type,
        )
        for r in records
    ]
    await conn.executemany(query, data)
    if transaction_depth == 0:
        await conn.commit()


async def get_attachments(
    conn: aiosqlite.Connection, conversation_id: str
) -> list[AttachmentRecord]:
    """Get all attachments for a conversation."""
    cursor = await conn.execute(
        """
        SELECT a.*, r.message_id
        FROM attachments a
        JOIN attachment_refs r ON a.attachment_id = r.attachment_id
        WHERE r.conversation_id = ?
        """,
        (conversation_id,),
    )
    rows = await cursor.fetchall()
    return [
        AttachmentRecord(
            attachment_id=row["attachment_id"],
            conversation_id=ConversationId(conversation_id),
            message_id=row["message_id"],
            mime_type=row["mime_type"],
            size_bytes=row["size_bytes"],
            path=row["path"],
            provider_meta=_parse_json(
                row["provider_meta"], field="provider_meta", record_id=row["attachment_id"]
            ),
        )
        for row in rows
    ]


async def get_attachments_batch(
    conn: aiosqlite.Connection, conversation_ids: list[str]
) -> dict[str, list[AttachmentRecord]]:
    """Get attachments for multiple conversations in a single query."""
    if not conversation_ids:
        return {}
    result: dict[str, list[AttachmentRecord]] = {cid: [] for cid in conversation_ids}
    placeholders = ",".join("?" for _ in conversation_ids)
    cursor = await conn.execute(
        f"""
        SELECT a.*, r.message_id, r.conversation_id
        FROM attachments a
        JOIN attachment_refs r ON a.attachment_id = r.attachment_id
        WHERE r.conversation_id IN ({placeholders})
        """,
        conversation_ids,
    )
    rows = await cursor.fetchall()
    for row in rows:
        cid = row["conversation_id"]
        if cid in result:
            result[cid].append(
                AttachmentRecord(
                    attachment_id=row["attachment_id"],
                    conversation_id=ConversationId(cid),
                    message_id=row["message_id"],
                    mime_type=row["mime_type"],
                    size_bytes=row["size_bytes"],
                    path=row["path"],
                    provider_meta=_parse_json(
                        row["provider_meta"],
                        field="provider_meta",
                        record_id=row["attachment_id"],
                    ),
                )
            )
    return result


async def save_attachments(
    conn: aiosqlite.Connection,
    records: list[AttachmentRecord],
    transaction_depth: int,
) -> None:
    """Persist attachment records with reference counting."""
    if not records:
        return
    att_query = """
        INSERT INTO attachments (
            attachment_id, mime_type, size_bytes, path, ref_count, provider_meta
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(attachment_id) DO UPDATE SET
            mime_type = COALESCE(excluded.mime_type, attachments.mime_type),
            size_bytes = COALESCE(excluded.size_bytes, attachments.size_bytes),
            path = COALESCE(excluded.path, attachments.path),
            provider_meta = COALESCE(excluded.provider_meta, attachments.provider_meta)
    """
    att_data = [
        (r.attachment_id, r.mime_type, r.size_bytes, r.path, 0, _json_or_none(r.provider_meta))
        for r in records
    ]
    await conn.executemany(att_query, att_data)

    ref_query = """
        INSERT OR IGNORE INTO attachment_refs (
            ref_id, attachment_id, conversation_id, message_id, provider_meta
        ) VALUES (?, ?, ?, ?, ?)
    """
    ref_data = []
    for r in records:
        ref_id = _make_ref_id(r.attachment_id, r.conversation_id, r.message_id)
        ref_data.append(
            (ref_id, r.attachment_id, r.conversation_id, r.message_id, _json_or_none(r.provider_meta))
        )
    await conn.executemany(ref_query, ref_data)

    affected_aids = list({r.attachment_id for r in records})
    placeholders = ", ".join("?" for _ in affected_aids)
    await conn.execute(
        f"""
        UPDATE attachments
        SET ref_count = (
            SELECT COUNT(*)
            FROM attachment_refs
            WHERE attachment_refs.attachment_id = attachments.attachment_id
        )
        WHERE attachment_id IN ({placeholders})
        """,
        tuple(affected_aids),
    )
    if transaction_depth == 0:
        await conn.commit()


async def prune_attachments(
    conn: aiosqlite.Connection,
    conversation_id: str,
    keep_attachment_ids: set[str],
    transaction_depth: int,
) -> None:
    """Remove attachment refs not in keep set and clean up orphaned attachments."""
    if keep_attachment_ids:
        placeholders = ",".join("?" * len(keep_attachment_ids))
        cursor = await conn.execute(
            f"""
            SELECT attachment_id FROM attachment_refs
            WHERE conversation_id = ? AND attachment_id NOT IN ({placeholders})
            """,
            (conversation_id, *keep_attachment_ids),
        )
        refs_to_remove = await cursor.fetchall()
    else:
        cursor = await conn.execute(
            "SELECT attachment_id FROM attachment_refs WHERE conversation_id = ?",
            (conversation_id,),
        )
        refs_to_remove = await cursor.fetchall()

    if not refs_to_remove:
        return

    attachment_ids_to_check = {row[0] for row in refs_to_remove}

    if keep_attachment_ids:
        placeholders = ",".join("?" * len(keep_attachment_ids))
        await conn.execute(
            f"DELETE FROM attachment_refs WHERE conversation_id = ? AND attachment_id NOT IN ({placeholders})",
            (conversation_id, *keep_attachment_ids),
        )
    else:
        await conn.execute(
            "DELETE FROM attachment_refs WHERE conversation_id = ?",
            (conversation_id,),
        )

    aids_list = list(attachment_ids_to_check)
    aid_placeholders = ", ".join("?" for _ in aids_list)
    await conn.execute(
        f"""
        UPDATE attachments
        SET ref_count = (
            SELECT COUNT(*)
            FROM attachment_refs
            WHERE attachment_refs.attachment_id = attachments.attachment_id
        )
        WHERE attachment_id IN ({aid_placeholders})
        """,
        tuple(aids_list),
    )
    await conn.execute("DELETE FROM attachments WHERE ref_count <= 0")

    if transaction_depth == 0:
        await conn.commit()

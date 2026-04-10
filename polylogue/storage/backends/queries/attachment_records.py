"""Attachment read-query helpers."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.backends.queries.mappers import _parse_json
from polylogue.storage.store import AttachmentRecord
from polylogue.types import ConversationId


def _build_attachment_record(row, *, conversation_id: str) -> AttachmentRecord:
    return AttachmentRecord(
        attachment_id=row["attachment_id"],
        conversation_id=ConversationId(conversation_id),
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


async def get_attachments(
    conn: aiosqlite.Connection,
    conversation_id: str,
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
    return [_build_attachment_record(row, conversation_id=conversation_id) for row in rows]


async def get_attachments_batch(
    conn: aiosqlite.Connection,
    conversation_ids: list[str],
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
            result[cid].append(_build_attachment_record(row, conversation_id=cid))
    return result


__all__ = [
    "get_attachments",
    "get_attachments_batch",
]

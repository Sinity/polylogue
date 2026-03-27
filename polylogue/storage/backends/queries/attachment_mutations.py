"""Attachment mutation helpers."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.store import AttachmentRecord, _json_or_none, _make_ref_id


async def save_attachments(
    conn: aiosqlite.Connection,
    records: list[AttachmentRecord],
    transaction_depth: int,
) -> None:
    """Persist attachment records with reference counting."""
    if not records:
        return
    await conn.executemany(
        """
        INSERT INTO attachments (
            attachment_id, mime_type, size_bytes, path, ref_count, provider_meta
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(attachment_id) DO UPDATE SET
            mime_type = COALESCE(excluded.mime_type, attachments.mime_type),
            size_bytes = COALESCE(excluded.size_bytes, attachments.size_bytes),
            path = COALESCE(excluded.path, attachments.path),
            provider_meta = COALESCE(excluded.provider_meta, attachments.provider_meta)
        """,
        [
            (
                record.attachment_id,
                record.mime_type,
                record.size_bytes,
                record.path,
                0,
                _json_or_none(record.provider_meta),
            )
            for record in records
        ],
    )

    ref_data = []
    for record in records:
        ref_data.append(
            (
                _make_ref_id(record.attachment_id, record.conversation_id, record.message_id),
                record.attachment_id,
                record.conversation_id,
                record.message_id,
                _json_or_none(record.provider_meta),
            )
        )
    await conn.executemany(
        """
        INSERT OR IGNORE INTO attachment_refs (
            ref_id, attachment_id, conversation_id, message_id, provider_meta
        ) VALUES (?, ?, ?, ?, ?)
        """,
        ref_data,
    )

    affected_ids = tuple({record.attachment_id for record in records})
    placeholders = ", ".join("?" for _ in affected_ids)
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
        affected_ids,
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


__all__ = [
    "prune_attachments",
    "save_attachments",
]

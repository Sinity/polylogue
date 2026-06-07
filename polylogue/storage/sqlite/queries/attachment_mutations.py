"""Attachment mutation helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import aiosqlite

from polylogue.storage.runtime import AttachmentRecord


def _blob_hash(value: str) -> bytes:
    try:
        blob_hash = bytes.fromhex(value)
    except ValueError:
        blob_hash = hashlib.sha256(value.encode("utf-8")).digest()
    if len(blob_hash) != 32:
        blob_hash = hashlib.sha256(value.encode("utf-8")).digest()
    return blob_hash


def _display_name(record: AttachmentRecord) -> str | None:
    if not record.path:
        return None
    name = Path(record.path).name
    return name or None


def _source_url(record: AttachmentRecord) -> str | None:
    if record.path and record.path.startswith(("http://", "https://")):
        return record.path
    return None


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
            attachment_id, display_name, media_type, byte_count, blob_hash, ref_count
        ) VALUES (?, ?, ?, ?, ?, 0)
        ON CONFLICT(attachment_id) DO UPDATE SET
            display_name = COALESCE(excluded.display_name, attachments.display_name),
            media_type = COALESCE(excluded.media_type, attachments.media_type),
            byte_count = MAX(attachments.byte_count, excluded.byte_count),
            blob_hash = excluded.blob_hash
        """,
        [
            (
                record.attachment_id,
                _display_name(record),
                record.mime_type,
                record.size_bytes or 0,
                _blob_hash(str(record.attachment_id)),
            )
            for record in records
        ],
    )

    next_position: dict[str, int] = {}
    for record in records:
        if record.message_id is None:
            raise ValueError("attachment refs require message_id in the current archive schema")
        message_id = str(record.message_id)
        if message_id not in next_position:
            cursor = await conn.execute(
                "SELECT COALESCE(MAX(position) + 1, 0) FROM attachment_refs WHERE message_id = ?",
                (message_id,),
            )
            row = await cursor.fetchone()
            next_position[message_id] = int(row[0] or 0) if row is not None else 0
        position = next_position[message_id]
        next_position[message_id] = position + 1
        await conn.execute(
            """
            INSERT INTO attachment_refs (
                attachment_id, session_id, message_id, position, upload_origin, source_url
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id, position) DO UPDATE SET
                attachment_id = excluded.attachment_id,
                session_id = excluded.session_id,
                upload_origin = excluded.upload_origin,
                source_url = excluded.source_url
            """,
            (
                record.attachment_id,
                record.session_id,
                record.message_id,
                position,
                record.upload_origin,
                _source_url(record),
            ),
        )
        cursor = await conn.execute(
            "SELECT ref_id FROM attachment_refs WHERE message_id = ? AND position = ?",
            (message_id, position),
        )
        ref_row = await cursor.fetchone()
        if ref_row is None:
            raise RuntimeError("attachment ref insert did not produce a ref_id")
        ref_id = str(ref_row["ref_id"])
        native_rows = [
            ("attachment", record.attachment_native_id),
            ("file", record.file_native_id),
            ("drive", record.drive_native_id),
        ]
        if record.path:
            native_rows.append(("source", record.path))
        await conn.executemany(
            """
            INSERT OR IGNORE INTO attachment_native_ids (ref_id, id_kind, native_id)
            VALUES (?, ?, ?)
            """,
            [(ref_id, kind, native_id) for kind, native_id in native_rows if native_id],
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
    session_id: str,
    keep_attachment_ids: set[str],
    transaction_depth: int,
) -> None:
    """Remove attachment refs not in keep set and clean up orphaned attachments."""
    if keep_attachment_ids:
        placeholders = ",".join("?" * len(keep_attachment_ids))
        cursor = await conn.execute(
            f"""
            SELECT attachment_id FROM attachment_refs
            WHERE session_id = ? AND attachment_id NOT IN ({placeholders})
            """,
            (session_id, *keep_attachment_ids),
        )
        refs_to_remove = list(await cursor.fetchall())
    else:
        cursor = await conn.execute(
            "SELECT attachment_id FROM attachment_refs WHERE session_id = ?",
            (session_id,),
        )
        refs_to_remove = list(await cursor.fetchall())

    if not refs_to_remove:
        return

    attachment_ids_to_check = {row[0] for row in refs_to_remove}
    if keep_attachment_ids:
        placeholders = ",".join("?" * len(keep_attachment_ids))
        await conn.execute(
            f"DELETE FROM attachment_refs WHERE session_id = ? AND attachment_id NOT IN ({placeholders})",
            (session_id, *keep_attachment_ids),
        )
    else:
        await conn.execute(
            "DELETE FROM attachment_refs WHERE session_id = ?",
            (session_id,),
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

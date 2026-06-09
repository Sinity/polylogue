"""Attachment mutation helpers."""

from __future__ import annotations

import aiosqlite


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
]

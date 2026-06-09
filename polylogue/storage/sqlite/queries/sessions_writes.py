"""Session write/delete helpers."""

from __future__ import annotations

import aiosqlite


async def session_exists_by_hash(conn: aiosqlite.Connection, content_hash: str) -> bool:
    cursor = await conn.execute(
        "SELECT 1 FROM sessions WHERE content_hash = ? LIMIT 1",
        (content_hash,),
    )
    row = await cursor.fetchone()
    return row is not None


async def delete_session_sql(
    conn: aiosqlite.Connection,
    session_id: str,
    transaction_depth: int,
) -> bool:
    cursor = await conn.execute(
        "SELECT parent_session_id FROM sessions WHERE session_id = ?",
        (session_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return False

    parent_session_id = row[0]

    await conn.execute(
        """
        UPDATE sessions
        SET parent_session_id = ?
        WHERE parent_session_id = ?
        """,
        (parent_session_id, session_id),
    )

    cursor = await conn.execute(
        """SELECT DISTINCT ar.attachment_id FROM attachment_refs ar
           JOIN messages m ON ar.message_id = m.message_id
           WHERE m.session_id = ?""",
        (session_id,),
    )
    affected_attachments = [r[0] for r in await cursor.fetchall()]

    await conn.execute(
        "DELETE FROM sessions WHERE session_id = ?",
        (session_id,),
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
    "session_exists_by_hash",
    "delete_session_sql",
]

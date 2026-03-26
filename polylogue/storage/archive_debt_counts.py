"""Archive cleanup debt counters."""

from __future__ import annotations

import sqlite3


def count_orphaned_messages_sync(conn: sqlite3.Connection) -> int:
    orphan_cids = conn.execute(
        """
        SELECT DISTINCT m.conversation_id FROM messages m
        WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = m.conversation_id)
        """
    ).fetchall()
    if not orphan_cids:
        return 0
    placeholders = ",".join("?" for _ in orphan_cids)
    return int(
        conn.execute(
            f"SELECT COUNT(*) FROM messages WHERE conversation_id IN ({placeholders})",
            [row[0] for row in orphan_cids],
        ).fetchone()[0]
    )


def count_orphaned_content_blocks_sync(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM content_blocks cb
            WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = cb.conversation_id)
               OR NOT EXISTS (SELECT 1 FROM messages m WHERE m.message_id = cb.message_id)
            """
        ).fetchone()[0]
    )


def count_empty_conversations_sync(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            """
            SELECT COUNT(*) FROM conversations c
            WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.conversation_id)
            """
        ).fetchone()[0]
    )


def count_orphaned_attachments_sync(conn: sqlite3.Connection) -> int:
    orphaned_refs = int(
        conn.execute(
            """
            SELECT COUNT(*) FROM attachment_refs ar
            WHERE (ar.message_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.message_id = ar.message_id))
               OR NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = ar.conversation_id)
            """
        ).fetchone()[0]
    )
    unreferenced_attachments = int(
        conn.execute(
            """
            SELECT COUNT(*) FROM attachments a
            WHERE NOT EXISTS (SELECT 1 FROM attachment_refs ar WHERE ar.attachment_id = a.attachment_id)
            """
        ).fetchone()[0]
    )
    return orphaned_refs + unreferenced_attachments


__all__ = [
    "count_empty_conversations_sync",
    "count_orphaned_attachments_sync",
    "count_orphaned_content_blocks_sync",
    "count_orphaned_messages_sync",
]

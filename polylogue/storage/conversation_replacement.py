"""Conversation-scoped replacement helpers for runtime archive rows."""

from __future__ import annotations

import sqlite3

import aiosqlite


def _table_exists_sync(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


async def _table_exists_async(conn: aiosqlite.Connection, table_name: str) -> bool:
    row = await (
        await conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ?",
            (table_name,),
        )
    ).fetchone()
    return row is not None


async def _ensure_sqlite_vec_async(conn: aiosqlite.Connection) -> bool:
    try:
        import sqlite_vec

        await conn.enable_load_extension(True)
        try:
            await conn.load_extension(sqlite_vec.loadable_path())
            return True
        finally:
            await conn.enable_load_extension(False)
    except ImportError:
        return False
    except Exception:
        return False


def _invalidate_embedding_state_sync(conn: sqlite3.Connection, conversation_id: str) -> None:
    if _table_exists_sync(conn, "embeddings_meta"):
        conn.execute(
            """
            DELETE FROM embeddings_meta
            WHERE (target_type = 'message' AND target_id IN (
                SELECT message_id FROM messages WHERE conversation_id = ?
            ))
               OR (target_type = 'conversation' AND target_id = ?)
            """,
            (conversation_id, conversation_id),
        )
    if _table_exists_sync(conn, "embedding_status"):
        conn.execute("DELETE FROM embedding_status WHERE conversation_id = ?", (conversation_id,))
    if _table_exists_sync(conn, "message_embeddings"):
        conn.execute("DELETE FROM message_embeddings WHERE conversation_id = ?", (conversation_id,))


async def _invalidate_embedding_state_async(conn: aiosqlite.Connection, conversation_id: str) -> None:
    if await _table_exists_async(conn, "embeddings_meta"):
        await conn.execute(
            """
            DELETE FROM embeddings_meta
            WHERE (target_type = 'message' AND target_id IN (
                SELECT message_id FROM messages WHERE conversation_id = ?
            ))
               OR (target_type = 'conversation' AND target_id = ?)
            """,
            (conversation_id, conversation_id),
        )
    if await _table_exists_async(conn, "embedding_status"):
        await conn.execute("DELETE FROM embedding_status WHERE conversation_id = ?", (conversation_id,))
    if await _table_exists_async(conn, "message_embeddings") and await _ensure_sqlite_vec_async(conn):
        await conn.execute("DELETE FROM message_embeddings WHERE conversation_id = ?", (conversation_id,))


def replace_conversation_runtime_state_sync(
    conn: sqlite3.Connection,
    conversation_id: str,
) -> set[str]:
    """Remove message-scoped runtime rows before re-materializing a conversation."""
    affected_attachment_ids = {
        str(row[0])
        for row in conn.execute(
            "SELECT DISTINCT attachment_id FROM attachment_refs WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchall()
    }
    _invalidate_embedding_state_sync(conn, conversation_id)
    conn.execute("DELETE FROM attachment_refs WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM conversation_stats WHERE conversation_id = ?", (conversation_id,))
    return affected_attachment_ids


async def replace_conversation_runtime_state_async(
    conn: aiosqlite.Connection,
    conversation_id: str,
) -> set[str]:
    """Remove message-scoped runtime rows before re-materializing a conversation."""
    rows = await (
        await conn.execute(
            "SELECT DISTINCT attachment_id FROM attachment_refs WHERE conversation_id = ?",
            (conversation_id,),
        )
    ).fetchall()
    affected_attachment_ids = {str(row[0]) for row in rows}
    await _invalidate_embedding_state_async(conn, conversation_id)
    await conn.execute("DELETE FROM attachment_refs WHERE conversation_id = ?", (conversation_id,))
    await conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    await conn.execute("DELETE FROM conversation_stats WHERE conversation_id = ?", (conversation_id,))
    return affected_attachment_ids


def recount_and_prune_attachments_sync(
    conn: sqlite3.Connection,
    attachment_ids: set[str],
) -> None:
    """Recompute ref counts for affected attachments and remove orphans."""
    if not attachment_ids:
        return
    ids = tuple(sorted(attachment_ids))
    placeholders = ", ".join("?" for _ in ids)
    conn.execute(
        f"""
        UPDATE attachments
        SET ref_count = (
            SELECT COUNT(*)
            FROM attachment_refs
            WHERE attachment_refs.attachment_id = attachments.attachment_id
        )
        WHERE attachment_id IN ({placeholders})
        """,
        ids,
    )
    conn.execute(
        f"DELETE FROM attachments WHERE attachment_id IN ({placeholders}) AND ref_count <= 0",
        ids,
    )


async def recount_and_prune_attachments_async(
    conn: aiosqlite.Connection,
    attachment_ids: set[str],
) -> None:
    """Recompute ref counts for affected attachments and remove orphans."""
    if not attachment_ids:
        return
    ids = tuple(sorted(attachment_ids))
    placeholders = ", ".join("?" for _ in ids)
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
        ids,
    )
    await conn.execute(
        f"DELETE FROM attachments WHERE attachment_id IN ({placeholders}) AND ref_count <= 0",
        ids,
    )


__all__ = [
    "recount_and_prune_attachments_async",
    "recount_and_prune_attachments_sync",
    "replace_conversation_runtime_state_async",
    "replace_conversation_runtime_state_sync",
]

"""Session-scoped replacement helpers for runtime archive rows."""

from __future__ import annotations

import sqlite3

import aiosqlite

from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec_async


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


def _trigger_exists_sync(conn: sqlite3.Connection, trigger_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = ?",
        (trigger_name,),
    ).fetchone()
    return row is not None


async def _trigger_exists_async(conn: aiosqlite.Connection, trigger_name: str) -> bool:
    row = await (
        await conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = ?",
            (trigger_name,),
        )
    ).fetchone()
    return row is not None


async def _ensure_sqlite_vec_async(conn: aiosqlite.Connection) -> bool:
    loaded, _error = await try_load_sqlite_vec_async(conn)
    return loaded


def _purge_message_fts_sync(conn: sqlite3.Connection, session_id: str) -> None:
    if _trigger_exists_sync(conn, "messages_fts_ad"):
        return
    if not _table_exists_sync(conn, "messages_fts") or not _table_exists_sync(conn, "messages_fts_docsize"):
        return
    from polylogue.storage.fts.sql import delete_session_rows_sql

    conn.execute(delete_session_rows_sql(1), (session_id,))


async def _purge_message_fts_async(conn: aiosqlite.Connection, session_id: str) -> None:
    if await _trigger_exists_async(conn, "messages_fts_ad"):
        return
    if not await _table_exists_async(conn, "messages_fts") or not await _table_exists_async(
        conn, "messages_fts_docsize"
    ):
        return
    from polylogue.storage.fts.sql import delete_session_rows_sql

    await conn.execute(delete_session_rows_sql(1), (session_id,))


def _invalidate_embedding_state_sync(conn: sqlite3.Connection, session_id: str) -> None:
    if _table_exists_sync(conn, "message_embeddings_meta"):
        conn.execute(
            """
            DELETE FROM message_embeddings_meta
            WHERE message_id IN (
                SELECT message_id FROM messages WHERE session_id = ?
            )
            """,
            (session_id,),
        )
    if _table_exists_sync(conn, "message_embeddings"):
        conn.execute("DELETE FROM message_embeddings WHERE session_id = ?", (session_id,))


async def _invalidate_embedding_state_async(conn: aiosqlite.Connection, session_id: str) -> None:
    if await _table_exists_async(conn, "message_embeddings_meta"):
        await conn.execute(
            """
            DELETE FROM message_embeddings_meta
            WHERE message_id IN (
                SELECT message_id FROM messages WHERE session_id = ?
            )
            """,
            (session_id,),
        )
    if await _table_exists_async(conn, "message_embeddings") and await _ensure_sqlite_vec_async(conn):
        await conn.execute("DELETE FROM message_embeddings WHERE session_id = ?", (session_id,))


def replace_session_runtime_state_sync(
    conn: sqlite3.Connection,
    session_id: str,
) -> set[str]:
    """Remove message-scoped runtime rows before re-materializing a session."""
    affected_attachment_ids = {
        str(row[0])
        for row in conn.execute(
            "SELECT DISTINCT attachment_id FROM attachment_refs WHERE session_id = ?",
            (session_id,),
        ).fetchall()
    }
    _invalidate_embedding_state_sync(conn, session_id)
    conn.execute("DELETE FROM attachment_refs WHERE session_id = ?", (session_id,))
    _purge_message_fts_sync(conn, session_id)
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    return affected_attachment_ids


async def replace_session_runtime_state_async(
    conn: aiosqlite.Connection,
    session_id: str,
) -> set[str]:
    """Remove message-scoped runtime rows before re-materializing a session."""
    rows = await (
        await conn.execute(
            "SELECT DISTINCT attachment_id FROM attachment_refs WHERE session_id = ?",
            (session_id,),
        )
    ).fetchall()
    affected_attachment_ids = {str(row[0]) for row in rows}
    await _invalidate_embedding_state_async(conn, session_id)
    await conn.execute("DELETE FROM attachment_refs WHERE session_id = ?", (session_id,))
    await _purge_message_fts_async(conn, session_id)
    await conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
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
    "replace_session_runtime_state_async",
    "replace_session_runtime_state_sync",
]

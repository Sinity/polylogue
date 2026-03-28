"""Async full-text search index management for SQLite.

Provides async/await API for FTS5 index creation, rebuilding, and status checking.
All operations use the SQLiteBackend for non-blocking database access.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from polylogue.storage.backends.async_sqlite import SQLiteBackend


async def async_ensure_index(backend: SQLiteBackend) -> None:
    """Create FTS5 index table if it doesn't exist.

    Args:
        backend: Async SQLite backend instance
    """
    async with backend._get_connection() as conn:
        await conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                message_id UNINDEXED,
                conversation_id UNINDEXED,
                content
            );
            """
        )


async def async_rebuild_index(backend: SQLiteBackend) -> None:
    """Rebuild the entire FTS5 search index from scratch.

    Args:
        backend: Async SQLite backend instance
    """
    async with backend._get_connection() as conn:
        await async_ensure_index(backend)
        await conn.execute("DELETE FROM messages_fts")
        await conn.execute(
            """
            INSERT INTO messages_fts (message_id, conversation_id, content)
            SELECT messages.message_id, messages.conversation_id, messages.text
            FROM messages
            WHERE messages.text IS NOT NULL
            """
        )
        await conn.commit()


async def async_update_index_for_conversations(
    conversation_ids: Sequence[str], backend: SQLiteBackend
) -> None:
    """Update FTS5 search index for specific conversations.

    Optimized for batch operations using a single delete then batch insert.

    Args:
        conversation_ids: List of conversation IDs to re-index
        backend: Async SQLite backend instance
    """
    if not conversation_ids:
        return

    async with backend._get_connection() as conn:
        await async_ensure_index(backend)

        all_ids = list(conversation_ids)
        placeholders = ", ".join("?" for _ in all_ids)

        # SQLite FTS Update - single delete then batch insert
        await conn.execute(
            f"DELETE FROM messages_fts WHERE conversation_id IN ({placeholders})",
            tuple(all_ids),
        )

        # Fetch all messages to index in one query
        cursor = await conn.execute(
            f"""
            SELECT message_id, conversation_id, text
            FROM messages
            WHERE text IS NOT NULL AND conversation_id IN ({placeholders})
            """,
            tuple(all_ids),
        )
        message_rows = await cursor.fetchall()

        # Build batch for executemany
        fts_batch = [(row["message_id"], row["conversation_id"], row["text"]) for row in message_rows]

        if fts_batch:
            await conn.executemany(
                "INSERT INTO messages_fts (message_id, conversation_id, content) VALUES (?, ?, ?)",
                fts_batch,
            )

        await conn.commit()


def _chunked(items: Sequence[str], *, size: int) -> Iterable[Sequence[str]]:
    """Split a sequence into chunks of given size."""
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


async def async_index_status(backend: SQLiteBackend) -> dict[str, object]:
    """Get FTS5 index status information.

    Args:
        backend: Async SQLite backend instance

    Returns:
        Dictionary with 'exists' (bool) and 'count' (int) keys
    """
    async with backend._get_connection() as conn:
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
        )
        row = await cursor.fetchone()
        exists = bool(row)
        count = 0
        if exists:
            # COUNT(*) on FTS virtual table is O(N) and extremely slow (minutes on large DBs).
            # The backing docsize table has one row per indexed document and counts instantly.
            cursor = await conn.execute("SELECT COUNT(*) FROM messages_fts_docsize")
            row = await cursor.fetchone()
            count = row[0] if row else 0
        return {"exists": exists, "count": int(count)}


__all__ = [
    "async_ensure_index",
    "async_rebuild_index",
    "async_update_index_for_conversations",
    "async_index_status",
]

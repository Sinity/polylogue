from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Sequence

from .db import connection_context, open_connection
from .search_providers import create_search_provider, create_vector_provider
from .store import MessageRecord


def ensure_index(conn) -> None:
    """Create FTS5 index table if it doesn't exist.

    This function is maintained for backward compatibility. New code should
    use the search provider abstraction.

    Args:
        conn: Active SQLite database connection
    """
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            message_id UNINDEXED,
            conversation_id UNINDEXED,
            provider_name UNINDEXED,
            content
        );
        """
    )


def rebuild_index(conn: sqlite3.Connection | None = None) -> None:
    """Rebuild the entire search index from scratch.

    This function is maintained for backward compatibility. It rebuilds both
    FTS5 and Qdrant indexes if configured.

    Args:
        conn: Optional SQLite connection. If None, creates a new connection.
    """
    def _do(db_conn):
        ensure_index(db_conn)
        db_conn.execute("DELETE FROM messages_fts")
        db_conn.execute(
            """
            INSERT INTO messages_fts (message_id, conversation_id, provider_name, content)
            SELECT messages.message_id, messages.conversation_id, conversations.provider_name, messages.text
            FROM messages
            JOIN conversations ON conversations.conversation_id = messages.conversation_id
            WHERE messages.text IS NOT NULL
            """
        )

        # Optional Qdrant support via vector provider
        vector_provider = create_vector_provider()
        if vector_provider:
            from .index_qdrant import update_qdrant_for_conversations

            rows = db_conn.execute("SELECT conversation_id FROM conversations").fetchall()
            ids = [row["conversation_id"] for row in rows]
            update_qdrant_for_conversations(ids, db_conn)

        db_conn.commit()

    with connection_context(conn) as db_conn:
        _do(db_conn)


def update_index_for_conversations(conversation_ids: Sequence[str], conn: sqlite3.Connection | None = None) -> None:
    """Update search indexes for specific conversations.

    This function is maintained for backward compatibility. It updates both
    FTS5 and Qdrant indexes if configured.

    Args:
        conversation_ids: List of conversation IDs to re-index
        conn: Optional SQLite connection. If None, creates a new connection.
    """
    if not conversation_ids:
        return

    def _do(db_conn):
        ensure_index(db_conn)
        # SQLite FTS Update
        for chunk in _chunked(conversation_ids, size=200):
            placeholders = ", ".join("?" for _ in chunk)
            db_conn.execute(
                f"DELETE FROM messages_fts WHERE conversation_id IN ({placeholders})",
                tuple(chunk),
            )
            db_conn.execute(
                f"""
                INSERT INTO messages_fts (message_id, conversation_id, provider_name, content)
                SELECT messages.message_id, messages.conversation_id, conversations.provider_name, messages.text
                FROM messages
                JOIN conversations ON conversations.conversation_id = messages.conversation_id
                WHERE messages.text IS NOT NULL AND messages.conversation_id IN ({placeholders})
                """,
                tuple(chunk),
            )

        # Optional Qdrant support via vector provider
        vector_provider = create_vector_provider()
        if vector_provider:
            from .index_qdrant import update_qdrant_for_conversations

            update_qdrant_for_conversations(conversation_ids, db_conn)

        db_conn.commit()

    with connection_context(conn) as db_conn:
        _do(db_conn)


def _chunked(items: Sequence[str], *, size: int) -> Iterable[Sequence[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def index_status() -> dict:
    with open_connection(None) as conn:
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()
        exists = bool(row)
        count = 0
        if exists:
            count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        return {"exists": exists, "count": int(count)}


__all__ = [
    "rebuild_index",
    "update_index_for_conversations",
    "index_status",
    "ensure_index",
]

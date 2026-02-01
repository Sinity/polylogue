from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence

from .backends.sqlite import connection_context, open_connection
from .search_providers import create_vector_provider


def ensure_index(conn: sqlite3.Connection) -> None:
    """Create FTS5 index table if it doesn't exist.

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

    Rebuilds both FTS5 and Qdrant indexes if configured.

    Args:
        conn: Optional SQLite connection. If None, creates a new connection.
    """

    def _do(db_conn: sqlite3.Connection) -> None:
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

    Updates both FTS5 and Qdrant indexes if configured.
    Optimized for batch operations using a single provider_map query and executemany.

    Args:
        conversation_ids: List of conversation IDs to re-index
        conn: Optional SQLite connection. If None, creates a new connection.
    """
    if not conversation_ids:
        return

    def _do(db_conn: sqlite3.Connection) -> None:
        ensure_index(db_conn)

        # Build provider_map in a single query for all conversation IDs
        all_ids = list(conversation_ids)
        placeholders = ", ".join("?" for _ in all_ids)
        provider_map_rows = db_conn.execute(
            f"SELECT conversation_id, provider_name FROM conversations WHERE conversation_id IN ({placeholders})",
            tuple(all_ids),
        ).fetchall()
        provider_map = {row["conversation_id"]: row["provider_name"] for row in provider_map_rows}

        # SQLite FTS Update - single delete then batch insert
        db_conn.execute(
            f"DELETE FROM messages_fts WHERE conversation_id IN ({placeholders})",
            tuple(all_ids),
        )

        # Fetch all messages to index in one query
        message_rows = db_conn.execute(
            f"""
            SELECT message_id, conversation_id, text
            FROM messages
            WHERE text IS NOT NULL AND conversation_id IN ({placeholders})
            """,
            tuple(all_ids),
        ).fetchall()

        # Build batch for executemany
        fts_batch = [
            (row["message_id"], row["conversation_id"], provider_map.get(row["conversation_id"], ""), row["text"])
            for row in message_rows
            if row["conversation_id"] in provider_map
        ]

        if fts_batch:
            db_conn.executemany(
                "INSERT INTO messages_fts (message_id, conversation_id, provider_name, content) VALUES (?, ?, ?, ?)",
                fts_batch,
            )

        # Optional Qdrant support via vector provider
        vector_provider = create_vector_provider()
        if vector_provider:
            from .index_qdrant import update_qdrant_for_conversations

            update_qdrant_for_conversations(all_ids, db_conn)

        db_conn.commit()

    with connection_context(conn) as db_conn:
        _do(db_conn)


def _chunked(items: Sequence[str], *, size: int) -> Iterable[Sequence[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def index_status() -> dict[str, object]:
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

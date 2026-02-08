from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence

from .backends.sqlite import connection_context, open_connection


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
            content
        );
        """
    )


def rebuild_index(conn: sqlite3.Connection | None = None) -> None:
    """Rebuild the entire FTS5 search index from scratch.

    Args:
        conn: Optional SQLite connection. If None, creates a new connection.
    """

    def _do(db_conn: sqlite3.Connection) -> None:
        ensure_index(db_conn)
        db_conn.execute("DELETE FROM messages_fts")
        db_conn.execute(
            """
            INSERT INTO messages_fts (message_id, conversation_id, content)
            SELECT messages.message_id, messages.conversation_id, messages.text
            FROM messages
            WHERE messages.text IS NOT NULL
            """
        )
        db_conn.commit()

    with connection_context(conn) as db_conn:
        _do(db_conn)


def update_index_for_conversations(conversation_ids: Sequence[str], conn: sqlite3.Connection | None = None) -> None:
    """Update FTS5 search index for specific conversations.

    Optimized for batch operations using a single delete then batch insert.

    Args:
        conversation_ids: List of conversation IDs to re-index
        conn: Optional SQLite connection. If None, creates a new connection.
    """
    if not conversation_ids:
        return

    def _do(db_conn: sqlite3.Connection) -> None:
        ensure_index(db_conn)

        all_ids = list(conversation_ids)
        placeholders = ", ".join("?" for _ in all_ids)

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
        fts_batch = [(row["message_id"], row["conversation_id"], row["text"]) for row in message_rows]

        if fts_batch:
            db_conn.executemany(
                "INSERT INTO messages_fts (message_id, conversation_id, content) VALUES (?, ?, ?)",
                fts_batch,
            )

        db_conn.commit()

    with connection_context(conn) as db_conn:
        _do(db_conn)


def _chunked(items: Sequence[str], *, size: int) -> Iterable[Sequence[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def index_status(conn: sqlite3.Connection | None = None) -> dict[str, object]:
    def _query(c: sqlite3.Connection) -> dict[str, object]:
        row = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()
        exists = bool(row)
        count = 0
        if exists:
            count = c.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        return {"exists": exists, "count": int(count)}

    if conn is not None:
        return _query(conn)
    with open_connection(None) as fallback_conn:
        return _query(fallback_conn)


__all__ = [
    "rebuild_index",
    "update_index_for_conversations",
    "index_status",
    "ensure_index",
]

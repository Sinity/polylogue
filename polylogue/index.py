from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Sequence

from .db import open_connection


def ensure_index(conn) -> None:
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


import sqlite3


def rebuild_index(conn: sqlite3.Connection | None = None) -> None:
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
        # Optional Qdrant support
        if os.environ.get("QDRANT_URL") or os.environ.get("QDRANT_API_KEY"):
            from .index_qdrant import update_qdrant_for_conversations

            rows = db_conn.execute("SELECT conversation_id FROM conversations").fetchall()
            ids = [row["conversation_id"] for row in rows]
            update_qdrant_for_conversations(ids, db_conn)
        db_conn.commit()

    if conn:
        _do(conn)
    else:
        with open_connection(None) as new_conn:
            _do(new_conn)


def update_index_for_conversations(conversation_ids: Sequence[str], conn: sqlite3.Connection | None = None) -> None:
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
        # Optional Qdrant support
        if os.environ.get("QDRANT_URL") or os.environ.get("QDRANT_API_KEY"):
            from .index_qdrant import update_qdrant_for_conversations

            update_qdrant_for_conversations(conversation_ids, db_conn)
        db_conn.commit()

    if conn:
        _do(conn)
    else:
        with open_connection(None) as new_conn:
            _do(new_conn)


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

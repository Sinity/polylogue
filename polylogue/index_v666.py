from __future__ import annotations

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


def rebuild_index() -> None:
    with open_connection(None) as conn:
        ensure_index(conn)
        conn.execute("DELETE FROM messages_fts")
        conn.execute(
            """
            INSERT INTO messages_fts (message_id, conversation_id, provider_name, content)
            SELECT messages.message_id, messages.conversation_id, conversations.provider_name, messages.text
            FROM messages
            JOIN conversations ON conversations.conversation_id = messages.conversation_id
            WHERE messages.text IS NOT NULL
            """
        )
        conn.commit()


def index_status() -> dict:
    with open_connection(None) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
        ).fetchone()
        exists = bool(row)
        count = 0
        if exists:
            count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        return {"exists": exists, "count": int(count)}


__all__ = ["rebuild_index", "index_status", "ensure_index"]

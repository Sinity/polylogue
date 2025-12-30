from __future__ import annotations

from typing import Iterable, Sequence

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


def update_index_for_conversations(conversation_ids: Sequence[str]) -> None:
    if not conversation_ids:
        return
    with open_connection(None) as conn:
        ensure_index(conn)
        for chunk in _chunked(conversation_ids, size=200):
            placeholders = ", ".join("?" for _ in chunk)
            conn.execute(
                f"DELETE FROM messages_fts WHERE conversation_id IN ({placeholders})",
                tuple(chunk),
            )
            conn.execute(
                f"""
                INSERT INTO messages_fts (message_id, conversation_id, provider_name, content)
                SELECT messages.message_id, messages.conversation_id, conversations.provider_name, messages.text
                FROM messages
                JOIN conversations ON conversations.conversation_id = messages.conversation_id
                WHERE messages.text IS NOT NULL AND messages.conversation_id IN ({placeholders})
                """,
                tuple(chunk),
            )
        conn.commit()


def _chunked(items: Sequence[str], *, size: int) -> Iterable[Sequence[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


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

__all__ = [
    "rebuild_index",
    "update_index_for_conversations",
    "index_status",
    "ensure_index",
]

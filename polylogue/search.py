from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .db import open_connection


@dataclass
class SearchHit:
    conversation_id: str
    provider_name: str
    message_id: str
    title: Optional[str]
    timestamp: Optional[str]
    snippet: str
    conversation_path: Path


@dataclass
class SearchResult:
    hits: List[SearchHit]


def search_messages(query: str, *, archive_root: Path, limit: int = 20) -> SearchResult:
    with open_connection(None) as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
        ).fetchone()
        if not exists:
            raise RuntimeError("Search index not built. Run `polylogue run` with index enabled.")
        rows = conn.execute(
            """
            SELECT
                messages_fts.message_id,
                messages_fts.conversation_id,
                messages_fts.provider_name,
                conversations.title,
                messages.timestamp,
                snippet(messages_fts, 3, '[', ']', '…', 12) AS snippet
            FROM messages_fts
            JOIN conversations ON conversations.conversation_id = messages_fts.conversation_id
            JOIN messages ON messages.message_id = messages_fts.message_id
            WHERE messages_fts MATCH ?
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()

    hits: List[SearchHit] = []
    for row in rows:
        conversation_path = (
            archive_root / "render" / row["provider_name"] / row["conversation_id"] / "conversation.md"
        )
        hits.append(
            SearchHit(
                conversation_id=row["conversation_id"],
                provider_name=row["provider_name"],
                message_id=row["message_id"],
                title=row["title"],
                timestamp=row["timestamp"],
                snippet=row["snippet"],
                conversation_path=conversation_path,
            )
        )
    return SearchResult(hits=hits)


__all__ = ["SearchHit", "SearchResult", "search_messages"]

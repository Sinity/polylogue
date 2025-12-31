from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import List, Optional

from .db import open_connection
from .render_paths import legacy_render_root, render_root


@dataclass
class SearchHit:
    conversation_id: str
    provider_name: str
    source_name: Optional[str]
    message_id: str
    title: Optional[str]
    timestamp: Optional[str]
    snippet: str
    conversation_path: Path


@dataclass
class SearchResult:
    hits: List[SearchHit]


def _resolve_conversation_path(
    archive_root: Path,
    provider_name: str,
    conversation_id: str,
) -> Path:
    safe_root = render_root(archive_root, provider_name, conversation_id)
    safe_md = safe_root / "conversation.md"
    if safe_md.exists():
        return safe_md
    legacy_root = legacy_render_root(archive_root, provider_name, conversation_id)
    if legacy_root:
        legacy_md = legacy_root / "conversation.md"
        if legacy_md.exists():
            return legacy_md
    return safe_md


def search_messages(query: str, *, archive_root: Path, limit: int = 20) -> SearchResult:
    with open_connection(None) as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
        ).fetchone()
        if not exists:
            raise RuntimeError("Search index not built. Run `polylogue run` with index enabled.")
        try:
            rows = conn.execute(
                """
                SELECT
                    messages_fts.message_id,
                    messages_fts.conversation_id,
                    messages_fts.provider_name,
                    conversations.provider_meta,
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
        except sqlite3.Error as exc:
            raise RuntimeError(f"Invalid search query: {exc}") from exc

    hits: List[SearchHit] = []
    for row in rows:
        conversation_path = _resolve_conversation_path(
            archive_root,
            row["provider_name"],
            row["conversation_id"],
        )
        source_name = None
        meta = row["provider_meta"]
        if isinstance(meta, str) and meta:
            try:
                payload = json.loads(meta)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                source_name = payload.get("source")
        hits.append(
            SearchHit(
                conversation_id=row["conversation_id"],
                provider_name=row["provider_name"],
                source_name=source_name,
                message_id=row["message_id"],
                title=row["title"],
                timestamp=row["timestamp"],
                snippet=row["snippet"],
                conversation_path=conversation_path,
            )
        )
    return SearchResult(hits=hits)


__all__ = ["SearchHit", "SearchResult", "search_messages"]

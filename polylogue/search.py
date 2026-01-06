from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
    render_root_path: Optional[Path],
    provider_name: str,
    conversation_id: str,
) -> Path:
    output_root = render_root_path or (archive_root / "render")
    safe_root = render_root(output_root, provider_name, conversation_id)
    safe_md = safe_root / "conversation.md"
    if safe_md.exists():
        return safe_md
    legacy_root = legacy_render_root(output_root, provider_name, conversation_id)
    if legacy_root:
        legacy_md = legacy_root / "conversation.md"
        if legacy_md.exists():
            return legacy_md
    return safe_md


def search_messages(
    query: str,
    *,
    archive_root: Path,
    render_root_path: Optional[Path] = None,
    limit: int = 20,
    source: Optional[str] = None,
    since: Optional[str] = None,
) -> SearchResult:
    with open_connection(None) as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
        ).fetchone()
        if not exists:
            raise RuntimeError("Search index not built. Run `polylogue run` with index enabled.")
        
        sql = """
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
        """
        params: List[object] = [query]
        
        if source:
            # We filter by provider_name OR source in metadata
            # Note: exact match on provider_name is fast, metadata check is slower
            # For simplicity, we just check provider_name here.
            # If 'source' refers to the config source name, it often matches provider_name
            # except for multiple sources of same provider.
            # To support 'source' properly we need to check provider_meta.
            sql += " AND (messages_fts.provider_name = ? OR json_extract(conversations.provider_meta, '$.source') = ?)"
            params.extend([source, source])
            
        if since:
            try:
                dt = datetime.fromisoformat(since)
                # messages.timestamp can be ISO string or float-like string.
                # Standardize on string comparison if ISO, or basic string compare.
                # Assuming timestamp is ISO string in DB.
                sql += " AND messages.timestamp >= ?"
                params.append(since)
            except ValueError:
                pass # Ignore invalid date

        sql += " LIMIT ?"
        params.append(limit)

        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise RuntimeError(f"Invalid search query: {exc}") from exc

    hits: List[SearchHit] = []
    for row in rows:
        conversation_path = _resolve_conversation_path(
            archive_root,
            render_root_path,
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

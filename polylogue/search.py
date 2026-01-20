from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .db import DatabaseError, open_connection
from .render_paths import legacy_render_root, render_root

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    conversation_id: str
    provider_name: str
    source_name: str | None
    message_id: str
    title: str | None
    timestamp: str | None
    snippet: str
    conversation_path: Path


@dataclass
class SearchResult:
    hits: list[SearchHit]


def _resolve_conversation_path(
    archive_root: Path,
    render_root_path: Path | None,
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
    render_root_path: Path | None = None,
    limit: int = 20,
    source: str | None = None,
    since: str | None = None,
) -> SearchResult:
    with open_connection(None) as conn:
        exists = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()
        if not exists:
            raise DatabaseError("Search index not built. Run `polylogue run` with index enabled.")

        sql = """
            SELECT
                messages_fts.message_id,
                messages_fts.conversation_id,
                messages_fts.provider_name,
                conversations.provider_meta,
                conversations.source_name,
                conversations.title,
                messages.timestamp,
                snippet(messages_fts, 3, '[', ']', '…', 12) AS snippet
            FROM messages_fts
            JOIN conversations ON conversations.conversation_id = messages_fts.conversation_id
            JOIN messages ON messages.message_id = messages_fts.message_id
            WHERE messages_fts MATCH ?
        """
        params: list[object] = [query]

        if source:
            # Filter by provider_name OR source_name (computed column from provider_meta)
            # The computed column with index makes this much faster than json_extract
            sql += " AND (messages_fts.provider_name = ? OR conversations.source_name = ?)"
            params.extend([source, source])

        if since:
            try:
                since_dt = datetime.fromisoformat(since)
            except ValueError as exc:
                raise ValueError(f"Invalid --since date '{since}': {exc}. Use ISO format (e.g., 2023-01-01)") from exc
            since_ts = since_dt.timestamp()
            # messages.timestamp can be ISO string or float-like string.
            # Use a CASE expression to normalize to numeric comparison:
            # - If timestamp looks numeric, cast directly
            # - Otherwise, convert ISO string to unix timestamp via strftime
            sql += """
                AND CASE
                    WHEN messages.timestamp GLOB '*[^0-9.]*'
                    THEN CAST(strftime('%s', messages.timestamp) AS REAL)
                    ELSE CAST(messages.timestamp AS REAL)
                END >= ?
            """
            params.append(since_ts)

        sql += " LIMIT ?"
        params.append(limit)

        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Invalid search query: {exc}") from exc

    hits: list[SearchHit] = []
    for row in rows:
        conversation_path = _resolve_conversation_path(
            archive_root,
            render_root_path,
            row["provider_name"],
            row["conversation_id"],
        )
        # Use computed source_name column directly instead of parsing JSON
        source_name = row["source_name"] if "source_name" in row.keys() else None
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

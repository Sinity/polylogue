"""Cached runtime execution for FTS-backed message search."""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

from polylogue.errors import DatabaseError
from polylogue.maintenance.targets import build_maintenance_target_catalog
from polylogue.storage.fts.fts_lifecycle import check_fts_readiness, message_fts_search_readiness_sync
from polylogue.storage.search.cache import SearchCacheKey
from polylogue.storage.search.models import SearchHit, SearchResult
from polylogue.storage.search.query_builders import build_ranked_session_search_query, session_web_url
from polylogue.storage.search.query_support import sort_key_to_iso
from polylogue.storage.sqlite.connection import open_read_connection

_MAINTENANCE_TARGET_CATALOG = build_maintenance_target_catalog()
_MESSAGE_SEARCH_REPAIR_HINT = _MAINTENANCE_TARGET_CATALOG.repair_hint(("dangling_fts",), include_run_all=True)


@lru_cache(maxsize=128)
def search_messages_cached(cache_key: SearchCacheKey) -> SearchResult:
    """Internal cached implementation of search_messages."""
    return search_messages_impl(
        query=cache_key.query,
        archive_root=Path(cache_key.archive_root),
        db_path=Path(cache_key.db_path) if cache_key.db_path else None,
        limit=cache_key.limit,
        source=cache_key.source,
        since=cache_key.since,
    )


def search_messages_impl(
    query: str,
    archive_root: Path,
    db_path: Path | None,
    limit: int,
    source: str | None,
    since: str | None,
) -> SearchResult:
    query_spec = build_ranked_session_search_query(
        query=query,
        limit=limit,
        scope_names=[source] if source else None,
        since=since,
        include_snippet=True,
    )
    if query_spec is None:
        return SearchResult(hits=[])

    sql, params = query_spec.sql, query_spec.params
    with open_read_connection(db_path) as conn:
        readiness = message_fts_search_readiness_sync(conn)
        check_fts_readiness(readiness, _MESSAGE_SEARCH_REPAIR_HINT)
        try:
            rows = conn.execute(sql, tuple(params)).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Invalid search query: {exc}") from exc
        fallback_snippets = {
            row["message_id"]: _fallback_snippet(conn, row["message_id"], query)
            for row in rows
            if row["snippet"] is None
        }

    hits: list[SearchHit] = []
    for row in rows:
        session_id = row["session_id"]
        hits.append(
            SearchHit(
                session_id=session_id,
                source_name=row["source_name"],
                message_id=row["message_id"],
                title=row["title"],
                timestamp=sort_key_to_iso(row["sort_key"]),
                snippet=str(row["snippet"] or fallback_snippets.get(row["message_id"]) or ""),
                session_url=session_web_url(session_id),
            )
        )
    return SearchResult(hits=hits)


def _fallback_snippet(conn: sqlite3.Connection, message_id: str, query: str) -> str | None:
    row = conn.execute("SELECT text FROM messages WHERE message_id = ?", (message_id,)).fetchone()
    parts: list[str] = []
    if row is not None and row["text"]:
        parts.append(str(row["text"]))
    block_rows = conn.execute(
        """
        SELECT text, tool_input, metadata
        FROM content_blocks
        WHERE message_id = ?
        ORDER BY block_index
        """,
        (message_id,),
    ).fetchall()
    for block in block_rows:
        parts.extend(str(block[key]) for key in ("text", "tool_input", "metadata") if block[key])
    text = "\n".join(parts).strip()
    if not text:
        return None
    needle = query.strip().strip('"').split()[0].lower() if query.strip() else ""
    offset = text.lower().find(needle) if needle else -1
    if offset < 0:
        return text[:200]
    start = max(0, offset - 80)
    end = min(len(text), offset + 120)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def search_messages(
    query: str,
    *,
    archive_root: Path,
    db_path: Path | None = None,
    limit: int = 20,
    source: str | None = None,
    since: str | None = None,
) -> SearchResult:
    """Search for messages using FTS5 full-text search."""
    cache_key = SearchCacheKey.create(
        query=query,
        archive_root=archive_root,
        db_path=db_path,
        limit=limit,
        source=source,
        since=since,
    )
    return search_messages_cached(cache_key)


__all__ = ["open_read_connection", "search_messages", "search_messages_cached", "search_messages_impl"]

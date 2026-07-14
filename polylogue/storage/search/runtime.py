"""Cached runtime execution for FTS-backed message search."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from polylogue.errors import DatabaseError
from polylogue.storage.fts.fts_lifecycle import check_fts_readiness, message_fts_search_readiness_sync
from polylogue.storage.search.cache import SearchCacheKey
from polylogue.storage.search.models import SearchHit, SearchResult
from polylogue.storage.search.query_builders import build_ranked_session_search_query, session_web_url
from polylogue.storage.search.query_support import normalize_fts5_query, sort_key_to_iso
from polylogue.storage.sqlite.connection import open_read_connection
from polylogue.storage.sqlite.introspection import (
    table_exists,
)

_MESSAGE_SEARCH_REPAIR_HINT = "Run `polylogued run`."


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
            if _table_exists(conn, "messages_fts") and _table_exists(conn, "blocks"):
                rows = _search_archive_blocks(conn, query=query, limit=limit, source=source, since=since)
            else:
                rows = conn.execute(sql, tuple(params)).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Invalid search query: {exc}") from exc
        try:
            fallback_snippets = {
                row["message_id"]: (
                    _row_text(row, "fallback_text") or _fallback_snippet(conn, row["message_id"], query)
                )
                for row in rows
                if not row["snippet"]
            }
        except sqlite3.Error as exc:
            raise DatabaseError(f"Invalid search query: {exc}") from exc

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


def _search_archive_blocks(
    conn: sqlite3.Connection,
    *,
    query: str,
    limit: int,
    source: str | None,
    since: str | None,
) -> list[sqlite3.Row]:
    normalized = normalize_fts5_query(query)
    if normalized is None:
        return []
    clauses = ["messages_fts MATCH ?"]
    params: list[object] = [normalized]
    if source is not None:
        clauses.append("s.origin = ?")
        params.append(source)
    if since is not None:
        # A row with no reliable timestamp anywhere in its fallback chain is
        # not evidence it falls outside a --since window -- include it
        # rather than let SQL's NULL propagation silently exclude it
        # (polylogue-s5mm, sort_key_ms COALESCE audit).
        clauses.append(
            "(COALESCE(m.occurred_at_ms, s.sort_key_ms, s.updated_at_ms, s.created_at_ms) IS NULL"
            " OR COALESCE(m.occurred_at_ms, s.sort_key_ms, s.updated_at_ms, s.created_at_ms) >= ?)"
        )
        try:
            since_ms = int(datetime.fromisoformat(since).timestamp() * 1000)
        except ValueError as exc:
            raise ValueError(f"Invalid --since date '{since}': {exc}. Use ISO format") from exc
        params.append(since_ms)
    params.append(limit)
    return conn.execute(
        f"""
        WITH candidate_hits AS (
            SELECT
                b.message_id AS message_id,
                b.session_id AS session_id,
                s.origin AS source_name,
                s.title AS title,
                COALESCE(m.occurred_at_ms, s.sort_key_ms, s.updated_at_ms, s.created_at_ms) / 1000.0 AS sort_key,
                b.search_text AS fallback_text,
                snippet(messages_fts, 4, '[', ']', '...', 24) AS snippet,
                bm25(messages_fts) AS relevance
            FROM messages_fts
            JOIN blocks AS b ON b.rowid = messages_fts.rowid
            JOIN messages AS m ON m.message_id = b.message_id
            JOIN sessions AS s ON s.session_id = b.session_id
            WHERE {" AND ".join(clauses)}
        ),
        ranked_hits AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY session_id
                    ORDER BY relevance ASC, sort_key DESC, message_id ASC
                ) AS session_rank
            FROM candidate_hits
        )
        SELECT message_id, session_id, source_name, title, sort_key, snippet, fallback_text
        FROM ranked_hits
        WHERE session_rank = 1
        ORDER BY relevance ASC, sort_key DESC, message_id ASC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()


def _fallback_snippet(conn: sqlite3.Connection, message_id: str, query: str) -> str | None:
    parts: list[str] = []
    if not _table_exists(conn, "blocks"):
        return None
    block_rows = conn.execute(
        """
        SELECT text, tool_input
        FROM blocks
        WHERE message_id = ?
        ORDER BY position
        """,
        (message_id,),
    ).fetchall()
    for block in block_rows:
        parts.extend(str(block[key]) for key in ("text", "tool_input") if block[key])
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


def _row_text(row: sqlite3.Row, key: str) -> str | None:
    try:
        value = row[key]
    except (IndexError, KeyError):
        return None
    return str(value) if value else None


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

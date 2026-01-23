from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from .db import DatabaseError, open_connection
from .search_cache import SearchCacheKey
from polylogue.render_paths import legacy_render_root, render_root

logger = logging.getLogger(__name__)

# FTS5 special characters that need escaping or quoting (+ is inclusion operator)
_FTS5_SPECIAL = re.compile(r'[":*^(){}[\]|&!+\-]')
# FTS5 boolean/special operators that should be treated as literals when alone
_FTS5_OPERATORS = {'AND', 'OR', 'NOT', 'NEAR'}
# Pattern to detect queries that are only asterisks (dangerous wildcard-only)
_ASTERISK_ONLY = re.compile(r'^\*+$')
# Pattern to detect dangerous operator positions (start, end, consecutive)
_OPERATOR_PATTERN = re.compile(r'\b(AND|OR|NOT|NEAR)\b', re.IGNORECASE)


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


def escape_fts5_query(query: str) -> str:
    """Escape a query string for safe use in FTS5 MATCH clause.

    Handles:
    - Special characters (quotes, operators, parentheses, etc.)
    - Boolean operators (AND, OR, NOT, NEAR) when used as literals
    - Unbalanced quotes and dangerous patterns
    - Empty queries

    Strategy:
    - Return empty string literal for empty/whitespace queries
    - Quote bare operators to treat them as literals
    - Quote any query containing FTS5 special characters
    - Escape internal quotes by doubling them
    - Pass through simple alphanumeric queries as-is

    Args:
        query: Raw user query string

    Returns:
        Escaped query safe for FTS5 MATCH clause

    Examples:
        escape_fts5_query('hello') -> 'hello'
        escape_fts5_query('AND') -> '"AND"'
        escape_fts5_query('"quoted"') -> '"\"quoted\""'
        escape_fts5_query('*') -> '""'
    """
    if not query or not query.strip():
        return '""'  # Empty query -> empty phrase

    query = query.strip()

    # Check for asterisk-only queries (dangerous wildcard-only pattern)
    if _ASTERISK_ONLY.match(query):
        return '""'  # Treat as empty query

    # Check for bare operators - these should be quoted to be treated as literals
    if query.upper() in _FTS5_OPERATORS:
        return f'"{query}"'

    # Check if query contains FTS5 special characters
    if _FTS5_SPECIAL.search(query):
        # Escape internal quotes by doubling them (FTS5 escaping)
        escaped = query.replace('"', '""')
        return f'"{escaped}"'

    # Check for dangerous operator positions that would cause FTS5 syntax errors:
    # - Query starting with an operator (e.g., "OR test")
    # - Query ending with an operator (e.g., "test OR")
    # - Consecutive operators (e.g., "a AND AND b")
    words = query.split()
    if len(words) > 1:
        # Check first/last word
        if words[0].upper() in _FTS5_OPERATORS or words[-1].upper() in _FTS5_OPERATORS:
            escaped = query.replace('"', '""')
            return f'"{escaped}"'
        # Check for consecutive operators
        for i in range(len(words) - 1):
            if words[i].upper() in _FTS5_OPERATORS and words[i + 1].upper() in _FTS5_OPERATORS:
                escaped = query.replace('"', '""')
                return f'"{escaped}"'

    # Simple alphanumeric query with spaces - safe as-is
    return query


@lru_cache(maxsize=128)
def _search_messages_cached(cache_key: SearchCacheKey) -> SearchResult:
    """Internal cached implementation of search_messages.

    Args:
        cache_key: Immutable cache key containing all search parameters

    Returns:
        SearchResult with hits
    """
    # Reconstruct parameters from cache key
    query = cache_key.query
    archive_root = Path(cache_key.archive_root)
    render_root_path = Path(cache_key.render_root_path) if cache_key.render_root_path else None
    limit = cache_key.limit
    source = cache_key.source
    since = cache_key.since

    return _search_messages_impl(query, archive_root, render_root_path, limit, source, since)


def _search_messages_impl(
    query: str,
    archive_root: Path,
    render_root_path: Path | None,
    limit: int,
    source: str | None,
    since: str | None,
) -> SearchResult:
    with open_connection(None) as conn:
        exists = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone()
        if not exists:
            raise DatabaseError("Search index not built. Run `polylogue run` with index enabled.")

        # Escape the query to avoid syntax errors with special characters
        fts_query = escape_fts5_query(query)

        # We fetch more than the limit to allow for deduplication in Python
        # since FTS5 doesn't easily support snippet() with GROUP BY.
        sql_limit = limit * 5

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
        params: list[object] = [fts_query]

        if source:
            # Case-insensitive comparison for provider_name or source_name
            sql += " AND (messages_fts.provider_name = ? COLLATE NOCASE OR conversations.source_name = ? COLLATE NOCASE)"
            params.extend([source, source])

        if since:
            try:
                since_dt = datetime.fromisoformat(since)
            except ValueError as exc:
                raise ValueError(f"Invalid --since date '{since}': {exc}. Use ISO format (e.g., 2023-01-01)") from exc
            since_ts = since_dt.timestamp()
            sql += """
                AND CASE
                    WHEN messages.timestamp GLOB '*[^0-9.]*'
                    THEN CAST(strftime('%s', messages.timestamp) AS REAL)
                    ELSE CAST(messages.timestamp AS REAL)
                END >= ?
            """
            params.append(since_ts)

        # Sort by rank (relevance)
        sql += " ORDER BY rank"

        sql += " LIMIT ?"
        params.append(sql_limit)

        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Invalid search query: {exc}") from exc

    hits: list[SearchHit] = []
    seen_conversations: set[str] = set()
    
    for row in rows:
        cid = row["conversation_id"]
        if cid in seen_conversations:
            continue
        
        seen_conversations.add(cid)
        
        conversation_path = _resolve_conversation_path(
            archive_root,
            render_root_path,
            row["provider_name"],
            cid,
        )
        # Use computed source_name column directly instead of parsing JSON
        source_name = row["source_name"] if "source_name" in row.keys() else None
        hits.append(
            SearchHit(
                conversation_id=cid,
                provider_name=row["provider_name"],
                source_name=source_name,
                message_id=row["message_id"],
                title=row["title"],
                timestamp=row["timestamp"],
                snippet=row["snippet"],
                conversation_path=conversation_path,
            )
        )
        
        if len(hits) >= limit:
            break
            
    return SearchResult(hits=hits)


def search_messages(
    query: str,
    *,
    archive_root: Path,
    render_root_path: Path | None = None,
    limit: int = 20,
    source: str | None = None,
    since: str | None = None,
) -> SearchResult:
    """Search for messages using FTS5 full-text search.

    This function uses an LRU cache for repeated queries. The cache is
    automatically invalidated when conversations are re-ingested.

    Args:
        query: Search query string (automatically escaped for FTS5)
        archive_root: Root directory for archived conversations
        render_root_path: Optional root for rendered output
        limit: Maximum number of results to return (default: 20)
        source: Optional source/provider filter
        since: Optional timestamp filter (ISO format)

    Returns:
        SearchResult with matching conversations

    Raises:
        DatabaseError: If search index doesn't exist
        ValueError: If since date is invalid
    """
    # Create cache key
    cache_key = SearchCacheKey.create(
        query=query,
        archive_root=archive_root,
        render_root_path=render_root_path,
        limit=limit,
        source=source,
        since=since,
    )

    # Use cached implementation
    return _search_messages_cached(cache_key)


__all__ = ["SearchHit", "SearchResult", "search_messages", "escape_fts5_query"]

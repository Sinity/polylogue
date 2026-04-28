"""Cached runtime execution for FTS-backed message search."""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

from polylogue.errors import DatabaseError
from polylogue.maintenance.targets import build_maintenance_target_catalog
from polylogue.storage.backends.connection import open_read_connection
from polylogue.storage.fts.fts_lifecycle import check_fts_readiness, message_fts_readiness_sync
from polylogue.storage.search.cache import SearchCacheKey
from polylogue.storage.search.models import SearchHit, SearchResult
from polylogue.storage.search.query_builders import build_ranked_conversation_search_query, resolve_conversation_path
from polylogue.storage.search.query_support import sort_key_to_iso

_MAINTENANCE_TARGET_CATALOG = build_maintenance_target_catalog()
_MESSAGE_SEARCH_REPAIR_HINT = _MAINTENANCE_TARGET_CATALOG.repair_hint(("dangling_fts",), include_run_all=True)


@lru_cache(maxsize=128)
def search_messages_cached(cache_key: SearchCacheKey) -> SearchResult:
    """Internal cached implementation of search_messages."""
    return search_messages_impl(
        query=cache_key.query,
        archive_root=Path(cache_key.archive_root),
        render_root_path=Path(cache_key.render_root_path) if cache_key.render_root_path else None,
        db_path=Path(cache_key.db_path) if cache_key.db_path else None,
        limit=cache_key.limit,
        source=cache_key.source,
        since=cache_key.since,
    )


def search_messages_impl(
    query: str,
    archive_root: Path,
    render_root_path: Path | None,
    db_path: Path | None,
    limit: int,
    source: str | None,
    since: str | None,
) -> SearchResult:
    query_spec = build_ranked_conversation_search_query(
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
        readiness = message_fts_readiness_sync(conn, exact_counts=True)
        check_fts_readiness(readiness, _MESSAGE_SEARCH_REPAIR_HINT)
        try:
            rows = conn.execute(sql, tuple(params)).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Invalid search query: {exc}") from exc

    hits: list[SearchHit] = []
    for row in rows:
        conversation_id = row["conversation_id"]
        hits.append(
            SearchHit(
                conversation_id=conversation_id,
                provider_name=row["provider_name"],
                source_name=row["source_name"],
                message_id=row["message_id"],
                title=row["title"],
                timestamp=sort_key_to_iso(row["sort_key"]),
                snippet=row["snippet"],
                conversation_path=resolve_conversation_path(
                    archive_root,
                    render_root_path,
                    row["provider_name"],
                    conversation_id,
                ),
            )
        )
    return SearchResult(hits=hits)


def search_messages(
    query: str,
    *,
    archive_root: Path,
    render_root_path: Path | None = None,
    db_path: Path | None = None,
    limit: int = 20,
    source: str | None = None,
    since: str | None = None,
) -> SearchResult:
    """Search for messages using FTS5 full-text search."""
    cache_key = SearchCacheKey.create(
        query=query,
        archive_root=archive_root,
        render_root_path=render_root_path,
        db_path=db_path,
        limit=limit,
        source=source,
        since=since,
    )
    return search_messages_cached(cache_key)


__all__ = ["open_read_connection", "search_messages", "search_messages_cached", "search_messages_impl"]

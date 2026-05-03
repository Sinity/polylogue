"""Public FTS5 search surface."""

from __future__ import annotations

from pathlib import Path

from polylogue.storage.search import runtime as _search_runtime
from polylogue.storage.search.models import SearchHit, SearchResult
from polylogue.storage.search.query_builders import (
    build_ranked_action_search_query,
    build_ranked_conversation_search_query,
)
from polylogue.storage.search.query_support import _FTS5_SPECIAL, escape_fts5_query, normalize_fts5_query
from polylogue.storage.sqlite.connection import open_read_connection as open_connection


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
    """Search for messages using the patch-safe runtime implementation."""
    _search_runtime.open_read_connection = open_connection
    return _search_runtime.search_messages(
        query=query,
        archive_root=archive_root,
        render_root_path=render_root_path,
        db_path=db_path,
        limit=limit,
        source=source,
        since=since,
    )


__all__ = [
    "SearchHit",
    "SearchResult",
    "_FTS5_SPECIAL",
    "build_ranked_action_search_query",
    "build_ranked_conversation_search_query",
    "escape_fts5_query",
    "normalize_fts5_query",
    "open_connection",
    "search_messages",
]

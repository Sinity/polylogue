"""Public FTS5 search surface."""

from polylogue.storage import search_runtime as _search_runtime
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.search_models import SearchHit, SearchResult
from polylogue.storage.search_query_builders import (
    build_ranked_action_search_query,
    build_ranked_conversation_search_query,
)
from polylogue.storage.search_query_support import _FTS5_SPECIAL, escape_fts5_query, normalize_fts5_query


def search_messages(*args, **kwargs):
    """Search for messages using the patch-safe runtime implementation."""
    _search_runtime.open_connection = open_connection
    return _search_runtime.search_messages(*args, **kwargs)

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

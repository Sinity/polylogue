"""Search result caching for improved query performance.

This module provides an LRU cache for search results that can be invalidated
when conversations are re-ingested or modified.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
# Global cache state
_cache_lock = threading.Lock()
_cache_version = 0


@dataclass(frozen=True)
class SearchCacheKey:
    """Immutable key for search result caching.

    Uses a hash of the query parameters to create a cache-friendly key.
    """

    query: str
    archive_root: str
    render_root_path: str | None
    db_path: str | None
    limit: int
    source: str | None
    since: str | None
    cache_version: int

    @classmethod
    def create(
        cls,
        query: str,
        archive_root: Path,
        render_root_path: Path | None = None,
        db_path: Path | None = None,
        limit: int = 20,
        source: str | None = None,
        since: str | None = None,
    ) -> SearchCacheKey:
        """Create a cache key from search parameters.

        Args:
            query: Search query string
            archive_root: Archive root path
            render_root_path: Optional render root path
            db_path: Optional database path (for testing isolation)
            limit: Maximum results
            source: Optional source filter
            since: Optional timestamp filter

        Returns:
            Immutable cache key
        """
        with _cache_lock:
            current_version = _cache_version

        return cls(
            query=query,
            archive_root=str(archive_root),
            render_root_path=str(render_root_path) if render_root_path else None,
            db_path=str(db_path) if db_path else None,
            limit=limit,
            source=source,
            since=since,
            cache_version=current_version,
        )


def invalidate_search_cache() -> None:
    """Invalidate the entire search cache.

    Call this when conversations are re-ingested or modified to ensure
    fresh results on the next search.

    This is thread-safe and uses a version counter to invalidate all
    cached entries without clearing the cache dict (which would require
    accessing internal lru_cache state).
    """
    global _cache_version
    with _cache_lock:
        _cache_version += 1


def get_cache_stats() -> dict[str, int]:
    """Get cache version for invalidation tracking.

    Returns:
        Dictionary with current cache_version counter.
    """
    with _cache_lock:
        return {
            "cache_version": _cache_version,
        }


# Note: The actual caching is implemented via decorator in search_providers
# This module provides the key generation and invalidation infrastructure
__all__ = ["SearchCacheKey", "invalidate_search_cache", "get_cache_stats"]

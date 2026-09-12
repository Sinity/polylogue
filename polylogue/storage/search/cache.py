"""Search result caching for improved query performance.

This module provides an LRU cache for search results that can be invalidated
when sessions are re-parsed or modified.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.result_cache_metrics import publish_result_cache_stats

# Global cache state
_cache_lock = threading.Lock()
_cache_version = 0
_result_cache: OrderedDict[tuple[str, str, str, int, str], tuple[str, int]] = OrderedDict()
_result_cache_bytes = 0
_result_cache_hits = 0
_result_cache_misses = 0
_result_cache_evictions = 0

# The daemon is a resident process, so an unbounded result cache would turn a
# latency optimization into a memory leak.  JSON payloads are copied through
# the wire representation both to keep callers from mutating cached values
# and to make the byte budget deterministic.
RESULT_CACHE_MAX_ENTRIES = 256
RESULT_CACHE_MAX_BYTES = 64 * 1024 * 1024


def _publish_result_cache_stats() -> None:
    publish_result_cache_stats(
        entries=len(_result_cache),
        bytes=_result_cache_bytes,
        hits=_result_cache_hits,
        misses=_result_cache_misses,
        evictions=_result_cache_evictions,
    )


@dataclass(frozen=True)
class SearchCacheKey:
    """Immutable key for search result caching.

    Uses a hash of the query parameters to create a cache-friendly key.
    """

    query: str
    archive_root: str
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
        db_path: Path | None = None,
        limit: int = 20,
        source: str | None = None,
        since: str | None = None,
    ) -> SearchCacheKey:
        """Create a cache key from search parameters.

        Args:
            query: Search query string
            archive_root: Archive root path
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
            db_path=str(db_path) if db_path else None,
            limit=limit,
            source=source,
            since=since,
            cache_version=current_version,
        )


def invalidate_search_cache() -> None:
    """Invalidate the entire search cache.

    Call this when sessions are re-parsed or modified to ensure
    fresh results on the next search.

    This is thread-safe and uses a version counter to invalidate all
    cached entries without clearing the cache dict (which would require
    accessing internal lru_cache state).
    """
    global _cache_version, _result_cache_bytes
    with _cache_lock:
        _cache_version += 1
        _result_cache.clear()
        _result_cache_bytes = 0
        _publish_result_cache_stats()


def _result_cache_key(
    operation: str,
    payload: dict[str, object],
    *,
    archive_root: Path,
    generation: str,
    cache_version: int,
) -> tuple[str, str, str, int, str]:
    """Build a stable key for a pinned daemon read.

    ``generation`` is the resolved active index path, not the configured
    archive root.  This prevents a result from one promoted index generation
    being served by a long-lived daemon after a pointer swap.  The global
    invalidation version covers ordinary ingest/index writes between
    generations.
    """
    fingerprint = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (operation, str(archive_root.resolve()), generation, cache_version, fingerprint)


def get_cached_result(
    operation: str,
    payload: dict[str, object],
    *,
    archive_root: Path,
    generation: str,
) -> dict[str, object] | None:
    """Return a defensive copy of a cached JSON result, if present.

    The caller supplies the already-pinned generation identity.  A missing
    entry is deliberately indistinguishable from a cold cache; callers must
    execute the canonical read and populate it with :func:`put_cached_result`.
    """
    global _result_cache_hits, _result_cache_misses
    with _cache_lock:
        current_version = _cache_version
        key = _result_cache_key(
            operation,
            payload,
            archive_root=archive_root,
            generation=generation,
            cache_version=current_version,
        )
        encoded = _result_cache.get(key)
        if encoded is None:
            _result_cache_misses += 1
            _publish_result_cache_stats()
            return None
        _result_cache.move_to_end(key)
        _result_cache_hits += 1
        try:
            decoded = json.loads(encoded[0])
        except (TypeError, ValueError):
            # A corrupt in-memory entry must never turn a read into an error.
            _result_cache.pop(key, None)
            _result_cache_misses += 1
            _publish_result_cache_stats()
            return None
        _publish_result_cache_stats()
    if not isinstance(decoded, dict):
        return None
    return decoded


def put_cached_result(
    operation: str,
    payload: dict[str, object],
    result: dict[str, object],
    *,
    archive_root: Path,
    generation: str,
) -> None:
    """Store one bounded daemon read result.

    Oversized responses are intentionally not cached: returning a complete
    response remains correct, while retaining it would crowd every other
    interactive result out of the resident cache.
    """
    global _result_cache_bytes, _result_cache_evictions
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    size = len(encoded.encode("utf-8"))
    if size > RESULT_CACHE_MAX_BYTES:
        return
    with _cache_lock:
        key = _result_cache_key(
            operation,
            payload,
            archive_root=archive_root,
            generation=generation,
            cache_version=_cache_version,
        )
        previous = _result_cache.pop(key, None)
        if previous is not None:
            _result_cache_bytes -= previous[1]
        _result_cache[key] = (encoded, size)
        _result_cache_bytes += size
        while _result_cache and (
            len(_result_cache) > RESULT_CACHE_MAX_ENTRIES or _result_cache_bytes > RESULT_CACHE_MAX_BYTES
        ):
            _old_key, (_old_value, old_size) = _result_cache.popitem(last=False)
            _result_cache_bytes -= old_size
            _result_cache_evictions += 1
        _publish_result_cache_stats()


def get_cache_stats() -> dict[str, int]:
    """Get cache version for invalidation tracking.

    Returns:
        Dictionary with current cache_version counter.
    """
    with _cache_lock:
        return {
            "cache_version": _cache_version,
            "result_cache_entries": len(_result_cache),
            "result_cache_bytes": _result_cache_bytes,
            "result_cache_hits": _result_cache_hits,
            "result_cache_misses": _result_cache_misses,
            "result_cache_evictions": _result_cache_evictions,
        }


# The cache is intentionally kept at the read boundary rather than in a
# provider: all canonical daemon read routes share the same revision boundary.
__all__ = [
    "RESULT_CACHE_MAX_BYTES",
    "RESULT_CACHE_MAX_ENTRIES",
    "SearchCacheKey",
    "get_cached_result",
    "get_cache_stats",
    "invalidate_search_cache",
    "put_cached_result",
]

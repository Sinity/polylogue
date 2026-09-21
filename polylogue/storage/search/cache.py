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


@dataclass(frozen=True, slots=True)
class ReadViewIdentity:
    """The archive view one read was evaluated against.

    ``generation`` is the resolved active index path, not the configured
    archive root, so a result from one promoted index generation is never
    served by a long-lived daemon after a pointer swap.  ``epoch`` is the
    global invalidation counter, which covers ordinary ingest/index writes
    *within* one generation -- the index path does not move for those.

    The whole point of this being one value is that the same instance travels
    from lookup through execution to insertion.  Reading the epoch a second
    time at insertion is what let a result computed before an invalidation be
    stamped with the epoch that followed it, and then be served to the next
    read as if it were fresh.
    """

    archive_root: str
    generation: str
    epoch: int


def current_cache_epoch() -> int:
    """Read the global invalidation counter once, under the cache lock."""
    with _cache_lock:
        return _cache_version


def capture_read_view(*, archive_root: Path, generation: str, epoch: int | None = None) -> ReadViewIdentity:
    """Name the view a read is about to be evaluated against.

    Callers that establish the database snapshot capture this at that same
    boundary and pass the result through every later cache call.  ``epoch``
    is accepted so a caller can prove the counter did not move across its own
    snapshot pin and reuse the value it already observed.
    """
    return ReadViewIdentity(
        archive_root=str(archive_root.resolve()),
        generation=generation,
        epoch=current_cache_epoch() if epoch is None else epoch,
    )


def _result_cache_key(
    operation: str,
    payload: dict[str, object],
    *,
    view: ReadViewIdentity,
) -> tuple[str, str, str, int, str]:
    """Build a stable key for a pinned daemon read from its own view."""
    fingerprint = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (operation, view.archive_root, view.generation, view.epoch, fingerprint)


def get_cached_result(
    operation: str,
    payload: dict[str, object],
    *,
    view: ReadViewIdentity,
) -> dict[str, object] | None:
    """Return a defensive copy of a cached JSON result, if present.

    The caller supplies the view its snapshot was pinned against.  A missing
    entry is deliberately indistinguishable from a cold cache; callers must
    execute the canonical read and populate it with :func:`put_cached_result`
    under the *same* view.  Because the epoch is part of the key, a read
    pinned to an older view can never consume an answer computed for a newer
    one, nor the reverse.
    """
    global _result_cache_hits, _result_cache_misses
    with _cache_lock:
        key = _result_cache_key(operation, payload, view=view)
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
    view: ReadViewIdentity,
) -> None:
    """Store one bounded daemon read result under the view that computed it.

    The entry is keyed on the caller's own ``view``, never on the epoch that
    happens to be current when the query finishes.  If the active view moved
    while the query ran, the answer describes a view nothing will ask for
    again, so it is declined rather than relabelled: relabelling is exactly
    how a pre-invalidation answer came to be served to a post-invalidation
    read without executing its query body.

    Oversized responses are intentionally not cached: returning a complete
    response remains correct, while retaining it would crowd every other
    interactive result out of the resident cache.  A declined entry costs
    latency only; every declined path still returns the complete answer.
    """
    global _result_cache_bytes, _result_cache_evictions
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    size = len(encoded.encode("utf-8"))
    if size > RESULT_CACHE_MAX_BYTES:
        return
    with _cache_lock:
        if view.epoch != _cache_version:
            return
        key = _result_cache_key(operation, payload, view=view)
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
    "ReadViewIdentity",
    "SearchCacheKey",
    "capture_read_view",
    "current_cache_epoch",
    "get_cached_result",
    "get_cache_stats",
    "invalidate_search_cache",
    "put_cached_result",
]

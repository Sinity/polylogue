"""Process-local metrics bridge for resident read-result caches.

The cache lives below the daemon surface and the Prometheus renderer lives in
the daemon package. This tiny core-owned bridge keeps that observation path
from creating a substrate-to-surface import edge.
"""

from __future__ import annotations

from threading import Lock

_LOCK = Lock()
_STATS: dict[str, int] = {
    "entries": 0,
    "bytes": 0,
    "hits": 0,
    "misses": 0,
    "evictions": 0,
}


def publish_result_cache_stats(*, entries: int, bytes: int, hits: int, misses: int, evictions: int) -> None:
    """Publish the latest bounded-cache counters for the local metrics view."""
    with _LOCK:
        _STATS.update(
            entries=max(0, int(entries)),
            bytes=max(0, int(bytes)),
            hits=max(0, int(hits)),
            misses=max(0, int(misses)),
            evictions=max(0, int(evictions)),
        )


def result_cache_metrics() -> dict[str, int]:
    """Return a stable copy of the latest cache counters."""
    with _LOCK:
        return dict(_STATS)


__all__ = ["publish_result_cache_stats", "result_cache_metrics"]

"""Genuine multi-thread stress test for the live parse-prefetch cache.

polylogue-xikl (free-threading audit): the packaged daemon runs a
free-threaded (no-GIL) CPython build by default, and the live watcher
dispatches real OS threads through
``polylogue.sources.live.parse_prefetch.LiveParsePrefetchCache``. A
single-threaded functional test cannot detect a race that only manifests
when two OS threads interleave inside a lock-guarded method without the GIL
serializing their bytecode.

This test hammers the real ``threading.Lock``-guarded admission from many
concurrent ``ThreadPoolExecutor`` workers and asserts the documented budget
invariant still holds afterward: no lost update on the shared inflight byte
counter. A failure means the lock discipline has a gap; a pass is positive
evidence (not proof) that the seam is correctly synchronized.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from polylogue.sources.live.parse_prefetch import LiveParsePrefetchCache


def test_live_prefetch_cache_concurrent_try_admit_never_loses_or_duplicates_budget() -> None:
    """Concurrent admissions of distinct paths must neither lose nor
    double-count inflight bytes."""
    entry_count = 300
    payload = b"x" * 10
    budget = entry_count * len(payload)
    cache = LiveParsePrefetchCache(max_inflight_bytes=budget)

    accepted = []
    lock = threading.Lock()

    def _admit(i: int) -> None:
        ok = cache.try_admit(f"path-{i}", [], payload=payload)
        if ok:
            with lock:
                accepted.append(i)

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(_admit, range(entry_count)))

    assert len(accepted) == entry_count
    assert len(cache) == entry_count
    assert cache._inflight_bytes == entry_count * len(payload)

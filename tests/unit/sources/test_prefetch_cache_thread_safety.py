"""Genuine multi-thread stress tests for the parse-prefetch caches.

polylogue-xikl (free-threading audit): the packaged daemon now runs a
free-threaded (no-GIL) CPython build by default, and a full index rebuild
dispatches up to 16 real OS threads through
``polylogue.sources.revision_backfill.RawParsePrefetchCache``,
``polylogue.sources.live.parse_prefetch.LiveParsePrefetchCache``, and
``polylogue.daemon.parse_prefetch.DaemonParseStage``'s parsed-tree-bytes
ledger. Every existing test for these classes (``test_revision_backfill.py``,
``test_parse_prefetch.py``) exercises them from a single calling thread --
useful for functional correctness, but it cannot detect a race that only
manifests when two OS threads interleave inside a lock-guarded method
without the GIL serializing their bytecode.

These tests hammer the real ``threading.Lock``-guarded methods from many
concurrent ``ThreadPoolExecutor`` workers and assert the documented budget
invariants still hold afterward: no lost update on the shared
inflight/content-cache byte counters, no double-admission of the same key,
and no negative or over-budget accounting. A failure here means the
existing lock discipline in production code has a gap; a pass is positive
evidence (not proof) that the audited seams are correctly synchronized.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from polylogue.archive.revision_authority import RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.daemon.parse_prefetch import DaemonParseStage
from polylogue.sources.live.parse_prefetch import LiveParsePrefetchCache
from polylogue.sources.revision_backfill import RawParsePrefetchCache


def test_raw_prefetch_cache_concurrent_try_admit_never_loses_or_duplicates_budget() -> None:
    """Many threads racing ``try_admit`` on DISTINCT keys must not corrupt
    the shared ``_inflight_bytes`` counter (the non-atomic
    check-budget-then-insert hazard the GIL used to paper over)."""
    entry_count = 400
    payload_bytes = 10
    budget = entry_count * payload_bytes  # exact fit if every admit succeeds
    cache = RawParsePrefetchCache(max_inflight_bytes=budget)

    accepted = []
    lock = threading.Lock()

    def _admit(i: int) -> None:
        ok = cache.try_admit(f"raw-{i}", [], payload_bytes=payload_bytes, revision_kind=RawRevisionKind.FULL)
        if ok:
            with lock:
                accepted.append(i)

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(_admit, range(entry_count)))

    # Every distinct key fits the exact budget -- none should be rejected,
    # and the cache's own bookkeeping must exactly match what was accepted.
    assert len(accepted) == entry_count
    assert len(cache) == entry_count
    assert cache._inflight_bytes == entry_count * payload_bytes


def test_raw_prefetch_cache_concurrent_admit_same_key_admits_exactly_once() -> None:
    """Many threads racing ``try_admit`` on the SAME key must not double-admit
    (the classic check-then-act TOCTOU: ``if raw_id in self._entries`` then
    insert)."""
    cache = RawParsePrefetchCache(max_inflight_bytes=1_000_000)
    successes = []
    lock = threading.Lock()

    def _admit(_i: int) -> None:
        ok = cache.try_admit("shared-raw", [], payload_bytes=123, revision_kind=RawRevisionKind.FULL)
        if ok:
            with lock:
                successes.append(_i)

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(_admit, range(200)))

    assert len(successes) == 1
    assert len(cache) == 1
    assert cache._inflight_bytes == 123


def test_raw_prefetch_cache_concurrent_admit_and_pop_keeps_budget_consistent() -> None:
    """Interleaved admit/pop from many threads must leave ``_inflight_bytes``
    exactly equal to the sum of payload bytes for entries still resident --
    never drifting positive (a phantom budget leak that would eventually
    wedge every future admission) or negative (an accounting underflow)."""
    cache = RawParsePrefetchCache(max_inflight_bytes=1_000_000)
    n = 300

    def _churn(i: int) -> None:
        raw_id = f"raw-{i}"
        cache.try_admit(raw_id, [], payload_bytes=17, revision_kind=RawRevisionKind.FULL)
        # Immediately race a pop against admissions still in flight on other
        # threads -- this is the admit-then-maybe-pop pattern the daemon's
        # writer-held pass exercises against the warmer's concurrent warm().
        cache.pop(raw_id)
        # Re-admit half the keys so the cache ends up with a known population.
        if i % 2 == 0:
            cache.try_admit(raw_id, [], payload_bytes=17, revision_kind=RawRevisionKind.FULL)

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(_churn, range(n)))

    expected_resident = sum(1 for i in range(n) if i % 2 == 0)
    assert len(cache) == expected_resident
    assert cache._inflight_bytes == expected_resident * 17
    assert cache._inflight_bytes >= 0


def test_raw_prefetch_content_cache_concurrent_put_get_stays_within_budget() -> None:
    """The LRU content cache's ``put_content``/``get_content`` under
    concurrent hammering from many threads must never exceed its byte
    budget and must keep ``_content_bytes`` exactly equal to the sum of
    resident entries (the eviction-loop TOCTOU: read total size, then
    mutate, without the GIL to make that atomic)."""
    budget = 1000
    cache = RawParsePrefetchCache(max_inflight_bytes=1_000_000, max_content_cache_bytes=budget)
    keys = [(Provider.CODEX, f"hash-{i}", "", None) for i in range(50)]

    def _worker(_i: int) -> None:
        for key in keys:
            cache.put_content(key, [], payload_bytes=25, revision_kind=RawRevisionKind.FULL)
            cache.get_content(key)

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(_worker, range(64)))

    with cache._lock:
        actual_total = sum(entry.payload_bytes for entry in cache._content_entries.values())
        assert cache._content_bytes == actual_total
        assert cache._content_bytes <= budget


def test_live_prefetch_cache_concurrent_try_admit_never_loses_or_duplicates_budget() -> None:
    """Same lost-update hazard as the census cache, for the live watcher's
    path-keyed variant (distinct implementation, same lock discipline to
    verify)."""
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


def test_daemon_parse_stage_tree_byte_ledger_stays_consistent_under_concurrent_registration() -> None:
    """``DaemonParseStage``'s parsed-tree-bytes side ledger
    (``_register_cached_tree_bytes``) reconciles against the raw cache's
    own admission/eviction from a SEPARATE lock. Concurrently registering
    many raws (each first admitted to the raw cache, mirroring
    ``warm_raw_ids``'s real call order) must leave
    ``_cached_tree_bytes_total`` exactly equal to the sum of the entries
    the ledger still tracks, and never above the configured budget."""
    stage = DaemonParseStage(max_workers=1, max_inflight_bytes=10_000_000, max_cached_tree_bytes=1000)
    try:
        n = 200
        tree_bytes = 13

        def _register(i: int) -> None:
            raw_id = f"raw-{i}"
            # Mirrors warm_raw_ids: admit to the raw cache first, then
            # register the estimated tree size against the ledger.
            stage.cache.try_admit(raw_id, [], payload_bytes=1, revision_kind=RawRevisionKind.FULL)
            stage._register_cached_tree_bytes(raw_id, tree_bytes)

        with ThreadPoolExecutor(max_workers=32) as pool:
            list(pool.map(_register, range(n)))

        with stage._tree_bytes_lock:
            actual_total = sum(stage._tree_bytes_by_raw_id.values())
            assert stage._cached_tree_bytes_total == actual_total
            assert stage._cached_tree_bytes_total <= stage._max_cached_tree_bytes
            # Every tracked raw_id must still actually be resident in the
            # raw cache -- the ledger must never claim bytes for an entry
            # the cache itself has already evicted/forgotten.
            for raw_id in stage._tree_bytes_by_raw_id:
                assert stage.cache.contains(raw_id)
    finally:
        stage.shutdown()

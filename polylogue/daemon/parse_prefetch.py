"""Daemon-owned parse-stage extraction: parse census candidates off the writer hold.

polylogue-m6tp phase (a). The raw-materialization conveyor's writer hold
(``DaemonWriteCoordinator.run_sync``) used to cover BOTH the CPU-bound
blob->``ParsedSession`` decode (census parse) and the SQLite writes that
record it, so a large or slow parse extended the writer hold by exactly as
long as the parse took -- starving every other write actor (live ingest,
status snapshots, insight convergence) queued behind the same coordinator.

This module lets the daemon pre-parse the NEXT pass's candidate raws in a
bounded ``ThreadPoolExecutor`` BEFORE the writer hold is ever requested. The
writer-held pass then finds those results already warmed in a
``RawParsePrefetchCache`` (``polylogue.sources.revision_backfill``) and skips
reparsing them -- see that class's docstring for why a miss (empty cache,
budget-rejected entry, or the flag simply being off) always degrades to the
exact unmodified parse path rather than incorrect behavior.

Why threads are safe here even on a standard (GIL) build: the polylogue-7mtf
control-run measurement (``parallel_threads_effective`` in
``polylogue.pipeline.services.process_pool``) found threaded parse gives no
GIL-build speedup AND inflates a *concurrently write-holding* thread's commit
latency ~5000x. That hazard is specifically about a parse thread running
WHILE a writer thread is active. This module never does that: ``warm()`` is
called by the conveyor BEFORE it ever asks the write coordinator for the
writer hold, so there is no writer thread to contend with. On a GIL build
this still gives little or no wall-clock parse speedup (CPython serializes
the CPU-bound decode across threads) -- that is expected and is the point of
phase (a): prove the parse/apply seam is correct and equivalence-safe ahead
of the free-threaded 3.14t deploy (phase (b), polylogue-m6tp), which is what
turns the same code path into a real speedup.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from polylogue.config import Config
from polylogue.logging import get_logger
from polylogue.pipeline.parsed_tree_size import estimate_parsed_tree_bytes
from polylogue.sources import revision_backfill
from polylogue.sources.dispatch import is_stream_record_provider
from polylogue.sources.revision_backfill import RawParsePrefetchCache
from polylogue.storage.repair import (
    raw_materialization_pending_census_raw_ids,
    raw_materialization_readonly_descriptors,
)

logger = get_logger(__name__)

# Floor/ceiling for the adaptive whale-memory budget below. The original
# fixed 64 MiB default starved bulk-scale warm on whale corpora: measured
# live 2026-07-20 on the 50K-raw archive, a 2000-raw page warmed 139 raws in
# 376s (0.37 raws/s, pool stalled on cache admission) under 64 MiB versus
# 500 raws in 8.8s (56.7 raws/s) with the budget raised — the workers were
# blocked on `try_admit`, not on parsing. The budget's purpose is bounding
# transient memory beside a live daemon, so it scales with the machine
# instead of a one-size constant: 1/16 of physical RAM, clamped to
# [64 MiB, 2 GiB].
_MIN_MAX_INFLIGHT_BYTES = 64 * 1024 * 1024  # 64 MiB
_MAX_MAX_INFLIGHT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB

# CodeRabbit (PR #3168): as_completed()/future.result() had no timeout, so one
# hung worker (e.g. an unresponsive filesystem read) would block warm()
# forever -- and warm() is awaited directly ahead of run_sync in the periodic
# raw-materialization loop, so a stuck warm pass would stall every subsequent
# drain pass indefinitely, not just this one. 300s (5 min) is generous for
# the happy path (a bounded batch of already-published local blob reads) and
# only ever matters on a genuine hang. On timeout, still-pending raws are
# simply left uncached -- the writer-held pass reparses them normally, the
# same graceful-degradation guarantee as any other prefetch miss. A
# ThreadPoolExecutor cannot forcibly kill a running worker thread, so a truly
# wedged worker keeps occupying one pool slot until it (eventually) returns;
# that is an inherent limitation of thread-based cancellation, not something
# this bound can fix -- the bound's job is only to stop the CONVEYOR LOOP
# from waiting on it forever, which it does.
_DEFAULT_WARM_TIMEOUT_SECONDS = 300.0


def daemon_parse_stage_worker_count() -> int:
    """Bounded worker cap for the daemon-owned pre-parse thread pool.

    ``cpu_count - 1`` leaves one core free for the daemon's own event loop,
    mirroring ``resolve_parse_worker_count``'s cpu-1 convention (see
    ``polylogue.pipeline.services.process_pool``). Override with
    ``POLYLOGUE_DAEMON_PARSE_STAGE_WORKERS``.
    """
    raw = os.environ.get("POLYLOGUE_DAEMON_PARSE_STAGE_WORKERS")
    if raw is not None:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    return max(1, (os.cpu_count() or 2) - 1)


def _physical_memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size


def daemon_parse_stage_max_inflight_bytes() -> int:
    """Whale-memory budget for parsed sessions held in the prefetch cache.

    Adaptive: 1/16 of physical RAM clamped to [64 MiB, 2 GiB] (see the
    constants above for the measured starvation the old fixed 64 MiB default
    caused). Override with ``POLYLOGUE_DAEMON_PARSE_STAGE_MAX_INFLIGHT_BYTES``.
    """
    raw = os.environ.get("POLYLOGUE_DAEMON_PARSE_STAGE_MAX_INFLIGHT_BYTES")
    if raw is not None:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    physical = _physical_memory_bytes()
    if physical is None:
        return _MIN_MAX_INFLIGHT_BYTES
    return max(_MIN_MAX_INFLIGHT_BYTES, min(_MAX_MAX_INFLIGHT_BYTES, physical // 16))


# polylogue-xb4i: the inflight-bytes budget above (and RawParsePrefetchCache's
# own admission gate) both account raw PAYLOAD bytes, because payload size is
# the only thing known BEFORE a raw is parsed. But a parsed ``ParsedSession``
# tree resident in the cache is not the same size as the payload it was
# parsed from -- Pydantic model instances, per-block dicts, and Python object
# overhead inflate a compact JSON/JSONL payload substantially. Two earlyoom
# kills (19.3G and 20.2G RSS peaks, 2026-07-20) happened on a whale-dense
# page precisely because the cache retained a whole 2000-raw page of PARSED
# TREES while only the raw payload bytes were budgeted -- clamping the
# inflight (pre-parse) budget did nothing, since the memory pressure came
# from trees already sitting in the cache post-parse, not from parses in
# flight.
#
# Floor/ceiling mirror the inflight-bytes budget's adaptive-RAM shape: 1/8 of
# physical RAM (trees are the bigger of the two budgets since they are what
# actually sits resident) clamped to [256 MiB, 4 GiB].
_MIN_MAX_CACHED_TREE_BYTES = 256 * 1024 * 1024  # 256 MiB
_MAX_MAX_CACHED_TREE_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB


def daemon_parse_stage_max_cached_tree_bytes() -> int:
    """Whole-cache budget for ESTIMATED parsed-tree bytes (not payload bytes).

    Distinct from :func:`daemon_parse_stage_max_inflight_bytes`, which caps
    raw PAYLOAD bytes admitted while parses are in flight (the only thing
    knowable pre-parse). This is the budget that actually bounds what a
    quiet daemon holds resident in ``DaemonParseStage.cache`` between warm()
    passes -- see the calibration comment on ``_ESTIMATOR_BYTES_PER_CHAR``
    above for why the two budgets can diverge by 10x+ on the same page.
    Adaptive: 1/8 of physical RAM clamped to [256 MiB, 4 GiB]. Override with
    ``POLYLOGUE_DAEMON_PARSE_STAGE_MAX_CACHED_TREE_BYTES``.
    """
    raw = os.environ.get("POLYLOGUE_DAEMON_PARSE_STAGE_MAX_CACHED_TREE_BYTES")
    if raw is not None:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    physical = _physical_memory_bytes()
    if physical is None:
        return _MIN_MAX_CACHED_TREE_BYTES
    return max(_MIN_MAX_CACHED_TREE_BYTES, min(_MAX_MAX_CACHED_TREE_BYTES, physical // 8))


def daemon_parse_stage_warm_timeout_seconds() -> float:
    """Bound on how long ``warm()`` waits for its dispatched workers.

    Override with ``POLYLOGUE_DAEMON_PARSE_STAGE_WARM_TIMEOUT_SECONDS``. See
    ``_DEFAULT_WARM_TIMEOUT_SECONDS`` for why this exists and what it does
    (and does not) guarantee.
    """
    raw = os.environ.get("POLYLOGUE_DAEMON_PARSE_STAGE_WARM_TIMEOUT_SECONDS")
    if raw is not None:
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        if value > 0:
            return value
    return _DEFAULT_WARM_TIMEOUT_SECONDS


class DaemonParseStage:
    """Owns the daemon's bounded pre-parse ``ThreadPoolExecutor`` and cache.

    One instance lives for the daemon process's lifetime (created lazily by
    the raw-materialization conveyor loop when ``daemon_parse_stage_split``
    is enabled). ``warm`` is synchronous/blocking -- callers run it off the
    event loop (``asyncio.to_thread``), exactly like every other conveyor
    pass, and NEVER under ``daemon_write_coordinator().run_sync``: doing so
    would defeat the entire point, since the pre-parse must run without the
    writer hold held.
    """

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        max_inflight_bytes: int | None = None,
        warm_timeout_seconds: float | None = None,
        max_cached_tree_bytes: int | None = None,
    ) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers if max_workers is not None else daemon_parse_stage_worker_count(),
            thread_name_prefix="polylogue-parse-stage",
        )
        self.cache = RawParsePrefetchCache(
            max_inflight_bytes=(
                max_inflight_bytes if max_inflight_bytes is not None else daemon_parse_stage_max_inflight_bytes()
            )
        )
        self._warm_timeout_seconds = (
            warm_timeout_seconds if warm_timeout_seconds is not None else daemon_parse_stage_warm_timeout_seconds()
        )
        # polylogue-xb4i: a SECOND budget tracked alongside ``self.cache``,
        # keyed on the same raw_ids but accounting ESTIMATED PARSED-TREE
        # bytes instead of the raw cache's payload bytes. ``self.cache``
        # itself is not touched/subclassed (it is a shared type consumed
        # directly by other callers -- ``bulk_rebuild.py`` hands
        # ``stage.cache`` straight to ``RebuildIndexRequest.prefetch_cache``
        # -- so this stays a side ledger that reconciles against the raw
        # cache's own admission/eviction rather than replacing it.
        self._max_cached_tree_bytes = (
            max_cached_tree_bytes if max_cached_tree_bytes is not None else daemon_parse_stage_max_cached_tree_bytes()
        )
        self._tree_bytes_lock = threading.Lock()
        self._tree_bytes_by_raw_id: dict[str, int] = {}
        self._cached_tree_bytes_total = 0

    @property
    def cached_tree_bytes_total(self) -> int:
        """Sum of estimated parsed-tree bytes currently tracked as cached."""
        with self._tree_bytes_lock:
            return self._cached_tree_bytes_total

    def _reconcile_stale_tree_tracking_locked(self) -> None:
        """Drop tracking for any raw_id no longer present in ``self.cache``.

        Consumers outside this class (the writer-held pass, via
        ``RawParsePrefetchCache.pop``) remove entries from ``self.cache``
        directly -- this class has no hook into that removal, so the tree-
        byte ledger can only be reconciled lazily, by checking membership
        before making an eviction decision. Cheap: proportional to the
        number of currently-tracked entries, each check a dict lookup under
        the raw cache's own lock.
        """
        stale = [raw_id for raw_id in self._tree_bytes_by_raw_id if not self.cache.contains(raw_id)]
        for raw_id in stale:
            self._drop_tree_bytes_locked(raw_id)

    def _drop_tree_bytes_locked(self, raw_id: str) -> None:
        tree_bytes = self._tree_bytes_by_raw_id.pop(raw_id, None)
        if tree_bytes is not None:
            self._cached_tree_bytes_total -= tree_bytes

    def _select_eviction_candidate_locked(self) -> str | None:
        """Largest entry wins; ties break to the oldest (dict preserves
        insertion order, and ``>`` -- not ``>=`` -- means the first-seen
        (oldest) entry at the max size is kept as the running candidate)."""
        best_id: str | None = None
        best_bytes = -1
        for raw_id, tree_bytes in self._tree_bytes_by_raw_id.items():
            if tree_bytes > best_bytes:
                best_bytes = tree_bytes
                best_id = raw_id
        return best_id

    def _register_cached_tree_bytes(self, raw_id: str, tree_bytes: int) -> None:
        """Record a newly-admitted raw's estimated tree size and evict
        largest-or-oldest entries (via ``self.cache.pop``, which releases
        both the raw cache's own payload-byte budget and this ledger's tree-
        byte budget) until back under ``self._max_cached_tree_bytes``."""
        evicted: list[str] = []
        with self._tree_bytes_lock:
            self._reconcile_stale_tree_tracking_locked()
            self._tree_bytes_by_raw_id[raw_id] = tree_bytes
            self._cached_tree_bytes_total += tree_bytes
            while self._cached_tree_bytes_total > self._max_cached_tree_bytes and self._tree_bytes_by_raw_id:
                candidate = self._select_eviction_candidate_locked()
                if candidate is None:
                    break
                self._drop_tree_bytes_locked(candidate)
                evicted.append(candidate)
        for evicted_id in evicted:
            self.cache.pop(evicted_id)
        if evicted:
            logger.info(
                "parse-stage prefetch: evicted %d cached parsed tree(s) to stay within the %d-byte "
                "estimated-tree-bytes budget after admitting raw_id=%s (%d bytes)",
                len(evicted),
                self._max_cached_tree_bytes,
                raw_id,
                tree_bytes,
            )

    def warm(self, config: Config, *, limit: int, max_payload_bytes: int) -> int:
        """Pre-parse up to ``limit`` pending census candidates outside any writer hold.

        Returns the number of raws newly admitted to the cache. Read-only
        end to end: candidate discovery and descriptor lookup both open
        ``mode=ro`` SQLite connections (``polylogue.storage.repair``);
        parsing reads only already-published blob bytes via a stateless
        ``ArchiveBlobPublisher``, mirroring the production census parse
        worker exactly (``census_parse_worker``, the same function the
        writer-held path dispatches to a process/thread pool). Nothing here
        writes to source.db, index.db, or takes the daemon's writer lease.
        """
        candidate_raw_ids = raw_materialization_pending_census_raw_ids(
            config, limit=limit, max_payload_bytes=max_payload_bytes
        )
        return self.warm_raw_ids(config, raw_ids=candidate_raw_ids, max_payload_bytes=max_payload_bytes)

    def warm_raw_ids(self, config: Config, *, raw_ids: Sequence[str], max_payload_bytes: int) -> int:
        """Pre-parse an explicit ``raw_ids`` list outside any writer hold.

        Same read-only, graceful-degradation contract as :meth:`warm`, but
        for a caller (polylogue-gd6v's daemon bulk-rebuild routing) that
        already knows exactly which raws its next bounded pass will select
        -- a resumable rebuild transaction's own paged cursor -- instead of
        querying the raw-materialization conveyor's own pending-census
        candidate set. :meth:`warm` is now a thin wrapper around this method.
        """
        raw_ids = [raw_id for raw_id in raw_ids if not self.cache.contains(raw_id)]
        if not raw_ids:
            return 0
        archive_root = config.archive_root
        descriptors = raw_materialization_readonly_descriptors(archive_root, raw_ids)
        blob_root_str = str(archive_root / "blob")
        source_db_path_str = str(archive_root / "source.db")

        futures = {}
        for raw_id in raw_ids:
            descriptor = descriptors.get(raw_id)
            if descriptor is None:
                continue
            provider, blob_hash, source_path, _kind, _size = descriptor
            future = self._executor.submit(
                revision_backfill.census_parse_worker,
                raw_id,
                provider.value,
                blob_hash,
                source_path,
                is_stream_record_provider(source_path, str(provider)),
                blob_root_str,
                source_db_path_str,
            )
            futures[future] = raw_id

        warmed = 0
        completed = 0
        try:
            for future in as_completed(futures, timeout=self._warm_timeout_seconds):
                completed += 1
                raw_id = futures[future]
                try:
                    _raw_id, sessions, error = future.result()
                except Exception:
                    logger.warning("parse-stage prefetch: worker failed for raw_id=%s", raw_id, exc_info=True)
                    continue
                if error is not None or sessions is None:
                    # Parse failures are intentionally NOT cached: the writer-held
                    # pass reparses (and correctly quarantines/records) this raw
                    # exactly as it would with the flag off. Prefetch only ever
                    # shortcuts the happy path.
                    continue
                _provider, _blob_hash, _source_path, kind, payload_size = descriptors[raw_id]
                # polylogue-xb4i: estimate the PARSED TREE size (not payload
                # size) before ever admitting to the cache. A tree bigger
                # than the whole tree-bytes budget is never retained at all
                # -- it does not even occupy a payload-bytes admission slot
                # -- so one whale raw can never pin gigabytes of resident
                # memory regardless of how much inflight-bytes headroom it
                # happened to fit under pre-parse. This is the fix for the
                # 2026-07-20 earlyoom kills: the inflight clamp bounded
                # PAYLOAD bytes in flight, but the cache retained whatever
                # trees resulted from that payload with no size check at all.
                tree_bytes = estimate_parsed_tree_bytes(sessions)
                if tree_bytes > self._max_cached_tree_bytes:
                    logger.warning(
                        "parse-stage prefetch: raw_id=%s estimated parsed-tree bytes %d exceed the "
                        "whole cache budget %d bytes; never retained -- the writer-held pass reparses "
                        "it normally, identical to any other prefetch miss",
                        raw_id,
                        tree_bytes,
                        self._max_cached_tree_bytes,
                    )
                    continue
                if self.cache.try_admit(raw_id, sessions, payload_bytes=payload_size, revision_kind=kind):
                    self._register_cached_tree_bytes(raw_id, tree_bytes)
                    warmed += 1
        except TimeoutError:
            # Bounds the CONVEYOR LOOP's wait, not the worker itself -- a
            # ThreadPoolExecutor cannot forcibly kill a running thread, so a
            # genuinely wedged worker keeps occupying one pool slot until it
            # eventually returns (see _DEFAULT_WARM_TIMEOUT_SECONDS). Every raw
            # not yet completed is simply left uncached: the writer-held pass
            # reparses it normally, identical to any other prefetch miss.
            pending = len(futures) - completed
            logger.warning(
                "parse-stage prefetch: warm() timed out after %.0fs waiting on %d of %d worker(s); "
                "leaving unfinished raw(s) uncached for the writer-held pass to reparse normally",
                self._warm_timeout_seconds,
                pending,
                len(futures),
            )
        return warmed

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = [
    "DaemonParseStage",
    "daemon_parse_stage_max_cached_tree_bytes",
    "daemon_parse_stage_max_inflight_bytes",
    "daemon_parse_stage_warm_timeout_seconds",
    "daemon_parse_stage_worker_count",
    "estimate_parsed_tree_bytes",
]

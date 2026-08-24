"""Shared off-writer-hold parse-stage engine: parse census candidates before any writer hold.

polylogue-m6tp phase (a), relocated to substrate (polylogue-czq2). Originally
lived in ``polylogue.daemon.parse_prefetch`` and was consulted by exactly one
caller (``daemon/bulk_rebuild.py``'s automagic bulk-rebuild routing) even
though the mechanism it provides -- pre-parsing a bounded set of raw ids in a
``ThreadPoolExecutor`` and handing the result to ``RawParsePrefetchCache`` --
has nothing daemon-specific about it. Every OTHER caller of the shared
rebuild engine (the offline ``polylogue ops maintenance rebuild-index`` CLI,
and the daemon's own ``/api/maintenance/rebuild-index`` HTTP route) threaded
``prefetch_cache=None`` and paid the full serial re-parse/spill-reload cost
this module exists to avoid -- see ``maintenance/rebuild_index.py``'s
``_warm_offline_prefetch_cache`` for the fix that consumes this module
directly instead of only through the daemon's bulk-rebuild loop.

``polylogue.daemon.parse_prefetch`` re-exports ``DaemonParseStage`` (an alias
of :class:`CensusParseStage` below) and every config-resolution helper from
here unchanged, so every existing daemon caller/test keeps its import path
and behavior byte-identical; this module is the substrate the daemon
consumes, not a daemon-owned implementation detail any more.

The writer-hold contention this was originally built to avoid does not apply
to the offline CLI or HTTP maintenance route the same way (there is no
sibling write actor sharing a coordinator in a one-shot offline rebuild
process; the HTTP route already sits inside the write bridge's held slot) --
but the SAME mechanism still avoids the double parse-then-spill round trip
those callers were paying: a raw already popped from the prefetch cache
skips ``_parse_retained_raws``'s own dispatch entirely (see that function's
docstring), so census output flows into ``_ParsedSessionSpill`` exactly once
either way, just without the caller having forgotten to ask for it.

Why threads are safe here even on a standard (GIL) build: the polylogue-7mtf
control-run measurement (``parallel_threads_effective`` in
``polylogue.pipeline.services.process_pool``) found threaded parse gives no
GIL-build speedup AND inflates a *concurrently write-holding* thread's commit
latency ~5000x. That hazard is specifically about a parse thread running
WHILE a writer thread is active. This module never does that: ``warm()``/
``warm_raw_ids()`` are called BEFORE any caller ever asks for (or already
holds) a writer hold, so there is no writer thread to contend with. On a GIL
build this still gives little or no wall-clock parse speedup (CPython
serializes the CPU-bound decode across threads) -- that is expected; it still
avoids the process-pool pickle-back round trip ``_parse_unique_retained_raws``
pays on a GIL build (threads share parsed object graphs by reference), and it
is what turns into a real multi-core speedup on the free-threaded 3.14t
deploy (phase (b), polylogue-m6tp).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sqlite3
import threading
import time
import weakref
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from concurrent.futures import thread as _thread_impl
from contextlib import closing
from pathlib import Path
from typing import Any

from polylogue.archive.revision_authority import RawRevisionKind
from polylogue.config import Config
from polylogue.logging import get_logger
from polylogue.pipeline.parsed_tree_size import (
    effective_physical_memory_bytes,
    estimate_parsed_tree_bytes,
)
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


class _ProcessBoundedThreadPoolExecutor(ThreadPoolExecutor):
    """Thread pool whose wedged workers cannot keep daemon process exit alive.

    ``ThreadPoolExecutor.shutdown(wait=False)`` stops admission but leaves a
    running worker alive until it returns; standard executor workers are
    non-daemon and therefore can still hold interpreter shutdown hostage. The
    parse stage only performs graceful-degradation work, so its workers are
    explicitly daemon threads. The coordinator and durable outbox remain the
    authority for all SQLite work and are unaffected by this containment.
    """

    def _adjust_thread_count(self) -> None:  # pragma: no cover - exercised by submit
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_worker_reference: object, queue: Any = self._work_queue) -> None:
            queue.put(None)

        num_threads = len(self._threads)
        if num_threads >= self._max_workers:
            return
        thread_name = f"{self._thread_name_prefix or self}_{num_threads}"
        worker = threading.Thread(
            name=thread_name,
            target=_thread_impl._worker,
            args=(
                weakref.ref(self, weakref_cb),
                self._create_worker_context(),  # type: ignore[attr-defined]
                self._work_queue,
            ),
            daemon=True,
        )
        worker.start()
        self._threads.add(worker)  # type: ignore[attr-defined]
        _thread_impl._threads_queues[worker] = self._work_queue  # type: ignore[index]

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Stop admission without registering running daemon workers for join."""
        super().shutdown(wait=wait, cancel_futures=cancel_futures)
        for worker in tuple(self._threads):
            _thread_impl._threads_queues.pop(worker, None)  # type: ignore[attr-defined]


def _resolve_readonly_native_ids(archive_root: Path, raw_ids: Sequence[str]) -> dict[str, str | None]:
    """Read-only ``native_id`` lookup for pre-parse dispatch (no writer needed).

    polylogue-6lyh1: mirrors ``ArchiveStore.raw_native_id`` (same column, same
    "blank means unknown" contract) but over a plain ``mode=ro`` connection,
    the same relationship ``raw_materialization_readonly_descriptors`` (in
    ``storage/repair.py``) has to ``ArchiveStore.raw_revision_descriptor`` --
    kept as a small dedicated query here rather than widening that shared
    helper's return shape, since its other callers do not need this column.
    """
    raw_ids = list(raw_ids)
    if not raw_ids:
        return {}
    placeholders = ",".join("?" for _ in raw_ids)
    result: dict[str, str | None] = {}
    with closing(sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            f"SELECT raw_id, native_id FROM raw_sessions WHERE raw_id IN ({placeholders})",
            raw_ids,
        ).fetchall()
    for row in rows:
        value = row[1]
        result[str(row[0])] = value if isinstance(value, str) and value.strip() else None
    return result


def daemon_parse_stage_worker_count() -> int:
    """Bounded worker cap for the daemon-owned pre-parse thread pool.

    ``cpu_count - 1`` leaves one core free for the daemon's own event loop,
    mirroring ``resolve_parse_worker_count``'s cpu-1 convention (see
    ``polylogue.pipeline.services.process_pool``). Override with
    ``POLYLOGUE_DAEMON_PARSE_STAGE_WORKERS``.
    """
    from polylogue.config import load_polylogue_config

    configured = load_polylogue_config().daemon_parse_stage_workers
    if configured is not None and configured > 0:
        return configured
    return max(1, (os.cpu_count() or 2) - 1)


def _physical_memory_bytes() -> int | None:
    return effective_physical_memory_bytes()


def daemon_parse_stage_max_inflight_bytes() -> int:
    """Whale-memory budget for parsed sessions held in the prefetch cache.

    Adaptive: 1/16 of physical RAM clamped to [64 MiB, 2 GiB] (see the
    constants above for the measured starvation the old fixed 64 MiB default
    caused). Override with ``POLYLOGUE_DAEMON_PARSE_STAGE_MAX_INFLIGHT_BYTES``.
    """
    from polylogue.config import load_polylogue_config

    configured = load_polylogue_config().daemon_parse_stage_max_inflight_bytes
    if configured is not None and configured > 0:
        return configured
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
    from polylogue.config import load_polylogue_config

    configured = load_polylogue_config().daemon_parse_stage_max_cached_tree_bytes
    if configured is not None and configured > 0:
        return configured
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
    from polylogue.config import load_polylogue_config

    configured = load_polylogue_config().daemon_parse_stage_warm_timeout_seconds
    if configured is not None and configured > 0:
        return configured
    return _DEFAULT_WARM_TIMEOUT_SECONDS


class CensusParseStage:
    """Owns a bounded pre-parse ``ThreadPoolExecutor`` and its prefetch cache.

    In the daemon, one instance lives for the process's lifetime (created
    lazily on first use by the raw-materialization conveyor loop, or by
    ``daemon/bulk_rebuild.py``'s bulk-rebuild routing). ``warm``/
    ``warm_raw_ids`` are synchronous/blocking -- a daemon caller runs them off
    the event loop (``asyncio.to_thread``), exactly like every other conveyor
    pass, and NEVER under ``daemon_write_coordinator().run_sync``: doing so
    would defeat the entire point, since the pre-parse must run without the
    writer hold held. An offline caller (``maintenance/rebuild_index.py``)
    instead constructs a short-lived instance scoped to one bounded pass's
    raw ids and discards it once ``warm_raw_ids`` returns -- see
    ``_warm_offline_prefetch_cache``.
    """

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        max_inflight_bytes: int | None = None,
        warm_timeout_seconds: float | None = None,
        max_cached_tree_bytes: int | None = None,
    ) -> None:
        self._executor = _ProcessBoundedThreadPoolExecutor(
            max_workers=max_workers if max_workers is not None else daemon_parse_stage_worker_count(),
            thread_name_prefix="polylogue-parse-stage",
        )
        self.cache = RawParsePrefetchCache(
            max_inflight_bytes=(
                max_inflight_bytes if max_inflight_bytes is not None else daemon_parse_stage_max_inflight_bytes()
            )
        )
        # A cancelled ``asyncio.to_thread(stage.warm, ...)`` cannot cancel a
        # running parse worker.  Track the whole warm operation so daemon
        # writer admission can reject while an orphaned worker still owns
        # parse resources, rather than assuming the caller's cancellation
        # released the executor.
        self._warm_state_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._active_warm_operations = 0
        self._warm_idle = threading.Event()
        self._warm_idle.set()
        self._writer_admission_ready = threading.Event()
        self._writer_admission_ready.set()
        self._background_workers: set[Future[object]] = set()
        # Admission is global to this stage, not just to one ``warm`` call.
        # A timed-out thread cannot be cancelled, and its future can remain in
        # the executor queue (or occupy a worker) after the caller returns.
        # Reserve its source-payload bytes until that exact future completes so
        # retries neither duplicate raw ids nor grow an unbounded queue.
        self._pending_payload_by_raw_id: dict[str, int] = {}
        self._pending_payload_bytes = 0
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

    def _refresh_lifecycle_state_locked(self) -> None:
        """Publish the single state transition for completed warm work."""
        if self._active_warm_operations != 0 or self._background_workers or self._pending_payload_by_raw_id:
            return
        self._warm_idle.set()
        if not self._stop_requested.is_set():
            self._writer_admission_ready.set()

    def _begin_warm_operation(self) -> None:
        with self._warm_state_lock:
            self._active_warm_operations += 1
            self._warm_idle.clear()
            if not self._stop_requested.is_set():
                self._writer_admission_ready.clear()

    def _end_warm_operation(self) -> None:
        with self._warm_state_lock:
            self._active_warm_operations -= 1
            self._refresh_lifecycle_state_locked()

    def _track_background_workers(self, futures: Sequence[Future[Any]]) -> None:
        pending = [future for future in futures if not future.done()]
        if not pending:
            return
        with self._warm_state_lock:
            self._background_workers.update(pending)
            self._warm_idle.clear()
        for future in pending:
            future.add_done_callback(self._background_worker_done)

    def _background_worker_done(self, future: Future[object]) -> None:
        with self._warm_state_lock:
            self._background_workers.discard(future)
            self._refresh_lifecycle_state_locked()

    def _release_pending_payload(self, raw_id: str, _future: Future[Any] | None) -> None:
        """Release one global source-payload reservation after completion."""
        with self._warm_state_lock:
            reserved = self._pending_payload_by_raw_id.get(raw_id)
            if reserved is None:
                return
            self._pending_payload_by_raw_id.pop(raw_id, None)
            self._pending_payload_bytes -= reserved
            self._refresh_lifecycle_state_locked()

    def _discard_future_result(self, future: Future[Any], *, raw_id: str) -> None:
        """Consume/discard one result before releasing its payload reservation."""
        with contextlib.suppress(BaseException):
            future.result()
        self._release_pending_payload(raw_id, future)

    def _retain_cleanup_callbacks(self, futures: dict[Future[Any], str]) -> None:
        """Cancel queued work and drain every admitted future exactly once."""
        for future in futures:
            future.cancel()
        for future, raw_id in futures.items():
            if future.done():
                self._discard_future_result(future, raw_id=raw_id)
                continue

            def discard(done: Future[Any], *, rid: str = raw_id) -> None:
                self._discard_future_result(done, raw_id=rid)

            future.add_done_callback(discard)
        self._track_background_workers(list(futures))

    def _reserve_pending_payload(self, raw_id: str, payload_bytes: int) -> bool:
        """Atomically admit a raw against cache plus all outstanding workers."""
        with self._warm_state_lock:
            if raw_id in self._pending_payload_by_raw_id or self.cache.contains(raw_id):
                return False
            if self.cache.inflight_bytes + self._pending_payload_bytes + payload_bytes > self.cache.max_inflight_bytes:
                return False
            self._pending_payload_by_raw_id[raw_id] = payload_bytes
            self._pending_payload_bytes += payload_bytes
            self._warm_idle.clear()
            return True

    def writer_admission_ready(self) -> bool:
        """Whether the writer may proceed despite an invalidated warm worker."""
        return self._writer_admission_ready.is_set()

    def wait_until_idle(self, *, timeout: float | None = None) -> bool:
        """Wait for all warm callers and their parse workers to finish."""
        return self._warm_idle.wait(timeout)

    @property
    def warm_timeout_seconds(self) -> float:
        """Return the bounded wait used for worker admission and warm()."""
        return self._warm_timeout_seconds

    def max_inflight_bytes(self) -> int:
        """Return the source-payload admission budget for warm workers."""
        return self.cache.max_inflight_bytes

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

    def request_stop(self) -> None:
        """Invalidate warm controllers and restore writer admission immediately."""
        self._stop_requested.set()
        self._writer_admission_ready.set()

    async def warm_async(
        self,
        config: Config,
        *,
        limit: int,
        max_payload_bytes: int,
        raw_artifact_id: str | None = None,
    ) -> int:
        """Run blocking warm orchestration on a daemon controller thread."""
        loop = asyncio.get_running_loop()
        result: asyncio.Future[int] = loop.create_future()

        def complete_exception(exc: BaseException) -> None:
            if not result.done():
                result.set_exception(exc)

        def complete_result(value: int) -> None:
            if not result.done():
                result.set_result(value)

        def run() -> None:
            try:
                value = self.warm(
                    config,
                    limit=limit,
                    max_payload_bytes=max_payload_bytes,
                    raw_artifact_id=raw_artifact_id,
                )
            except BaseException as exc:
                if not loop.is_closed():
                    loop.call_soon_threadsafe(complete_exception, exc)
            else:
                if not loop.is_closed():
                    loop.call_soon_threadsafe(complete_result, value)

        threading.Thread(target=run, name="polylogue-parse-warm-controller", daemon=True).start()
        try:
            return await result
        except asyncio.CancelledError:
            self.request_stop()
            raise

    def warm(
        self,
        config: Config,
        *,
        limit: int,
        max_payload_bytes: int,
        raw_artifact_id: str | None = None,
    ) -> int:
        """Run one tracked off-writer warm operation."""
        self._begin_warm_operation()
        try:
            return self._warm_impl(
                config,
                limit=limit,
                max_payload_bytes=max_payload_bytes,
                raw_artifact_id=raw_artifact_id,
            )
        finally:
            self._end_warm_operation()

    def _warm_impl(
        self,
        config: Config,
        *,
        limit: int,
        max_payload_bytes: int,
        raw_artifact_id: str | None = None,
    ) -> int:
        """Pre-parse pending census candidates outside any writer hold.

        When ``raw_artifact_id`` is supplied, discovery is narrowed to the
        authority component containing that seed.  The daemon's whale path
        uses this candidate-specific scope rather than reusing the ordinary
        archive-wide preview; the default keeps the ordinary conveyor
        contract unchanged.

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
            config,
            limit=limit,
            max_payload_bytes=max_payload_bytes,
            raw_artifact_id=raw_artifact_id,
        )
        return self._warm_raw_ids_impl(config, raw_ids=candidate_raw_ids, max_payload_bytes=max_payload_bytes)

    def warm_raw_ids(self, config: Config, *, raw_ids: Sequence[str], max_payload_bytes: int) -> int:
        """Pre-parse explicit raws under the same admission fence as ``warm``."""
        self._begin_warm_operation()
        try:
            return self._warm_raw_ids_impl(config, raw_ids=raw_ids, max_payload_bytes=max_payload_bytes)
        finally:
            self._end_warm_operation()

    def _warm_raw_ids_impl(self, config: Config, *, raw_ids: Sequence[str], max_payload_bytes: int) -> int:
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
        # polylogue-6lyh1: resolve the same APPEND fallback-identity hint the
        # sequential path recovers, so this warmer's parsed output matches
        # the writer-held pass exactly -- see census_parse_worker's docstring.
        append_raw_ids = [
            raw_id
            for raw_id in raw_ids
            if (descriptors.get(raw_id) is not None and descriptors[raw_id][3] is RawRevisionKind.APPEND)
        ]
        native_ids = _resolve_readonly_native_ids(archive_root, append_raw_ids)

        # Source descriptors provide the only safe pre-parse size proof.
        # Select a deterministic subset whose aggregate payload bytes fit the
        # cache's remaining admission budget before submitting any worker.
        # This bounds concurrent full-payload decodes; a raw that is too large
        # for the remaining budget is left for the writer-held fallback path.
        futures: dict[Future[Any], str] = {}
        payload_budget = min(self.cache.max_inflight_bytes, max_payload_bytes)
        reserved_this_warm = 0
        try:
            for raw_id in raw_ids:
                if self._stop_requested.is_set():
                    break
                descriptor = descriptors.get(raw_id)
                if descriptor is None:
                    continue
                provider, blob_hash, source_path, kind, payload_size = descriptor
                if payload_size > payload_budget - reserved_this_warm or not self._reserve_pending_payload(
                    raw_id, payload_size
                ):
                    continue
                reserved_this_warm += payload_size
                if self._stop_requested.is_set():
                    self._release_pending_payload(raw_id, None)
                    break
                native_id = native_ids.get(raw_id) if kind is RawRevisionKind.APPEND else None
                try:
                    future = self._executor.submit(
                        revision_backfill.census_parse_worker,
                        raw_id,
                        provider.value,
                        blob_hash,
                        source_path,
                        is_stream_record_provider(source_path, str(provider)),
                        blob_root_str,
                        source_db_path_str,
                        kind.value,
                        native_id,
                    )
                except BaseException:
                    self._release_pending_payload(raw_id, None)
                    raise
                futures[future] = raw_id
        except BaseException:
            # A failed mid-batch submit may leave a mix of queued and running
            # workers. Cancel queued futures, consume completed results, and
            # retain callbacks for running futures before the warm operation
            # releases its active-operation marker.
            self._retain_cleanup_callbacks(futures)
            raise

        warmed = 0
        completed = 0
        consumed: set[Future[Any]] = set()
        remaining = set(futures)
        deadline = time.monotonic() + self._warm_timeout_seconds
        while remaining:
            if self._stop_requested.is_set():
                break
            wait_timeout = min(0.1, max(0.0, deadline - time.monotonic()))
            if wait_timeout <= 0:
                break
            done, _ = wait(remaining, timeout=wait_timeout)
            if not done:
                continue
            for future in done:
                remaining.discard(future)
                completed += 1
                consumed.add(future)
                raw_id = futures[future]
                try:
                    try:
                        _raw_id, sessions, error = future.result()
                    except Exception:
                        logger.warning("parse-stage prefetch: worker failed for raw_id=%s", raw_id, exc_info=True)
                        continue
                    if error is not None or sessions is None:
                        continue
                    _provider, _blob_hash, _source_path, kind, payload_size = descriptors[raw_id]
                    tree_bytes = estimate_parsed_tree_bytes(sessions)
                    if tree_bytes > self._max_cached_tree_bytes:
                        logger.warning(
                            "parse-stage prefetch: raw_id=%s estimated parsed tree exceeds cache budget; "
                            "leaving it uncached for the writer-held fallback",
                            raw_id,
                        )
                        continue
                    if self.cache.try_admit(raw_id, sessions, payload_bytes=payload_size, revision_kind=kind):
                        self._register_cached_tree_bytes(raw_id, tree_bytes)
                        warmed += 1
                finally:
                    self._release_pending_payload(raw_id, future)

        if remaining:
            self._retain_cleanup_callbacks({future: futures[future] for future in remaining})
            pending = len(remaining)
            reason = "stop requested" if self._stop_requested.is_set() else "warm timeout"
            logger.warning(
                "parse-stage prefetch: %s; leaving %d unfinished raw(s) uncached",
                reason,
                pending,
            )
        return warmed

    def shutdown(self) -> None:
        self.request_stop()
        self._executor.shutdown(wait=False, cancel_futures=True)


#: Back-compat alias: every existing daemon call site and test imports this
#: class as ``DaemonParseStage`` (from ``polylogue.daemon.parse_prefetch``,
#: which re-exports it from here). Keeping the same name here too means
#: ``polylogue.sources.census_parse_stage.DaemonParseStage`` and
#: ``polylogue.daemon.parse_prefetch.DaemonParseStage`` are the exact same
#: object, not two types that happen to look alike.
DaemonParseStage = CensusParseStage


__all__ = [
    "CensusParseStage",
    "DaemonParseStage",
    "daemon_parse_stage_max_cached_tree_bytes",
    "daemon_parse_stage_max_inflight_bytes",
    "daemon_parse_stage_warm_timeout_seconds",
    "daemon_parse_stage_worker_count",
    "estimate_parsed_tree_bytes",
]

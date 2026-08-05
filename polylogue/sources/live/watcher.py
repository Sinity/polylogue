"""Live JSONL session watcher.

Watches one or more roots for ``*.jsonl`` changes via ``watchfiles`` and
ingests new or grown files through the archive pipeline. Idempotent via
content-hash dedup; the cursor table suppresses re-work when the stored
content fingerprint and parser fingerprint still match the file.

Files are batched: all changed files within a debounce window are collected
and ingested in a single pipeline call. This avoids the O(n²) problem where
each file triggered a full source-tree rescan via ``parse_file()``.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import stat as stat_module
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from polylogue.core.enums import Origin
from polylogue.core.sources import provider_from_origin
from polylogue.logging import get_logger
from polylogue.sources.hooks import drain_hook_event_spool, hook_spool_root, pending_hook_spool_dir
from polylogue.sources.live.acquisition_log import log_unclaimed_file
from polylogue.sources.live.batch import (
    CursorAuthorityBlockedError,
    LiveBatchEventEmitter,
    LiveBatchProcessor,
    fingerprint_file,
)
from polylogue.sources.live.batch_support import (
    _archive_blob_exists,
    cursor_ctime_ns,
    cursor_prefix_hash,
    cursor_tail_hash,
    encode_cursor_hash_authority,
    sha256_range_from_path,
    tail_hash_and_last_complete_newline_from_path,
    tail_hash_from_path,
)
from polylogue.sources.live.cursor import CursorObservationRebase, CursorRecord, CursorStore
from polylogue.sources.live.deferred_cursor import record_deferred_append_cursor
from polylogue.sources.live.metrics import LiveBatchMetrics
from polylogue.sources.live.parse_prefetch import LiveParseStage
from polylogue.sources.sqlite_snapshot import is_sqlite_path, sqlite_database_for_sidecar, sqlite_source_revision

if TYPE_CHECKING:
    from polylogue.api import Polylogue

logger = get_logger(__name__)
_PARSER_FINGERPRINT = "live-batched-v2"
# One bounded writer hold per hook-spool drain batch; the drain loops until
# the backlog is gone, releasing the writer between batches.
_HOOK_SPOOL_DRAIN_BATCH_LIMIT = 250
# A catch-up writer owns the only archive writer for the whole chunk.  The
# former 50-file/64-MiB envelope held it for 14+ minutes on the real archive,
# starving fresh watcher events.  Keep historical convergence fair by
# yielding after a handful of files; one individually large source still owns
# one bounded logical-session write, but cannot be bundled with dozens more.
_CATCH_UP_MAX_BATCH_FILES = 4
_CATCH_UP_MAX_BATCH_BYTES = 16 * 1024 * 1024
_CATCH_UP_HOT_FILE_AGE_S = 60.0 * 60.0
# polylogue-11cg9: the file/byte caps above bound a catch-up chunk's *size*
# but not the *time* a single full-ingest pass can hold the sole archive
# writer -- a handful of files, or one slow-to-parse file, can still exceed
# them by any margin (the original de2a incident was an in-size-bounds 7 MB
# chunk that held the writer for 860s). Mirrors de2a's
# ``_RAW_MATERIALIZATION_MAX_PASS_SECONDS`` / qlae's
# ``_DRIVE_CATCHUP_MAX_PASS_SECONDS`` constant and value -- checked *between*
# full-ingest records/groups (a single session write cannot be split
# mid-transaction), never mid-record.
_LIVE_INGEST_MAX_PASS_SECONDS = 20.0
_INCOMPLETE_APPEND_PROBE_BYTES = 64 * 1024 * 1024
# polylogue-2qrx: minimum age a deferred incomplete-tail observation must
# reach (``cursor.updated_at`` unchanged, i.e. the stat-match fast path kept
# firing) before escalating to an unbounded full-tail probe. An ordinary
# in-progress writer routinely leaves a genuinely-incomplete trailing record
# (no closing newline yet) between two polls milliseconds apart -- that must
# NOT be treated as "the writer is done" just because the stat happened to
# match on a second, immediate check. Only a source that has been sitting at
# the exact same byte state for a long time is plausibly finished rather than
# merely paused; matches the periodic catch-up safety net's own duty cycle
# (``_PERIODIC_CATCH_UP_MAX_INTERVAL_S``) so the escalation cannot fire
# before that safety net would have re-observed the file anyway.
_STUCK_DEFERRED_APPEND_AGE_S = 60.0 * 60.0
# Filesystem notifications are the real-time delivery path.  This sweep is a
# recovery mechanism for notifications missed while the daemon was unavailable
# or a watch backend was briefly unhealthy.  Keeping it at the watch cadence
# made a large, otherwise-idle archive continuously rescan itself.
_PERIODIC_CATCH_UP_INTERVAL_S = 5.0 * 60.0
_PERIODIC_CATCH_UP_MAX_INTERVAL_S = 60.0 * 60.0
INBOX_SOURCE_SUFFIXES = (".jsonl", ".zip", ".json", ".ndjson", ".db", ".sqlite", ".sqlite3")


class _ArchivedCursorReconciliation(str, Enum):
    """Whether archive evidence can safely restore a live cursor."""

    RECONCILED = "reconciled"
    UNAVAILABLE = "unavailable"
    INCOMPATIBLE = "incompatible"


def _stage_timing_summary(stage_timings_s: dict[str, float], *, limit: int = 8) -> str:
    if not stage_timings_s:
        return "none"
    ordered = sorted(stage_timings_s.items(), key=lambda item: item[1], reverse=True)
    shown = ",".join(f"{name}:{elapsed:.3f}" for name, elapsed in ordered[:limit])
    omitted = len(ordered) - limit
    if omitted <= 0:
        return shown
    return f"{shown},+{omitted} more"


def _log_ingest_metrics(prefix: str, metrics: LiveBatchMetrics) -> None:
    """Log actual live-ingest read work separately from candidate file size."""
    input_bytes = getattr(metrics, "input_bytes", 0)
    source_payload_read_bytes = getattr(metrics, "source_payload_read_bytes", 0)
    read_amp = source_payload_read_bytes / input_bytes if input_bytes > 0 else 0.0
    stage_timings_s = getattr(metrics, "stage_timings_s", {})
    stage_summary = _stage_timing_summary(stage_timings_s if isinstance(stage_timings_s, dict) else {})
    logger.info(
        "%s complete: read=%.1f MB input=%.1f MB read_amp=%.6fx append_files=%d full_files=%d "
        "succeeded=%d failed=%d parse_s=%.3f convergence_s=%.3f stages=%s "
        "wal_before_checkpoint=%.1f MB wal_after_checkpoint=%.1f MB wal_busy_pages=%d time_budget_exceeded=%s",
        prefix,
        source_payload_read_bytes / 1e6,
        input_bytes / 1e6,
        read_amp,
        getattr(metrics, "append_file_count", 0),
        getattr(metrics, "full_file_count", 0),
        getattr(metrics, "succeeded_file_count", 0),
        getattr(metrics, "failed_file_count", 0),
        getattr(metrics, "parse_time_s", 0.0),
        getattr(metrics, "convergence_time_s", 0.0),
        stage_summary,
        getattr(metrics, "wal_bytes_before_checkpoint_max", 0) / 1e6,
        getattr(metrics, "wal_bytes_after_checkpoint_max", 0) / 1e6,
        getattr(metrics, "wal_busy_pages_total", 0),
        getattr(metrics, "time_budget_exceeded", False),
    )
    if getattr(metrics, "time_budget_exceeded", False):
        logger.info(
            "%s: max_pass_seconds budget exceeded -- remaining files deferred to the next tick (polylogue-11cg9)",
            prefix,
        )


def _log_unclaimed_catch_up_candidate(path: Path, *, source_name: str, reason: str) -> None:
    """Log one file the catch-up scan reached but no source suffix accepted.

    Best-effort ``stat`` for size/mtime -- a file that vanished between the
    ``os.walk`` listing and this call is still worth a log record (it WAS
    seen and unclaimed), just without size/mtime detail.
    """
    try:
        stat_result = path.stat()
        size: int | None = stat_result.st_size
        mtime: float | None = stat_result.st_mtime
    except OSError:
        size, mtime = None, None
    log_unclaimed_file(path=path, size=size, mtime=mtime, reason=reason, source_name=source_name)


@dataclass(frozen=True, slots=True)
class WatchSource:
    """A directory to watch for live session files."""

    name: str
    root: Path
    suffixes: tuple[str, ...] = (".jsonl",)
    ignored_dir_names: frozenset[str] = frozenset({".git", "__pycache__", "node_modules", "venv", ".venv"})

    def exists(self) -> bool:
        return self.root.exists()

    def accepts(self, path: Path) -> bool:
        name = path.name.lower()
        return any(name.endswith(suffix) for suffix in self.suffixes)

    def ignores_directory(self, path: Path) -> bool:
        """Return whether a subtree cannot contain a live source artifact."""
        return path.name in self.ignored_dir_names


@dataclass(frozen=True, slots=True)
class CandidateSourceFile:
    """One statted source file candidate from a catch-up scan."""

    path: Path
    source_name: str
    suffix: str
    stat: os.stat_result


@dataclass(frozen=True, slots=True)
class CatchUpPlan:
    """Planned catch-up work after bulk cursor comparison."""

    candidates: tuple[CandidateSourceFile, ...]
    needed: tuple[Path, ...]
    skipped_file_count: int
    needed_bytes: int


class LiveWatcher:
    """Async watcher that ingests grown JSONL files in batches.

    On startup (catch-up), all files across all roots are fingerprinted
    and the changed ones are ingested in a single batch. During live
    watching, files that change within the debounce window are batched
    together.
    """

    def __init__(
        self,
        polylogue: Polylogue,
        sources: Iterable[WatchSource],
        *,
        debounce_s: float = 2.0,
        cursor: CursorStore | None = None,
        max_workers: int | None = None,
        converger: object | None = None,  # DaemonConverger | None — avoids circular import
        event_emitter: LiveBatchEventEmitter | None = None,
        catch_up_event_emitter: Callable[..., None] | None = None,
        write_coordinator: object | None = None,
        parse_stage: LiveParseStage | None = None,
    ) -> None:
        self._polylogue = polylogue
        self._sources = tuple(sources)
        self._debounce_s = debounce_s
        self._cursor = cursor or CursorStore(
            _cursor_db_path(polylogue),
            initialize=write_coordinator is None,
            ops_db_path=Path(polylogue.archive_root) / "ops.db",
        )
        self._max_workers = max_workers
        self._converger = converger
        self._write_coordinator = write_coordinator
        self._catch_up_event_emitter = catch_up_event_emitter
        # polylogue-wf8a: always on -- pre-parsing runs entirely BEFORE the
        # write coordinator is ever asked for the writer hold
        # (``LiveBatchProcessor._ingest_full_paths``), so it never contends
        # with an active writer thread for the GIL regardless of interpreter
        # build (see ``polylogue.sources.live.parse_prefetch`` for the full
        # safety argument, identical in shape to ``DaemonParseStage``). An
        # explicit ``parse_stage`` always wins (tests / callers that want to
        # own the stage's lifecycle themselves); otherwise one is created
        # here, owned by this watcher, and shut down in ``stop()``.
        self._owns_parse_stage = parse_stage is None
        self._parse_stage: LiveParseStage | None = parse_stage if parse_stage is not None else LiveParseStage()
        self._pending_paths: set[Path] = set()
        self._pending_scheduled = False
        self._drain_task: asyncio.Task[None] | None = None
        self._failed_retry_task: asyncio.Task[None] | None = None
        self._periodic_catch_up_task: asyncio.Task[None] | None = None
        self._failed_retry_deadline: float | None = None
        self._last_enqueue_at = 0.0
        self._last_batch_at: float = 0.0
        self._batch_lock = asyncio.Lock()
        self._ingest_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._catch_up_complete = asyncio.Event()
        self._archived_cursor_conns: tuple[sqlite3.Connection, sqlite3.Connection] | None = None
        # Set once per reconciliation scope: True when the index tier has no
        # materialized sessions at all despite source.db holding successfully
        # parsed raw material -- the post-index-reset signature (polylogue-emx2).
        # While true, cursor-trust skip decisions must be corroborated
        # per-candidate against index presence instead of being taken on faith.
        self._archived_cursor_index_untrusted = False
        self._batch_processor = LiveBatchProcessor(
            polylogue,
            self._sources,
            cursor=self._cursor,
            parser_fingerprint=lambda: _PARSER_FINGERPRINT,
            converger=converger,
            stop_requested=self._stop.is_set,
            event_emitter=event_emitter,
            sync_runner=self._run_writer_sync,
            parse_stage=self._parse_stage,
        )

    async def _run_writer_sync(
        self,
        actor: str,
        function: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run blocking watcher writes without joining the loop executor at exit."""
        run_sync = getattr(self._write_coordinator, "run_sync", None)
        if callable(run_sync):
            return await run_sync(actor, function, *args, **kwargs)
        return await asyncio.to_thread(function, *args, **kwargs)

    @property
    def catch_up_complete(self) -> asyncio.Event:
        return self._catch_up_complete

    async def run(self) -> None:
        # Hook commands create their first pending envelope lazily.  Ensure the
        # nested root exists before ``awatch`` snapshots its roots, otherwise a
        # daemon that starts before the first hook event never sees that file.
        for source in self._sources:
            if source.name == "hooks":
                source.root.mkdir(parents=True, exist_ok=True)
        roots = [s.root for s in self._sources if s.exists()]
        if not roots:
            logger.warning("live.watcher: no source roots exist; nothing to watch")
            self._catch_up_complete.set()
            return

        # Register the filesystem watch before catch-up.  Starting it only
        # after catch-up left a blind interval where a writer could append
        # after its file was scanned but before ``awatch`` took its snapshot.
        watch_task = asyncio.create_task(self._watch_changes(roots))
        await asyncio.sleep(0)
        try:
            try:
                await self._catch_up(roots)
            finally:
                self._catch_up_complete.set()
            self._schedule_failed_retry_scan()
            self._ensure_pending_scheduled()
            self._periodic_catch_up_task = asyncio.create_task(self._periodic_catch_up(roots))

            logger.info("live.watcher: watching %s", ", ".join(str(r) for r in roots))
            await watch_task
        finally:
            if not watch_task.done():
                watch_task.cancel()
            with suppress(asyncio.CancelledError):
                await watch_task
            self._cancel_periodic_catch_up()

    async def _watch_changes(self, roots: list[Path]) -> None:
        from watchfiles import Change, awatch

        async for changes in awatch(
            *roots,
            watch_filter=self._watch_filter,
            stop_event=self._stop,
            recursive=True,
        ):
            for change, raw_path in changes:
                if change is Change.deleted:
                    continue
                path = self._canonical_watch_path(Path(raw_path))
                if path is None:
                    continue
                if not self._source_accepts(path):
                    continue
                if self._is_hook_spool_path(path):
                    await self._drain_hook_spool()
                    continue
                self._enqueue(path)

    def stop(self) -> None:
        self._stop.set()
        self._cancel_failed_retry_task()
        self._cancel_periodic_catch_up()
        if self._parse_stage is not None and self._owns_parse_stage:
            self._parse_stage.shutdown()

    def cancel_pending(self) -> None:
        task = self._drain_task
        if task is not None and not task.done():
            task.cancel()
        self._drain_task = None
        self._pending_scheduled = False
        self._cancel_failed_retry_task()
        self._cancel_periodic_catch_up()

    async def _periodic_catch_up(self, roots: list[Path]) -> None:
        delay_s = _PERIODIC_CATCH_UP_INTERVAL_S
        while not self._stop.is_set():
            await asyncio.sleep(delay_s)
            if self._stop.is_set():
                return
            try:
                await self._catch_up(roots)
            except sqlite3.OperationalError as exc:
                if not _is_database_locked(exc):
                    raise
                logger.warning("live.watcher: archive busy during periodic catch-up; will retry")
            # A periodic pass is deliberately a low-duty-cycle safety net.
            # Event-driven batches and explicit failed-file retry wakeups keep
            # normal writes and known failures prompt; repeatedly walking every
            # source tree does neither.
            delay_s = min(delay_s * 2, _PERIODIC_CATCH_UP_MAX_INTERVAL_S)

    def _cancel_periodic_catch_up(self) -> None:
        task = self._periodic_catch_up_task
        if task is not None and not task.done():
            task.cancel()
        self._periodic_catch_up_task = None

    # ------------------------------------------------------------------
    # Catch-up: batch all changed files
    # ------------------------------------------------------------------

    async def _catch_up(self, roots: list[Path]) -> None:
        candidates = self._scan_catch_up_candidates(roots)
        if not candidates:
            await self._drain_hook_spool()
            return

        now = time.time()
        hot_candidates = tuple(
            candidate for candidate in candidates if now - candidate.stat.st_mtime <= _CATCH_UP_HOT_FILE_AGE_S
        )
        hot_paths = {candidate.path for candidate in hot_candidates}
        cold_candidates = tuple(candidate for candidate in candidates if candidate.path not in hot_paths)
        if hot_candidates:
            logger.info(
                "live.watcher: prioritizing %d recently modified source file(s) before backlog catch-up",
                len(hot_candidates),
            )
        for group in (hot_candidates, cold_candidates):
            if self._stop.is_set() or not group:
                break
            await self._catch_up_candidates(group)
        await self._drain_hook_spool()

    async def _catch_up_candidates(self, candidates: tuple[CandidateSourceFile, ...]) -> None:
        """Plan and ingest one priority class of catch-up candidates."""
        plan_holder: list[CatchUpPlan] = []

        async def prepare_catch_up() -> None:
            # The preflight must precede both cursor initialization and the
            # planning pass. ``_plan_catch_up`` can reconcile missing cursors
            # and rebase matching filesystem observations before ingestion has
            # a chance to apply its own gate.
            self._batch_processor.require_cursor_authority()
            await self._run_writer_sync("watcher.catch_up.cursor_initialize", self._cursor.initialize)
            self._batch_processor.require_cursor_authority()
            logger.info("live.watcher: catch-up scan over %d file(s)", len(candidates))
            plan_holder.append(self._plan_catch_up(candidates))

        try:
            await self._run_coordinated("watcher.catch_up.prefilter", prepare_catch_up)
        except CursorAuthorityBlockedError as exc:
            logger.warning("live.watcher: catch-up planning refused by cursor authority: %s", exc)
            return
        plan = plan_holder[0]
        if not plan.needed:
            return
        operation_id = f"watcher-catch-up:{uuid.uuid4()}"
        cycle_started = time.perf_counter()
        await self._emit_catch_up_cycle(
            operation_id=operation_id,
            phase="start",
            backlog_start=len(plan.candidates),
            backlog_end=len(plan.candidates),
            discovered=len(plan.candidates),
            attempted=0,
            skipped=0,
            ingested=0,
            quarantine_count=0,
            errors_by_kind={},
            cursor_before=None,
            cursor_after=None,
            duration_ms=0.0,
            stage_timings_s={},
            repair=None,
        )
        candidate_by_path = {candidate.path: candidate for candidate in plan.candidates}
        chunks = tuple(self._chunk_catch_up_paths(plan.needed, candidate_by_path))
        attempted = 0
        ingested = 0
        failed = 0
        stage_timings_s: dict[str, float] = {}
        logger.info(
            "live.watcher: catch-up ingesting %d file(s) (%.1f MB), skipped=%d, chunks=%d",
            len(plan.needed),
            plan.needed_bytes / 1e6,
            plan.skipped_file_count,
            len(chunks),
        )
        try:
            for index, chunk in enumerate(chunks, start=1):
                if self._stop.is_set():
                    await self._emit_catch_up_terminal(
                        operation_id, "stopped", plan, attempted, ingested, failed, stage_timings_s, cycle_started
                    )
                    return
                chunk_bytes = sum(candidate_by_path[path].stat.st_size for path in chunk)
                logger.info(
                    "live.watcher: catch-up chunk %d/%d ingesting %d file(s) (%.1f MB)",
                    index,
                    len(chunks),
                    len(chunk),
                    chunk_bytes / 1e6,
                )
                chunk_index = index
                chunk_paths = list(chunk)

                async def ingest_chunk(
                    chunk_index: int = chunk_index,
                    chunk_paths: list[Path] = chunk_paths,
                ) -> None:
                    nonlocal attempted, ingested, failed
                    metrics = await self._ingest_files(
                        chunk_paths,
                        queued_file_count=len(plan.candidates) if chunk_index == 1 else len(chunk_paths),
                        skipped_file_count=plan.skipped_file_count if chunk_index == 1 else 0,
                    )
                    if metrics is not None:
                        _log_ingest_metrics(f"live.watcher: catch-up chunk {chunk_index}/{len(chunks)}", metrics)
                        attempted += metrics.needed_file_count
                        ingested += metrics.succeeded_file_count
                        failed += metrics.failed_file_count
                        for stage, elapsed_s in metrics.stage_timings_s.items():
                            stage_timings_s[stage] = stage_timings_s.get(stage, 0.0) + elapsed_s
                        if (
                            getattr(metrics, "succeeded_file_count", 0) == 0
                            and getattr(metrics, "failed_file_count", 0) == 0
                        ):
                            self._defer_unaccounted_failed_retries(chunk_paths)

                await self._run_coordinated("watcher.catch_up.chunk", ingest_chunk)
            if self._stop.is_set():
                await self._emit_catch_up_terminal(
                    operation_id, "stopped", plan, attempted, ingested, failed, stage_timings_s, cycle_started
                )
                return
            backlog_end = max(0, len(plan.candidates) - plan.skipped_file_count - ingested)
            await self._emit_catch_up_cycle(
                operation_id=operation_id,
                phase="end",
                backlog_start=len(plan.candidates),
                backlog_end=backlog_end,
                discovered=len(plan.candidates),
                attempted=attempted,
                skipped=plan.skipped_file_count,
                ingested=ingested,
                quarantine_count=0,
                errors_by_kind={"ingest_failed": failed} if failed else {},
                cursor_before=None,
                cursor_after=None,
                duration_ms=(time.perf_counter() - cycle_started) * 1000.0,
                stage_timings_s=stage_timings_s,
                repair={"required": failed, "performed": 0, "remaining": backlog_end},
            )
            await self._emit_catch_up_terminal(
                operation_id, "success", plan, attempted, ingested, failed, stage_timings_s, cycle_started, backlog_end
            )
            self._schedule_failed_retry_scan()
        except asyncio.CancelledError:
            await self._emit_catch_up_terminal(
                operation_id, "cancelled", plan, attempted, ingested, failed, stage_timings_s, cycle_started
            )
            raise
        except CursorAuthorityBlockedError as exc:
            logger.warning("live.watcher: catch-up refused by cursor authority: %s", exc)
            return
        except BaseException:
            await self._emit_catch_up_terminal(
                operation_id, "failure", plan, attempted, ingested, failed, stage_timings_s, cycle_started
            )
            raise

    async def _drain_hook_spool(self) -> None:
        """Acknowledge hook envelopes only after their source-tier write commits.

        Drains in bounded batches, releasing the writer between them, so a
        large spool backlog cannot monopolize the single writer against
        live ingest and catch-up chunks.
        """

        try:
            self._batch_processor.require_cursor_authority()
        except CursorAuthorityBlockedError as exc:
            logger.warning("live.watcher: hook-spool drain refused by cursor authority: %s", exc)
            return

        total_acknowledged = 0
        while True:
            result = await self._run_writer_sync(
                "watcher.hook_spool.drain",
                drain_hook_event_spool,
                Path(self._polylogue.archive_root),
                root=self._hook_spool_root(),
                limit=_HOOK_SPOOL_DRAIN_BATCH_LIMIT,
            )
            total_acknowledged += result.acknowledged
            if result.failed:
                logger.warning(
                    "live.watcher: hook spool drain left %d event(s) pending",
                    result.failed,
                )
            if result.acknowledged == 0 or result.remaining <= result.failed:
                break
        if total_acknowledged:
            logger.info("live.watcher: acknowledged %d hook spool event(s)", total_acknowledged)

    def _scan_catch_up_candidates(self, roots: list[Path]) -> tuple[CandidateSourceFile, ...]:
        root_set = {root.resolve() for root in roots}
        candidates: list[CandidateSourceFile] = []
        for source in self._sources:
            if not source.exists() or source.root.resolve() not in root_set:
                continue
            if source.name == "hooks":
                continue
            for directory, dirnames, filenames in os.walk(source.root, followlinks=False):
                dirnames[:] = [
                    dirname for dirname in dirnames if not source.ignores_directory(Path(directory) / dirname)
                ]
                for filename in filenames:
                    path = Path(directory) / filename
                    if not source.accepts(path):
                        # Unclaimed-file sweep (mission item 2): a file this
                        # source's own root walk reached but whose suffix no
                        # detector is configured to accept at all. Logged here
                        # -- the real catch-up scan, run at daemon startup and
                        # on every periodic sweep -- rather than only from a
                        # standalone diagnostic, so the record exists whether
                        # or not an operator remembers to run one.
                        _log_unclaimed_catch_up_candidate(
                            path,
                            source_name=source.name,
                            reason=f"suffix not in watched set {source.suffixes} for source {source.name!r}",
                        )
                        continue
                    try:
                        stat = path.stat()
                    except FileNotFoundError:
                        continue
                    if not stat_module.S_ISREG(stat.st_mode):
                        continue
                    candidates.append(
                        CandidateSourceFile(
                            path=path,
                            source_name=source.name,
                            suffix=path.suffix,
                            stat=stat,
                        )
                    )
        return tuple(_interleave_by_source(candidates))

    def _plan_catch_up(self, candidates: tuple[CandidateSourceFile, ...]) -> CatchUpPlan:
        if not candidates:
            return CatchUpPlan(candidates=(), needed=(), skipped_file_count=0, needed_bytes=0)
        cursor_records = self._cursor.get_records(candidate.path for candidate in candidates)
        needed: list[Path] = []
        rebases: list[CursorObservationRebase] = []
        skipped = 0
        needed_bytes = 0
        with self._archived_cursor_reconciliation_scope():
            for candidate in candidates:
                if self._stop.is_set():
                    break
                if self._needs_work_from_state(
                    candidate.path,
                    stat=candidate.stat,
                    cursor=cursor_records.get(candidate.path),
                    rebase_queue=rebases,
                ):
                    needed.append(candidate.path)
                    needed_bytes += candidate.stat.st_size
                else:
                    skipped += 1
        if rebases:
            self._cursor.rebase_authoritative_observations(rebases)
        return CatchUpPlan(
            candidates=candidates,
            needed=tuple(needed),
            skipped_file_count=skipped,
            needed_bytes=needed_bytes,
        )

    def _chunk_catch_up_paths(
        self,
        paths: tuple[Path, ...],
        candidate_by_path: dict[Path, CandidateSourceFile],
    ) -> tuple[tuple[Path, ...], ...]:
        chunks: list[tuple[Path, ...]] = []
        current: list[Path] = []
        current_bytes = 0
        for path in paths:
            size = candidate_by_path[path].stat.st_size
            would_exceed_count = len(current) >= _CATCH_UP_MAX_BATCH_FILES
            would_exceed_bytes = current_bytes > 0 and current_bytes + size > _CATCH_UP_MAX_BATCH_BYTES
            if current and (would_exceed_count or would_exceed_bytes):
                chunks.append(tuple(current))
                current = []
                current_bytes = 0
            current.append(path)
            current_bytes += size
        if current:
            chunks.append(tuple(current))
        return tuple(chunks)

    # ------------------------------------------------------------------
    # Live: debounced batch scheduling
    # ------------------------------------------------------------------

    def _enqueue(self, path: Path) -> None:
        """Enqueue a path for batched ingestion after debounce."""
        self._pending_paths.add(path)
        self._last_enqueue_at = time.monotonic()
        self._ensure_pending_scheduled()

    def _ensure_pending_scheduled(self) -> None:
        if not self._pending_paths or self._stop.is_set():
            return
        if self._drain_task is None or self._drain_task.done():
            self._pending_scheduled = True
            self._drain_task = asyncio.create_task(self._debounced_batch())

    def _schedule_failed_retry_scan(self) -> None:
        if self._stop.is_set():
            return
        due_paths: list[Path] = []
        next_retry_at: datetime | None = None
        for record in self._cursor.list_retry_records():
            path = Path(record.source_path)
            if not self._source_accepts(path):
                continue
            if _retry_due(record.next_retry_at):
                due_paths.append(path)
                continue
            retry_at = _parse_retry_at(record.next_retry_at)
            if retry_at is not None and (next_retry_at is None or retry_at < next_retry_at):
                next_retry_at = retry_at
        if due_paths:
            logger.info("live.watcher: scheduling %d failed file(s) whose retry is due", len(due_paths))
            self._pending_paths.update(due_paths)
            self._ensure_pending_scheduled()
        if next_retry_at is not None:
            self._schedule_failed_retry_wakeup(next_retry_at)

    def _schedule_failed_retry_wakeup(self, retry_at: datetime) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        delay_s = max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
        deadline = loop.time() + delay_s
        if (
            self._failed_retry_task is not None
            and not self._failed_retry_task.done()
            and self._failed_retry_deadline is not None
            and self._failed_retry_deadline <= deadline
        ):
            return
        self._cancel_failed_retry_task()
        self._failed_retry_deadline = deadline
        self._failed_retry_task = asyncio.create_task(self._wake_failed_retries(delay_s))

    async def _wake_failed_retries(self, delay_s: float) -> None:
        try:
            await asyncio.sleep(delay_s)
            self._failed_retry_deadline = None
            self._failed_retry_task = None
            self._schedule_failed_retry_scan()
        except asyncio.CancelledError:
            raise

    def _cancel_failed_retry_task(self) -> None:
        task = self._failed_retry_task
        if task is not None and not task.done():
            task.cancel()
        self._failed_retry_task = None
        self._failed_retry_deadline = None

    async def _debounced_batch(self) -> None:
        """Wait for the debounce window, then drain pending paths serially."""
        try:
            await asyncio.sleep(self._debounce_s)

            while not self._stop.is_set():
                await self._wait_for_pending_quiet()

                flushed = await self._flush_pending()
                if not flushed:
                    break
        finally:
            if self._pending_paths and not self._stop.is_set():
                self._drain_task = asyncio.create_task(self._debounced_batch())
            else:
                self._pending_scheduled = False
                self._drain_task = None

    async def _wait_for_pending_quiet(self) -> None:
        """Wait until no enqueue event lands during the quiet window."""
        quiet_s = self._debounce_s
        while self._pending_paths and not self._stop.is_set():
            last_enqueue_at = self._last_enqueue_at
            elapsed_s = time.monotonic() - last_enqueue_at
            if elapsed_s >= quiet_s:
                return
            await asyncio.sleep(max(quiet_s - elapsed_s, 0.01))
            if self._last_enqueue_at == last_enqueue_at:
                return

    async def _flush_pending(self) -> bool:
        """Flush one pending path snapshot and report whether work ran."""
        async with self._batch_lock:
            if not self._pending_paths:
                return False
            paths = list(self._pending_paths)
            self._pending_paths.clear()

        async def flush_batch() -> None:
            # Filtering a changed-file batch invokes cursor reconciliation and
            # lifecycle actuators, so the source-selection proof must be
            # consumed before initialization or any stateful decision.
            self._batch_processor.require_cursor_authority()
            await self._run_writer_sync("watcher.live_batch.cursor_initialize", self._cursor.initialize)
            self._batch_processor.require_cursor_authority()
            # Filter to files that actually need work.
            cursor_records = self._cursor.get_records(paths)
            needed = []
            with self._archived_cursor_reconciliation_scope():
                for path in paths:
                    try:
                        stat = path.stat()
                    except FileNotFoundError:
                        continue
                    if self._needs_work_from_state(path, stat=stat, cursor=cursor_records.get(path)):
                        needed.append(path)
            if not needed:
                self._defer_unaccounted_failed_retries(paths)
                return

            logger.info("live.watcher: batching %d changed file(s)", len(needed))
            metrics = await self._ingest_files(
                needed,
                queued_file_count=len(paths),
                skipped_file_count=len(paths) - len(needed),
            )
            if metrics is not None:
                _log_ingest_metrics("live.watcher: changed-file batch", metrics)
                if (
                    getattr(metrics, "succeeded_file_count", 0) == 0
                    and getattr(metrics, "failed_file_count", 0) == 0
                    and needed
                ):
                    self._defer_unaccounted_failed_retries(needed)
            self._schedule_failed_retry_scan()

        try:
            await self._run_coordinated("watcher.live_batch", flush_batch)
        except CursorAuthorityBlockedError as exc:
            logger.warning("live.watcher: changed-file batch refused by cursor authority: %s", exc)
            # Authority denial must leave both durable cursor state and the
            # in-memory work queue intact.  Otherwise the source is invisible
            # until a later catch-up scan instead of retrying on the next
            # authorized debounce flush.
            async with self._batch_lock:
                self._pending_paths.update(paths)
            return True
        except sqlite3.OperationalError as exc:
            if not _is_database_locked(exc):
                raise
            logger.warning("live.watcher: archive busy; requeueing %d changed file(s)", len(paths))
            async with self._batch_lock:
                self._pending_paths.update(paths)
            await asyncio.sleep(self._debounce_s)
        return True

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _defer_unaccounted_failed_retries(self, paths: list[Path]) -> None:
        """Move no-op failed retries back into backoff instead of hot-looping."""
        deferred = 0
        for path in paths:
            record = self._cursor.get_record(path)
            if record is None or record.excluded or record.failure_count <= 0:
                continue
            if not _retry_due(record.next_retry_at):
                continue
            self._cursor.mark_failed(path)
            deferred += 1
        if deferred:
            logger.warning(
                "live.watcher: deferred %d failed retry file(s) after no-op ingest batch",
                deferred,
            )

    def _needs_work(self, path: Path) -> bool:
        """Return True if the file is new, grown, or fingerprint-changed."""
        try:
            stat = path.stat()
        except FileNotFoundError:
            return False
        cursor = self._cursor.get_record(path)
        return self._needs_work_from_state(path, stat=stat, cursor=cursor)

    def _needs_work_from_state(
        self,
        path: Path,
        *,
        stat: os.stat_result,
        cursor: CursorRecord | None,
        rebase_queue: list[CursorObservationRebase] | None = None,
    ) -> bool:
        """Decide whether ``path`` needs (re-)ingestion, corroborated against the index.

        Delegates the byte/fingerprint-level decision to
        :meth:`_needs_work_from_state_uncorroborated`. When that would skip
        the file (trusting a cursor claim) but the index tier globally shows
        no corroborating material (:attr:`_archived_cursor_index_untrusted`,
        set once per catch-up scan), the skip is demoted to "needed" unless a
        per-file existence check on ``path`` specifically finds its raw
        material already materialized -- see polylogue-emx2.
        """
        needs_work = self._needs_work_from_state_uncorroborated(
            path, stat=stat, cursor=cursor, rebase_queue=rebase_queue
        )
        if needs_work or not self._archived_cursor_index_untrusted:
            return needs_work
        if self._cursor_skip_corroborated_by_index(path):
            return False
        logger.warning(
            "live.watcher: demoting uncorroborated cursor skip to needed for %s (index cannot show materialized raw)",
            path,
        )
        return True

    def _needs_work_from_state_uncorroborated(
        self,
        path: Path,
        *,
        stat: os.stat_result,
        cursor: CursorRecord | None,
        rebase_queue: list[CursorObservationRebase] | None = None,
    ) -> bool:
        size = stat.st_size
        if cursor is None:
            if not self._reconcile_archived_cursor(path, stat=stat):
                return True
            cursor = self._cursor.get_record(path)
            return cursor is not None and size > cursor.byte_offset
        if cursor.excluded:
            identity_unchanged = (
                cursor.byte_size,
                cursor.st_dev,
                cursor.st_ino,
                cursor.mtime_ns,
            ) == (size, stat.st_dev, stat.st_ino, stat.st_mtime_ns)
            # polylogue-ix5r: exclusion revival was previously bound only to
            # file identity, so a parser fix could never revive a cursor
            # excluded before that fix shipped -- the file itself never
            # changes, so ``identity_unchanged`` stays True forever and the
            # cursor stays dark until someone manually re-ingests it. A
            # parser fingerprint change is exactly the other legitimate
            # reason a previously-poisoned observation deserves a fresh
            # attempt: the code that failed to parse it no longer exists.
            if identity_unchanged and cursor.parser_fingerprint == _PARSER_FINGERPRINT:
                return False
            self._cursor.revive_replaced_exclusion(
                path,
                byte_size=size,
                st_dev=stat.st_dev,
                st_ino=stat.st_ino,
                mtime_ns=stat.st_mtime_ns,
                current_parser_fingerprint=_PARSER_FINGERPRINT,
            )
            return True
        if cursor.failure_count == 0 and cursor.content_fingerprint is None and cursor.next_retry_at is not None:
            if not _retry_due(cursor.next_retry_at):
                return False
            reconciliation = self._reconcile_archived_cursor_outcome(path, stat=stat)
            if reconciliation is _ArchivedCursorReconciliation.RECONCILED:
                reconciled = self._cursor.get_record(path)
                return reconciled is not None and size > reconciled.byte_offset
            if reconciliation is _ArchivedCursorReconciliation.UNAVAILABLE:
                self._cursor.defer_full_cursor_reconciliation(path)
                return False
            self._invalidate_deferred_full_cursor(path, stat=stat)
            return True
        if cursor.failure_count > 0:
            if self._reconcile_archived_cursor(path, stat=stat):
                cursor = self._cursor.get_record(path)
                return cursor is not None and size > cursor.byte_offset
            return _retry_due(cursor.next_retry_at)
        parser_matches = cursor.parser_fingerprint == _PARSER_FINGERPRINT
        if not parser_matches:
            return True
        if self._is_hermes_database(path):
            return cursor.tail_hash != sqlite_source_revision(path)
        if size == cursor.byte_size and cursor.content_fingerprint is not None:
            # Only an exact recorded observation authorizes the hot skip.
            # A bounded tail cannot prove that an earlier same-size prefix was
            # not rewritten, so any changed observation with modern tail
            # authority must return to the full route.
            if _cursor_stat_matches(cursor, stat):
                if cursor.byte_offset >= cursor.byte_size:
                    return False
                # polylogue-2qrx: ``byte_size`` matches the current file size
                # but ``byte_offset`` lags behind it. This is exactly the
                # state ``record_deferred_append_cursor`` leaves after a
                # bounded incomplete-tail probe (``_defer_incomplete_jsonl_
                # append``): it advances ``byte_size`` to the observed file
                # size but deliberately leaves ``byte_offset`` where it was,
                # pending a complete trailing record. Once the file then
                # stops changing (the writer finished), the stat-match check
                # above returned ``False`` unconditionally here forever --
                # measured live as 211 files / 414MB of stalled append
                # backlog, up to 329h stale on one 94.8MB lag, with zero
                # durable signal.
                #
                # An ordinary in-progress writer also leaves this exact state
                # between two close-together polls (a trailing record simply
                # not terminated *yet*), so escalate only once the deferred
                # observation is old enough to be implausible as "still
                # being written" -- ``cursor.updated_at`` is the timestamp of
                # that original bounded-probe deferral (nothing else touches
                # this row while the stat keeps matching).
                if not _cursor_age_exceeds(cursor, _STUCK_DEFERRED_APPEND_AGE_S):
                    return False
                # A real complete record past the original bounded window
                # reopens the normal append path; if truly nothing is there,
                # record a durable failure instead of parking silently again.
                if not self._defer_incomplete_jsonl_append(path, stat=stat, cursor=cursor, probe_bytes=None):
                    return True
                logger.warning(
                    "live.watcher: %s has no complete trailing record across its entire "
                    "outstanding tail (%d bytes past offset %d) and stopped changing; "
                    "marking failed for durable visibility instead of parking silently",
                    path,
                    cursor.byte_size - cursor.byte_offset,
                    cursor.byte_offset,
                )
                self._cursor.mark_failed(path, failed_stat=stat)
                return False
            prefix_hash = cursor_prefix_hash(cursor.tail_hash)
            if prefix_hash is None:
                if self._reconcile_archived_cursor(path, stat=stat):
                    reconciled = self._cursor.get_record(path)
                    return reconciled is None or size > reconciled.byte_offset
                return True
            try:
                current_prefix_hash, _bytes_read = sha256_range_from_path(
                    path,
                    start_offset=0,
                    end_offset=cursor.byte_offset,
                )
                final_stat = path.stat()
            except (EOFError, OSError):
                return True
            observation_changed = (
                final_stat.st_dev,
                final_stat.st_ino,
                final_stat.st_size,
                final_stat.st_mtime_ns,
                final_stat.st_ctime_ns,
            ) != (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if current_prefix_hash != prefix_hash or observation_changed:
                return True
            tail_hash = cursor_tail_hash(cursor.tail_hash)
            if tail_hash is None:
                return True
            rebase = CursorObservationRebase(
                path=path,
                expected=cursor,
                st_dev=final_stat.st_dev,
                st_ino=final_stat.st_ino,
                mtime_ns=final_stat.st_mtime_ns,
                tail_hash=encode_cursor_hash_authority(prefix_hash, tail_hash, ctime_ns=final_stat.st_ctime_ns),
            )
            if rebase_queue is None:
                self._cursor.rebase_authoritative_observations((rebase,))
            else:
                rebase_queue.append(rebase)
            return False
        if size > cursor.byte_offset:
            # A previous incomplete-tail probe recorded this exact filesystem
            # state.  The next useful observation is a write notification (or
            # a periodic scan that sees different stat evidence), not another
            # probe of the same unfinished record.  In particular, a single
            # oversized JSONL record must not make the 15-second safety scan
            # reread its first 64 MiB forever.
            if _cursor_stat_matches(cursor, stat):
                return False
            return not self._defer_incomplete_jsonl_append(path, stat=stat, cursor=cursor)
        if cursor.content_fingerprint is None:
            return True
        try:
            fingerprint, _last_nl = fingerprint_file(path)
        except FileNotFoundError:
            return False
        return not (size == cursor.byte_size and fingerprint == cursor.content_fingerprint)

    def _defer_incomplete_jsonl_append(
        self,
        path: Path,
        *,
        stat: os.stat_result,
        cursor: CursorRecord,
        probe_bytes: int | None = _INCOMPLETE_APPEND_PROBE_BYTES,
    ) -> bool:
        """Record a grown-but-incomplete JSONL tail without scheduling ingest.

        ``probe_bytes=None`` (polylogue-2qrx escalation) reads the *entire*
        outstanding tail instead of the bounded default. The bounded probe
        exists so a routine watch/periodic tick never rereads a large
        unfinished record on every pass; but that same bound means a
        complete trailing record sitting just past the probe window (e.g. a
        multi-hundred-MB rollout whose byte range past ``cursor.byte_offset``
        happens to open with one oversized blob) is never found, and once
        the source file stops changing (the writing session ends), the
        stat-matches fast path below skips this cursor forever with zero
        durable signal -- see the 211-file, 414MB stalled-cursor backlog
        this bead measured. ``probe_bytes=None`` is only used for that one
        already-deferred-with-unchanged-stat case, so the full-tail read
        happens at most once per genuine state (immediately superseded by
        either a real append or a durable failure record, never repeated
        against the same stat).
        """
        if path.suffix.lower() not in {".jsonl", ".ndjson"}:
            return False
        if cursor.content_fingerprint is None:
            return False
        if cursor.st_dev is not None and cursor.st_dev != stat.st_dev:
            return False
        if cursor.st_ino is not None and cursor.st_ino != stat.st_ino:
            return False
        start_offset = max(cursor.byte_offset, 0)
        if stat.st_size <= start_offset:
            return False
        remaining_bytes = stat.st_size - start_offset
        bytes_to_probe = remaining_bytes if probe_bytes is None else min(remaining_bytes, probe_bytes)
        try:
            with path.open("rb") as handle:
                handle.seek(start_offset)
                payload = handle.read(bytes_to_probe)
        except OSError:
            return True
        if b"\n" in payload:
            return False
        # The bounded probe can prove that no complete record begins at the
        # cursor, even when the unfinished record exceeds the probe budget.
        # Record this observed state so unchanged periodic catch-up scans skip
        # it; a subsequent append changes stat evidence and reopens the probe.
        # polylogue-hat0: this probe found no complete trailing record, not a
        # resolved authority state -- preserve any existing pending-authority
        # marker unchanged rather than clearing it.
        record_deferred_append_cursor(
            self._cursor,
            path,
            cursor=cursor,
            parser_fingerprint=_PARSER_FINGERPRINT,
            source_name=self._source_name_for(path),
            deferred_end_offset=cursor.deferred_end_offset,
        )
        return True

    def _invalidate_deferred_full_cursor(self, path: Path, *, stat: os.stat_result) -> None:
        """Clear a busy-handoff defer when current bytes reject archive authority."""

        updated = self._cursor.set(
            path,
            stat.st_size,
            byte_offset=0,
            last_complete_newline=0,
            parser_fingerprint=_PARSER_FINGERPRINT,
            content_fingerprint=None,
            tail_hash=None,
            source_name=self._source_name_for(path),
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
            failure_count=0,
            next_retry_at=None,
            excluded=False,
            allow_backward=True,
        )
        if not updated:
            raise sqlite3.OperationalError(f"failed to invalidate deferred cursor for {path}")

    def _reconcile_archived_cursor(self, path: Path, *, stat: os.stat_result) -> bool:
        """Restore a missing/stale cursor from proven archive raw state."""

        return self._reconcile_archived_cursor_outcome(path, stat=stat) is _ArchivedCursorReconciliation.RECONCILED

    @contextmanager
    def _archived_cursor_reconciliation_scope(self) -> Iterator[None]:
        """Share one read-only connection pair across a bulk planning pass.

        Cursor reconciliation runs per cursor-less file; during a 20k-file
        catch-up plan, opening source.db+index.db fresh for every file cost
        ~10 minutes of silent startup CPU (observed live 2026-07-18). The
        scope is deliberately bounded to ONE planning pass — a long-lived
        cached connection would keep reading a replaced index.db inode
        across a blue-green generation swap.
        """
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        source_db = archive_root / "source.db"
        index_db = archive_root / "index.db"
        conns: tuple[sqlite3.Connection, sqlite3.Connection] | None = None
        if source_db.exists() and index_db.exists():
            try:
                source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=1.0)
                try:
                    index_conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True, timeout=1.0)
                except sqlite3.Error:
                    source_conn.close()
                    raise
                conns = (source_conn, index_conn)
            except sqlite3.Error:
                conns = None
        self._archived_cursor_conns = conns
        self._archived_cursor_index_untrusted = (
            self._index_lacks_all_corroboration(source_conn=conns[0], index_conn=conns[1])
            if conns is not None
            else False
        )
        try:
            yield
        finally:
            self._archived_cursor_conns = None
            self._archived_cursor_index_untrusted = False
            if conns is not None:
                for conn in conns:
                    with suppress(sqlite3.Error):
                        conn.close()

    @staticmethod
    def _index_lacks_all_corroboration(
        *,
        source_conn: sqlite3.Connection,
        index_conn: sqlite3.Connection,
    ) -> bool:
        """True when the index tier holds zero sessions despite acquired raw material.

        A full ``index.db`` reset/rebuild leaves ``ops.db`` ingest cursors
        pointing at file offsets the daemon already acquired and parsed --
        cursor state that catch-up's cursor-trust fast paths would otherwise
        take on faith and skip re-materializing (polylogue-emx2, Finding 8:
        14,879 cursors skipped 100% of files against an empty post-reset
        index). One cheap pair of existence probes per catch-up scan detects
        this and forces every candidate through a per-file corroboration
        check instead.
        """
        has_parsed_raw = source_conn.execute(
            "SELECT 1 FROM raw_sessions WHERE parsed_at_ms IS NOT NULL AND parse_error IS NULL LIMIT 1"
        ).fetchone()
        if has_parsed_raw is None:
            return False
        has_any_session = index_conn.execute("SELECT 1 FROM sessions LIMIT 1").fetchone()
        return has_any_session is None

    @staticmethod
    def _archived_cursor_row(
        path: Path,
        *,
        source_conn: sqlite3.Connection,
        index_conn: sqlite3.Connection,
    ) -> tuple[object, ...] | None:
        """Newest parsed raw row for ``path`` that the index actually contains."""
        rows = source_conn.execute(
            """
            SELECT raw_id, origin, blob_hash, blob_size
            FROM raw_sessions
            WHERE source_path = ?
              AND COALESCE(source_index, 0) >= 0
              AND parsed_at_ms IS NOT NULL
              AND parse_error IS NULL
            ORDER BY acquired_at_ms DESC, raw_id DESC
            """,
            (str(path),),
        ).fetchall()
        return next(
            (
                candidate
                for candidate in rows
                if index_conn.execute(
                    "SELECT 1 FROM sessions WHERE raw_id = ? LIMIT 1",
                    (candidate[0],),
                ).fetchone()
                is not None
            ),
            None,
        )

    @classmethod
    def _path_corroborated_by_index(
        cls,
        path: Path,
        *,
        source_conn: sqlite3.Connection,
        index_conn: sqlite3.Connection,
    ) -> bool:
        """True unless ``path`` has parsed raw material the index cannot show."""
        has_parsed_raw = source_conn.execute(
            """
            SELECT 1 FROM raw_sessions
            WHERE source_path = ?
              AND COALESCE(source_index, 0) >= 0
              AND parsed_at_ms IS NOT NULL
              AND parse_error IS NULL
            LIMIT 1
            """,
            (str(path),),
        ).fetchone()
        if has_parsed_raw is None:
            # Nothing parsed for this path yet -- not this check's concern.
            return True
        return cls._archived_cursor_row(path, source_conn=source_conn, index_conn=index_conn) is not None

    def _cursor_skip_corroborated_by_index(self, path: Path) -> bool:
        """Whether a cursor-trust skip for ``path`` is backed by a materialized session.

        Only consulted while :attr:`_archived_cursor_index_untrusted` is set
        (index tier globally empty relative to acquired source material). A
        path with no successfully parsed raw row yet is not this bead's
        concern (nothing for the index to have lost) and is treated as
        corroborated so unrelated cursor kinds are unaffected.
        """
        shared = self._archived_cursor_conns
        try:
            if shared is not None:
                return self._path_corroborated_by_index(path, source_conn=shared[0], index_conn=shared[1])
            archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
            source_db = archive_root / "source.db"
            index_db = archive_root / "index.db"
            if not source_db.exists() or not index_db.exists():
                return True
            with (
                closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=1.0)) as source_conn,
                closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True, timeout=1.0)) as index_conn,
            ):
                return self._path_corroborated_by_index(path, source_conn=source_conn, index_conn=index_conn)
        except sqlite3.Error:
            # Cannot prove absence on a transient DB error -- don't force a
            # spurious re-ingest of an otherwise-healthy cursor.
            return True

    def _reconcile_archived_cursor_outcome(
        self,
        path: Path,
        *,
        stat: os.stat_result,
    ) -> _ArchivedCursorReconciliation:
        """Restore a missing/stale cursor from proven archive raw state.

        A daemon interruption can leave the archive source tier populated but
        the live cursor absent. Without this repair, startup catch-up replays
        the whole source file through the archive writer again. The archive row
        proves the stored prefix for the exact source path; if the live file
        has grown since that row was written, the cursor is restored to the
        archived prefix so catch-up can take the append path instead of
        parsing the whole active JSONL again.
        """
        shared = self._archived_cursor_conns
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        try:
            if shared is not None:
                row = self._archived_cursor_row(path, source_conn=shared[0], index_conn=shared[1])
            else:
                source_db = archive_root / "source.db"
                index_db = archive_root / "index.db"
                if not source_db.exists() or not index_db.exists():
                    return _ArchivedCursorReconciliation.UNAVAILABLE
                with (
                    closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=1.0)) as source_conn,
                    closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True, timeout=1.0)) as index_conn,
                ):
                    row = self._archived_cursor_row(path, source_conn=source_conn, index_conn=index_conn)
        except sqlite3.Error:
            return _ArchivedCursorReconciliation.UNAVAILABLE
        if row is None:
            return _ArchivedCursorReconciliation.INCOMPATIBLE
        _raw_id, origin, blob_hash, blob_size = row
        archived_size = int(cast("int | None", blob_size) or 0)
        current_size = int(stat.st_size)
        if archived_size <= 0 or archived_size > current_size:
            return _ArchivedCursorReconciliation.INCOMPATIBLE
        if isinstance(blob_hash, bytes):
            content_fingerprint = blob_hash.hex()
        elif isinstance(blob_hash, str):
            content_fingerprint = blob_hash.lower()
        else:
            return _ArchivedCursorReconciliation.INCOMPATIBLE
        if not _archive_blob_exists(archive_root, content_fingerprint):
            return _ArchivedCursorReconciliation.INCOMPATIBLE
        try:
            current_fingerprint, _fingerprint_bytes = sha256_range_from_path(
                path,
                start_offset=0,
                end_offset=archived_size,
            )
            if current_fingerprint != content_fingerprint:
                return _ArchivedCursorReconciliation.INCOMPATIBLE
            if archived_size == current_size:
                tail_hash, last_complete_newline, _bytes_read = tail_hash_and_last_complete_newline_from_path(
                    path, current_size
                )
                if path.suffix.lower() not in {".jsonl", ".ndjson"}:
                    last_complete_newline = archived_size
            else:
                tail_hash, _bytes_read = tail_hash_from_path(path, archived_size)
                last_complete_newline = archived_size
            post_read_stat = path.stat()
        except OSError:
            return _ArchivedCursorReconciliation.UNAVAILABLE
        if (
            post_read_stat.st_dev,
            post_read_stat.st_ino,
            post_read_stat.st_size,
            post_read_stat.st_mtime_ns,
            post_read_stat.st_ctime_ns,
        ) != (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns):
            return _ArchivedCursorReconciliation.UNAVAILABLE
        self._cursor.set(
            path,
            archived_size,
            byte_offset=last_complete_newline,
            last_complete_newline=last_complete_newline,
            parser_fingerprint=_PARSER_FINGERPRINT,
            content_fingerprint=content_fingerprint,
            tail_hash=encode_cursor_hash_authority(
                content_fingerprint,
                tail_hash,
                ctime_ns=stat.st_ctime_ns,
            ),
            source_name=provider_from_origin(Origin.from_string(str(origin))).value
            if origin is not None
            else self._source_name_for(path),
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
        )
        self._cursor.reset_failures(path)
        logger.info("live.watcher: reconciled cursor from archive source row for %s", path)
        return _ArchivedCursorReconciliation.RECONCILED

    async def _ingest_files(
        self,
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
    ) -> LiveBatchMetrics:
        """Ingest files through the reusable daemon live batch processor."""
        self._batch_processor.require_cursor_authority()
        async with self._ingest_lock:

            async def ingest() -> LiveBatchMetrics:
                return await self._batch_processor.ingest_files(
                    paths,
                    queued_file_count=queued_file_count,
                    skipped_file_count=skipped_file_count,
                    max_pass_seconds=_LIVE_INGEST_MAX_PASS_SECONDS,
                )

            run = getattr(self._write_coordinator, "run", None)
            metrics = await run("watcher.live_ingest", ingest) if callable(run) else await ingest()
        return metrics

    async def _emit_catch_up_terminal(
        self,
        operation_id: str,
        outcome: str,
        plan: CatchUpPlan,
        attempted: int,
        ingested: int,
        failed: int,
        stage_timings_s: Mapping[str, float],
        cycle_started: float,
        backlog_end: int | None = None,
    ) -> None:
        resolved_backlog_end = (
            max(0, len(plan.candidates) - plan.skipped_file_count - ingested) if backlog_end is None else backlog_end
        )
        await self._emit_catch_up_cycle(
            operation_id=operation_id,
            phase="terminal",
            backlog_start=len(plan.candidates),
            backlog_end=resolved_backlog_end,
            discovered=len(plan.candidates),
            attempted=attempted,
            skipped=plan.skipped_file_count,
            ingested=ingested,
            quarantine_count=0,
            errors_by_kind={"ingest_failed": failed} if failed else {},
            cursor_before=None,
            cursor_after=None,
            duration_ms=(time.perf_counter() - cycle_started) * 1000.0,
            stage_timings_s=stage_timings_s,
            repair={"required": failed, "performed": 0, "remaining": resolved_backlog_end},
            terminal_outcome=outcome,
        )

    async def _emit_catch_up_cycle(self, **kwargs: object) -> None:
        """Persist lifecycle facts through the daemon's write coordinator."""
        if self._catch_up_event_emitter is not None:
            await self._run_writer_sync(
                "watcher.catch_up.event",
                self._catch_up_event_emitter,
                **kwargs,
            )

    async def _run_coordinated(self, actor: str, operation: Callable[[], Awaitable[None]]) -> None:
        """Run a complete watcher write batch under the injected coordinator."""
        run = getattr(self._write_coordinator, "run", None)
        if callable(run):
            await run(actor, operation)
            return
        await operation()

    def _source_name_for(self, path: Path) -> str:
        resolved = path.resolve()
        for source in self._sources:
            try:
                if resolved.is_relative_to(source.root.resolve()):
                    return source.name
            except OSError:
                continue
        return path.parent.name

    def _source_accepts(self, path: Path) -> bool:
        resolved = path.resolve()
        for source in self._sources:
            try:
                if resolved.is_relative_to(source.root.resolve()):
                    return source.accepts(path)
            except OSError:
                continue
        return path.suffix == ".jsonl"

    def _is_hook_spool_path(self, path: Path) -> bool:
        for source in self._sources:
            if source.name != "hooks":
                continue
            try:
                return path.resolve().is_relative_to(source.root.resolve())
            except OSError:
                return False
        return False

    def _hook_spool_root(self) -> Path:
        """Return the root paired with this watcher's hook source."""

        for source in self._sources:
            if source.name == "hooks":
                return source.root.parent
        return hook_spool_root()

    def _is_hermes_database(self, path: Path) -> bool:
        resolved = path.resolve()
        for source in self._sources:
            if source.name != "hermes":
                continue
            try:
                if resolved.is_relative_to(source.root.resolve()) and source.accepts(path):
                    return is_sqlite_path(path)
            except OSError:
                continue
        return False

    def _canonical_watch_path(self, path: Path) -> Path | None:
        if self._source_accepts(path):
            return path
        database = sqlite_database_for_sidecar(path)
        if database is not None and self._is_hermes_database(database):
            return database
        return None

    def _watch_filter(self, _change: object, path: str) -> bool:
        """Accept configured source files under hidden canonical roots.

        watchfiles' default filter ignores hidden directories. Polylogue's
        normal roots live under paths such as ~/.claude, ~/.codex, ~/.local,
        and repo-local .cache, so the generic filter silently drops real source
        writes. This filter keeps the project's own source/suffix predicate as
        the gate instead.
        """
        return self._canonical_watch_path(Path(path)) is not None


def _interleave_by_source(candidates: list[CandidateSourceFile]) -> list[CandidateSourceFile]:
    """Round-robin candidates across source families (#1616).

    Plain alphabetical sort by path puts all of one source's files
    before any of another's, so a long-source-first catch-up hides
    small-source ingestion progress for hours. Bucket by source_name,
    sort each bucket by path for determinism, then round-robin across
    buckets so the first chunk contains some of every present family.

    Exception: browser-capture spool files drain FIRST. The raw
    materialization conveyor yields the writer while any spool file
    lacks a cursor (daemon/cli.py), so interleaving them across the
    whole plan parks source→index self-healing for the entire catch-up
    (observed live 2026-07-18: conveyor idle for hours behind ~600
    spooled captures spread over ~700 chunks).
    """
    buckets: dict[str, list[CandidateSourceFile]] = {}
    for candidate in candidates:
        buckets.setdefault(candidate.source_name, []).append(candidate)
    for source_name in buckets:
        buckets[source_name].sort(key=lambda candidate: candidate.path)
    ordered: list[CandidateSourceFile] = list(buckets.pop("browser-capture", []))
    iterators = [iter(buckets[name]) for name in sorted(buckets)]
    while iterators:
        next_round = []
        for it in iterators:
            picked = next(it, None)
            if picked is not None:
                ordered.append(picked)
                next_round.append(it)
        iterators = next_round
    return ordered


def default_sources(*, hermes_root: Path | None = None, beads_roots: tuple[Path, ...] = ()) -> tuple[WatchSource, ...]:
    """Discover the default live-source roots from XDG/home conventions.

    Includes the archive inbox so that ``polylogue ingest PATH``
    (which stages to ``archive_root()/inbox``) is observed by the
    daemon-owned watcher.

    ``beads_roots`` are repository roots (not ``.beads`` directories
    themselves) whose append-only ``.beads/interactions.jsonl`` ledger
    should be watched. Beads is opt-in and unlike every other source here:
    a single global runtime directory does not exist for it (each git
    repository owns its own ledger), so there is no home-relative default
    -- callers must supply the repository roots explicitly (see
    ``PolylogueConfig.beads_roots`` / ``sources.beads_roots`` in
    ``polylogue.toml``). A repository without a ``.beads/interactions.jsonl``
    ledger yet is still watched (``WatchSource.exists()`` reports it
    absent) so a ledger created later is picked up without a daemon
    restart.
    """
    from polylogue.core.enums import Provider
    from polylogue.paths import (
        antigravity_path,
        archive_root,
        browser_capture_spool_root,
        claude_code_path,
        claude_code_todos_path,
        codex_path,
        gemini_cli_path,
        hermes_sessions_path,
    )
    from polylogue.sources.origin_specs import artifact_suffixes_for_provider

    beads_sources = tuple(
        WatchSource(
            name=f"beads:{repository_root.name}",
            root=repository_root / ".beads",
            # Scoped to the append-only interaction ledger, never the
            # mutable current-state ``issues.jsonl``/backup snapshots that
            # also live under ``.beads/`` -- those are not evidence
            # timelines and would misrepresent issue history if ingested.
            suffixes=("interactions.jsonl",),
        )
        for repository_root in beads_roots
    )

    return (
        WatchSource(
            name="claude-code",
            root=claude_code_path(),
            suffixes=artifact_suffixes_for_provider(Provider.CLAUDE_CODE, defaults=(".jsonl",)),
        ),
        # polylogue-t0p: Claude Code's live plan-snapshot directory
        # (~/.claude/todos/) is a sibling of claude_code_path(), not nested
        # under it -- a second, narrower WatchSource rooted there, same
        # precedent as "codex-state" below, so the main claude-code root
        # doesn't have to widen its own suffix/path assumptions to reach a
        # completely different directory tree.
        WatchSource(
            name="claude-code-todos",
            root=claude_code_todos_path(),
            suffixes=(".json",),
        ),
        WatchSource(name="codex", root=codex_path()),
        # polylogue-0jf4: Codex also keeps live SQLite state (thread titles,
        # spawn topology, goals, memories) as siblings of the sessions/
        # directory, not under it -- a second, narrower WatchSource rooted at
        # ~/.codex (codex_path().parent) rather than widening the "codex"
        # source's own root, so a broadened suffix set never has to reason
        # about history.jsonl/config.toml/log/ under the shared root. Suffix
        # filtering alone (".sqlite"/".db") keeps this cheap; the acquisition
        # path (sources/live/batch.py) re-verifies table shape by name and
        # structure before treating anything as in-scope evidence.
        WatchSource(
            name="codex-state",
            root=codex_path().parent,
            suffixes=(".sqlite", ".db"),
        ),
        WatchSource(name="gemini-cli", root=gemini_cli_path(), suffixes=(".json", ".jsonl")),
        # Hermes emits four independently durable source classes under its
        # runtime root: state.db, optional session snapshots, NeMo Relay ATIF
        # documents, and append-only ATOF JSONL.  The ledger database is
        # admitted as a live SQLite source too; parsing it remains a separate
        # fidelity/normalization contract rather than an implicit filename
        # fallback.
        WatchSource(
            name="hermes",
            root=hermes_root if hermes_root is not None else hermes_sessions_path(),
            suffixes=(".json", ".jsonl", ".db", ".sqlite", ".sqlite3"),
        ),
        WatchSource(name="antigravity", root=antigravity_path(), suffixes=(".metadata.json",)),
        WatchSource(name="browser-capture", root=browser_capture_spool_root(), suffixes=(".json",)),
        # #1683: inbox accepts archive, zip, and json-line formats so that
        # GDPR exports (typically .zip) and raw .json dumps are observed.
        WatchSource(name="inbox", root=archive_root() / "inbox", suffixes=INBOX_SOURCE_SUFFIXES),
        WatchSource(name="hooks", root=pending_hook_spool_dir(), suffixes=(".json",)),
        *beads_sources,
    )


def _cursor_db_path(polylogue: Polylogue) -> Path:
    """Use the archive ops tier for daemon cursor state."""
    backend = getattr(polylogue, "backend", None)
    db_path = getattr(backend, "db_path", None)
    if isinstance(db_path, Path):
        return db_path
    return Path(polylogue.archive_root) / "ops.db"


def _is_database_locked(exc: sqlite3.OperationalError) -> bool:
    return "database is locked" in str(exc).lower()


def _cursor_age_exceeds(cursor: CursorRecord, min_age_s: float) -> bool:
    """Return True when ``cursor.updated_at`` is older than ``min_age_s``.

    A malformed/missing timestamp is treated as old (fail toward escalating
    rather than toward parking silently forever, matching this check's own
    purpose).
    """
    try:
        updated_at = datetime.fromisoformat(cursor.updated_at)
    except ValueError:
        return True
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    return (datetime.now(UTC) - updated_at).total_seconds() >= min_age_s


def _retry_due(next_retry_at: str | None) -> bool:
    if not next_retry_at:
        return True
    retry_at = _parse_retry_at(next_retry_at)
    if retry_at is None:
        return True
    return retry_at <= datetime.now(UTC)


def _parse_retry_at(next_retry_at: str | None) -> datetime | None:
    if not next_retry_at:
        return None
    try:
        retry_at = datetime.fromisoformat(next_retry_at)
    except ValueError:
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return retry_at


def _cursor_stat_matches(cursor: CursorRecord, stat: os.stat_result) -> bool:
    """Return True when the cursor was written for this exact file state."""

    return (
        cursor.st_dev == stat.st_dev
        and cursor.st_ino == stat.st_ino
        and cursor.mtime_ns == stat.st_mtime_ns
        and cursor_ctime_ns(cursor.tail_hash) == stat.st_ctime_ns
    )


__all__ = ["LiveWatcher", "WatchSource", "default_sources"]

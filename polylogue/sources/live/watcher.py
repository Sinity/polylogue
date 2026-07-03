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
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.core.enums import Origin
from polylogue.core.sources import provider_from_origin
from polylogue.logging import get_logger
from polylogue.sources.live.batch import LiveBatchEventEmitter, LiveBatchProcessor, fingerprint_file
from polylogue.sources.live.batch_support import (
    _archive_blob_exists,
    tail_hash_and_last_complete_newline_from_path,
    tail_hash_from_path,
)
from polylogue.sources.live.cursor import CursorRecord, CursorStore
from polylogue.sources.live.deferred_cursor import record_deferred_append_cursor
from polylogue.sources.live.metrics import LiveBatchMetrics

if TYPE_CHECKING:
    from polylogue.api import Polylogue

logger = get_logger(__name__)
_PARSER_FINGERPRINT = "live-batched-v2"
_CATCH_UP_MAX_BATCH_FILES = 50
_CATCH_UP_MAX_BATCH_BYTES = 64 * 1024 * 1024
_INCOMPLETE_APPEND_PROBE_BYTES = 64 * 1024 * 1024
INBOX_SOURCE_SUFFIXES = (".jsonl", ".zip", ".json", ".ndjson")


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
        "wal_before_checkpoint=%.1f MB wal_after_checkpoint=%.1f MB wal_busy_pages=%d",
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
    )


@dataclass(frozen=True, slots=True)
class WatchSource:
    """A directory to watch for live session files."""

    name: str
    root: Path
    suffixes: tuple[str, ...] = (".jsonl",)

    def exists(self) -> bool:
        return self.root.exists()

    def accepts(self, path: Path) -> bool:
        name = path.name.lower()
        return any(name.endswith(suffix) for suffix in self.suffixes)


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
    ) -> None:
        self._polylogue = polylogue
        self._sources = tuple(sources)
        self._debounce_s = debounce_s
        self._cursor = cursor or CursorStore(_cursor_db_path(polylogue))
        self._max_workers = max_workers
        self._converger = converger
        self._pending_paths: set[Path] = set()
        self._pending_scheduled = False
        self._drain_task: asyncio.Task[None] | None = None
        self._failed_retry_task: asyncio.Task[None] | None = None
        self._failed_retry_deadline: float | None = None
        self._last_enqueue_at = 0.0
        self._last_batch_at: float = 0.0
        self._batch_lock = asyncio.Lock()
        self._ingest_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._catch_up_complete = asyncio.Event()
        self._batch_processor = LiveBatchProcessor(
            polylogue,
            self._sources,
            cursor=self._cursor,
            parser_fingerprint=lambda: _PARSER_FINGERPRINT,
            converger=converger,
            stop_requested=self._stop.is_set,
            event_emitter=event_emitter,
        )

    @property
    def catch_up_complete(self) -> asyncio.Event:
        return self._catch_up_complete

    async def run(self) -> None:
        from watchfiles import Change, awatch

        roots = [s.root for s in self._sources if s.exists()]
        if not roots:
            logger.warning("live.watcher: no source roots exist; nothing to watch")
            self._catch_up_complete.set()
            return

        try:
            await self._catch_up(roots)
        finally:
            self._catch_up_complete.set()
        self._schedule_failed_retry_scan()
        self._ensure_pending_scheduled()

        logger.info("live.watcher: watching %s", ", ".join(str(r) for r in roots))
        async for changes in awatch(*roots, stop_event=self._stop, recursive=True):
            for change, raw_path in changes:
                if change is Change.deleted:
                    continue
                path = Path(raw_path)
                if not self._source_accepts(path):
                    continue
                self._enqueue(path)

    def stop(self) -> None:
        self._stop.set()
        self._cancel_failed_retry_task()

    def cancel_pending(self) -> None:
        task = self._drain_task
        if task is not None and not task.done():
            task.cancel()
        self._drain_task = None
        self._pending_scheduled = False
        self._cancel_failed_retry_task()

    # ------------------------------------------------------------------
    # Catch-up: batch all changed files
    # ------------------------------------------------------------------

    async def _catch_up(self, roots: list[Path]) -> None:
        candidates = self._scan_catch_up_candidates(roots)
        if not candidates:
            return
        logger.info("live.watcher: catch-up scan over %d file(s)", len(candidates))
        plan = self._plan_catch_up(candidates)

        if plan.needed:
            candidate_by_path = {candidate.path: candidate for candidate in plan.candidates}
            chunks = tuple(self._chunk_catch_up_paths(plan.needed, candidate_by_path))
            logger.info(
                "live.watcher: catch-up ingesting %d file(s) (%.1f MB), skipped=%d, chunks=%d",
                len(plan.needed),
                plan.needed_bytes / 1e6,
                plan.skipped_file_count,
                len(chunks),
            )
            for index, chunk in enumerate(chunks, start=1):
                if self._stop.is_set():
                    break
                chunk_bytes = sum(candidate_by_path[path].stat.st_size for path in chunk)
                logger.info(
                    "live.watcher: catch-up chunk %d/%d ingesting %d file(s) (%.1f MB)",
                    index,
                    len(chunks),
                    len(chunk),
                    chunk_bytes / 1e6,
                )
                metrics = await self._ingest_files(
                    list(chunk),
                    queued_file_count=len(plan.candidates) if index == 1 else len(chunk),
                    skipped_file_count=plan.skipped_file_count if index == 1 else 0,
                )
                if metrics is not None:
                    _log_ingest_metrics(f"live.watcher: catch-up chunk {index}/{len(chunks)}", metrics)
                    if (
                        getattr(metrics, "succeeded_file_count", 0) == 0
                        and getattr(metrics, "failed_file_count", 0) == 0
                    ):
                        self._defer_unaccounted_failed_retries(list(chunk))
            self._schedule_failed_retry_scan()

    def _scan_catch_up_candidates(self, roots: list[Path]) -> tuple[CandidateSourceFile, ...]:
        root_set = {root.resolve() for root in roots}
        candidates: list[CandidateSourceFile] = []
        for source in self._sources:
            if not source.exists() or source.root.resolve() not in root_set:
                continue
            for suffix in source.suffixes:
                for path in source.root.rglob(f"*{suffix}"):
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
                            suffix=suffix,
                            stat=stat,
                        )
                    )
        return tuple(_interleave_by_source(candidates))

    def _plan_catch_up(self, candidates: tuple[CandidateSourceFile, ...]) -> CatchUpPlan:
        if not candidates:
            return CatchUpPlan(candidates=(), needed=(), skipped_file_count=0, needed_bytes=0)
        cursor_records = self._cursor.get_records(candidate.path for candidate in candidates)
        needed: list[Path] = []
        skipped = 0
        needed_bytes = 0
        for candidate in candidates:
            if self._stop.is_set():
                break
            if self._needs_work_from_state(
                candidate.path,
                stat=candidate.stat,
                cursor=cursor_records.get(candidate.path),
            ):
                needed.append(candidate.path)
                needed_bytes += candidate.stat.st_size
            else:
                skipped += 1
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
        for record in self._cursor.list_failed_records():
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

        try:
            # Filter to files that actually need work.
            cursor_records = self._cursor.get_records(paths)
            needed = []
            for path in paths:
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                if self._needs_work_from_state(path, stat=stat, cursor=cursor_records.get(path)):
                    needed.append(path)
            if not needed:
                return bool(paths)

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

    def _needs_work_from_state(self, path: Path, *, stat: os.stat_result, cursor: CursorRecord | None) -> bool:
        size = stat.st_size
        if cursor is None:
            if not self._reconcile_archived_cursor(path, stat=stat):
                return True
            cursor = self._cursor.get_record(path)
            return cursor is not None and size > cursor.byte_offset
        if cursor.excluded:
            return False
        if cursor.failure_count > 0:
            if self._reconcile_archived_cursor(path, stat=stat):
                cursor = self._cursor.get_record(path)
                return cursor is not None and size > cursor.byte_offset
            return _retry_due(cursor.next_retry_at)
        parser_matches = cursor.parser_fingerprint == _PARSER_FINGERPRINT
        if not parser_matches:
            return True
        if size == cursor.byte_size and cursor.content_fingerprint is not None:
            # Stable path + size + recorded content fingerprint is the hot
            # catch-up skip path. Device/inode churn across bind mounts,
            # restored homes, or filesystem rebuilds must not force a full
            # content rehash of every historical session file.
            if _cursor_stat_matches(cursor, stat):
                return False
            if cursor.tail_hash is None:
                return False
            try:
                current_tail_hash, _bytes_read = tail_hash_from_path(path, size)
            except FileNotFoundError:
                return False
            return current_tail_hash != cursor.tail_hash
        if size > cursor.byte_offset:
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
    ) -> bool:
        """Record a grown-but-incomplete JSONL tail without scheduling ingest."""
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
        bytes_to_probe = min(remaining_bytes, _INCOMPLETE_APPEND_PROBE_BYTES)
        try:
            with path.open("rb") as handle:
                handle.seek(start_offset)
                payload = handle.read(bytes_to_probe)
        except OSError:
            return True
        if b"\n" in payload:
            return False
        if bytes_to_probe < remaining_bytes:
            return False
        record_deferred_append_cursor(
            self._cursor,
            path,
            cursor=cursor,
            parser_fingerprint=_PARSER_FINGERPRINT,
            source_name=self._source_name_for(path),
        )
        return True

    def _reconcile_archived_cursor(self, path: Path, *, stat: os.stat_result) -> bool:
        """Restore a missing/stale cursor from proven archive raw state.

        A daemon interruption can leave the archive source tier populated but
        the live cursor absent. Without this repair, startup catch-up replays
        the whole source file through the archive writer again. The archive row
        proves the stored prefix for the exact source path; if the live file
        has grown since that row was written, the cursor is restored to the
        archived prefix so catch-up can take the append path instead of
        parsing the whole active JSONL again.
        """
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        source_db = archive_root / "source.db"
        if not source_db.exists():
            return False
        try:
            with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=1.0) as conn:
                row = conn.execute(
                    """
                    SELECT origin, blob_hash, blob_size
                    FROM raw_sessions
                    WHERE source_path = ?
                      AND COALESCE(source_index, 0) >= 0
                    ORDER BY acquired_at_ms DESC, raw_id DESC
                    LIMIT 1
                    """,
                    (str(path),),
                ).fetchone()
        except sqlite3.Error:
            return False
        if row is None:
            return False
        origin, blob_hash, blob_size = row
        archived_size = int(blob_size or 0)
        current_size = int(stat.st_size)
        if archived_size <= 0 or archived_size > current_size:
            return False
        if isinstance(blob_hash, bytes):
            content_fingerprint = blob_hash.hex()
        elif isinstance(blob_hash, str):
            content_fingerprint = blob_hash.lower()
        else:
            return False
        if not _archive_blob_exists(archive_root, content_fingerprint):
            return False
        try:
            if archived_size == current_size:
                tail_hash, last_complete_newline, _bytes_read = tail_hash_and_last_complete_newline_from_path(
                    path, current_size
                )
                if path.suffix.lower() not in {".jsonl", ".ndjson"}:
                    last_complete_newline = archived_size
            else:
                tail_hash, _bytes_read = tail_hash_from_path(path, archived_size)
                last_complete_newline = archived_size
        except FileNotFoundError:
            return False
        self._cursor.set(
            path,
            archived_size,
            byte_offset=last_complete_newline,
            last_complete_newline=last_complete_newline,
            parser_fingerprint=_PARSER_FINGERPRINT,
            content_fingerprint=content_fingerprint,
            tail_hash=tail_hash,
            source_name=provider_from_origin(Origin.from_string(str(origin))).value
            if origin is not None
            else self._source_name_for(path),
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
        )
        self._cursor.reset_failures(path)
        logger.info("live.watcher: reconciled cursor from archive source row for %s", path)
        return True

    async def _ingest_files(
        self,
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
    ) -> LiveBatchMetrics:
        """Ingest files through the reusable daemon live batch processor."""
        async with self._ingest_lock:
            metrics = await self._batch_processor.ingest_files(
                paths,
                queued_file_count=queued_file_count,
                skipped_file_count=skipped_file_count,
            )
        return metrics

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


def _interleave_by_source(candidates: list[CandidateSourceFile]) -> list[CandidateSourceFile]:
    """Round-robin candidates across source families (#1616).

    Plain alphabetical sort by path puts all of one source's files
    before any of another's, so a long-source-first catch-up hides
    small-source ingestion progress for hours. Bucket by source_name,
    sort each bucket by path for determinism, then round-robin across
    buckets so the first chunk contains some of every present family.
    """
    buckets: dict[str, list[CandidateSourceFile]] = {}
    for candidate in candidates:
        buckets.setdefault(candidate.source_name, []).append(candidate)
    for source_name in buckets:
        buckets[source_name].sort(key=lambda candidate: candidate.path)
    ordered: list[CandidateSourceFile] = []
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


def default_sources() -> tuple[WatchSource, ...]:
    """Discover the default live-source roots from XDG/home conventions.

    Includes the archive inbox so that ``polylogue ingest PATH``
    (which stages to ``archive_root()/inbox``) is observed by the
    daemon-owned watcher.
    """
    from polylogue.paths import (
        antigravity_path,
        archive_root,
        browser_capture_spool_root,
        claude_code_path,
        codex_path,
        gemini_cli_path,
        hermes_sessions_path,
        hooks_sidecar_dir,
    )

    return (
        WatchSource(name="claude-code", root=claude_code_path()),
        WatchSource(name="codex", root=codex_path()),
        WatchSource(name="gemini-cli", root=gemini_cli_path(), suffixes=(".json", ".jsonl")),
        WatchSource(name="hermes", root=hermes_sessions_path(), suffixes=(".json",)),
        WatchSource(name="antigravity", root=antigravity_path(), suffixes=(".metadata.json",)),
        WatchSource(name="browser-capture", root=browser_capture_spool_root(), suffixes=(".json",)),
        # #1683: inbox accepts archive, zip, and json-line formats so that
        # GDPR exports (typically .zip) and raw .json dumps are observed.
        WatchSource(name="inbox", root=archive_root() / "inbox", suffixes=INBOX_SOURCE_SUFFIXES),
        WatchSource(name="hooks", root=hooks_sidecar_dir()),
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

    return cursor.st_dev == stat.st_dev and cursor.st_ino == stat.st_ino and cursor.mtime_ns == stat.st_mtime_ns


__all__ = ["LiveWatcher", "WatchSource", "default_sources"]

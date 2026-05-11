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
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.sources.live.batch import LiveBatchEventEmitter, LiveBatchProcessor, fingerprint_file
from polylogue.sources.live.cursor import CursorRecord, CursorStore

if TYPE_CHECKING:
    from polylogue.api import Polylogue

logger = get_logger(__name__)
_PARSER_FINGERPRINT = "live-batched-v2"


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
        self._last_batch_at: float = 0.0
        self._batch_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._batch_processor = LiveBatchProcessor(
            polylogue,
            self._sources,
            cursor=self._cursor,
            parser_fingerprint=lambda: _PARSER_FINGERPRINT,
            converger=converger,
            stop_requested=self._stop.is_set,
            event_emitter=event_emitter,
        )

    async def run(self) -> None:
        from watchfiles import Change, awatch

        roots = [s.root for s in self._sources if s.exists()]
        if not roots:
            logger.warning("live.watcher: no source roots exist; nothing to watch")
            return

        await self._catch_up(roots)

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

    def cancel_pending(self) -> None:
        pass  # No orphaned tasks with the batched model

    # ------------------------------------------------------------------
    # Catch-up: batch all changed files
    # ------------------------------------------------------------------

    async def _catch_up(self, roots: list[Path]) -> None:
        root_set = {root.resolve() for root in roots}
        files: list[Path] = []
        for source in self._sources:
            if not source.exists() or source.root.resolve() not in root_set:
                continue
            for suffix in source.suffixes:
                files.extend(p for p in source.root.rglob(f"*{suffix}") if p.is_file())
        if not files:
            return
        logger.info("live.watcher: catch-up scan over %d file(s)", len(files))
        cursor_records = self._cursor.get_records(files)
        needed: list[Path] = []
        skipped = 0
        needed_bytes = 0
        for path in files:
            if self._stop.is_set():
                return
            try:
                stat = path.stat()
            except FileNotFoundError:
                skipped += 1
                continue
            if self._needs_work_from_state(path, stat=stat, cursor=cursor_records.get(path)):
                needed.append(path)
                needed_bytes += stat.st_size
            else:
                skipped += 1

        if needed:
            logger.info(
                "live.watcher: catch-up ingesting %d file(s) (%.1f MB), skipped=%d",
                len(needed),
                needed_bytes / 1e6,
                skipped,
            )
            await self._ingest_files(needed, queued_file_count=len(files), skipped_file_count=skipped)

    # ------------------------------------------------------------------
    # Live: debounced batch scheduling
    # ------------------------------------------------------------------

    def _enqueue(self, path: Path) -> None:
        """Enqueue a path for batched ingestion after debounce."""
        self._pending_paths.add(path)
        if not self._pending_scheduled:
            self._pending_scheduled = True
            asyncio.create_task(self._debounced_batch())

    async def _debounced_batch(self) -> None:
        """Wait for the debounce window, then flush all pending paths."""
        await asyncio.sleep(self._debounce_s)

        # Re-check: collect any new arrivals during the sleep.
        while True:
            before = len(self._pending_paths)
            await asyncio.sleep(0.5)
            after = len(self._pending_paths)
            if after == before:
                break

        await self._flush_pending()

    async def _flush_pending(self) -> None:
        """Flush all pending paths in a single batch."""
        async with self._batch_lock:
            if not self._pending_paths:
                self._pending_scheduled = False
                return
            paths = list(self._pending_paths)
            self._pending_paths.clear()
            self._pending_scheduled = False

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
            return

        logger.info("live.watcher: batching %d changed file(s)", len(needed))
        await self._ingest_files(
            needed,
            queued_file_count=len(paths),
            skipped_file_count=len(paths) - len(needed),
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

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
            return True
        if cursor.excluded:
            return False
        if cursor.failure_count > 0:
            return _retry_due(cursor.next_retry_at)
        parser_matches = cursor.parser_fingerprint == _PARSER_FINGERPRINT
        if not parser_matches:
            return True
        same_inode = (cursor.st_dev is None or cursor.st_dev == stat.st_dev) and (
            cursor.st_ino is None or cursor.st_ino == stat.st_ino
        )
        if same_inode and size > cursor.byte_offset:
            return True
        if same_inode and size == cursor.byte_size and cursor.content_fingerprint is not None:
            # Same file (dev, ino), same size, fingerprint already recorded
            # — trust the cursor. Mtime drift alone (filesystem touch,
            # hardlink/rename-replace that preserves inode, fsync rounding,
            # nanosecond-resolution rounding across filesystems) is not a
            # content-change signal. Rehashing here costs O(file_size) per
            # touched file across catch-up and active monitoring; refuted
            # by the regression test in tests/integration/test_live_read_amplification.py.
            return False
        if cursor.content_fingerprint is None:
            return True
        try:
            fingerprint, _last_nl = fingerprint_file(path)
        except FileNotFoundError:
            return False
        return not (size == cursor.byte_size and fingerprint == cursor.content_fingerprint)

    async def _ingest_files(
        self,
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
    ) -> None:
        """Ingest files through the reusable daemon live batch processor."""
        await self._batch_processor.ingest_files(
            paths,
            queued_file_count=queued_file_count,
            skipped_file_count=skipped_file_count,
        )

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


def default_sources() -> tuple[WatchSource, ...]:
    """Discover the default live-source roots from XDG/home conventions.

    Includes the archive inbox so that ``polylogue ingest PATH``
    (which stages to ``archive_root()/inbox``) is observed by the
    daemon-owned watcher.
    """
    from polylogue.paths import (
        antigravity_path,
        archive_root,
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
        WatchSource(name="inbox", root=archive_root() / "inbox"),
        WatchSource(name="hooks", root=hooks_sidecar_dir()),
    )


def _cursor_db_path(polylogue: Polylogue) -> Path:
    """Use the archive database for daemon cursor state."""
    backend = getattr(polylogue, "backend", None)
    db_path = getattr(backend, "db_path", None)
    if isinstance(db_path, Path):
        return db_path
    return Path(polylogue.archive_root) / "polylogue.db"


def _retry_due(next_retry_at: str | None) -> bool:
    if not next_retry_at:
        return True
    try:
        retry_at = datetime.fromisoformat(next_retry_at)
    except ValueError:
        return True
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return retry_at <= datetime.now(UTC)


__all__ = ["LiveWatcher", "WatchSource", "default_sources"]

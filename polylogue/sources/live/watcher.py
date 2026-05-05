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
import hashlib
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.sources.live.cursor import CursorStore

if TYPE_CHECKING:
    from polylogue.api import Polylogue

logger = get_logger(__name__)
_PARSER_FINGERPRINT = "live-batched-v2"


@dataclass(frozen=True, slots=True)
class WatchSource:
    """A directory to watch for live JSONL session files."""

    name: str
    root: Path

    def exists(self) -> bool:
        return self.root.exists()


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
                if path.suffix != ".jsonl":
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
        files: list[Path] = []
        for root in roots:
            # Session files: {project}/{uuid}.jsonl
            files.extend(p for p in root.glob("*/*.jsonl") if p.is_file())
            # Sub-agent files: {project}/{uuid}/subagents/agent-*.jsonl
            files.extend(p for p in root.glob("*/*/subagents/agent-*.jsonl") if p.is_file())
        if not files:
            return
        logger.info("live.watcher: catch-up scan over %d file(s)", len(files))

        # Process in size-based chunks so a single 150 MB session
        # doesn't starve the batch alongside 199 tiny files.
        chunk_size_bytes = 50 * 1024 * 1024  # 50 MB
        chunk: list[Path] = []
        chunk_bytes: int = 0

        for i, path in enumerate(files, start=1):
            if self._stop.is_set():
                return
            if self._needs_work(path):
                try:
                    size = path.stat().st_size
                except FileNotFoundError:
                    size = 0
                chunk.append(path)
                chunk_bytes += size

            if chunk_bytes >= chunk_size_bytes and chunk:
                logger.info(
                    "live.watcher: chunk %d/%d — ingesting %d file(s) (%.1f MB)",
                    i,
                    len(files),
                    len(chunk),
                    chunk_bytes / 1e6,
                )
                await self._ingest_files(chunk)
                chunk.clear()
                chunk_bytes = 0

        if chunk:
            logger.info(
                "live.watcher: final chunk — ingesting %d file(s) (%.1f MB)",
                len(chunk),
                chunk_bytes / 1e6,
            )
            await self._ingest_files(chunk)

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
        needed = [p for p in paths if self._needs_work(p)]
        if not needed:
            return

        logger.info("live.watcher: batching %d changed file(s)", len(needed))
        await self._ingest_files(needed)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _needs_work(self, path: Path) -> bool:
        """Return True if the file is new, grown, or fingerprint-changed."""
        try:
            stat = path.stat()
        except FileNotFoundError:
            return False
        size = stat.st_size
        try:
            fingerprint, _last_nl = _fingerprint_file(path)
        except FileNotFoundError:
            return False
        cursor = self._cursor.get_record(path)
        if cursor is not None:
            if cursor.excluded:
                return False
            if cursor.failure_count > 0:
                return _retry_due(cursor.next_retry_at)
        return not (
            cursor is not None
            and size == cursor.byte_size
            and fingerprint == cursor.content_fingerprint
            and cursor.parser_fingerprint == _PARSER_FINGERPRINT
        )

    async def _ingest_files(self, paths: list[Path]) -> None:
        """Ingest files in batch, then run global convergence stages once."""
        from polylogue.config import Source

        # Phase 1: batched ingest via parse_sources() — efficient.
        by_source: dict[str, list[Path]] = {}
        for path in paths:
            by_source.setdefault(self._source_name_for(path), []).append(path)
        succeeded_paths: set[Path] = set()
        for source_name, source_paths in by_source.items():
            if self._stop.is_set():
                return
            sources = [Source(name=f"{source_name}:{p.parent.name}", path=p) for p in source_paths]
            t0 = time.perf_counter()
            try:
                await self._polylogue.parse_sources(sources=sources, download_assets=False)
            except Exception as exc:
                logger.warning("live.watcher: batch failed for %s: %s", source_name, exc)
                for p in source_paths:
                    self._record_failed_cursor(p)
                continue
            succeeded_paths.update(source_paths)
            elapsed = time.perf_counter() - t0
            logger.info(
                "live.watcher: batch ingested %s — %d in %.1fs (%.1f/s)",
                source_name,
                len(source_paths),
                elapsed,
                len(source_paths) / max(elapsed, 0.01),
            )

        # Phase 2: global convergence — run once after batch ingest.
        if self._converger is not None and succeeded_paths:
            try:
                hint_path = next(iter(succeeded_paths))
                if hint_path:
                    self._converger.converge_file(hint_path)  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning("live.watcher: post-ingest converge failed: %s", exc)

        # Update cursors only for paths that ingested successfully.
        for path in succeeded_paths:
            try:
                stat = path.stat()
                fp, last_nl = _fingerprint_file(path)
            except FileNotFoundError:
                continue
            self._cursor.set(
                path,
                stat.st_size,
                byte_offset=last_nl,
                last_complete_newline=last_nl,
                parser_fingerprint=_PARSER_FINGERPRINT,
                content_fingerprint=fp,
                source_name=self._source_name_for(path),
                st_dev=stat.st_dev,
                st_ino=stat.st_ino,
                mtime_ns=stat.st_mtime_ns,
            )
            self._cursor.reset_failures(path)

    def _record_failed_cursor(self, path: Path) -> None:
        try:
            stat = path.stat()
            fp, last_nl = _fingerprint_file(path)
        except FileNotFoundError:
            self._cursor.mark_failed(path)
            return
        existing = self._cursor.get_record(path)
        self._cursor.set(
            path,
            stat.st_size,
            byte_offset=last_nl,
            last_complete_newline=last_nl,
            parser_fingerprint=_PARSER_FINGERPRINT,
            content_fingerprint=fp,
            source_name=self._source_name_for(path),
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
            failure_count=existing.failure_count if existing else 0,
            next_retry_at=existing.next_retry_at if existing else None,
            excluded=bool(existing.excluded) if existing else False,
        )
        self._cursor.mark_failed(path)

    def _source_name_for(self, path: Path) -> str:
        resolved = path.resolve()
        for source in self._sources:
            try:
                if resolved.is_relative_to(source.root.resolve()):
                    return source.name
            except OSError:
                continue
        return path.parent.name


def default_sources() -> tuple[WatchSource, ...]:
    """Discover the default live-source roots from XDG/home conventions."""
    from polylogue.paths import claude_code_path, codex_path

    return (
        WatchSource(name="claude-code", root=claude_code_path()),
        WatchSource(name="codex", root=codex_path()),
    )


def _fingerprint_file(path: Path) -> tuple[str, int]:
    content = path.read_bytes()
    newline_at = content.rfind(b"\n")
    last_complete_newline = 0 if newline_at < 0 else newline_at + 1
    return hashlib.sha256(content).hexdigest(), last_complete_newline


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

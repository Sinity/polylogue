"""Live JSONL session watcher.

Watches one or more roots for ``*.jsonl`` changes via ``watchfiles`` and
re-parses each grown file through the existing ingest pipeline. Idempotent
via content-hash dedup; the cursor table only suppresses re-work when the
stored content fingerprint and parser fingerprint still match the file.

There's no concept of a "live" or "active" session here — any JSONL under the
roots may grow at any time, including ones years old (resume). The watcher
treats every grown file identically.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.sources.live.cursor import CursorStore

if TYPE_CHECKING:
    from polylogue.api import Polylogue

logger = get_logger(__name__)
_PARSER_FINGERPRINT = "live-jsonl-full-file-v1"


@dataclass(frozen=True, slots=True)
class WatchSource:
    """A directory to watch for live JSONL session files."""

    name: str
    root: Path

    def exists(self) -> bool:
        return self.root.exists()


@dataclass(slots=True)
class _PendingFile:
    path: Path
    last_change_at: float = 0.0
    pending_task: asyncio.Task[None] | None = field(default=None)


class LiveWatcher:
    """Async watcher that ingests grown JSONL files as they're appended-to."""

    def __init__(
        self,
        polylogue: Polylogue,
        sources: Iterable[WatchSource],
        *,
        debounce_s: float = 2.0,
        cursor: CursorStore | None = None,
    ) -> None:
        self._polylogue = polylogue
        self._sources = tuple(sources)
        self._debounce_s = debounce_s
        self._cursor = cursor or CursorStore(polylogue.archive_root / "polylogue.sqlite")
        self._pending: dict[Path, _PendingFile] = {}
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
                self._schedule(path)

    def stop(self) -> None:
        self._stop.set()

    def cancel_pending(self) -> None:
        """Cancel all orphaned debounced child tasks.

        Called during shutdown to clean up scheduled work that will never
        complete.
        """
        for entry in self._pending.values():
            if entry.pending_task is not None and not entry.pending_task.done():
                entry.pending_task.cancel()

    async def _catch_up(self, roots: list[Path]) -> None:
        files: list[Path] = []
        for root in roots:
            files.extend(p for p in root.rglob("*.jsonl") if p.is_file())
        if not files:
            return
        logger.info("live.watcher: catch-up scan over %d file(s)", len(files))
        for path in files:
            if self._stop.is_set():
                logger.info("live.watcher: catch-up interrupted by stop event")
                return
            await self._ingest_if_grown(path)

    def _schedule(self, path: Path) -> None:
        entry = self._pending.get(path)
        if entry is None:
            entry = _PendingFile(path=path)
            self._pending[path] = entry

        loop = asyncio.get_running_loop()
        entry.last_change_at = loop.time()
        if entry.pending_task is None or entry.pending_task.done():
            entry.pending_task = asyncio.create_task(self._debounced(entry))

    async def _debounced(self, entry: _PendingFile) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(self._debounce_s)
            if loop.time() - entry.last_change_at >= self._debounce_s:
                break
        await self._ingest_if_grown(entry.path)

    async def _ingest_if_grown(self, path: Path) -> None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return
        size = stat.st_size
        try:
            fingerprint, last_complete_newline = _fingerprint_file(path)
        except FileNotFoundError:
            return
        cursor = self._cursor.get_record(path)
        if (
            cursor is not None
            and size == cursor.byte_size
            and fingerprint == cursor.content_fingerprint
            and cursor.parser_fingerprint == _PARSER_FINGERPRINT
        ):
            return
        source_name = self._source_name_for(path)
        try:
            await self._polylogue.parse_file(path, source_name=source_name)
        except Exception as exc:
            logger.warning("live.watcher: parse failed for %s: %s", path, exc)
            self._cursor.mark_failed(path)
            return
        self._cursor.set(
            path,
            size,
            byte_offset=last_complete_newline,
            last_complete_newline=last_complete_newline,
            parser_fingerprint=_PARSER_FINGERPRINT,
            content_fingerprint=fingerprint,
            source_name=source_name,
            st_dev=getattr(stat, "st_dev", None),
            st_ino=getattr(stat, "st_ino", None),
            mtime_ns=getattr(stat, "st_mtime_ns", None),
        )
        self._cursor.reset_failures(path)
        logger.debug("live.watcher: ingested %s (size=%d, source=%s)", path, size, source_name)

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


__all__ = ["LiveWatcher", "WatchSource", "default_sources"]

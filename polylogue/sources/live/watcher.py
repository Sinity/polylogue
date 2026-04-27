"""Live JSONL session watcher.

Watches one or more roots for ``*.jsonl`` changes via ``watchfiles`` and
re-parses each grown file through the existing ingest pipeline. Idempotent
via content-hash dedup; the cursor table only suppresses re-work when a file
hasn't grown since we last saw it.

There's no concept of a "live" or "active" session here — any JSONL under the
roots may grow at any time, including ones years old (resume). The watcher
treats every grown file identically.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.sources.live.cursor import CursorStore

if TYPE_CHECKING:
    from polylogue.api import Polylogue

logger = get_logger(__name__)


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

    async def _catch_up(self, roots: list[Path]) -> None:
        files: list[Path] = []
        for root in roots:
            files.extend(p for p in root.rglob("*.jsonl") if p.is_file())
        if not files:
            return
        logger.info("live.watcher: catch-up scan over %d file(s)", len(files))
        for path in files:
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
        cursor = self._cursor.get(path)
        if size == cursor:
            return
        try:
            await self._polylogue.parse_file(path, source_name=path.parent.name)
        except Exception as exc:
            logger.warning("live.watcher: parse failed for %s: %s", path, exc)
            return
        self._cursor.set(path, size)
        logger.debug("live.watcher: ingested %s (size=%d)", path, size)


def default_sources() -> tuple[WatchSource, ...]:
    """Discover the default live-source roots from XDG/home conventions."""
    from polylogue.paths import claude_code_path, codex_path

    return (
        WatchSource(name="claude-code", root=claude_code_path()),
        WatchSource(name="codex", root=codex_path()),
    )


__all__ = ["LiveWatcher", "WatchSource", "default_sources"]

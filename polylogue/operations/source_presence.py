"""Whether the daemon's watch set would find any material to acquire."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from polylogue.sources.live.discovery import _bounded_source_paths
from polylogue.sources.live.watcher import POLYLOGUE_OWNED_SOURCE_NAMES, WatchSource, daemon_watch_sources
from polylogue.sources.walk_faults import WalkRefusedError


@dataclass(frozen=True, slots=True)
class WatchedSourcePresence:
    """The canonical tool locations checked, and whether any would be acquired."""

    present: bool
    tool_roots: tuple[Path, ...]


def watched_source_presence(*, hermes_root: Path | None) -> WatchedSourcePresence:
    """Probe the daemon's own watch set (``daemon_watch_sources``), as ``polylogued run`` builds it.

    That set includes secondary canonical roots, such as Codex's state
    database beside its sessions directory. A tool's location counts once it
    exists, since the daemon acquires what the tool writes there. The
    Polylogue-owned inbox and capture spool exist regardless, so they count
    only when they hold a file their watch source admits. Hook carriers
    (the sources with a topology identity) are evidence about sessions, not
    a chat source.
    """
    sources = tuple(source for source in daemon_watch_sources(hermes_root=hermes_root) if source.source_id is None)
    return WatchedSourcePresence(
        present=any(_would_acquire(source, sources) for source in sources),
        tool_roots=tuple(source.root for source in sources if source.name not in POLYLOGUE_OWNED_SOURCE_NAMES),
    )


def _would_acquire(source: WatchSource, sources: tuple[WatchSource, ...]) -> bool:
    if source.name not in POLYLOGUE_OWNED_SOURCE_NAMES:
        if source.exact_paths is not None:
            return any(path.exists() for path in source.exact_paths)
        return source.exists()
    try:
        return bool(_bounded_source_paths(source, sources, limit=1, after=None))
    except (WalkRefusedError, OSError):
        return False


__all__ = ["WatchedSourcePresence", "watched_source_presence"]

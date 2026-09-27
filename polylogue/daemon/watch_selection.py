"""The running daemon's live-watch selection, as status reports it.

``polylogued run`` can narrow the typed default sources (``--default-source``,
``--no-default-sources``). Status used to describe every default source as a
live source regardless, so a narrowed selection that left a source out -- its
files present, never acquired -- was invisible. The run command records the
selection it actually watches here, and status reads it back to report both the
watched sources and every default source that exists on disk but is not
selected.
"""

from __future__ import annotations

from threading import Lock

from polylogue.sources.live.watcher import WatchSource, default_sources

_LOCK = Lock()
_ACTIVE: tuple[WatchSource, ...] | None = None
_UNSELECTED: tuple[WatchSource, ...] = ()


def record_active_watch_sources(
    sources: tuple[WatchSource, ...],
    *,
    unselected: tuple[WatchSource, ...] = (),
) -> None:
    """Record the sources this daemon watches and the defaults it leaves out."""
    global _ACTIVE, _UNSELECTED
    with _LOCK:
        _ACTIVE = tuple(sources)
        _UNSELECTED = tuple(unselected)


def clear_active_watch_sources() -> None:
    """Forget the selection once the daemon's services have stopped."""
    global _ACTIVE, _UNSELECTED
    with _LOCK:
        _ACTIVE = None
        _UNSELECTED = ()


def active_watch_sources() -> tuple[WatchSource, ...] | None:
    """The recorded selection, or ``None`` outside a running daemon."""
    with _LOCK:
        return _ACTIVE


def recorded_unselected_sources() -> tuple[WatchSource, ...]:
    """Default sources the running daemon found on disk but does not watch."""
    with _LOCK:
        return _UNSELECTED


def unselected_default_sources(
    watched: tuple[WatchSource, ...],
    defaults: tuple[WatchSource, ...] | None = None,
) -> tuple[WatchSource, ...]:
    """Default sources present on disk that ``watched`` does not cover."""
    watched_names = {source.name for source in watched}
    watched_roots = {source.root.resolve(strict=False) for source in watched}
    candidates = default_sources() if defaults is None else defaults
    return tuple(
        source
        for source in candidates
        if source.name not in watched_names
        and source.root.resolve(strict=False) not in watched_roots
        and source.exists()
    )


__all__ = [
    "active_watch_sources",
    "clear_active_watch_sources",
    "record_active_watch_sources",
    "recorded_unselected_sources",
    "unselected_default_sources",
]

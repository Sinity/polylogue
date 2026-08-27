"""Live filesystem watching for append-only JSONL session sources.

Tails ``~/.claude/projects/`` and ``~/.codex/sessions/`` and feeds new or
updated JSONL files into the regular parse/upsert pipeline. Idempotent:
re-ingesting the same file is a no-op via content-hash dedup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.sources.live.batch import LiveBatchMetrics, LiveBatchProcessor
    from polylogue.sources.live.watcher import LiveWatcher, WatchSource

__all__ = ["LiveBatchMetrics", "LiveBatchProcessor", "LiveWatcher", "WatchSource"]


def __getattr__(name: str) -> object:
    """Load live surfaces lazily so source adapters can import shared contracts.

    ``source_acquisition_components`` is used by Drive and imports the small
    admission contract from this package.  Eagerly importing ``batch`` here
    makes that path recurse back into ``source_acquisition_components`` before
    its detection constants exist.  Lazy exports preserve the public package
    API while keeping adapter import order acyclic.
    """
    if name in {"LiveBatchMetrics", "LiveBatchProcessor"}:
        from polylogue.sources.live.batch import LiveBatchMetrics, LiveBatchProcessor

        return {"LiveBatchMetrics": LiveBatchMetrics, "LiveBatchProcessor": LiveBatchProcessor}[name]
    if name in {"LiveWatcher", "WatchSource"}:
        from polylogue.sources.live.watcher import LiveWatcher, WatchSource

        return {"LiveWatcher": LiveWatcher, "WatchSource": WatchSource}[name]
    raise AttributeError(name)

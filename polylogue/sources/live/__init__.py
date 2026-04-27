"""Live filesystem watching for append-only JSONL session sources.

Tails ``~/.claude/projects/`` and ``~/.codex/sessions/`` and feeds new or
updated JSONL files into the regular parse/upsert pipeline. Idempotent:
re-ingesting the same file is a no-op via content-hash dedup.
"""

from __future__ import annotations

from polylogue.sources.live.watcher import LiveWatcher, WatchSource

__all__ = ["LiveWatcher", "WatchSource"]

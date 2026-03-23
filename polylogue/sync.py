"""Synchronous bridge for the Polylogue async facade.

Wraps the async ``Polylogue`` facade so that synchronous callers
(like Lynchpin's trajectory pipeline) can consume session profiles,
summaries, and conversations without managing an event loop.

Example::

    from polylogue.sync import SyncPolylogue

    poly = SyncPolylogue()
    summaries = poly.list_summaries(since="2026-01-01")
    conv = poly.get_conversation("abc12345")
    poly.close()
"""

from __future__ import annotations

from collections.abc import Awaitable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from polylogue.sync_bridge import run_coroutine_sync

if TYPE_CHECKING:
    from polylogue.facade import ArchiveStats
    from polylogue.lib.models import Conversation, ConversationSummary
    from polylogue.storage.search import SearchResult

T = TypeVar("T")


def _run(coro: Awaitable[T]) -> T:
    """Run a Polylogue coroutine from synchronous callers.

    This remains the canonical sync execution seam for module-local sync helpers
    and external consumers that build async filter pipelines before forcing the
    terminal awaitable.
    """

    return run_coroutine_sync(coro)


class SyncPolylogue:
    """Synchronous wrapper around the async ``Polylogue`` facade.

    All methods delegate to ``Polylogue`` through the shared sync bridge.
    """

    def __init__(
        self,
        archive_root: str | Path | None = None,
        db_path: str | Path | None = None,
    ):
        from polylogue.facade import Polylogue

        self._facade = Polylogue(archive_root=archive_root, db_path=db_path)

    def close(self) -> None:
        """Release database connections."""
        _run(self._facade.close())

    def __enter__(self) -> SyncPolylogue:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # --- Queries ---

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Fetch a single conversation by full or prefix ID."""
        return _run(self._facade.get_conversation(conversation_id))

    def get_conversations(self, conversation_ids: list[str]) -> list[Conversation]:
        """Batch fetch conversations."""
        return _run(self._facade.get_conversations(conversation_ids))

    def list_conversations(
        self,
        provider: str | None = None,
        limit: int | None = None,
    ) -> list[Conversation]:
        """List conversations with optional filtering."""
        return _run(self._facade.list_conversations(provider=provider, limit=limit))

    def list_summaries(
        self,
        *,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
        provider: str | None = None,
        limit: int | None = None,
    ) -> list[ConversationSummary]:
        """List lightweight conversation summaries."""
        filt = self._facade.filter()
        if provider:
            filt = filt.provider(provider)
        if since:
            filt = filt.since(since)
        if until:
            filt = filt.until(until)
        if limit:
            filt = filt.limit(limit)
        return _run(filt.list_summaries())

    def search(
        self,
        query: str,
        *,
        limit: int = 100,
        source: str | None = None,
        since: str | None = None,
    ) -> SearchResult:
        """Search conversations."""
        return _run(self._facade.search(query, limit=limit, source=source, since=since))

    def stats(self) -> ArchiveStats:
        """Get archive statistics."""
        return _run(self._facade.stats())

    def filter(self):
        """Create a fluent filter builder (terminal methods are still async)."""
        return self._facade.filter()

    def __repr__(self) -> str:
        return f"SyncPolylogue(facade={self._facade!r})"


__all__ = ["SyncPolylogue", "_run"]

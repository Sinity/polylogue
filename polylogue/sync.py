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
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, TypeVar

from polylogue.sync_bridge import run_coroutine_sync
from polylogue.sync_conversation_queries import SyncConversationQueriesMixin
from polylogue.sync_product_queries import SyncProductQueriesMixin

if TYPE_CHECKING:
    from polylogue.facade import Polylogue
    from polylogue.lib.filters import ConversationFilter

T = TypeVar("T")


def _run(coro: Awaitable[T]) -> T:
    """Run a Polylogue coroutine from synchronous callers."""
    return run_coroutine_sync(coro)


class SyncPolylogue(SyncConversationQueriesMixin, SyncProductQueriesMixin):
    """Synchronous wrapper around the async ``Polylogue`` facade."""

    _facade: Polylogue

    def __init__(
        self,
        archive_root: str | Path | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        from polylogue.facade import Polylogue

        self._facade = Polylogue(archive_root=archive_root, db_path=db_path)

    def close(self) -> None:
        """Release database connections."""
        _run(self._facade.close())

    def __enter__(self) -> SyncPolylogue:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def filter(self) -> ConversationFilter:
        """Create a fluent filter builder (terminal methods are still async)."""
        return self._facade.filter()

    def __repr__(self) -> str:
        return f"SyncPolylogue(facade={self._facade!r})"


__all__ = ["SyncPolylogue", "_run"]

"""Async indexing service for pipeline operations."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable
from typing import TYPE_CHECKING

from typing_extensions import TypedDict

from polylogue.logging import get_logger
from polylogue.storage.fts.fts_lifecycle import (
    ensure_fts_index_async,
    fts_index_status_async,
    rebuild_fts_index_async,
    repair_fts_index_async,
)
from polylogue.storage.search.cache import invalidate_search_cache

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.core.protocols import ProgressCallback
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

logger = get_logger(__name__)

__all__ = ["IndexService"]


class IndexStatus(TypedDict):
    exists: bool
    count: int


def _status_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


async def ensure_index(backend: SQLiteBackend) -> None:
    """Ensure the FTS5 table exists on the archive backend."""
    async with backend.connection() as conn:
        await ensure_fts_index_async(conn)


async def rebuild_index(
    backend: SQLiteBackend,
    *,
    session_ids: list[str] | None = None,
    phase_count: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Rebuild the entire FTS5 index from persisted message rows."""
    session_id_list = (
        session_ids if session_ids is not None else [session_id async for session_id in backend.iter_session_ids()]
    )
    async with backend.connection() as conn:
        del phase_count
        phase_total = len(session_id_list)
        if progress_callback is not None and session_id_list:
            progress_callback(
                0,
                desc=f"Indexing: full-text search 0/{phase_total:,}",
            )
        await rebuild_fts_index_async(
            conn,
            session_ids=session_id_list,
            progress_callback=progress_callback,
            progress_desc=(
                _fts_progress_desc_factory(phase_total=phase_total) if progress_callback is not None else None
            ),
        )
        await conn.commit()
    invalidate_search_cache()


async def update_index_for_sessions(
    session_ids: Iterable[str] | AsyncIterable[str],
    backend: SQLiteBackend,
    *,
    phase_count: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Repair FTS rows for the provided sessions from persisted message rows."""
    session_id_list = [session_id async for session_id in _iter_ids(session_ids)]
    changed = bool(session_id_list)
    async with backend.connection() as conn:
        del phase_count
        phase_total = len(session_id_list)
        if progress_callback is not None and session_id_list:
            progress_callback(
                0,
                desc=f"Indexing: full-text search 0/{phase_total:,}",
            )
        await repair_fts_index_async(
            conn,
            session_id_list,
            progress_callback=progress_callback,
            progress_desc=(
                _fts_progress_desc_factory(phase_total=phase_total) if progress_callback is not None else None
            ),
        )
        await conn.commit()
    if changed:
        invalidate_search_cache()


async def _iter_ids(items: Iterable[str] | AsyncIterable[str]) -> AsyncIterator[str]:
    if isinstance(items, AsyncIterable):
        async for item in items:
            yield item
        return

    for item in items:
        yield item


def _fts_progress_desc_factory(*, phase_total: int) -> Callable[[int, int], str]:
    def describe(processed: int, total: int) -> str:
        del total
        return f"Indexing: full-text search {processed:,}/{phase_total:,}"

    return describe


async def index_status(backend: SQLiteBackend) -> IndexStatus:
    """Return whether the FTS5 index exists and how many docs it contains."""
    async with backend.connection() as conn:
        raw_status = await fts_index_status_async(conn)
    return {
        "exists": bool(raw_status.get("exists", False)),
        "count": _status_int(raw_status.get("count", 0)),
    }


class IndexService:
    """Service for managing full-text and vector search indices (async version)."""

    def __init__(
        self,
        config: Config,
        backend: SQLiteBackend | None = None,
    ) -> None:
        """Initialize the async indexing service.

        Args:
            config: Application configuration
            backend: Optional async database backend to use
        """
        self.config = config
        self.backend = backend

    async def update_index(
        self,
        session_ids: Iterable[str] | AsyncIterable[str],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> bool:
        """Update the search index for specific sessions.

        Args:
            session_ids: Session IDs to index

        Returns:
            True if indexing succeeded, False otherwise
        """
        if self.backend is None:
            logger.error("Cannot update index without a backend")
            return False

        session_id_list = [session_id async for session_id in _iter_ids(session_ids)]
        try:
            if progress_callback is None:
                await update_index_for_sessions(
                    session_id_list,
                    self.backend,
                )
            else:
                await update_index_for_sessions(
                    session_id_list,
                    self.backend,
                    progress_callback=progress_callback,
                )
            return True
        except sqlite3.DatabaseError as exc:
            logger.error("Failed to update index", error=str(exc), exc_info=True)
            return False

    async def rebuild_index(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> bool:
        """Rebuild the entire search index from scratch.

        Returns:
            True if rebuild succeeded, False otherwise
        """
        if self.backend is None:
            logger.error("Cannot rebuild index without a backend")
            return False

        try:
            session_id_list = [session_id async for session_id in self.backend.iter_session_ids()]
            if progress_callback is None:
                await rebuild_index(
                    self.backend,
                    session_ids=session_id_list,
                )
            else:
                await rebuild_index(
                    self.backend,
                    session_ids=session_id_list,
                    progress_callback=progress_callback,
                )
            return True
        except sqlite3.DatabaseError as exc:
            logger.error("Failed to rebuild index", error=str(exc), exc_info=True)
            return False

    async def ensure_index_exists(self) -> bool:
        """Ensure the FTS5 index table exists.

        Returns:
            True if index exists or was created, False on error
        """
        try:
            if self.backend is not None:
                await ensure_index(self.backend)
            return True
        except sqlite3.DatabaseError as exc:
            logger.error("Failed to ensure index exists", error=str(exc), exc_info=True)
            return False

    async def get_index_status(self) -> IndexStatus:
        """Get the current status of the search index.

        Returns:
            Dictionary with 'exists' (bool) and 'count' (int) keys
        """
        if self.backend is None:
            logger.error("Cannot get index status without a backend")
            return {"exists": False, "count": 0}

        try:
            return await index_status(self.backend)
        except sqlite3.DatabaseError as exc:
            logger.error("Failed to get index status", error=str(exc), exc_info=True)
            return {"exists": False, "count": 0}

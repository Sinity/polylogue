"""Async indexing service for pipeline operations."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.storage.action_event_rebuild_runtime import (
    action_event_repair_candidates_async,
    rebuild_action_event_read_model_async,
)
from polylogue.storage.fts_lifecycle import (
    ensure_fts_index_async,
    fts_index_status_async,
    rebuild_fts_index_async,
    repair_fts_index_async,
)
from polylogue.storage.search_cache import invalidate_search_cache

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.protocols import ProgressCallback
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

logger = get_logger(__name__)

__all__ = ["IndexService"]


async def ensure_index(backend: SQLiteBackend) -> None:
    """Ensure the FTS5 table exists on the archive backend."""
    async with backend.connection() as conn:
        await ensure_fts_index_async(conn)


async def rebuild_index(
    backend: SQLiteBackend,
    *,
    conversation_ids: list[str] | None = None,
    phase_count: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Rebuild the entire FTS5 index from persisted message rows."""
    conversation_id_list = (
        conversation_ids
        if conversation_ids is not None
        else [conversation_id async for conversation_id in backend.queries.iter_conversation_ids()]
    )
    async with backend.connection() as conn:
        action_targets = await _action_event_repair_targets(conn, conversation_id_list)
        action_target_count = len(action_targets)
        del phase_count
        phase_total = len(conversation_id_list) + action_target_count
        if action_targets:
            if progress_callback is not None:
                progress_callback(0, desc=f"Indexing: action events 0/{phase_total:,}")
            await rebuild_action_event_read_model_async(
                conn,
                conversation_ids=action_targets,
                progress_callback=progress_callback,
                progress_desc=(
                    _action_progress_desc_factory(phase_total)
                    if progress_callback is not None
                    else None
                ),
            )
        if progress_callback is not None and conversation_id_list:
            progress_callback(
                0,
                desc=f"Indexing: full-text search {action_target_count:,}/{phase_total:,}",
            )
        await rebuild_fts_index_async(
            conn,
            conversation_ids=conversation_id_list,
            progress_callback=progress_callback,
            progress_desc=(
                _fts_progress_desc_factory(offset=action_target_count, phase_total=phase_total)
                if progress_callback is not None
                else None
            ),
        )
        await conn.commit()
    invalidate_search_cache()


async def update_index_for_conversations(
    conversation_ids: Iterable[str] | AsyncIterable[str],
    backend: SQLiteBackend,
    *,
    phase_count: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Repair FTS rows for the provided conversations from persisted message rows."""
    conversation_id_list = [conversation_id async for conversation_id in _iter_ids(conversation_ids)]
    changed = bool(conversation_id_list)
    async with backend.connection() as conn:
        action_targets = await _action_event_repair_targets(conn, conversation_id_list)
        action_target_count = len(action_targets)
        del phase_count
        phase_total = len(conversation_id_list) + action_target_count
        if action_targets:
            if progress_callback is not None:
                progress_callback(0, desc=f"Indexing: action events 0/{phase_total:,}")
            await rebuild_action_event_read_model_async(
                conn,
                conversation_ids=action_targets,
                progress_callback=progress_callback,
                progress_desc=(
                    _action_progress_desc_factory(phase_total)
                    if progress_callback is not None
                    else None
                ),
            )
        if progress_callback is not None and conversation_id_list:
            progress_callback(
                0,
                desc=f"Indexing: full-text search {action_target_count:,}/{phase_total:,}",
            )
        await repair_fts_index_async(
            conn,
            conversation_id_list,
            progress_callback=progress_callback,
            progress_desc=(
                _fts_progress_desc_factory(offset=action_target_count, phase_total=phase_total)
                if progress_callback is not None
                else None
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


def _action_progress_desc_factory(phase_total: int):
    def describe(processed: int, total: int) -> str:
        del total
        return f"Indexing: action events {processed:,}/{phase_total:,}"

    return describe


def _fts_progress_desc_factory(*, offset: int, phase_total: int):
    def describe(processed: int, total: int) -> str:
        del total
        return f"Indexing: full-text search {offset + processed:,}/{phase_total:,}"

    return describe


async def _action_event_repair_targets(
    conn,
    conversation_id_list: list[str],
) -> list[str]:
    if not conversation_id_list:
        return []
    candidate_ids = await action_event_repair_candidates_async(conn)
    if not candidate_ids:
        return []
    allowed = set(conversation_id_list)
    return [conversation_id for conversation_id in candidate_ids if conversation_id in allowed]


async def index_status(backend: SQLiteBackend) -> dict[str, object]:
    """Return whether the FTS5 index exists and how many docs it contains."""
    async with backend.connection() as conn:
        return await fts_index_status_async(conn)


class IndexService:
    """Service for managing full-text and vector search indices (async version)."""

    def __init__(
        self,
        config: Config,
        backend: SQLiteBackend | None = None,
    ):
        """Initialize the async indexing service.

        Args:
            config: Application configuration
            backend: Optional async database backend to use
        """
        self.config = config
        self.backend = backend

    async def update_index(
        self,
        conversation_ids: Iterable[str] | AsyncIterable[str],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> bool:
        """Update the search index for specific conversations.

        Args:
            conversation_ids: Conversation IDs to index

        Returns:
            True if indexing succeeded, False otherwise
        """
        if self.backend is None:
            logger.error("Cannot update index without a backend")
            return False

        conversation_id_list = [conversation_id async for conversation_id in _iter_ids(conversation_ids)]
        try:
            if progress_callback is None:
                await update_index_for_conversations(
                    conversation_id_list,
                    self.backend,
                )
            else:
                await update_index_for_conversations(
                    conversation_id_list,
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
            conversation_id_list = [
                conversation_id async for conversation_id in self.backend.queries.iter_conversation_ids()
            ]
            if progress_callback is None:
                await rebuild_index(
                    self.backend,
                    conversation_ids=conversation_id_list,
                )
            else:
                await rebuild_index(
                    self.backend,
                    conversation_ids=conversation_id_list,
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

    async def get_index_status(self) -> dict[str, object]:
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

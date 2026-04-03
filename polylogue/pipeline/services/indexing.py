"""Async indexing service for pipeline operations."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.storage.action_event_rebuild_runtime import rebuild_action_event_read_model_async
from polylogue.storage.fts_lifecycle import (
    ensure_fts_index_async,
    fts_index_status_async,
    rebuild_fts_index_async,
    repair_fts_index_async,
)
from polylogue.storage.search_cache import invalidate_search_cache
from polylogue.storage.search_providers import create_vector_provider

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
    phase_count: int = 2,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Rebuild the entire FTS5 index from persisted message rows."""
    conversation_id_list = (
        conversation_ids
        if conversation_ids is not None
        else [conversation_id async for conversation_id in backend.queries.iter_conversation_ids()]
    )
    phase_total = len(conversation_id_list) * 2
    if phase_count > 2:
        phase_total = len(conversation_id_list) * phase_count

    def action_progress_desc(processed: int, total: int) -> str:
        del total
        return f"Indexing: action events {processed:,}/{phase_total:,}"

    def fts_progress_desc(processed: int, total: int) -> str:
        del total
        return f"Indexing: full-text search {len(conversation_id_list) + processed:,}/{phase_total:,}"

    async with backend.connection() as conn:
        if progress_callback is not None and conversation_id_list:
            progress_callback(0, desc=f"Indexing: action events 0/{phase_total:,}")
        await rebuild_action_event_read_model_async(
            conn,
            conversation_ids=conversation_id_list,
            progress_callback=progress_callback,
            progress_desc=action_progress_desc if progress_callback is not None else None,
        )
        if progress_callback is not None and conversation_id_list:
            progress_callback(
                0,
                desc=f"Indexing: full-text search {len(conversation_id_list):,}/{phase_total:,}",
            )
        await rebuild_fts_index_async(
            conn,
            conversation_ids=conversation_id_list,
            progress_callback=progress_callback,
            progress_desc=fts_progress_desc if progress_callback is not None else None,
        )
        await conn.commit()
    invalidate_search_cache()


async def update_index_for_conversations(
    conversation_ids: Iterable[str] | AsyncIterable[str],
    backend: SQLiteBackend,
    *,
    phase_count: int = 2,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Repair FTS rows for the provided conversations from persisted message rows."""
    conversation_id_list = [conversation_id async for conversation_id in _iter_ids(conversation_ids)]
    changed = bool(conversation_id_list)
    phase_total = len(conversation_id_list) * 2
    if phase_count > 2:
        phase_total = len(conversation_id_list) * phase_count

    def action_progress_desc(processed: int, total: int) -> str:
        del total
        return f"Indexing: action events {processed:,}/{phase_total:,}"

    def fts_progress_desc(processed: int, total: int) -> str:
        del total
        return f"Indexing: full-text search {len(conversation_id_list) + processed:,}/{phase_total:,}"

    async with backend.connection() as conn:
        if progress_callback is not None and conversation_id_list:
            progress_callback(0, desc=f"Indexing: action events 0/{phase_total:,}")
        await rebuild_action_event_read_model_async(
            conn,
            conversation_ids=conversation_id_list,
            progress_callback=progress_callback,
            progress_desc=action_progress_desc if progress_callback is not None else None,
        )
        if progress_callback is not None and conversation_id_list:
            progress_callback(
                0,
                desc=f"Indexing: full-text search {len(conversation_id_list):,}/{phase_total:,}",
            )
        await repair_fts_index_async(
            conn,
            conversation_id_list,
            progress_callback=progress_callback,
            progress_desc=fts_progress_desc if progress_callback is not None else None,
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

    def _auto_embed_enabled(self) -> bool:
        return bool(self.config.index_config and self.config.index_config.auto_embed)

    async def _embed_indexed_conversations(
        self,
        conversation_ids: list[str],
        *,
        progress_callback: ProgressCallback | None = None,
        progress_offset: int = 0,
        phase_total: int = 0,
    ) -> bool:
        if not conversation_ids:
            return True
        if self.backend is None:
            logger.error("Cannot auto-embed without a backend")
            return False

        from polylogue.storage.repository import ConversationRepository

        vector_provider = create_vector_provider(config=self.config, db_path=self.backend.db_path)
        if vector_provider is None:
            logger.error("Auto-embed enabled but no vector provider is available")
            return False

        repository = ConversationRepository(backend=self.backend)
        if progress_callback is not None:
            progress_callback(0, desc=f"Indexing: embeddings {progress_offset:,}/{phase_total:,}")

        failed = False
        for processed, conversation_id in enumerate(conversation_ids, start=1):
            try:
                await repository.embed_conversation(
                    conversation_id,
                    vector_provider=vector_provider,
                )
            except Exception as exc:
                failed = True
                logger.error(
                    "Failed to embed conversation during index stage",
                    conversation_id=conversation_id,
                    error=str(exc),
                    exc_info=True,
                )
            if progress_callback is not None:
                progress_callback(
                    1,
                    desc=f"Indexing: embeddings {progress_offset + processed:,}/{phase_total:,}",
                )

        return not failed

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
            phase_count = 3 if self._auto_embed_enabled() and conversation_id_list else 2
            if progress_callback is None:
                await update_index_for_conversations(
                    conversation_id_list,
                    self.backend,
                    phase_count=phase_count,
                )
            else:
                await update_index_for_conversations(
                    conversation_id_list,
                    self.backend,
                    phase_count=phase_count,
                    progress_callback=progress_callback,
                )
            if self._auto_embed_enabled():
                return await self._embed_indexed_conversations(
                    conversation_id_list,
                    progress_callback=progress_callback,
                    progress_offset=len(conversation_id_list) * 2,
                    phase_total=len(conversation_id_list) * phase_count,
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
            conversation_id_list = [conversation_id async for conversation_id in self.backend.queries.iter_conversation_ids()]
            phase_count = 3 if self._auto_embed_enabled() and conversation_id_list else 2
            if progress_callback is None:
                await rebuild_index(
                    self.backend,
                    conversation_ids=conversation_id_list,
                    phase_count=phase_count,
                )
            else:
                await rebuild_index(
                    self.backend,
                    conversation_ids=conversation_id_list,
                    phase_count=phase_count,
                    progress_callback=progress_callback,
                )
            if self._auto_embed_enabled():
                return await self._embed_indexed_conversations(
                    conversation_id_list,
                    progress_callback=progress_callback,
                    progress_offset=len(conversation_id_list) * 2,
                    phase_total=len(conversation_id_list) * phase_count,
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

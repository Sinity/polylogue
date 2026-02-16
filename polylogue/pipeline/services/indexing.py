"""Async indexing service for pipeline operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.lib.log import get_logger
from polylogue.storage.async_index import (
    ensure_index,
    index_status,
    rebuild_index,
    update_index_for_conversations,
)

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

logger = get_logger(__name__)

__all__ = ["IndexService"]


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

    async def update_index(self, conversation_ids: list[str]) -> bool:
        """Update the search index for specific conversations.

        Args:
            conversation_ids: List of conversation IDs to index

        Returns:
            True if indexing succeeded, False otherwise
        """
        if not conversation_ids:
            if self.backend is not None:
                try:
                    await ensure_index(self.backend)
                except Exception as exc:
                    logger.error("Failed to ensure index", error=str(exc), exc_info=True)
                    return False
            return True

        if self.backend is None:
            logger.error("Cannot update index without a backend")
            return False

        try:
            await update_index_for_conversations(conversation_ids, self.backend)
            return True
        except Exception as exc:
            logger.error("Failed to update index", error=str(exc), exc_info=True)
            return False

    async def rebuild_index(self) -> bool:
        """Rebuild the entire search index from scratch.

        Returns:
            True if rebuild succeeded, False otherwise
        """
        if self.backend is None:
            logger.error("Cannot rebuild index without a backend")
            return False

        try:
            await rebuild_index(self.backend)
            return True
        except Exception as exc:
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
        except Exception as exc:
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
        except Exception as exc:
            logger.error("Failed to get index status", error=str(exc), exc_info=True)
            return {"exists": False, "count": 0}

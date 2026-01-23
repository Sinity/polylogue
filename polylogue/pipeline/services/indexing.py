"""Indexing service for pipeline operations."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from polylogue.core.log import get_logger
from polylogue.storage.index import ensure_index, index_status, rebuild_index, update_index_for_conversations

if TYPE_CHECKING:
    from polylogue.config import Config

logger = get_logger(__name__)


class IndexService:
    """Service for managing full-text and vector search indices."""

    def __init__(
        self,
        config: Config,
        conn: sqlite3.Connection | None = None,
    ):
        """Initialize the indexing service.

        Args:
            config: Application configuration
            conn: Optional database connection to use
        """
        self.config = config
        self.conn = conn

    def update_index(self, conversation_ids: list[str]) -> bool:
        """Update the search index for specific conversations.

        Args:
            conversation_ids: List of conversation IDs to index

        Returns:
            True if indexing succeeded, False otherwise
        """
        if not conversation_ids:
            ensure_index(self.conn)
            return True

        try:
            update_index_for_conversations(conversation_ids, self.conn)
            return True
        except Exception as exc:
            logger.error("Failed to update index", error=str(exc))
            return False

    def rebuild_index(self) -> bool:
        """Rebuild the entire search index from scratch.

        Returns:
            True if rebuild succeeded, False otherwise
        """
        try:
            rebuild_index(self.conn)
            return True
        except Exception as exc:
            logger.error("Failed to rebuild index", error=str(exc))
            return False

    def ensure_index_exists(self) -> bool:
        """Ensure the FTS5 index table exists.

        Returns:
            True if index exists or was created, False on error
        """
        try:
            ensure_index(self.conn)
            return True
        except Exception as exc:
            logger.error("Failed to ensure index exists", error=str(exc))
            return False

    def get_index_status(self) -> dict:
        """Get the current status of the search index.

        Returns:
            Dictionary with 'exists' (bool) and 'count' (int) keys
        """
        try:
            return index_status()
        except Exception as exc:
            logger.error("Failed to get index status", error=str(exc))
            return {"exists": False, "count": 0}

"""Async indexing service for pipeline operations."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from typing import TYPE_CHECKING

from polylogue.lib.log import get_logger

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

logger = get_logger(__name__)

__all__ = ["IndexService"]


async def ensure_index(backend: SQLiteBackend) -> None:
    """Create the FTS5 index table if it does not exist."""
    async with backend.connection() as conn:
        await conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                message_id UNINDEXED,
                conversation_id UNINDEXED,
                text,
                tokenize='unicode61'
            );
            """
        )


async def rebuild_index(backend: SQLiteBackend) -> None:
    """Rebuild the entire FTS5 index from message rows."""
    async with backend.connection() as conn:
        await ensure_index(backend)
        await conn.execute("DELETE FROM messages_fts")
        await conn.execute(
            """
            INSERT INTO messages_fts (message_id, conversation_id, text)
            SELECT messages.message_id, messages.conversation_id, messages.text
            FROM messages
            WHERE messages.text IS NOT NULL
            """
        )
        await conn.commit()


async def update_index_for_conversations(
    conversation_ids: Iterable[str] | AsyncIterable[str],
    backend: SQLiteBackend,
) -> None:
    """Update the FTS5 index for the provided conversations."""
    async with backend.connection() as conn:
        await ensure_index(backend)

        async for chunk_ids in _chunked_ids(conversation_ids, size=500):
            placeholders = ", ".join("?" for _ in chunk_ids)
            params = tuple(chunk_ids)

            await conn.execute(
                f"DELETE FROM messages_fts WHERE conversation_id IN ({placeholders})",
                params,
            )
            await conn.execute(
                f"""
                INSERT INTO messages_fts (message_id, conversation_id, text)
                SELECT message_id, conversation_id, text
                FROM messages
                WHERE text IS NOT NULL AND conversation_id IN ({placeholders})
                """,
                params,
            )

        await conn.commit()


async def _chunked_ids(
    items: Iterable[str] | AsyncIterable[str],
    *,
    size: int,
) -> AsyncIterator[list[str]]:
    chunk: list[str] = []
    async for item in _iter_ids(items):
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


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
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
        )
        row = await cursor.fetchone()
        exists = bool(row)
        count = 0
        if exists:
            cursor = await conn.execute("SELECT COUNT(*) FROM messages_fts_docsize")
            row = await cursor.fetchone()
            count = row[0] if row else 0
        return {"exists": exists, "count": int(count)}


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

        try:
            await update_index_for_conversations(conversation_ids, self.backend)
            return True
        except sqlite3.DatabaseError as exc:
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

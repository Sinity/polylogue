"""Async storage repository for conversation persistence.

Provides async/await interface for storing and retrieving conversations.
Wraps SQLiteBackend for parallel operations.

All methods are async and use eager loading (conversation_from_records)
instead of lazy loading, since async I/O already enables efficient parallel
fetching of conversations, messages, and attachments together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.protocols import ConversationReader, SearchStore, TagStore
from polylogue.storage.backends.async_sqlite import SQLiteBackend

if TYPE_CHECKING:
    from pathlib import Path

    from polylogue.storage.backends.query_store import SQLiteQueryStore

from polylogue.storage.repository_action_reads import RepositoryActionReadMixin
from polylogue.storage.repository_archive_reads import RepositoryArchiveReadMixin
from polylogue.storage.repository_product_profile_reads import (
    RepositoryProductProfileReadMixin,
)
from polylogue.storage.repository_product_summary_reads import (
    RepositoryProductSummaryReadMixin,
)
from polylogue.storage.repository_product_thread_reads import (
    RepositoryProductThreadReadMixin,
)
from polylogue.storage.repository_product_timeline_reads import (
    RepositoryProductTimelineReadMixin,
)
from polylogue.storage.repository_raw import RepositoryRawMixin
from polylogue.storage.repository_vectors import RepositoryVectorMixin
from polylogue.storage.repository_writes import RepositoryWriteMixin


class ConversationRepository(
    RepositoryArchiveReadMixin,
    RepositoryActionReadMixin,
    RepositoryProductProfileReadMixin,
    RepositoryProductTimelineReadMixin,
    RepositoryProductThreadReadMixin,
    RepositoryProductSummaryReadMixin,
    RepositoryRawMixin,
    RepositoryWriteMixin,
    RepositoryVectorMixin,
    ConversationReader,
    SearchStore,
    TagStore,
):
    """Async repository for conversation storage operations.

    Wraps SQLiteBackend to provide high-level async storage interface with
    full feature parity to sync ConversationRepository.

    All methods are async. Eager loading (conversation_from_records) is used
    for fetching conversations, enabling efficient parallel I/O via asyncio.gather()
    for conversations, messages, and attachments.

    Write safety is provided by SQLite's ``BEGIN IMMEDIATE`` transactions
    in the backend layer, combined with asyncio.Lock() serialization.

    Example:
        async with ConversationRepository() as repo:
            conv = await repo.get("claude-ai:abc123")
            convs = await repo.list(limit=10)
            await repo.save_conversation(conv_rec, msgs, atts)
    """

    def __init__(
        self,
        backend: SQLiteBackend | None = None,
        db_path: Path | None = None,
    ) -> None:
        """Initialize async storage repository.

        Args:
            backend: Optional SQLiteBackend instance. If provided, db_path is ignored.
            db_path: Optional path to database file. Used if backend is None.
        """
        if backend is not None:
            self._backend = backend
        else:
            self._backend = SQLiteBackend(db_path=db_path)

    async def __aenter__(self) -> ConversationRepository:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit async context manager."""
        await self.close()

    @property
    def backend(self) -> SQLiteBackend:
        """Access the underlying async storage backend."""
        return self._backend

    @property
    def queries(self) -> SQLiteQueryStore:
        """Access the canonical low-level query surface."""
        return self._backend.queries

    async def close(self) -> None:
        """Close database connections and release resources."""
        await self._backend.close()


__all__ = ["ConversationRepository"]

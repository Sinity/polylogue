"""Async storage repository for session persistence.

Provides async/await interface for storing and retrieving sessions.
Wraps SQLiteBackend for parallel operations.

All methods are async and use eager loading (session_from_records)
instead of lazy loading, since async I/O already enables efficient parallel
fetching of sessions, messages, and attachments together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

if TYPE_CHECKING:
    from pathlib import Path

from polylogue.storage.repository.archive.reads import RepositoryArchiveReadMixin
from polylogue.storage.repository.archive.repository_writes import RepositoryWriteMixin
from polylogue.storage.repository.insight.profile_reads import (
    RepositoryInsightProfileReadMixin,
)
from polylogue.storage.repository.insight.run_projection_reads import (
    RepositoryInsightRunProjectionReadMixin,
)
from polylogue.storage.repository.insight.summary_reads import (
    RepositoryInsightSummaryReadMixin,
)
from polylogue.storage.repository.insight.thread_reads import (
    RepositoryInsightThreadReadMixin,
)
from polylogue.storage.repository.insight.timeline_reads import (
    RepositoryInsightTimelineReadMixin,
)
from polylogue.storage.repository.insight.topology_reads import (
    RepositoryInsightTopologyReadMixin,
)
from polylogue.storage.repository.raw.repository_raw import RepositoryRawMixin
from polylogue.storage.repository.vectors.repository_vectors import RepositoryVectorMixin


class SessionRepository(
    RepositoryArchiveReadMixin,
    RepositoryInsightProfileReadMixin,
    RepositoryInsightRunProjectionReadMixin,
    RepositoryInsightTimelineReadMixin,
    RepositoryInsightThreadReadMixin,
    RepositoryInsightSummaryReadMixin,
    RepositoryInsightTopologyReadMixin,
    RepositoryRawMixin,
    RepositoryWriteMixin,
    RepositoryVectorMixin,
):
    """Async repository for session storage operations.

    Wraps SQLiteBackend to provide high-level async storage interface with
    full feature parity to sync SessionRepository.

    All methods are async. Eager loading (session_from_records) is used
    for fetching sessions, enabling efficient parallel I/O via asyncio.gather()
    for sessions, messages, and attachments.

    Write safety is provided by SQLite's ``BEGIN IMMEDIATE`` transactions
    in the backend layer, combined with asyncio.Lock() serialization.

    Example:
        async with SessionRepository() as repo:
            conv = await repo.get("claude-ai:abc123")
            convs = await repo.list(limit=10)
            await repo.save_parsed_session(parsed_session, content_hash)
    """

    def __init__(
        self,
        backend: SQLiteBackend | None = None,
        db_path: Path | None = None,
        archive_root: Path | None = None,
    ) -> None:
        """Initialize async storage repository.

        Args:
            backend: Optional SQLiteBackend instance. If provided, db_path is ignored.
            db_path: Optional path to database file. Used if backend is None.
        """
        active_backend = backend if backend is not None else SQLiteBackend(db_path=db_path)
        self._backend: SQLiteBackend = active_backend
        self._archive_root: Path | None = archive_root
        self._source_backend: SQLiteBackend | None = (
            SQLiteBackend(db_path=archive_root / "source.db") if archive_root is not None else None
        )
        self.queries = active_backend.queries

    async def __aenter__(self) -> SessionRepository:
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
    def source_backend(self) -> SQLiteBackend | None:
        """Access the durable source-tier backend when this repository owns one."""
        return self._source_backend

    async def close(self) -> None:
        """Close database connections and release resources."""
        await self._backend.close()
        if self._source_backend is not None:
            await self._source_backend.close()


__all__ = ["SessionRepository"]

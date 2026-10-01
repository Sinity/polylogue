"""sqlite-vec vector search provider implementation."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

from polylogue.paths import embeddings_db_path
from polylogue.storage.search_providers.sqlite_vec_embeddings import SqliteVecEmbeddingMixin
from polylogue.storage.search_providers.sqlite_vec_queries import SqliteVecQueryMixin
from polylogue.storage.search_providers.sqlite_vec_runtime import SqliteVecRuntimeMixin, _vector_snapshot_index_binding
from polylogue.storage.search_providers.sqlite_vec_support import (
    BATCH_SIZE,
    DEFAULT_DIMENSION,
    DEFAULT_MODEL,
    SqliteVecError,
    _serialize_f32,
)


class SqliteVecProvider(
    SqliteVecRuntimeMixin,
    SqliteVecEmbeddingMixin,
    SqliteVecQueryMixin,
):
    """VectorProvider implementation using sqlite-vec + Voyage AI embeddings."""

    def __init__(
        self,
        voyage_key: str | None,
        db_path: Path | None = None,
        model: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        archive_root: Path | None = None,
        snapshot_connection: sqlite3.Connection | None = None,
    ) -> None:
        if snapshot_connection is not None:
            # This provider is an operation-scoped reader. The archive owner
            # opened and pinned the handle; consume its recorded index proof
            # without certifying it by a later pathname or reopening it.
            self.db_path = Path("embeddings.db")
            self.archive_root = None
            self._snapshot_connection = snapshot_connection
            self._snapshot_thread_id = threading.get_ident()
            self._snapshot_index_path, self._snapshot_index_identity = _vector_snapshot_index_binding(
                snapshot_connection
            )
            self.voyage_key = voyage_key
            self.model = model
            self.dimension = dimension
            self._vec_available = True
            self._tables_ensured = True
            return
        self.db_path = (db_path or embeddings_db_path()).absolute()
        self.archive_root = archive_root.absolute() if archive_root is not None else None

        self.voyage_key = voyage_key
        self.model = model
        self.dimension = dimension
        self._vec_available: bool | None = None
        self._tables_ensured: bool = False
        self._snapshot_connection: sqlite3.Connection | None = None

    async def read_session_similarity(
        self,
        session_id: str,
        *,
        index_path: Path,
        project: Callable[[sqlite3.Connection, int, list[tuple[str, float]]], dict[str, object]],
        limit: int = 10,
    ) -> dict[str, object]:
        """Count and rank retained vectors in one operation-owned snapshot.

        The caller selects the same backend generation used to hydrate hits.
        Owned handles are acquired, queried and closed in one worker. Supplied
        handles and their projection stay on the creating thread.
        """
        if self._snapshot_connection is not None:
            if threading.get_ident() != self._snapshot_thread_id:
                raise SqliteVecError("operation vector snapshot must be read on its creating thread")
            return self._read_session_similarity(session_id, index_path=index_path, project=project, limit=limit)
        return await asyncio.to_thread(
            self._read_session_similarity, session_id, index_path=index_path, project=project, limit=limit
        )

    def _read_session_similarity(
        self,
        session_id: str,
        *,
        index_path: Path,
        project: Callable[[sqlite3.Connection, int, list[tuple[str, float]]], dict[str, object]],
        limit: int,
    ) -> dict[str, object]:
        with self._lifecycle_admission():
            connection = self._get_read_connection(index_path=index_path)
            try:
                reader = (
                    self
                    if connection is self._snapshot_connection
                    else self.from_vector_read_snapshot(
                        voyage_key=self.voyage_key, connection=connection, model=self.model, dimension=self.dimension
                    )
                )
                count = reader.count_session_embeddings(session_id)
                hits = reader.query_by_session(session_id, limit=limit) if count else []
                return project(connection, count, hits)
            finally:
                self._release_connection(connection)

    @classmethod
    def from_vector_read_snapshot(
        cls,
        *,
        voyage_key: str | None,
        connection: sqlite3.Connection,
        model: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
    ) -> SqliteVecProvider:
        """Bind reads to a handle returned by ``open_vector_read_snapshot``.

        The archive owner retains the handle's lifetime and creating thread.
        Its selected-index proof was captured at admission; unsupported raw
        connections visibly refuse rather than receiving a fresh-stat proof.
        """

        return cls(
            voyage_key,
            model=model,
            dimension=dimension,
            snapshot_connection=connection,
        )


__all__ = [
    "BATCH_SIZE",
    "DEFAULT_DIMENSION",
    "DEFAULT_MODEL",
    "SqliteVecError",
    "SqliteVecProvider",
    "_serialize_f32",
]

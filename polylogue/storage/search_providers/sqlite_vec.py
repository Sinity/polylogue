"""sqlite-vec vector search provider implementation."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import TypeVar

from polylogue.core.compute import compute_adapter
from polylogue.core.errors import EmbeddingRetrievalNotReadyError
from polylogue.paths import embeddings_db_path
from polylogue.storage.embeddings.identity import EmbeddingRecipe
from polylogue.storage.search_providers.sqlite_vec_embeddings import SqliteVecEmbeddingMixin
from polylogue.storage.search_providers.sqlite_vec_queries import SqliteVecQueryMixin
from polylogue.storage.search_providers.sqlite_vec_runtime import (
    SqliteVecRuntimeMixin,
    _assert_vec0_dimension,
    _vector_snapshot_binding,
    require_vector_seed_session,
)
from polylogue.storage.search_providers.sqlite_vec_support import (
    BATCH_SIZE,
    DEFAULT_DIMENSION,
    DEFAULT_MODEL,
    SqliteVecError,
    _serialize_f32,
)

_SimilarityT = TypeVar("_SimilarityT")


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
        query_recipe: EmbeddingRecipe | None = None,
    ) -> None:
        document_recipe = EmbeddingRecipe.current(model=model, dimensions=dimension)
        selected_query = query_recipe or EmbeddingRecipe.current(model=model, dimensions=dimension, input_type="query")
        if selected_query.input_type != "query" or not document_recipe.retrieval_compatible(selected_query):
            raise SqliteVecError("query and document recipes do not declare compatible retrieval contracts")
        self._query_recipe = query_recipe
        if snapshot_connection is not None:
            # This provider is an operation-scoped reader. The archive owner
            # opened and pinned the handle; consume its recorded index proof
            # without certifying it by a later pathname or reopening it.
            self.db_path = Path("embeddings.db")
            self.archive_root = None
            self._snapshot_connection = snapshot_connection
            self._snapshot_index_path, self._snapshot_index_identity, self._snapshot_thread_id = (
                _vector_snapshot_binding(snapshot_connection)
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

    @property
    def document_recipe(self) -> EmbeddingRecipe:
        """The actual document producer's current declared request contract."""
        return EmbeddingRecipe.current(model=self.model, dimensions=self.dimension)

    @property
    def query_recipe(self) -> EmbeddingRecipe:
        """An explicit query selection, or the current document model with query role."""
        return self._query_recipe or EmbeddingRecipe.current(
            model=self.model, dimensions=self.dimension, input_type="query"
        )

    async def read_similarity(
        self,
        *,
        index_path: Path,
        project: Callable[[sqlite3.Connection, int, list[tuple[str, float]]], _SimilarityT],
        text: str | None = None,
        seed_session_id: str | None = None,
        limit: int = 10,
    ) -> _SimilarityT:
        """Rank and hydrate a session page in one selected read snapshot."""
        if (text is None) == (seed_session_id is None):
            raise ValueError("similarity read requires exactly one seed")
        if self._snapshot_connection is not None:
            if threading.get_ident() != self._snapshot_thread_id:
                raise SqliteVecError("operation vector snapshot must be read on its creating thread")
            return self._read_similarity(
                index_path=index_path, project=project, text=text, seed_session_id=seed_session_id, limit=limit
            )
        return (
            await compute_adapter()
            .submit(
                partial(
                    self._read_similarity,
                    index_path=index_path,
                    project=project,
                    text=text,
                    seed_session_id=seed_session_id,
                    limit=limit,
                ),
                admission_class="interactive-read",
                estimated_bytes=len((text if text is not None else seed_session_id or "").encode("utf-8")),
            )
            .wait()
        )

    def _read_similarity(
        self,
        *,
        index_path: Path,
        project: Callable[[sqlite3.Connection, int, list[tuple[str, float]]], _SimilarityT],
        text: str | None,
        seed_session_id: str | None,
        limit: int,
    ) -> _SimilarityT:
        with self._lifecycle_admission():
            connection = self._get_read_connection(index_path=index_path)
            try:
                reader = (
                    self
                    if connection is self._snapshot_connection
                    else self.from_vector_read_snapshot(
                        voyage_key=self.voyage_key,
                        connection=connection,
                        model=self.model,
                        dimension=self.dimension,
                        query_recipe=self._query_recipe,
                    )
                )
                count = 0
                with closing(connection.cursor()) as cursor:
                    if seed_session_id is not None:
                        require_vector_seed_session(connection, seed_session_id)
                        count = reader.count_session_embeddings(seed_session_id)
                    hits: list[tuple[str, float]] = []
                    parameters: tuple[object, ...]
                    if limit > 0 and (seed_session_id is None or count):
                        if seed_session_id is not None:
                            _assert_vec0_dimension(connection, self.dimension)
                            self._require_seed(connection, seed_session_id)
                            parameters = (seed_session_id, seed_session_id, seed_session_id, limit)
                        else:
                            assert text is not None
                            if not self.voyage_key:
                                raise EmbeddingRetrievalNotReadyError(
                                    "text retrieval requires embedding acquisition credentials",
                                    readiness_status="disabled",
                                )
                            query_vector = self._query_vector(connection, text)
                            if query_vector is None:
                                return project(connection, count, [])
                            parameters = (query_vector, limit)
                        cursor.execute(
                            self._distance_sql(session_grain=True, session_seed=seed_session_id is not None)
                            + " LIMIT ?",
                            parameters,
                        )
                        hits = [(str(row["message_id"]), float(row["distance"])) for row in cursor]
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
        query_recipe: EmbeddingRecipe | None = None,
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
            query_recipe=query_recipe,
        )


__all__ = [
    "BATCH_SIZE",
    "DEFAULT_DIMENSION",
    "DEFAULT_MODEL",
    "SqliteVecError",
    "SqliteVecProvider",
    "_serialize_f32",
]

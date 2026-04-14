"""sqlite-vec vector search provider implementation."""

from __future__ import annotations

from pathlib import Path

from polylogue.paths import db_path as archive_db_path
from polylogue.storage.search_providers.sqlite_vec_embeddings import SqliteVecEmbeddingMixin
from polylogue.storage.search_providers.sqlite_vec_queries import SqliteVecQueryMixin
from polylogue.storage.search_providers.sqlite_vec_runtime import SqliteVecRuntimeMixin
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
        voyage_key: str,
        db_path: Path | None = None,
        model: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
    ) -> None:
        self.db_path = db_path or archive_db_path()
        self.voyage_key = voyage_key
        self.model = model
        self.dimension = dimension
        self._vec_available: bool | None = None
        self._tables_ensured: bool = False


__all__ = [
    "BATCH_SIZE",
    "DEFAULT_DIMENSION",
    "DEFAULT_MODEL",
    "SqliteVecError",
    "SqliteVecProvider",
    "_serialize_f32",
]

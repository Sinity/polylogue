"""sqlite-vec vector search provider implementation."""

from __future__ import annotations

from pathlib import Path

from polylogue.paths import embeddings_db_path
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
        archive_root: Path | None = None,
    ) -> None:
        explicit_path = db_path is not None
        self.db_path = (db_path or embeddings_db_path()).absolute()
        self.archive_root = archive_root.absolute() if archive_root is not None else None
        # Direct construction is retained for storage-level test fixtures; the
        # public factory is the security boundary for managed routes.
        self._legacy_compatibility = self.archive_root is None and explicit_path

        self.voyage_key = voyage_key
        self.model = model
        self.dimension = dimension
        self._vec_available: bool | None = None
        self._tables_ensured: bool = False


class _LegacySqliteVecProvider(SqliteVecProvider):
    """Private compatibility adapter for pre-split arbitrary test fixtures."""

    def __init__(
        self,
        voyage_key: str,
        db_path: Path,
        model: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
    ) -> None:
        self.db_path = db_path.absolute()
        self.archive_root = None
        self._legacy_compatibility = True
        self.voyage_key = voyage_key
        self.model = model
        self.dimension = dimension
        self._vec_available: bool | None = None
        self._tables_ensured = False


__all__ = [
    "BATCH_SIZE",
    "DEFAULT_DIMENSION",
    "DEFAULT_MODEL",
    "SqliteVecError",
    "SqliteVecProvider",
    "_serialize_f32",
]

"""Storage backend implementations for Polylogue.

Currently provides SQLiteBackend as the primary synchronous backend,
and AsyncSQLiteBackend as the async-first backend used by the pipeline.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.storage.backends.async_sqlite import AsyncSQLiteBackend
from polylogue.storage.backends.sqlite import SQLiteBackend


def create_backend(db_path: Path | None = None) -> SQLiteBackend:
    """Create a SQLite storage backend.

    Args:
        db_path: Optional path to database file. If None, uses default path.

    Returns:
        SQLiteBackend instance
    """
    return SQLiteBackend(db_path=db_path)


__all__ = [
    "create_backend",
    "SQLiteBackend",
    "AsyncSQLiteBackend",
]

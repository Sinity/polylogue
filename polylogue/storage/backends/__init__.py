"""Storage backend implementations for Polylogue.

This package provides different storage backend implementations that can be
used interchangeably via the StorageBackend protocol. Each backend handles
database-specific operations while maintaining a consistent interface.

Available backends:
- SQLiteBackend: SQLite-based storage (default)
- PostgreSQLBackend: PostgreSQL storage (future)
- DuckDBBackend: DuckDB storage (future)

The create_backend() factory function provides a convenient way to instantiate
the appropriate backend based on configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.storage.backends.sqlite import SQLiteBackend

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.protocols import StorageBackend


def create_backend(config: Config | None = None, db_path: Path | None = None) -> StorageBackend:
    """Create a storage backend based on configuration.

    This factory function instantiates the appropriate storage backend.
    Currently only SQLite is supported, but this provides the extension
    point for adding PostgreSQL, DuckDB, or other backends in the future.

    Args:
        config: Optional configuration object. If provided, backend selection
                will be based on config.backend_type (when implemented).
        db_path: Optional path to database file. Takes precedence over config.
                 If None, uses config.db_path or default path.

    Returns:
        StorageBackend instance (currently always SQLiteBackend)

    Example:
        # Use default SQLite backend with default path
        backend = create_backend()

        # Use specific database path
        backend = create_backend(db_path=Path("/path/to/db.sqlite"))

        # Use configuration (future: might select different backend types)
        from polylogue.config import load_config
        config = load_config()
        backend = create_backend(config)
    """
    # Future: check config.backend_type when we support multiple backends
    # if config and hasattr(config, 'backend_type'):
    #     if config.backend_type == 'postgresql':
    #         return PostgreSQLBackend(config)
    #     elif config.backend_type == 'duckdb':
    #         return DuckDBBackend(config)

    # For now, always return SQLite backend
    if db_path is None and config is not None and hasattr(config, "db_path"):
        db_path = config.db_path

    return SQLiteBackend(db_path=db_path)


__all__ = [
    "create_backend",
    "SQLiteBackend",
]

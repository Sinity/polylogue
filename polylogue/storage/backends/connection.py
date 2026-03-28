"""SQLite connection management and database utilities.

Provides thread-local connection caching for ``connection_context()``,
the ``_load_sqlite_vec()`` helper, and factory functions for the
default backend.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

import polylogue.paths as _paths
from polylogue.lib.log import get_logger
from polylogue.storage.backends.schema import _ensure_schema

LOGGER = get_logger(__name__)


def _load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Attempt to load sqlite-vec extension.

    Returns True if loaded successfully, False otherwise.
    The extension is optional - vector search is simply unavailable without it.
    Silent on failure since this is called on every connection.

    Note: enable_load_extension(True) is required before loading native SQLite
    extensions. We re-disable it after loading for security (prevents untrusted
    SQL from loading arbitrary extensions).
    """
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
            return True
        finally:
            conn.enable_load_extension(False)
    except ImportError:
        return False
    except Exception as exc:
        LOGGER.warning("sqlite-vec extension load failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Thread-local connection cache
# ---------------------------------------------------------------------------

_connection_cache: threading.local = threading.local()


def _get_cached_connection(path: Path) -> sqlite3.Connection:
    """Return a thread-local cached connection for the given path.

    Creates a new connection on first access per (thread, path) pair.
    Connections are configured with WAL, foreign keys, busy_timeout,
    sqlite-vec, and schema migrations — all exactly once per connection.
    """
    cache: dict[str, sqlite3.Connection] = getattr(_connection_cache, "conns", {})
    if not hasattr(_connection_cache, "conns"):
        _connection_cache.conns = cache

    key = str(path)
    if key in cache:
        return cache[key]

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    _load_sqlite_vec(conn)
    _ensure_schema(conn)

    cache[key] = conn
    return conn


def _clear_connection_cache() -> None:
    """Close all cached connections and clear the thread-local cache.

    This must be called before moving or deleting database files that
    may have open cached connections — otherwise SQLite WAL sidecar
    files (.db-wal, .db-shm) won't be checkpointed and the moved
    file will be corrupted.

    Also useful in test teardown to ensure test isolation.
    """
    cache: dict[str, sqlite3.Connection] = getattr(_connection_cache, "conns", {})
    for conn in cache.values():
        with suppress(Exception):
            conn.close()
    _connection_cache.conns = {}


@contextmanager
def connection_context(db_path: Path | str | sqlite3.Connection | None = None) -> Iterator[sqlite3.Connection]:
    """Context manager for thread-local, reusable sqlite3 connections.

    Connections are cached per (thread, db_path) pair, so repeated calls
    within the same thread reuse the same connection instead of opening
    and closing one each time.

    Args:
        db_path: Path to the database file, or an existing connection.
                 If None, uses default path.

    Yields:
        An open sqlite3.Connection with Row factory and WAL mode enabled.
        sqlite-vec extension is loaded if available.
    """
    if isinstance(db_path, sqlite3.Connection):
        _load_sqlite_vec(db_path)
        yield db_path
        return

    path = Path(db_path) if db_path else default_db_path()
    yield _get_cached_connection(path)


# Alias for backward compatibility
open_connection = connection_context


def default_db_path() -> Path:
    """Return the default database path.

    Uses XDG_DATA_HOME/polylogue/polylogue.db (semantic data, not ephemeral state).
    Reads from polylogue.paths at call time (not import time) so that
    tests can reload the paths module with monkeypatched XDG_DATA_HOME.
    """
    return _paths.data_home() / "polylogue.db"


def create_default_backend() -> object:
    """Create a SQLiteBackend with the default database path.

    This is a convenience function for creating backends when
    no custom path is needed.

    Returns:
        SQLiteBackend connected to the default database location
    """
    # Late import to avoid circular dependency
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

    return SQLiteBackend(db_path=None)


__all__ = [
    "connection_context",
    "open_connection",
    "_load_sqlite_vec",
    "_get_cached_connection",
    "_clear_connection_cache",
    "default_db_path",
    "create_default_backend",
]

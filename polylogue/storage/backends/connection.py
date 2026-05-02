"""SQLite connection management and database utilities."""

from __future__ import annotations

import atexit
import os
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

import polylogue.paths as _paths
from polylogue.logging import get_logger
from polylogue.storage.backends.connection_profile import (
    DB_TIMEOUT,
    READ_CACHE_SIZE_KIB,
    READ_CONNECTION_PRAGMA_STATEMENTS,
    READ_DB_TIMEOUT,
    READ_MMAP_SIZE_BYTES,
    WAL_AUTOCHECKPOINT_PAGES,
    WRITE_CACHE_SIZE_KIB,
    WRITE_CONNECTION_PRAGMA_STATEMENTS,
    WRITE_MMAP_SIZE_BYTES,
)
from polylogue.storage.backends.schema import _ensure_schema, assert_readable_archive_layout
from polylogue.storage.backends.sqlite_vec_extension import try_load_sqlite_vec

if TYPE_CHECKING:
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

logger = get_logger(__name__)


def _apply_pragma_statements(conn: sqlite3.Connection, statements: Sequence[str]) -> None:
    for statement in statements:
        conn.execute(statement)


def _load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Attempt to load sqlite-vec extension.

    Returns True if loaded successfully, False otherwise.
    The extension is optional - vector search is simply unavailable without it.
    Silent on failure since this is called on every connection.

    Note: enable_load_extension(True) is required before loading native SQLite
    extensions. We re-disable it after loading for security (prevents untrusted
    SQL from loading arbitrary extensions).
    """
    loaded, error = try_load_sqlite_vec(conn)
    if loaded:
        return True
    if isinstance(error, ImportError):
        return False
    if error is not None:
        logger.warning("sqlite-vec extension load failed: %s", error)
    return False


def _configure_read_connection(conn: sqlite3.Connection) -> None:
    """Apply read-safe settings without taking write-oriented locks."""
    conn.row_factory = sqlite3.Row
    _apply_pragma_statements(conn, READ_CONNECTION_PRAGMA_STATEMENTS)


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
    conn = sqlite3.connect(path, timeout=DB_TIMEOUT)
    os.chmod(path, 0o600)
    conn.row_factory = sqlite3.Row
    _apply_pragma_statements(conn, WRITE_CONNECTION_PRAGMA_STATEMENTS)
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


atexit.register(_clear_connection_cache)


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

    path = Path(db_path) if db_path else _paths.db_path()
    yield _get_cached_connection(path)


open_connection = connection_context


@contextmanager
def open_read_connection(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """Open a short-lived read-only connection when the DB already exists.

    This avoids writer-style setup (`journal_mode`, schema ensure) for read
    paths that should remain responsive while another process is bulk-ingesting.
    If the database does not exist yet, fall back to the normal connection path
    so first-run callers still get a usable empty archive.
    """
    path = Path(db_path) if db_path else _paths.db_path()
    if not path.exists():
        with open_connection(path) as conn:
            yield conn
        return

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=READ_DB_TIMEOUT)
    _configure_read_connection(conn)
    try:
        assert_readable_archive_layout(conn)
        yield conn
    finally:
        conn.close()


def create_default_backend() -> SQLiteBackend:
    """Create a SQLiteBackend with the default database path.

    This is a convenience function for creating backends when
    no custom path is needed.

    Returns:
        SQLiteBackend connected to the default database location
    """
    # Late import to avoid circular dependency
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

    return SQLiteBackend(db_path=None)


def _build_scope_filter(
    names: Sequence[str] | None,
    *,
    column: str,
) -> tuple[str, list[str]]:
    """Build a simple IN predicate for one scoped column."""
    if names is None:
        return "", []
    if not names:
        return "0", []

    placeholders = ",".join("?" for _ in names)
    return f"{column} IN ({placeholders})", list(names)


def _build_source_scope_filter(
    names: Sequence[str] | None,
    *,
    source_column: str = "source_name",
) -> tuple[str, list[str]]:
    """Build a source-name predicate. Source scoping is no longer conflated with providers."""
    return _build_scope_filter(names, column=source_column)


def _build_provider_scope_filter(
    names: Sequence[str] | None,
    *,
    provider_column: str = "provider_name",
) -> tuple[str, list[str]]:
    """Build a provider-name predicate."""
    return _build_scope_filter(names, column=provider_column)


__all__ = [
    "DB_TIMEOUT",
    "READ_CACHE_SIZE_KIB",
    "READ_CONNECTION_PRAGMA_STATEMENTS",
    "READ_MMAP_SIZE_BYTES",
    "READ_DB_TIMEOUT",
    "WAL_AUTOCHECKPOINT_PAGES",
    "WRITE_CACHE_SIZE_KIB",
    "WRITE_CONNECTION_PRAGMA_STATEMENTS",
    "WRITE_MMAP_SIZE_BYTES",
    "_build_provider_scope_filter",
    "_build_scope_filter",
    "_build_source_scope_filter",
    "connection_context",
    "create_default_backend",
    "open_connection",
    "open_read_connection",
]

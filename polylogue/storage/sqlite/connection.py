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
from polylogue.storage.sqlite.connection_profile import (
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
from polylogue.storage.sqlite.schema import _ensure_schema, assert_readable_archive_layout
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

if TYPE_CHECKING:
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

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


_SIBLING_TIER_ATTACHMENTS: tuple[tuple[str, str], ...] = (
    ("source_tier", "source.db"),
    ("user_tier", "user.db"),
    ("embeddings", "embeddings.db"),
    ("ops_tier", "ops.db"),
)


def _attach_sibling_tiers(conn: sqlite3.Connection) -> None:
    """Attach sibling archive tiers to an ``index.db`` connection (idempotent).

    Mirrors the async backend so cross-tier reads (e.g. ``raw_sessions`` in
    ``source.db``) resolve with unqualified table names. SQLite resolves
    unqualified names to ``main`` first, so index-tier tables are unaffected.
    """
    main_path: str | None = None
    attached: set[str] = set()
    for row in conn.execute("PRAGMA database_list").fetchall():
        schema_name = str(row[1])
        if schema_name == "main":
            main_path = str(row[2]) if row[2] else None
        else:
            attached.add(schema_name)
    if not main_path:
        return
    main = Path(main_path)
    if main.name != "index.db":
        return
    root = main.parent
    for schema_name, filename in _SIBLING_TIER_ATTACHMENTS:
        if schema_name in attached:
            continue
        sibling = root / filename
        if sibling.exists():
            conn.execute(f"ATTACH DATABASE ? AS {schema_name}", (str(sibling),))


def _configure_read_connection(conn: sqlite3.Connection) -> None:
    """Apply read-safe settings without taking write-oriented locks."""
    conn.row_factory = sqlite3.Row
    _apply_pragma_statements(conn, READ_CONNECTION_PRAGMA_STATEMENTS)
    _attach_sibling_tiers(conn)


# ---------------------------------------------------------------------------
# Thread-local connection cache
# ---------------------------------------------------------------------------

_connection_cache: threading.local = threading.local()
_schema_lock_guard = threading.Lock()
_schema_locks: dict[str, threading.Lock] = {}


def _schema_lock_for_path(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _schema_lock_guard:
        lock = _schema_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _schema_locks[key] = lock
        return lock


def _is_initialized_archive_index(path: Path) -> bool:
    if path.name != "index.db":
        return False
    root = path.parent
    return all((root / filename).exists() for filename in ("source.db", "index.db", "user.db", "ops.db"))


def _get_cached_connection(path: Path) -> sqlite3.Connection:
    """Return a thread-local cached connection for the given path.

    Creates a new connection on first access per (thread, path) pair.
    Connections are configured with WAL, foreign keys, busy_timeout,
    sqlite-vec, and connection-local runtime setup — all exactly once per connection.
    """
    cache: dict[str, sqlite3.Connection] = getattr(_connection_cache, "conns", {})
    if not hasattr(_connection_cache, "conns"):
        _connection_cache.conns = cache

    key = str(path)
    if key in cache:
        return cache[key]

    if path.name == "index.db" and not _is_initialized_archive_index(path):
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

        initialize_active_archive_root(path.parent)

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=DB_TIMEOUT)
    try:
        os.chmod(path, 0o600)
        conn.row_factory = sqlite3.Row
        _apply_pragma_statements(conn, WRITE_CONNECTION_PRAGMA_STATEMENTS)
        _load_sqlite_vec(conn)
        _attach_sibling_tiers(conn)
        with _schema_lock_for_path(path):
            if path.name == "index.db" and not _is_initialized_archive_index(path):
                raise RuntimeError(f"Archive root was not initialized for {path}")
            if path.name != "index.db":
                _ensure_schema(conn)
    except BaseException:
        # Pragma/schema setup can fail (e.g. locked database). Close the
        # just-opened connection before propagating so it is neither cached
        # nor orphaned.
        conn.close()
        raise

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
        if not _is_initialized_archive_index(path):
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
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

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
    provider_column: str = "source_name",
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

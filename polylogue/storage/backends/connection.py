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

logger = get_logger(__name__)

# Default SQLite connection timeout in seconds.  Used for both sync and async
# connections across the storage layer to prevent indefinite blocking when the
# database is locked by another writer.
DB_TIMEOUT = 30


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
        logger.warning("sqlite-vec extension load failed: %s", exc)
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
    conn = sqlite3.connect(path, timeout=DB_TIMEOUT)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout = {DB_TIMEOUT * 1000}")
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


def _iso_to_epoch(iso_str: str) -> float:
    """Convert an ISO date string to epoch seconds for SQL comparison."""
    from datetime import datetime

    try:
        return datetime.fromisoformat(iso_str).timestamp()
    except (ValueError, TypeError):
        try:
            return float(iso_str)
        except (ValueError, TypeError):
            return 0.0


def _build_conversation_filters(
    *,
    source: str | None = None,
    provider: str | None = None,
    providers: list[str] | None = None,
    parent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    title_contains: str | None = None,
    has_tool_use: bool = False,
    has_thinking: bool = False,
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
    has_file_ops: bool = False,
    has_git_ops: bool = False,
    has_subagent: bool = False,
) -> tuple[str, list[str | int | float]]:
    """Build WHERE clause and params for conversation queries.

    Stats-based filters (has_tool_use, has_thinking, min_messages, max_messages,
    min_words) emit a LEFT JOIN on conversation_stats and filter on cs columns.
    Semantic filters (has_file_ops, has_git_ops, has_subagent) emit EXISTS
    subqueries against content_blocks.semantic_type.
    Callers must prefix conversation columns with 'c.' when using stats filters.
    """
    where_clauses: list[str] = []
    params: list[str | int | float] = []

    if source is not None:
        where_clauses.append("source_name = ?")
        params.append(source)
    if provider is not None:
        where_clauses.append("provider_name = ?")
        params.append(provider)
    if providers:
        placeholders = ",".join("?" for _ in providers)
        where_clauses.append(
            f"(provider_name IN ({placeholders}) OR source_name IN ({placeholders}))"
        )
        params.extend(providers)
        params.extend(providers)
    if parent_id is not None:
        where_clauses.append("parent_conversation_id = ?")
        params.append(parent_id)
    if since is not None:
        where_clauses.append("sort_key >= ?")
        params.append(_iso_to_epoch(since))
    if until is not None:
        where_clauses.append("sort_key <= ?")
        params.append(_iso_to_epoch(until))
    if title_contains is not None:
        escaped = title_contains.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        where_clauses.append("title LIKE ? ESCAPE '\\'")
        params.append(f"%{escaped}%")

    # Stats-based filters (require conversation_stats JOIN)
    if has_tool_use:
        where_clauses.append("cs.tool_use_count > 0")
    if has_thinking:
        where_clauses.append("cs.thinking_count > 0")
    if min_messages is not None:
        where_clauses.append("cs.message_count >= ?")
        params.append(min_messages)
    if max_messages is not None:
        where_clauses.append("cs.message_count <= ?")
        params.append(max_messages)
    if min_words is not None:
        where_clauses.append("cs.word_count >= ?")
        params.append(min_words)

    # Semantic filters via EXISTS subquery on content_blocks.semantic_type
    conv_id_col = "c.conversation_id" if needs_stats_join else "conversation_id"
    if has_file_ops:
        where_clauses.append(
            f"EXISTS (SELECT 1 FROM content_blocks cb WHERE cb.conversation_id = {conv_id_col}"
            " AND cb.semantic_type IN ('file_read', 'file_write', 'file_edit'))"
        )
    if has_git_ops:
        where_clauses.append(
            f"EXISTS (SELECT 1 FROM content_blocks cb WHERE cb.conversation_id = {conv_id_col}"
            " AND cb.semantic_type = 'git')"
        )
    if has_subagent:
        where_clauses.append(
            f"EXISTS (SELECT 1 FROM content_blocks cb WHERE cb.conversation_id = {conv_id_col}"
            " AND cb.semantic_type = 'subagent')"
        )

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    return where_sql, params


__all__ = [
    "DB_TIMEOUT",
    "connection_context",
    "create_default_backend",
    "default_db_path",
    "open_connection",
]

"""SQLite connection management and database utilities."""

from __future__ import annotations

import atexit
import os
import sqlite3
import threading
from builtins import BaseExceptionGroup
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import polylogue.paths as _paths
from polylogue.core.sql_settlement import current_native_sql_lifetimes
from polylogue.logging import get_logger
from polylogue.storage.fts.pl_fold import register_pl_fold
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.connection_profile import (
    DB_TIMEOUT,
    READ_CACHE_SIZE_KIB,
    READ_CONNECTION_PRAGMA_STATEMENTS,
    READ_DB_TIMEOUT,
    READ_MMAP_SIZE_BYTES,
    WAL_AUTOCHECKPOINT_PAGES,
    WRITE_CACHE_SIZE_KIB,
    WRITE_CONNECTION_PROFILE,
    WRITE_MMAP_SIZE_BYTES,
    NativeSQLCustodyOwner,
    _attach_sibling_tiers,
    _close_failed_native_construction,
    configured_archive_root,
    execute_pragma_statement,
    open_readonly_connection,
    write_connection_pragma_statements,
)
from polylogue.storage.sqlite.schema import _ensure_schema, assert_readable_archive_layout
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec
from polylogue.storage.sqlite.write_lease import require_write_lease, write_lease

logger = get_logger(__name__)


def _apply_pragma_statements(conn: sqlite3.Connection, statements: Sequence[str]) -> None:
    for statement in statements:
        execute_pragma_statement(conn, statement)


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


def _configure_read_connection(conn: sqlite3.Connection, *, archive_root: Path) -> None:
    """Apply read-safe settings without taking write-oriented locks."""
    # The profiled reader already applied its pragmas before installing the
    # read authorizer, which denies re-assigning them here.
    conn.row_factory = sqlite3.Row
    _attach_sibling_tiers(conn, archive_root=archive_root)
    register_pl_fold(conn)


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


def _is_initialized_archive_index(path: Path, *, archive_root: Path | None = None) -> bool:
    if path.name != "index.db":
        return False
    root = archive_root if archive_root is not None else path.parent
    return all((root / filename).exists() for filename in ("source.db", "index.db", "user.db", "ops.db"))


def _get_cached_connection(path: Path, *, archive_root: Path) -> sqlite3.Connection:
    """Return the current admitted operation's cached connection for a path.

    Creates a new connection on first access per (operation, thread, path) pair.
    Connections are configured with WAL, foreign keys, busy_timeout,
    sqlite-vec, and connection-local runtime setup — all exactly once per connection.
    """
    cache: dict[str, NativeSQLCustodyOwner] = getattr(_connection_cache, "conns", {})
    from polylogue.storage.sqlite.population_admission import assert_population_admitted

    assert_population_admitted(path)
    if not hasattr(_connection_cache, "conns"):
        _connection_cache.conns = cache

    key = str(path)
    require_write_lease(f"cached write connection({path})", archive_root=archive_root)
    if key in cache:
        return cache[key].admit_cached_reuse()

    if path.name == "index.db":
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

        # File presence is not lineage admission: historical four-file roots
        # must pass the format-marker gate before any tier is opened.
        initialize_active_archive_root(archive_root)

    path.parent.mkdir(parents=True, exist_ok=True)
    require_write_lease(f"cached write connection({path})", archive_root=archive_root)
    conn = connect_measured(path, uri=True, timeout=DB_TIMEOUT)
    owner = NativeSQLCustodyOwner(conn, cache_entry=(cache, key))
    cache[key] = owner
    try:
        os.chmod(path, 0o600)
        conn.row_factory = sqlite3.Row
        _apply_pragma_statements(conn, write_connection_pragma_statements(WRITE_CONNECTION_PROFILE))
        _load_sqlite_vec(conn)
        _attach_sibling_tiers(conn, archive_root=archive_root)
        register_pl_fold(conn)
        with _schema_lock_for_path(path):
            if path.name == "index.db" and not _is_initialized_archive_index(path, archive_root=archive_root):
                raise RuntimeError(f"Archive root was not initialized for {path}")
            if path.name != "index.db":
                _ensure_schema(conn)
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise

    return conn


def _clear_connection_cache() -> None:
    """Close all cached connections and clear the thread-local cache.

    This must be called before moving or deleting database files that
    may have open cached connections — otherwise SQLite WAL sidecar
    files (.db-wal, .db-shm) won't be checkpointed and the moved
    file will be corrupted.

    Also useful in test teardown to ensure test isolation.
    """
    cache: dict[str, NativeSQLCustodyOwner] = getattr(_connection_cache, "conns", {})
    failures: list[BaseException] = []
    for owner in tuple(cache.values()):
        try:
            owner.close()
        except BaseException as error:
            failures.append(error)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("Cached connection cleanup failed", failures)


atexit.register(_clear_connection_cache)


@contextmanager
def connection_context(
    db_path: Path | str | sqlite3.Connection | None = None,
    *,
    archive_root: Path | None = None,
) -> Iterator[sqlite3.Connection]:
    """Reuse native connections only inside one admitted archive operation.

    A standalone context owns one physical write lease. Nested contexts reuse
    the current operation's connection, which outer settlement actually closes.
    A caller-owned transaction must settle before that outer operation ends.

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
    root = configured_archive_root(path, archive_root)
    require_write_lease(f"cached connection({path})", archive_root=root)
    if not path.parent.exists():
        if path.name == "index.db":
            from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

            initialize_active_archive_root(root)
        else:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with write_lease(f"cached connection({path})", archive_root=root):
        connection = _get_cached_connection(path, archive_root=root)
        owner = _connection_cache.conns[str(path)]
        try:
            yield connection
        except BaseException as primary:
            _close_failed_native_construction(owner, primary)
            raise


open_connection = connection_context


@contextmanager
def open_read_connection(
    db_path: Path | str | None = None,
    *,
    archive_root: Path | None = None,
) -> Iterator[sqlite3.Connection]:
    """Open a short-lived read-only connection when the DB already exists.

    This avoids writer-style setup (`journal_mode`, schema ensure) for read
    paths that should remain responsive while another process is bulk-ingesting.
    If the database does not exist yet, fall back to the normal connection path
    so first-run callers still get a usable empty archive.
    """
    path = Path(db_path) if db_path else _paths.db_path()
    if not path.exists():
        with open_connection(path, archive_root=archive_root) as conn:
            yield conn
        return

    conn = open_readonly_connection(path, timeout_class="interactive-read", validate_schema=False)
    owner = NativeSQLCustodyOwner(conn, lifetime_dependencies=current_native_sql_lifetimes())
    try:
        _configure_read_connection(conn, archive_root=configured_archive_root(path, archive_root))
        if not _is_initialized_archive_index(path, archive_root=configured_archive_root(path, archive_root)):
            assert_readable_archive_layout(conn)
        yield conn
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    else:
        owner.close()


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


def _build_source_path_scope_filter(
    source_paths: Sequence[str] | None,
) -> tuple[str, list[str]]:
    """Build a path-prefix predicate scoping raw rows to configured source roots.

    Configured inbox/source roots are filesystem paths, not origins, so raw
    selection must scope on the ``source_path`` column rather than ``origin``.
    Each root matches an exact ``source_path`` or any descendant beneath it.

    polylogue-gzxhi: the boundary is a literal, byte-exact path prefix, not a
    LIKE pattern. ``LIKE`` folds ASCII case, so a root ``/archive/Foo`` would
    also claim the distinct sibling ``/archive/foo``; its ``%``/``_``
    metacharacters would likewise let ``/archive/100%`` claim
    ``/archive/1000``. The half-open range ``[root + "/", root + "0")`` --
    ``"0"`` is the byte after ``"/"`` -- selects exactly the root's descendants
    under the default BINARY collation and stays index-usable.
    """
    if source_paths is None:
        return "", []
    if not source_paths:
        return "0", []

    predicates: list[str] = []
    params: list[str] = []
    for path in source_paths:
        root = path.rstrip("/") or path
        predicates.append("(source_path = ? OR (source_path >= ? AND source_path < ?))")
        params.extend([path, f"{root}/", f"{root}0"])
    return "(" + " OR ".join(predicates) + ")", params


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
    "write_connection_pragma_statements",
    "WRITE_MMAP_SIZE_BYTES",
    "_build_provider_scope_filter",
    "_build_scope_filter",
    "_build_source_path_scope_filter",
    "_build_source_scope_filter",
    "connection_context",
    "open_connection",
    "open_read_connection",
]

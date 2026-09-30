"""Async SQLite storage backend implementation using aiosqlite.

This backend provides async/await API for all database operations, enabling
concurrent queries and parallel processing without blocking.

Performance characteristics:
- Parallel reads: 5-10x faster for batch operations
- Write serialization: Still uses exclusive locks (SQLite limitation)
- Connection pooling: Each async context gets its own connection
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import aiosqlite

import polylogue.paths as _paths
from polylogue.core.errors import DatabaseError
from polylogue.storage.fts.pl_fold import pl_fold
from polylogue.storage.runtime import (
    SessionProfileRecord,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.async_sqlite_archive import SQLiteArchiveMixin
from polylogue.storage.sqlite.async_sqlite_raw import SQLiteRawMixin
from polylogue.storage.sqlite.connection_profile import (
    DB_TIMEOUT,
    READ_CONNECTION_PRAGMA_STATEMENTS,
    READ_DB_TIMEOUT,
    WRITE_CONNECTION_PROFILE,
    _authorize_read_operation,
    write_connection_pragma_statements,
)
from polylogue.storage.sqlite.population_admission import assert_population_admitted
from polylogue.storage.sqlite.queries import (
    session_insight_profile_writes as session_insight_profiles_q,
)
from polylogue.storage.sqlite.query_store import SQLiteQueryStore
from polylogue.storage.sqlite.schema import SCHEMA_DDL, ensure_schema_async
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec_async
from polylogue.storage.sqlite.write_lease import current_write_lease, require_write_lease, write_lease_enforced


async def _apply_pragma_statements_async(conn: aiosqlite.Connection, statements: tuple[str, ...]) -> None:
    for statement in statements:
        await conn.execute(statement)


# Sibling archive tiers attached to an ``index.db`` connection so that
# cross-tier reads (e.g. ``raw_sessions`` in ``source.db`` joined against
# ``sessions`` in ``index.db``) resolve with unqualified table names. The
# write path for each tier still uses its own dedicated connection; this
# only makes the index connection able to *read* sibling tables. SQLite
# resolves unqualified names to ``main`` first, so index-tier tables are
# unaffected; only tables that live solely in a sibling tier resolve to it.
_SIBLING_TIER_ATTACHMENTS: tuple[tuple[str, str], ...] = (
    ("source_tier", "source.db"),
    ("user_tier", "user.db"),
    ("embeddings", "embeddings.db"),
    ("ops_tier", "ops.db"),
)
_SIBLING_ARCHIVE_TIERS = {
    "source_tier": ArchiveTier.SOURCE,
    "user_tier": ArchiveTier.USER,
    "embeddings": ArchiveTier.EMBEDDINGS,
    "ops_tier": ArchiveTier.OPS,
}


async def _attach_sibling_tiers(conn: aiosqlite.Connection, *, read_only: bool = False) -> None:
    """Attach sibling archive tiers to an ``index.db`` connection (idempotent).

    A reader attaches each sibling through a ``mode=ro`` URI, so the read
    profile's authorizer and ``query_only`` are not the only barrier between
    a reader and a writable durable tier.
    """
    cursor = await conn.execute("PRAGMA database_list")
    rows = list(await cursor.fetchall())
    main_path: str | None = None
    attached: set[str] = set()
    for row in rows:
        schema_name = str(row[1])
        if schema_name == "main":
            main_path = str(row[2]) if row[2] else None
        else:
            attached.add(schema_name)
    if not main_path:
        return
    from pathlib import Path as _Path

    from polylogue.core.errors import SchemaSkew
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.schema_identity import DerivedTier, derived_schema_identity

    main = _Path(main_path)
    assert_population_admitted(main)
    if main.name != "index.db":
        return
    root = main.parent
    for schema_name, filename in _SIBLING_TIER_ATTACHMENTS:
        if schema_name in attached:
            continue
        sibling = root / filename
        assert_population_admitted(sibling)
        if sibling.exists():
            if schema_name == "embeddings":
                # ``message_embeddings`` is a vec0 virtual table: without the
                # extension every read of it fails with "no such module: vec0"
                # and embedding coverage reads as unmeasurable. Load it before
                # the attach and before a reader's authorizer is installed.
                # A failed load leaves that honest unmeasurable outcome.
                await try_load_sqlite_vec_async(conn)
            target = f"file:{quote(str(sibling))}?mode=ro" if read_only else str(sibling)
            await conn.execute(f"ATTACH DATABASE ? AS {schema_name}", (target,))
            tier = _SIBLING_ARCHIVE_TIERS[schema_name]
            cursor = await conn.execute(f"PRAGMA {schema_name}.user_version")
            version_row = await cursor.fetchone()
            found = int(version_row[0]) if version_row is not None else 0
            expected = ARCHIVE_VERSION_BY_TIER[tier]
            if found != expected:
                raise SchemaSkew(tier.value, expected, found)
            # Only the derived tiers carry a stamped schema identity
            # (``DerivedTier``). ``embeddings.db`` is repurchased, never
            # replayed, and has no identity row; its version check above is
            # the whole contract.
            if tier is ArchiveTier.OPS:
                identity_cursor = await conn.execute(
                    f"SELECT identity FROM {schema_name}.schema_identity WHERE tier = ?", (tier.value,)
                )
                identity_row = await identity_cursor.fetchone()
                identity = str(identity_row[0]) if identity_row is not None else None
                expected_identity = derived_schema_identity(DerivedTier(tier.value))
                if identity != expected_identity:
                    raise SchemaSkew(tier.value, expected_identity, identity)


async def _assert_connection_population_admitted(conn: aiosqlite.Connection) -> None:
    async with conn.execute("PRAGMA database_list") as cursor:
        for row in await cursor.fetchall():
            if row[2]:
                assert_population_admitted(row[2])


async def configure_connection(conn: aiosqlite.Connection) -> None:
    """Apply canonical connection settings.

    Performance pragmas (cache_size, synchronous, mmap_size) are critical
    for large databases. With a 28 GB DB and 2 MB default cache, every
    operation thrashes disk. These settings bring throughput from ~0.5/s
    to expected levels.
    """
    await _assert_connection_population_admitted(conn)
    conn.row_factory = aiosqlite.Row
    await _apply_pragma_statements_async(conn, write_connection_pragma_statements(WRITE_CONNECTION_PROFILE))
    await _attach_sibling_tiers(conn)
    await conn.create_function("pl_fold", 1, pl_fold, deterministic=True)


async def configure_read_connection(conn: aiosqlite.Connection) -> None:
    """Apply read-safe settings without mutating database-wide state."""
    await _assert_connection_population_admitted(conn)
    conn.row_factory = aiosqlite.Row
    await _apply_pragma_statements_async(conn, READ_CONNECTION_PRAGMA_STATEMENTS)
    await _attach_sibling_tiers(conn, read_only=True)
    await conn.create_function("pl_fold", 1, pl_fold, deterministic=True)
    # The same DB-boundary authorizer as the synchronous read profile: a
    # reader cannot re-enable writes, attach a writable file or mutate schema.
    await conn.set_authorizer(_authorize_read_operation)


@dataclass(frozen=True, slots=True)
class _ConnectionCloseResult:
    actual_closed: bool
    error: BaseException | None
    cancellation: asyncio.CancelledError | None


@dataclass(frozen=True, slots=True)
class _BackendConnectionOwner:
    backend: SQLiteBackend
    connection: aiosqlite.Connection
    thread: threading.Thread
    task: asyncio.Task[object] | None
    pid: int


# Strong custody survives loss of the caller after a failed raw close.
_BACKEND_CONNECTIONS: dict[int, _BackendConnectionOwner] = {}
_BACKEND_CONNECTIONS_LOCK = threading.Lock()


async def _settled_connection_operation(
    awaitable: object,
) -> tuple[BaseException | None, asyncio.CancelledError | None]:
    task = asyncio.ensure_future(awaitable)  # type: ignore[arg-type]
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
        except BaseException:
            break
    try:
        task.result()
        return None, cancellation
    except BaseException as exc:
        return exc, cancellation


async def _settle_connection_close(conn: aiosqlite.Connection, *, rollback: bool) -> _ConnectionCloseResult:
    """Retain the raw handle and worker until native close has actually settled."""
    error = None
    cancellation = None
    if rollback and conn._connection is not None:
        error, cancellation = await _settled_connection_operation(conn.rollback())
    if conn._connection is not None:

        def close_raw() -> None:
            conn._conn.close()
            conn._connection = None

        close_error, close_cancellation = await _settled_connection_operation(conn._execute(close_raw))
        error = error or close_error
        cancellation = cancellation or close_cancellation
    actual_closed = conn._connection is None
    if actual_closed and conn._thread.is_alive():
        try:
            stopped = conn.stop()
            stop_error = None
            if stopped is not None:
                stop_error, stop_cancellation = await _settled_connection_operation(stopped)
                error = error or stop_error
                cancellation = cancellation or stop_cancellation
            if stop_error is None:
                # aiosqlite schedules the stop future before breaking its worker
                # loop. After that sentinel succeeds the worker needs no further
                # event-loop response; join proves its actual exit.
                conn._thread.join()
        except BaseException as stop_error:
            error = error or stop_error
    return _ConnectionCloseResult(actual_closed, error, cancellation)


async def _close_backend_connection(conn: aiosqlite.Connection, *, rollback: bool = False) -> None:
    result = await _settle_connection_close(conn, rollback=rollback)
    if result.actual_closed and not conn._thread.is_alive():
        with _BACKEND_CONNECTIONS_LOCK:
            owner = _BACKEND_CONNECTIONS.pop(id(conn), None)
        if owner is not None:
            if owner.backend._txn_conn is conn:
                owner.backend._txn_conn = None
                owner.backend._transaction_depth = 0
            if owner.backend._bulk_conn is conn:
                owner.backend._bulk_conn = None
                owner.backend._transaction_depth = 0
    if result.error is not None:
        if result.cancellation is not None:
            result.error.add_note("caller cancellation also occurred during connection cleanup")
        raise result.error
    if result.cancellation is not None:
        raise result.cancellation


async def _cleanup_backend_connections(connections: list[aiosqlite.Connection], primary: BaseException | None) -> None:
    first_error = None
    for conn in connections:
        try:
            await _close_backend_connection(conn, rollback=True)
        except BaseException as exc:
            if primary is not None:
                primary.add_note(f"connection cleanup also failed: {exc}")
            first_error = first_error or exc
    if primary is None and first_error is not None:
        raise first_error


async def _open_configured_backend_connection(
    backend: SQLiteBackend, *, read_only: bool = False
) -> aiosqlite.Connection:
    assert_population_admitted(backend._db_path)
    target = backend._db_path.absolute().as_uri() + "?mode=ro" if read_only else backend._db_path
    conn = aiosqlite.connect(target, uri=read_only, timeout=READ_DB_TIMEOUT if read_only else DB_TIMEOUT)
    with _BACKEND_CONNECTIONS_LOCK:
        _BACKEND_CONNECTIONS[id(conn)] = _BackendConnectionOwner(
            backend, conn, threading.current_thread(), asyncio.current_task(), os.getpid()
        )
    try:
        error, cancellation = await _settled_connection_operation(conn)
        if error is not None:
            raise error
        if cancellation is not None:
            raise cancellation
        await (configure_read_connection if read_only else configure_connection)(conn)
        return conn
    except BaseException as primary:
        await _cleanup_backend_connections([conn], primary)
        raise


@asynccontextmanager
async def _scoped_backend_connection(
    backend: SQLiteBackend, *, read_only: bool = False
) -> AsyncIterator[aiosqlite.Connection]:
    conn = await _open_configured_backend_connection(backend, read_only=read_only)
    primary = None
    try:
        yield conn
    except BaseException as exc:
        primary = exc
        raise
    finally:
        await _cleanup_backend_connections([conn], primary)


async def _read_schema_ready(backend: SQLiteBackend) -> bool:
    """Check whether an existing database already has the archive schema."""
    assert_population_admitted(backend._db_path)
    if not backend._db_path.exists():
        return False

    async with _scoped_backend_connection(backend, read_only=True) as conn:
        cursor = await conn.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        if not row or row[0] <= 0:
            return False

        cursor = await conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'")
        return await cursor.fetchone() is not None


def _is_initialized_archive_index(path: Path) -> bool:
    if path.name != "index.db":
        return False
    root = path.parent
    return all((root / filename).exists() for filename in ("source.db", "index.db", "user.db", "ops.db"))


# ---------------------------------------------------------------------------
# Runtime/state helpers (formerly async_sqlite_runtime.py)
# ---------------------------------------------------------------------------


def initialize_backend_state(backend: SQLiteBackend, db_path: Path | None) -> None:
    """Initialize backend state and shared query accessors."""
    requested_path = Path(db_path) if db_path is not None else _paths.db_path()
    assert_population_admitted(requested_path)
    archive_root = requested_path.parent
    if archive_root.name == ".index-generations":
        archive_root = archive_root.parent
    elif archive_root.parent.name == ".index-generations":
        archive_root = archive_root.parent.parent
    backend._db_path = requested_path if requested_path.name == "index.db" else archive_root / "index.db"
    backend._source_db_path = archive_root / "source.db"
    needs_bootstrap = not _is_initialized_archive_index(backend._db_path)
    if needs_bootstrap:
        # Bootstrap itself creates writable tier files. Enforce archive
        # ownership before any directory or database mutation in constructor.
        require_write_lease(f"async backend bootstrap({backend._db_path})", archive_root=archive_root)
    # Existing tier files do not prove format lineage; admit the root through
    # its marker before constructing sync or async connections. A caller with
    # write authority admits it through the active-root bootstrap, which may
    # also settle pending bootstrap state. A daemon-armed caller without the
    # lease (the live batch probing ``Polylogue.backend``) may not write, so
    # it is admitted read-only: the format-marker proof plus a validating
    # read-only open of the index, which raises ``SchemaSkew`` for an index
    # this runtime cannot serve, and no filesystem mutation at all.
    if needs_bootstrap or current_write_lease() is not None or not write_lease_enforced():
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

        backend._db_path.parent.mkdir(parents=True, exist_ok=True)
        initialize_active_archive_root(archive_root)
        if backend._db_path.exists():
            backend._db_path.chmod(0o600)
    else:
        from polylogue.storage.sqlite.archive_tiers.archive_plan import assert_archive_format_lineage
        from polylogue.storage.sqlite.connection_profile import open_readonly_connection

        assert_archive_format_lineage(archive_root)
        open_readonly_connection(backend._db_path, tier=ArchiveTier.INDEX).close()

    backend._write_lock = asyncio.Lock()
    backend._schema_lock = asyncio.Lock()
    backend._schema_ensured = False
    backend._transaction_depth = 0
    backend._txn_conn = None
    backend._bulk_conn = None
    backend._read_pool = None

    backend.queries = SQLiteQueryStore(connection_factory=backend._get_read_connection)

    # Keep the shared DDL visibly anchored in the async backend module family.
    backend._shared_schema_ddl = SCHEMA_DDL


async def ensure_schema_once(backend: SQLiteBackend) -> None:
    """Ensure schema initialization runs exactly once."""
    assert_population_admitted(backend._db_path)
    if backend._schema_ensured:
        return
    async with backend._schema_lock:
        if backend._schema_ensured:
            return
        if _is_initialized_archive_index(backend._db_path):
            backend._schema_ensured = True
            return
        require_write_lease(
            f"async schema initialization({backend._db_path})", archive_root=backend._source_db_path.parent
        )
        async with aiosqlite.connect(backend._db_path, timeout=DB_TIMEOUT) as init_conn:
            os.chmod(backend._db_path, 0o600)
            await configure_connection(init_conn)
            await backend._ensure_schema(init_conn)
        backend._schema_ensured = True


# ---------------------------------------------------------------------------
# Transaction helpers (formerly async_sqlite_transactions.py)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _backend_transaction(backend: SQLiteBackend) -> AsyncIterator[None]:
    """Context manager for database transactions.

    When a bulk_connection is active, acts as a nested savepoint within
    the bulk transaction instead of trying to open a new connection.
    """
    assert_population_admitted(backend._db_path)
    if backend._bulk_conn is not None:
        # Inside bulk_connection: use savepoint on the bulk connection
        sp_name = f"sp_bulk_{backend._transaction_depth}"
        backend._transaction_depth += 1
        try:
            await backend._bulk_conn.execute(f"SAVEPOINT {sp_name}")
            yield
            await backend._bulk_conn.execute(f"RELEASE SAVEPOINT {sp_name}")
        except BaseException:
            with suppress(Exception):
                await backend._bulk_conn.execute(f"ROLLBACK TO SAVEPOINT {sp_name}")
            raise
        finally:
            backend._transaction_depth -= 1
        return

    async with backend._write_lock:
        if backend._txn_conn is None:
            require_write_lease(f"async transaction({backend._db_path})", archive_root=backend._source_db_path.parent)
            backend._txn_conn = await _open_configured_backend_connection(backend)

        await _backend_begin(backend)
        try:
            yield
            await _backend_commit(backend)
        except BaseException as primary:
            try:
                await _backend_rollback(backend)
            except BaseException as cleanup_error:
                primary.add_note(f"transaction rollback also failed: {cleanup_error}")
            raise


async def _backend_begin(backend: SQLiteBackend) -> None:
    """Begin a transaction or nested savepoint."""
    await backend._ensure_schema_once()
    if backend._txn_conn is None:
        require_write_lease(f"async transaction begin({backend._db_path})", archive_root=backend._source_db_path.parent)
        backend._txn_conn = await _open_configured_backend_connection(backend)

    if backend._transaction_depth == 0:
        await backend._txn_conn.execute("BEGIN IMMEDIATE")
    else:
        await backend._txn_conn.execute(f"SAVEPOINT sp_{backend._transaction_depth}")
    backend._transaction_depth += 1


async def _backend_commit(backend: SQLiteBackend) -> None:
    """Commit the current transaction or release savepoint."""
    if backend._transaction_depth <= 0:
        raise DatabaseError("No active transaction to commit")
    if backend._txn_conn is None:
        raise DatabaseError("No transaction connection")

    backend._transaction_depth -= 1

    if backend._transaction_depth == 0:
        await backend._txn_conn.commit()
        await _close_backend_connection(backend._txn_conn)
        backend._txn_conn = None
    else:
        await backend._txn_conn.execute(f"RELEASE SAVEPOINT sp_{backend._transaction_depth}")


async def _backend_rollback(backend: SQLiteBackend) -> None:
    """Rollback to the last begin() or savepoint."""
    if backend._transaction_depth <= 0:
        raise DatabaseError("No active transaction to rollback")
    if backend._txn_conn is None:
        raise DatabaseError("No transaction connection")

    backend._transaction_depth -= 1

    if backend._transaction_depth == 0:
        await _close_backend_connection(backend._txn_conn, rollback=True)
        backend._txn_conn = None
    else:
        await backend._txn_conn.execute(f"ROLLBACK TO SAVEPOINT sp_{backend._transaction_depth}")


async def _close_backend(backend: SQLiteBackend) -> None:
    """Attempt retirement of every retained handle, including failed admission."""
    with _BACKEND_CONNECTIONS_LOCK:
        connections = [entry.connection for entry in _BACKEND_CONNECTIONS.values() if entry.backend is backend]
    if backend._txn_conn is not None and backend._txn_conn not in connections:
        connections.append(backend._txn_conn)
    try:
        await _cleanup_backend_connections(connections, None)
    finally:
        if backend._txn_conn is not None and backend._txn_conn._connection is None:
            backend._txn_conn = None
            backend._transaction_depth = 0


# ---------------------------------------------------------------------------
# Connection lifecycle helpers (formerly async_sqlite_connections.py)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _backend_connection(backend: SQLiteBackend) -> AsyncIterator[aiosqlite.Connection]:
    """Public connection context for read/query helpers.

    When a bulk_connection is active, reuses it instead of opening a new
    connection — avoids "database is locked" errors from competing for
    the write lock.
    """
    assert_population_admitted(backend._db_path)
    if backend._bulk_conn is not None:
        yield backend._bulk_conn
    else:
        async with backend._get_connection() as conn:
            yield conn


@asynccontextmanager
async def _bulk_connection(backend: SQLiteBackend) -> AsyncIterator[None]:
    """Keep a single connection alive for many sequential operations."""
    await backend._ensure_schema_once()
    require_write_lease(f"async bulk transaction({backend._db_path})", archive_root=backend._source_db_path.parent)
    conn = await _open_configured_backend_connection(backend)
    began = False
    primary = None
    try:
        await conn.execute("BEGIN IMMEDIATE")
        backend._bulk_conn = conn
        backend._transaction_depth += 1
        began = True
        yield
        await conn.commit()
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if began:
            backend._transaction_depth -= 1
            backend._bulk_conn = None
        await _cleanup_backend_connections([conn], primary)


@asynccontextmanager
async def _read_pool(backend: SQLiteBackend, size: int = 4) -> AsyncIterator[None]:
    """Open a pool of reusable read connections for concurrent operations."""
    await backend._ensure_schema_once()
    pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue()
    connections: list[aiosqlite.Connection] = []
    primary = None
    try:
        for _ in range(size):
            conn = await _open_configured_backend_connection(backend, read_only=True)
            connections.append(conn)
            pool.put_nowait(conn)
        backend._read_pool = pool
        yield
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if backend._read_pool is pool:
            backend._read_pool = None
        await _cleanup_backend_connections(connections, primary)


@asynccontextmanager
async def _get_connection(backend: SQLiteBackend) -> AsyncIterator[aiosqlite.Connection]:
    """Get async database connection with schema ensured."""
    await backend._ensure_schema_once()

    if backend._txn_conn is not None and backend._transaction_depth > 0:
        yield backend._txn_conn
        return

    if backend._bulk_conn is not None:
        yield backend._bulk_conn
        return

    if backend._read_pool is not None:
        pool = backend._read_pool
        conn = await pool.get()
        try:
            yield conn
        finally:
            if backend._read_pool is pool:
                pool.put_nowait(conn)
        return

    # This fallback is the writable connection used when no transaction or
    # bulk handle is active.  Keep it behind the same archive-bound lease as
    # the explicit transaction factories; otherwise an async caller can open
    # a second writer while the daemon coordinator is holding the gate.
    require_write_lease(f"async connection({backend._db_path})", archive_root=backend._source_db_path.parent)
    async with _scoped_backend_connection(backend) as conn:
        os.chmod(backend._db_path, 0o600)
        yield conn


@asynccontextmanager
async def _get_read_connection(backend: SQLiteBackend) -> AsyncIterator[aiosqlite.Connection]:
    """Get a read-oriented connection that stays responsive during bulk writes."""
    assert_population_admitted(backend._db_path)
    if not backend._schema_ensured:
        if await _read_schema_ready(backend):
            backend._schema_ensured = True
        else:
            await backend._ensure_schema_once()

    if backend._txn_conn is not None and backend._transaction_depth > 0:
        yield backend._txn_conn
        return

    if backend._bulk_conn is not None:
        yield backend._bulk_conn
        return

    if backend._read_pool is not None:
        pool = backend._read_pool
        conn = await pool.get()
        try:
            yield conn
        finally:
            if backend._read_pool is pool:
                pool.put_nowait(conn)
        return

    async with _scoped_backend_connection(backend, read_only=True) as conn:
        yield conn


class SQLiteBackend(
    SQLiteArchiveMixin,
    SQLiteRawMixin,
):
    """Async SQLite storage backend implementation.

    This backend provides async/await API for database operations, enabling
    true concurrency for read operations while maintaining write safety.
    """

    _db_path: Path
    _source_db_path: Path
    _write_lock: asyncio.Lock
    _schema_lock: asyncio.Lock
    _schema_ensured: bool
    _transaction_depth: int
    _txn_conn: aiosqlite.Connection | None
    _bulk_conn: aiosqlite.Connection | None
    _read_pool: asyncio.Queue[aiosqlite.Connection] | None
    queries: SQLiteQueryStore
    _shared_schema_ddl: str

    def __init__(self, db_path: Path | None = None) -> None:
        initialize_backend_state(self, db_path)

    @property
    def db_path(self) -> Path:
        """Return the backing SQLite database path."""
        return self._db_path

    @property
    def transaction_depth(self) -> int:
        """Return the active transaction nesting depth."""
        return self._transaction_depth

    # -- Connection lifecycle -----------------------------------------------

    def connection(self) -> AbstractAsyncContextManager[aiosqlite.Connection]:
        """Public connection context for read/query helpers."""
        return _backend_connection(self)

    def read_connection(self) -> AbstractAsyncContextManager[aiosqlite.Connection]:
        """Public read-oriented connection context for query/report helpers."""
        return _get_read_connection(self)

    async def _ensure_schema_once(self) -> None:
        """Ensure schema is initialized exactly once (thread-safe via asyncio lock)."""
        await ensure_schema_once(self)

    def bulk_connection(self) -> AbstractAsyncContextManager[None]:
        """Keep a single connection alive for many sequential operations."""
        return _bulk_connection(self)

    def read_pool(self, size: int = 4) -> AbstractAsyncContextManager[None]:
        """Open a pool of reusable read connections for concurrent operations."""
        return _read_pool(self, size=size)

    def _get_connection(self) -> AbstractAsyncContextManager[aiosqlite.Connection]:
        """Get async database connection with schema ensured."""
        return _get_connection(self)

    def _get_read_connection(self) -> AbstractAsyncContextManager[aiosqlite.Connection]:
        """Get async read connection with read-only semantics when possible."""
        return _get_read_connection(self)

    async def _ensure_schema(self, conn: aiosqlite.Connection) -> None:
        """Ensure database schema exists and is at the current schema version."""
        await ensure_schema_async(conn)

    # -- Transaction management ---------------------------------------------

    def transaction(self) -> AbstractAsyncContextManager[None]:
        """Context manager for database transactions."""
        return _backend_transaction(self)

    async def begin(self) -> None:
        """Begin a transaction or nested savepoint."""
        await _backend_begin(self)

    async def commit(self) -> None:
        """Commit the current transaction or release savepoint."""
        await _backend_commit(self)

    async def rollback(self) -> None:
        """Rollback to the last begin() or savepoint."""
        await _backend_rollback(self)

    async def close(self) -> None:
        """Close database connections."""
        await _close_backend(self)

    # -- Derived insights (formerly SQLiteDerivedInsightsMixin) --------------

    async def replace_session_profile(
        self,
        record: SessionProfileRecord,
    ) -> None:
        """Replace one durable session-profile row."""
        async with self._get_connection() as conn:
            await session_insight_profiles_q.replace_session_profile(
                conn,
                record,
                self._transaction_depth,
            )


__all__ = [
    "SCHEMA_DDL",
    "SQLiteBackend",
    "configure_connection",
]

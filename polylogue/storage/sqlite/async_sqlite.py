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
import sqlite3
import threading
from builtins import BaseExceptionGroup
from collections.abc import AsyncIterator, Awaitable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import quote

import aiosqlite

import polylogue.paths as _paths
from polylogue.core.errors import DatabaseError
from polylogue.core.sql_settlement import SQLSettlementRetry
from polylogue.storage.fts.pl_fold import pl_fold
from polylogue.storage.io_phase_metrics import connect_measured
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
    configured_archive_root,
    write_connection_pragma_statements,
)
from polylogue.storage.sqlite.population_admission import assert_population_admitted
from polylogue.storage.sqlite.queries import (
    session_insight_profile_writes as session_insight_profiles_q,
)
from polylogue.storage.sqlite.query_store import SQLiteQueryStore
from polylogue.storage.sqlite.schema import SCHEMA_DDL, ensure_schema_async
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec_async
from polylogue.storage.sqlite.write_lease import (
    UnleasedWriteError,
    WriteLease,
    WriteLeaseThreadGrant,
    _close_async_archive_custody,
    _settle_task,
    async_write_lease,
    bind_write_lease_thread,
    current_write_lease,
    grant_write_lease_thread,
    require_write_lease,
)


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
    grant: WriteLeaseThreadGrant | None = None
    cleanup_task: asyncio.Task[object] | None = None
    cleanup_attempt: asyncio.Future[None] | None = None


# Strong custody survives loss of the caller after a failed raw close.
_BACKEND_CONNECTIONS: dict[int, _BackendConnectionOwner] = {}
_BACKEND_CONNECTIONS_LOCK = threading.RLock()

_FORK_ABANDONED_BACKEND_CONNECTIONS: list[_BackendConnectionOwner] = []


def _before_backend_connection_fork() -> None:
    _BACKEND_CONNECTIONS_LOCK.acquire()


def _after_backend_connection_fork_parent() -> None:
    _BACKEND_CONNECTIONS_LOCK.release()


def _abandon_backend_connections_after_fork() -> None:
    global _BACKEND_CONNECTIONS, _BACKEND_CONNECTIONS_LOCK
    # No inherited mutex or SQLite call is safe here. Keep copied handles
    # unreachable for product use without running their finalizers in this
    # hook; fresh child writers receive a fresh metadata registry and lock.
    _FORK_ABANDONED_BACKEND_CONNECTIONS.extend(_BACKEND_CONNECTIONS.values())
    _BACKEND_CONNECTIONS = {}
    _BACKEND_CONNECTIONS_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_before_backend_connection_fork,
        after_in_parent=_after_backend_connection_fork_parent,
        after_in_child=_abandon_backend_connections_after_fork,
    )


async def _settled_connection_operation(
    awaitable: Awaitable[object],
) -> tuple[BaseException | None, asyncio.CancelledError | None]:
    task = asyncio.ensure_future(awaitable)
    cancellation = None
    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
        except BaseException:
            break
    try:
        task.result()
        return None, cancellation
    except BaseException as exc:
        return exc, cancellation


async def _clear_connection_progress_guard(conn: aiosqlite.Connection) -> None:
    """Clear interruption on the actual worker before rollback or native close."""
    raw = conn._connection
    if raw is not None:
        # aiosqlite's public annotation omits SQLite's documented None callback.
        await conn._execute(raw.set_progress_handler, None, 0)  # type: ignore[no-untyped-call]


async def _settle_connection_close(conn: aiosqlite.Connection, *, rollback: bool) -> _ConnectionCloseResult:
    """Retain the raw handle and worker until native close has actually settled."""
    errors: list[BaseException] = []
    cancellation = None
    if conn._connection is not None:
        error, cancellation = await _settled_connection_operation(_clear_connection_progress_guard(conn))
        if error is not None:
            errors.append(error)
    if rollback and conn._connection is not None:
        rollback_error, rollback_cancellation = await _settled_connection_operation(conn.rollback())
        if rollback_error is not None:
            errors.append(rollback_error)
        cancellation = cancellation or rollback_cancellation
    if conn._connection is not None:

        def close_raw() -> None:
            conn._conn.close()
            conn._connection = None

        close_error, close_cancellation = await _settled_connection_operation(
            conn._execute(close_raw)  # type: ignore[no-untyped-call]
        )
        if close_error is not None:
            errors.append(close_error)
        cancellation = cancellation or close_cancellation
    actual_closed = conn._connection is None
    if actual_closed and conn._thread.is_alive():
        try:
            # Failed aiosqlite connection construction already queued its stop
            # sentinel. A second sentinel has no worker left to settle it.
            stopped = conn.stop() if conn._running else None
            stop_error = None
            if stopped is not None:
                stop_error, stop_cancellation = await _settled_connection_operation(stopped)
                if stop_error is not None:
                    errors.append(stop_error)
                cancellation = cancellation or stop_cancellation
            if stop_error is None:
                # aiosqlite schedules the stop future before breaking its worker
                # loop. After that sentinel succeeds the worker needs no further
                # event-loop response; join proves its actual exit.
                conn._thread.join()
        except BaseException as stop_error:
            errors.append(stop_error)
    error = (
        errors[0]
        if len(errors) == 1
        else BaseExceptionGroup("SQLite worker cleanup failed", errors)
        if errors
        else None
    )
    return _ConnectionCloseResult(actual_closed, error, cancellation)


async def _await_settled(awaitable: Awaitable[object]) -> None:
    """Propagate cancellation only after the canonical queued operation settles."""
    error, cancellation = await _settled_connection_operation(awaitable)
    if error is not None:
        if cancellation is not None:
            raise BaseExceptionGroup("SQLite work failed during cancellation", [error, cancellation])
        raise error
    if cancellation is not None:
        raise cancellation


def _new_write_connection(backend: SQLiteBackend, purpose: str) -> aiosqlite.Connection:
    """Create a writer whose actual aiosqlite thread receives one owner grant."""
    require_write_lease(purpose, archive_root=backend._source_db_path.parent)
    grant = grant_write_lease_thread()
    try:
        conn = aiosqlite.Connection(lambda: _connect_write_thread(backend, grant), iter_chunk_size=64)
    except BaseException as primary:
        try:
            grant.complete()
        except BaseException as cleanup:
            raise BaseExceptionGroup("SQLite construction and grant cleanup failed", [primary, cleanup]) from primary
        raise
    with _BACKEND_CONNECTIONS_LOCK:
        _BACKEND_CONNECTIONS[id(conn)] = _BackendConnectionOwner(
            backend, conn, threading.current_thread(), _current_async_task(), os.getpid(), grant
        )
    return conn


def _connect_write_thread(backend: SQLiteBackend, grant: WriteLeaseThreadGrant) -> sqlite3.Connection:
    bind_write_lease_thread(grant)
    return connect_measured(backend._db_path, timeout=DB_TIMEOUT)


def _retire_backend_connection_owner(conn: aiosqlite.Connection) -> None:
    with _BACKEND_CONNECTIONS_LOCK:
        entry = _BACKEND_CONNECTIONS.get(id(conn))
    if entry is None:
        return
    # Returning the last grant can expose a descriptor close failure. Keep
    # this exact backend owner reachable until that physical custody settles.
    if entry.grant is not None:
        if not entry.grant.custody_retired:
            entry.grant.complete()
        custody = entry.grant.lease.custody
        if custody is not None and custody._pending_descriptor_closes:
            raise UnleasedWriteError("async connection retains unsettled archive descriptor custody")
    with _BACKEND_CONNECTIONS_LOCK:
        _BACKEND_CONNECTIONS.pop(id(conn), None)
    backend = entry.backend
    if getattr(backend, "_txn_conn", None) is conn:
        backend._txn_conn = None
        backend._transaction_depth = 0
        backend._transaction_owner_task = None
    if getattr(backend, "_bulk_conn", None) is conn:
        backend._bulk_conn = None
        backend._transaction_depth = 0
        backend._transaction_owner_task = None


async def _close_backend_connection(conn: aiosqlite.Connection, *, rollback: bool = False) -> None:
    task = _current_async_task()
    with _BACKEND_CONNECTIONS_LOCK:
        owner = _BACKEND_CONNECTIONS.get(id(conn))
        if owner is not None:
            if (
                owner.pid != os.getpid()
                or owner.thread is not threading.current_thread()
                or not _task_can_settle(owner.task)
                or (owner.cleanup_task is not None and owner.cleanup_task is not task and not owner.cleanup_task.done())
            ):
                raise UnleasedWriteError("async SQLite cleanup belongs to its original task and thread")
            attempt: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            attempt.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            owner = replace(owner, cleanup_task=task, cleanup_attempt=attempt)
            _BACKEND_CONNECTIONS[id(conn)] = owner
    try:
        await _finish_backend_connection_close(conn, owner=owner, rollback=rollback)
    except BaseException as error:
        if owner is not None and owner.cleanup_attempt is not None:
            owner.cleanup_attempt.set_exception(error)
        raise
    else:
        if owner is not None and owner.cleanup_attempt is not None:
            owner.cleanup_attempt.set_result(None)


async def _finish_backend_connection_close(
    conn: aiosqlite.Connection, *, owner: _BackendConnectionOwner | None, rollback: bool
) -> None:
    custody = owner.grant.lease.custody if owner is not None and owner.grant is not None else None
    if custody is not None and custody.settlement_retry is None:
        custody.settlement_retry = SQLSettlementRetry()
    retry = custody.settlement_retry if custody is not None else None
    observed = retry.generation() if retry is not None else 0
    result = await _settle_connection_close(conn, rollback=rollback)
    errors = [result.error] if result.error is not None else []
    cancellation = result.cancellation
    if cancellation is not None and retry is not None:
        retry.request()
    if result.actual_closed and not conn._thread.is_alive():
        first_retirement_error: BaseException | None = None
        while True:
            try:
                _retire_backend_connection_owner(conn)
                break
            except BaseException as retirement_error:
                first_retirement_error = first_retirement_error or retirement_error
            if custody is None or retry is None:
                break
            wait = asyncio.create_task(asyncio.to_thread(retry.wait_after, observed))
            value, interrupted, wait_error = await _settle_task(wait, retry=retry)
            cancellation = cancellation or interrupted
            if wait_error is not None:
                errors.append(wait_error)
                break
            observed = int(value)
            if custody._pending_descriptor_closes:
                try:
                    await _close_async_archive_custody(custody, initial_observed_generation=observed)
                except BaseException as cleanup_error:
                    errors.append(cleanup_error)
                # The original loop task has retained custody until the same
                # owner's descriptor evidence certifies physical retirement.
                continue
            # No descriptor was released by this failed callback. The next
            # loop iteration retries only that still-held grant reference.
        if first_retirement_error is not None:
            errors.append(first_retirement_error)
    if cancellation is not None:
        errors.append(cancellation)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("Async SQLite cleanup failed", errors)


async def _close_backend_connection_preserving(
    conn: aiosqlite.Connection, *, rollback: bool, primary: BaseException
) -> None:
    try:
        await _close_backend_connection(conn, rollback=rollback)
    except BaseException as cleanup_error:
        raise BaseExceptionGroup("SQLite operation and cleanup failed", [primary, cleanup_error]) from primary


async def _cleanup_backend_connections(connections: list[aiosqlite.Connection], primary: BaseException | None) -> None:
    errors: list[BaseException] = []
    for connection in connections:
        try:
            await _close_backend_connection(connection, rollback=True)
        except BaseException as error:
            errors.append(error)
    if errors:
        if primary is not None:
            errors.insert(0, primary)
        if len(errors) == 1:
            raise errors[0]
        raise BaseExceptionGroup("SQLite operation and owned connection cleanup failed", errors)


async def _open_configured_backend_connection(
    backend: SQLiteBackend,
    *,
    read_only: bool = False,
    purpose: str = "async configured connection",
) -> aiosqlite.Connection:
    _require_backend_process(backend)
    assert_population_admitted(backend._db_path)
    if read_only:
        grant = grant_write_lease_thread() if current_write_lease() is not None else None

        def connect_reader() -> sqlite3.Connection:
            if grant is not None:
                bind_write_lease_thread(grant)
            return connect_measured(
                backend._db_path.absolute().as_uri() + "?mode=ro",
                uri=True,
                timeout=READ_DB_TIMEOUT,
            )

        try:
            connection = aiosqlite.Connection(connect_reader, iter_chunk_size=64)
        except BaseException as primary:
            if grant is not None:
                try:
                    grant.complete()
                except BaseException as cleanup:
                    raise BaseExceptionGroup(
                        "SQLite reader construction and grant cleanup failed", [primary, cleanup]
                    ) from primary
            raise
        with _BACKEND_CONNECTIONS_LOCK:
            _BACKEND_CONNECTIONS[id(connection)] = _BackendConnectionOwner(
                backend,
                connection,
                threading.current_thread(),
                _current_async_task(),
                os.getpid(),
                grant,
            )
    else:
        connection = _new_write_connection(backend, purpose)
    try:
        await _await_settled(connection)
        await _await_settled(
            (configure_read_connection if read_only else configure_connection)(
                connection, archive_root=backend._source_db_path.parent
            )
        )
        return connection
    except BaseException as primary:
        await _cleanup_backend_connections([connection], primary)
        raise


@asynccontextmanager
async def _scoped_backend_connection(
    backend: SQLiteBackend,
    *,
    read_only: bool = False,
) -> AsyncIterator[aiosqlite.Connection]:
    connection = await _open_configured_backend_connection(backend, read_only=read_only)
    primary: BaseException | None = None
    try:
        yield connection
    except BaseException as error:
        primary = error
        raise
    finally:
        await _cleanup_backend_connections([connection], primary)


@asynccontextmanager
async def _backend_write_lease(backend: SQLiteBackend, actor: str) -> AsyncIterator[None]:
    """Prepare a configured fresh root, then serialize its first archive SQL."""
    _require_transaction_reader(backend)
    assert_population_admitted(backend._db_path)
    from polylogue.core.compute_cancel import compute_cancel_requested

    if compute_cancel_requested():
        raise asyncio.CancelledError("async archive mutation cancelled before admission")
    root = backend._source_db_path.parent
    require_write_lease(f"async writer admission({actor})", archive_root=root)
    if backend._bootstrap_needed:
        root.mkdir(parents=True, exist_ok=True)
    async with async_write_lease(actor, archive_root=root):
        yield


async def _apply_pragma_statements_async(conn: aiosqlite.Connection, statements: tuple[str, ...]) -> None:
    for statement in statements:
        if statement.startswith("PRAGMA journal_mode="):
            # A mode pragma rewrites the header even when nothing changes;
            # bootstrap established the mode (see execute_pragma_statement).
            async with conn.execute("PRAGMA journal_mode") as cursor:
                row = await cursor.fetchone()
            if row is not None and str(row[0]).lower() == statement.split("=", 1)[1].strip().lower():
                continue
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


async def _attach_sibling_tiers(conn: aiosqlite.Connection, *, archive_root: Path, read_only: bool = False) -> None:
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
    root = configured_archive_root(main, archive_root)
    from polylogue.storage.sqlite.write_lease import current_sql_custody

    custody = current_sql_custody()
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
    if custody is not None:
        custody.assert_namespace()


async def _assert_connection_population_admitted(conn: aiosqlite.Connection) -> None:
    async with conn.execute("PRAGMA database_list") as cursor:
        for row in await cursor.fetchall():
            if row[2]:
                assert_population_admitted(row[2])


async def configure_connection(conn: aiosqlite.Connection, *, archive_root: Path) -> None:
    """Apply canonical connection settings.

    Performance pragmas (cache_size, synchronous, mmap_size) are critical
    for large databases. With a 28 GB DB and 2 MB default cache, every
    operation thrashes disk. These settings bring throughput from ~0.5/s
    to expected levels.
    """
    await _assert_connection_population_admitted(conn)
    conn.row_factory = aiosqlite.Row
    await _apply_pragma_statements_async(conn, write_connection_pragma_statements(WRITE_CONNECTION_PROFILE))
    await _attach_sibling_tiers(conn, archive_root=archive_root)
    await conn.create_function("pl_fold", 1, pl_fold, deterministic=True)


async def configure_read_connection(conn: aiosqlite.Connection, *, archive_root: Path) -> None:
    """Apply read-safe settings without mutating database-wide state."""
    await _assert_connection_population_admitted(conn)
    conn.row_factory = aiosqlite.Row
    await _apply_pragma_statements_async(conn, READ_CONNECTION_PRAGMA_STATEMENTS)
    await _attach_sibling_tiers(conn, archive_root=archive_root, read_only=True)
    await conn.create_function("pl_fold", 1, pl_fold, deterministic=True)
    # The same DB-boundary authorizer as the synchronous read profile: a
    # reader cannot re-enable writes, attach a writable file or mutate schema.
    await conn.set_authorizer(_authorize_read_operation)


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
    backend._owner_pid = os.getpid()
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
    # Admission/bootstrap performs writable validation even for existing
    # files; defer it until the first write scope instead of doing it from a
    # synchronous constructor that may be running on an event loop.
    backend._bootstrap_needed = needs_bootstrap
    # Backend construction is synchronous and may run inside an event loop.
    # It only observes an existing archive; bootstrap waits for the async
    # writer scope in ensure_schema_once.
    if not needs_bootstrap:
        from polylogue.storage.sqlite.archive_tiers.archive_plan import assert_archive_format_lineage
        from polylogue.storage.sqlite.connection_profile import open_readonly_connection

        assert_archive_format_lineage(archive_root)
        open_readonly_connection(backend._db_path, tier=ArchiveTier.INDEX).close()

    backend._write_lock = asyncio.Lock()
    backend._schema_lock = asyncio.Lock()
    backend._schema_ensured = False
    backend._transaction_depth = 0
    backend._transaction_owner_task = None
    backend._txn_conn = None
    backend._bulk_conn = None
    backend._read_pool = None
    backend._manual_lease_cm = None
    backend._manual_owner_task = None

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
        if _is_initialized_archive_index(backend._db_path) and not backend._bootstrap_needed:
            backend._schema_ensured = True
            return
        async with _backend_write_lease(backend, f"async schema initialization({backend._db_path})"):
            if backend._bootstrap_needed:
                from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

                initialize_active_archive_root(backend._source_db_path.parent)
                if backend._db_path.exists():
                    backend._db_path.chmod(0o600)
                backend._bootstrap_needed = False
            conn = await _open_configured_backend_connection(
                backend, purpose=f"async schema initialization({backend._db_path})"
            )
            try:
                await _await_settled(backend._ensure_schema(conn))
            except BaseException as primary:
                await _close_backend_connection_preserving(conn, rollback=True, primary=primary)
                raise
            else:
                await _close_backend_connection(conn, rollback=False)
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
    async with _backend_write_lease(backend, "async.sqlite.transaction"):
        if backend._bulk_conn is not None:
            # Inside bulk_connection: use savepoint on the bulk connection.
            sp_name = f"sp_bulk_{backend._transaction_depth}"
            depth_entered = False
            try:
                await _await_settled(backend._bulk_conn.execute(f"SAVEPOINT {sp_name}"))
                backend._transaction_depth += 1
                depth_entered = True
                yield
            except BaseException as primary:
                errors = [primary]
                try:
                    await _await_settled(_clear_connection_progress_guard(backend._bulk_conn))
                except BaseException as cleanup:
                    errors.append(cleanup)
                for statement in (f"ROLLBACK TO SAVEPOINT {sp_name}", f"RELEASE SAVEPOINT {sp_name}"):
                    try:
                        await _await_settled(backend._bulk_conn.execute(statement))
                    except BaseException as cleanup:
                        errors.append(cleanup)
                if len(errors) > 1:
                    raise BaseExceptionGroup("Bulk transaction and savepoint cleanup failed", errors) from primary
                raise
            else:
                await _await_settled(backend._bulk_conn.execute(f"RELEASE SAVEPOINT {sp_name}"))
            finally:
                if depth_entered:
                    backend._transaction_depth -= 1
            return

        async with backend._write_lock:
            try:
                await _backend_begin(backend)
                yield
                await _backend_commit(backend)
            except BaseException as primary:
                errors = [primary]
                conn = backend._txn_conn
                with _BACKEND_CONNECTIONS_LOCK:
                    entry = _BACKEND_CONNECTIONS.get(id(conn)) if conn is not None else None
                # Commit/rollback already attempted terminal cleanup on this
                # connection. Its retained owner is for an explicit retry;
                # unwinding the transaction must not attempt it a second time.
                if conn is not None and (entry is None or entry.cleanup_task is None):
                    if backend._transaction_depth > 0:
                        try:
                            await _backend_rollback(backend)
                        except BaseException as rollback_error:
                            errors.append(rollback_error)
                            with _BACKEND_CONNECTIONS_LOCK:
                                entry = _BACKEND_CONNECTIONS.get(id(conn))
                            if entry is not None and entry.cleanup_task is None:
                                try:
                                    await _close_backend_connection(conn, rollback=False)
                                except BaseException as close_error:
                                    errors.append(close_error)
                    else:
                        try:
                            await _close_backend_connection(conn, rollback=True)
                        except BaseException as close_error:
                            errors.append(close_error)
                if len(errors) > 1:
                    raise BaseExceptionGroup("Transaction and owned cleanup failed", errors) from primary
                raise


async def _backend_begin(backend: SQLiteBackend) -> None:
    """Begin a transaction or nested savepoint."""
    await backend._ensure_schema_once()
    if backend._txn_conn is None:
        backend._txn_conn = await _open_configured_backend_connection(
            backend, purpose=f"async transaction begin({backend._db_path})"
        )

    if backend._transaction_depth == 0:
        await _await_settled(backend._txn_conn.execute("BEGIN IMMEDIATE"))
        backend._transaction_owner_task = _current_async_task()
    else:
        await _await_settled(backend._txn_conn.execute(f"SAVEPOINT sp_{backend._transaction_depth}"))
    backend._transaction_depth += 1


async def _backend_commit(backend: SQLiteBackend) -> None:
    """Commit the current transaction or release savepoint."""
    if backend._transaction_depth <= 0:
        raise DatabaseError("No active transaction to commit")
    if backend._txn_conn is None:
        raise DatabaseError("No transaction connection")
    if backend._transaction_owner_task is not _current_async_task():
        raise UnleasedWriteError("async transaction commit must run in its owning task")

    if backend._transaction_depth == 0:
        raise DatabaseError("No active transaction to commit")
    if backend._transaction_depth == 1:
        conn = backend._txn_conn
        try:
            await _await_settled(conn.commit())
        except BaseException as primary:
            try:
                await _close_backend_connection(conn, rollback=True)
            except BaseException as cleanup_error:
                raise BaseExceptionGroup("Transaction commit and cleanup failed", [primary, cleanup_error]) from primary
            else:
                backend._txn_conn = None
                backend._transaction_depth = 0
                backend._transaction_owner_task = None
            raise
        await _close_backend_connection(conn, rollback=False)
        backend._txn_conn = None
        backend._transaction_depth = 0
        backend._transaction_owner_task = None
    else:
        next_depth = backend._transaction_depth - 1
        await _await_settled(backend._txn_conn.execute(f"RELEASE SAVEPOINT sp_{next_depth}"))
        backend._transaction_depth = next_depth


async def _backend_rollback(backend: SQLiteBackend) -> None:
    """Rollback to the last begin() or savepoint."""
    if backend._transaction_depth <= 0:
        raise DatabaseError("No active transaction to rollback")
    if backend._txn_conn is None:
        raise DatabaseError("No transaction connection")
    if backend._transaction_owner_task is not _current_async_task():
        raise UnleasedWriteError("async transaction rollback must run in its owning task")

    if backend._transaction_depth == 1:
        conn = backend._txn_conn
        await _close_backend_connection(conn, rollback=True)
        backend._txn_conn = None
        backend._transaction_depth = 0
        backend._transaction_owner_task = None
    else:
        next_depth = backend._transaction_depth - 1
        await _await_settled(_clear_connection_progress_guard(backend._txn_conn))
        await _await_settled(backend._txn_conn.execute(f"ROLLBACK TO SAVEPOINT sp_{next_depth}"))
        await _await_settled(backend._txn_conn.execute(f"RELEASE SAVEPOINT sp_{next_depth}"))
        backend._transaction_depth = next_depth


def retained_write_backends_on_current_thread(*, lease: WriteLease | None = None) -> tuple[SQLiteBackend, ...]:
    """Observe actual async handles admitted on this terminal writer thread."""
    with _BACKEND_CONNECTIONS_LOCK:
        backends = {
            id(entry.backend): entry.backend
            for entry in _BACKEND_CONNECTIONS.values()
            if entry.pid == os.getpid()
            and entry.thread is threading.current_thread()
            and entry.grant is not None
            and (lease is None or entry.grant.lease is lease)
        }
    return tuple(backends.values())


async def _close_backend(backend: SQLiteBackend) -> None:
    """Close database connections, retaining custody when settlement fails."""
    _require_backend_process(backend)
    with _BACKEND_CONNECTIONS_LOCK:
        pending_attempts = tuple(
            entry.cleanup_attempt
            for entry in _BACKEND_CONNECTIONS.values()
            if entry.backend is backend
            and entry.pid == os.getpid()
            and entry.thread is threading.current_thread()
            and entry.cleanup_task is not _current_async_task()
            and entry.cleanup_attempt is not None
            and not entry.cleanup_attempt.done()
        )
    if pending_attempts:
        backend.request_sql_settlement()
        attempt_errors: list[BaseException] = []
        for attempt in pending_attempts:
            try:
                await _await_settled(attempt)
            except BaseException as error:
                attempt_errors.append(error)
        if len(attempt_errors) == 1:
            raise attempt_errors[0]
        if attempt_errors:
            raise BaseExceptionGroup("Original backend cleanup attempts failed", attempt_errors)
    _require_manual_owner(backend, "close", cleanup=True)
    if backend._transaction_owner_task is not None and not _task_can_settle(backend._transaction_owner_task):
        raise UnleasedWriteError("a live async transaction task owns its cleanup")
    errors: list[BaseException] = []
    with _BACKEND_CONNECTIONS_LOCK:
        owned_connections = tuple(_BACKEND_CONNECTIONS.values())
    for entry in owned_connections:
        if entry.backend is backend and (
            entry.pid != os.getpid()
            or entry.thread is not threading.current_thread()
            or not _task_can_settle(entry.task)
        ):
            raise UnleasedWriteError("async writer cleanup belongs to the task that admitted its actual handle")
    for entry in owned_connections:
        if entry.backend is backend:
            try:
                await _close_backend_connection(entry.connection, rollback=True)
            except BaseException as exc:
                errors.append(exc)
    with _BACKEND_CONNECTIONS_LOCK:
        txn_registered = backend._txn_conn is not None and id(backend._txn_conn) in _BACKEND_CONNECTIONS
        has_unsettled = any(entry.backend is backend for entry in _BACKEND_CONNECTIONS.values())
    if backend._txn_conn is not None and not txn_registered:
        backend._txn_conn = None
    if not has_unsettled:
        backend._txn_conn = None
        backend._transaction_depth = 0
        backend._transaction_owner_task = None
    if backend._txn_conn is None:
        try:
            await _release_manual_lease(backend)
        except BaseException as exc:
            errors.append(exc)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("Backend and manual lease cleanup failed", errors)


def _current_async_task() -> asyncio.Task[object] | None:
    return asyncio.current_task()


def _require_backend_process(backend: SQLiteBackend) -> None:
    if backend._owner_pid != os.getpid():
        raise UnleasedWriteError("async SQLite backend cannot use handles inherited across fork")


def _require_transaction_reader(backend: SQLiteBackend) -> None:
    _require_backend_process(backend)
    if backend._transaction_owner_task is not None and backend._transaction_owner_task is not _current_async_task():
        raise UnleasedWriteError("an async transaction handle belongs to its exact task")


def _task_can_settle(owner: asyncio.Task[object] | None) -> bool:
    return owner is _current_async_task() or (owner is not None and owner.done())


def _require_manual_owner(backend: SQLiteBackend, action: str, *, cleanup: bool = False) -> None:
    if backend._manual_lease_cm is None:
        return
    owner = backend._manual_owner_task
    if owner is not _current_async_task() and not (cleanup and _task_can_settle(owner)):
        raise UnleasedWriteError(f"async manual transaction {action} must run in its owning task")


async def _manual_begin(backend: SQLiteBackend) -> None:
    """Keep custody across the public begin/commit/rollback transaction API."""
    if backend._transaction_depth == 0 and backend._manual_lease_cm is None and current_write_lease() is None:
        lease_cm = _backend_write_lease(backend, f"async manual transaction({backend._db_path})")
        await lease_cm.__aenter__()
        backend._manual_lease_cm = lease_cm
        backend._manual_owner_task = _current_async_task()
    _require_manual_owner(backend, "begin")
    try:
        await _backend_begin(backend)
    except BaseException as primary:
        errors = [primary]
        if backend._txn_conn is not None:
            conn = backend._txn_conn
            try:
                await _close_backend_connection(conn, rollback=True)
            except BaseException as cleanup_error:
                errors.append(cleanup_error)
            else:
                backend._txn_conn = None
                backend._transaction_depth = 0
                backend._transaction_owner_task = None
        if backend._txn_conn is None:
            try:
                await _release_manual_lease(backend)
            except BaseException as cleanup_error:
                errors.append(cleanup_error)
        if len(errors) > 1:
            raise BaseExceptionGroup("Manual transaction and cleanup failed", errors) from primary
        raise


async def _release_manual_lease(backend: SQLiteBackend) -> None:
    lease_cm = backend._manual_lease_cm
    if lease_cm is None:
        return
    backend._manual_lease_cm = None
    backend._manual_owner_task = None
    await lease_cm.__aexit__(None, None, None)  # type: ignore[attr-defined]


async def _manual_commit(backend: SQLiteBackend) -> None:
    _require_manual_owner(backend, "commit")
    primary: BaseException | None = None
    try:
        await _backend_commit(backend)
    except BaseException as error:
        primary = error
        raise
    finally:
        if backend._transaction_depth == 0 and backend._txn_conn is None:
            try:
                await _release_manual_lease(backend)
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup(
                        "Manual transaction and lease cleanup failed", [primary, cleanup]
                    ) from primary
                raise


async def _manual_rollback(backend: SQLiteBackend) -> None:
    _require_manual_owner(backend, "rollback")
    primary: BaseException | None = None
    try:
        await _backend_rollback(backend)
    except BaseException as error:
        primary = error
        raise
    finally:
        if backend._transaction_depth == 0 and backend._txn_conn is None:
            try:
                await _release_manual_lease(backend)
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup(
                        "Manual transaction and lease cleanup failed", [primary, cleanup]
                    ) from primary
                raise


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
    _require_transaction_reader(backend)
    assert_population_admitted(backend._db_path)
    if backend._bulk_conn is not None:
        yield backend._bulk_conn
    else:
        async with backend._get_connection() as conn:
            yield conn


@asynccontextmanager
async def _bulk_connection(backend: SQLiteBackend) -> AsyncIterator[None]:
    """Keep a single connection alive for many sequential operations."""
    async with _backend_write_lease(backend, "async.sqlite.bulk"):
        await backend._ensure_schema_once()
        conn = await _open_configured_backend_connection(backend, purpose=f"async bulk transaction({backend._db_path})")
        began = False
        backend._bulk_conn = None
        try:
            await _await_settled(conn.execute("BEGIN IMMEDIATE"))
            began = True
            backend._bulk_conn = conn
            backend._transaction_owner_task = _current_async_task()
            backend._transaction_depth += 1
            try:
                yield
            except BaseException as primary:
                await _close_backend_connection_preserving(conn, rollback=True, primary=primary)
                raise
            else:
                try:
                    await _await_settled(conn.commit())
                except BaseException as primary:
                    await _close_backend_connection_preserving(conn, rollback=True, primary=primary)
                    raise
        except BaseException as primary:
            if not began:
                await _close_backend_connection_preserving(conn, rollback=True, primary=primary)
            raise
        else:
            await _close_backend_connection(conn, rollback=False)
        # Successful actual close retires backend state centrally. A failed
        # close leaves this handle, depth and exact owner available for cleanup.


@asynccontextmanager
async def _read_pool(backend: SQLiteBackend, size: int = 4) -> AsyncIterator[None]:
    """Open a pool of reusable read connections for concurrent operations."""
    if not await _read_schema_ready(backend):
        raise DatabaseError(f"archive index is not initialized for read pooling: {backend._db_path}")
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
    _require_transaction_reader(backend)
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
    async with _backend_write_lease(backend, "async.sqlite.connection"):
        conn = await _open_configured_backend_connection(backend, purpose=f"async connection({backend._db_path})")
        try:
            os.chmod(backend._db_path, 0o600)
            yield conn
        except BaseException as primary:
            await _close_backend_connection_preserving(conn, rollback=True, primary=primary)
            raise
        else:
            await _close_backend_connection(conn, rollback=True)


@asynccontextmanager
async def _get_read_connection(backend: SQLiteBackend) -> AsyncIterator[aiosqlite.Connection]:
    """Get a read-oriented connection that stays responsive during bulk writes."""
    _require_transaction_reader(backend)
    assert_population_admitted(backend._db_path)
    if not backend._schema_ensured:
        if not await _read_schema_ready(backend):
            raise DatabaseError(f"archive index is not initialized for read access: {backend._db_path}")
        backend._schema_ensured = True

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

    async with _scoped_backend_connection(backend, read_only=True) as connection:
        yield connection


class SQLiteBackend(
    SQLiteArchiveMixin,
    SQLiteRawMixin,
):
    """Async SQLite storage backend implementation.

    This backend provides async/await API for database operations, enabling
    true concurrency for read operations while maintaining write safety.
    """

    _db_path: Path
    _owner_pid: int
    _source_db_path: Path
    _write_lock: asyncio.Lock
    _schema_lock: asyncio.Lock
    _schema_ensured: bool
    _bootstrap_needed: bool
    _transaction_depth: int
    _transaction_owner_task: asyncio.Task[object] | None
    _txn_conn: aiosqlite.Connection | None
    _bulk_conn: aiosqlite.Connection | None
    _read_pool: asyncio.Queue[aiosqlite.Connection] | None
    _manual_lease_cm: object | None
    _manual_owner_task: asyncio.Task[object] | None
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
        await _manual_begin(self)

    async def commit(self) -> None:
        """Commit the current transaction or release savepoint."""
        await _manual_commit(self)

    async def rollback(self) -> None:
        """Rollback to the last begin() or savepoint."""
        await _manual_rollback(self)

    def request_sql_settlement(self) -> None:
        """Wake only this backend's existing creator cleanup attempts."""
        _require_backend_process(self)
        with _BACKEND_CONNECTIONS_LOCK:
            custodies = {
                id(entry.grant.lease.custody): entry.grant.lease.custody
                for entry in _BACKEND_CONNECTIONS.values()
                if entry.backend is self
                and entry.thread is threading.current_thread()
                and entry.cleanup_attempt is not None
                and not entry.cleanup_attempt.done()
                and entry.grant is not None
                and entry.grant.lease.custody is not None
            }
        for custody in custodies.values():
            custody.request_sql_settlement()

    async def close(self) -> None:
        """Close database connections or retry their existing creator cleanup."""
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

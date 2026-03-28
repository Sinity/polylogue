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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

import polylogue.paths as _paths
from polylogue.errors import DatabaseError
from polylogue.storage.backends.async_sqlite_archive import SQLiteArchiveMixin
from polylogue.storage.backends.async_sqlite_raw import SQLiteRawMixin
from polylogue.storage.backends.async_sqlite_schema import ensure_schema as ensure_current_schema
from polylogue.storage.backends.connection import DB_TIMEOUT
from polylogue.storage.backends.queries import action_events as action_events_q
from polylogue.storage.backends.queries import (
    session_product_profile_writes as session_product_profiles_q,
)
from polylogue.storage.backends.queries import (
    session_product_thread_queries as session_product_threads_q,
)
from polylogue.storage.backends.queries import (
    session_product_timeline_writes as session_product_timelines_q,
)
from polylogue.storage.backends.queries import stats as stats_q
from polylogue.storage.backends.query_store import SQLiteQueryStore
from polylogue.storage.backends.schema import SCHEMA_DDL
from polylogue.storage.store import (
    ActionEventRecord,
    MessageRecord,
    SessionPhaseRecord,
    SessionProfileRecord,
    SessionWorkEventRecord,
    WorkThreadRecord,
)

# ---------------------------------------------------------------------------
# Shared connection helpers (formerly async_sqlite_support.py)
# ---------------------------------------------------------------------------


def default_db_path() -> Path:
    """Return the default database path (same as sync backend)."""
    return _paths.data_home() / "polylogue.db"


async def configure_connection(conn: aiosqlite.Connection) -> None:
    """Apply canonical connection settings.

    Performance pragmas (cache_size, synchronous, mmap_size) are critical
    for large databases. With a 28 GB DB and 2 MB default cache, every
    operation thrashes disk. These settings bring throughput from ~0.5/s
    to expected levels.
    """
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys = ON")
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute(f"PRAGMA busy_timeout = {DB_TIMEOUT * 1000}")
    # Performance: 128 MB page cache (default is 2 MB — unusable for large DBs)
    await conn.execute("PRAGMA cache_size = -131072")
    # Performance: NORMAL sync is safe with WAL and avoids fsync per write
    await conn.execute("PRAGMA synchronous = NORMAL")
    # Performance: 256 MB memory-mapped I/O for faster reads
    await conn.execute("PRAGMA mmap_size = 268435456")
    # Performance: keep temp tables in memory
    await conn.execute("PRAGMA temp_store = MEMORY")


# ---------------------------------------------------------------------------
# Runtime/state helpers (formerly async_sqlite_runtime.py)
# ---------------------------------------------------------------------------


def initialize_backend_state(backend: SQLiteBackend, db_path: Path | None) -> None:
    """Initialize backend state and shared query accessors."""
    backend._db_path = Path(db_path) if db_path is not None else default_db_path()
    backend._db_path.parent.mkdir(parents=True, exist_ok=True)

    backend._write_lock = asyncio.Lock()
    backend._schema_lock = asyncio.Lock()
    backend._schema_ensured = False
    backend._transaction_depth = 0
    backend._txn_conn = None
    backend._bulk_conn = None
    backend._read_pool = None

    backend.queries = SQLiteQueryStore(connection_factory=backend._get_connection)

    # Keep the shared DDL visibly anchored in the async backend module family.
    backend._shared_schema_ddl = SCHEMA_DDL


async def ensure_schema_once(backend: SQLiteBackend) -> None:
    """Ensure schema initialization runs exactly once."""
    if backend._schema_ensured:
        return
    async with backend._schema_lock:
        if backend._schema_ensured:
            return
        async with aiosqlite.connect(backend._db_path, timeout=DB_TIMEOUT) as init_conn:
            await configure_connection(init_conn)
            await backend._ensure_schema(init_conn)
        backend._schema_ensured = True


# ---------------------------------------------------------------------------
# Transaction helpers (formerly async_sqlite_transactions.py)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _backend_transaction(backend: SQLiteBackend) -> AsyncIterator[None]:
    """Context manager for database transactions."""
    async with backend._write_lock:
        if backend._txn_conn is None:
            backend._txn_conn = await aiosqlite.connect(backend._db_path, timeout=DB_TIMEOUT)
            await configure_connection(backend._txn_conn)

        await _backend_begin(backend)
        try:
            yield
            await _backend_commit(backend)
        except Exception:
            await _backend_rollback(backend)
            raise


async def _backend_begin(backend: SQLiteBackend) -> None:
    """Begin a transaction or nested savepoint."""
    await backend._ensure_schema_once()
    if backend._txn_conn is None:
        backend._txn_conn = await aiosqlite.connect(backend._db_path, timeout=DB_TIMEOUT)
        await configure_connection(backend._txn_conn)

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
        await backend._txn_conn.close()
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
        await backend._txn_conn.rollback()
        await backend._txn_conn.close()
        backend._txn_conn = None
    else:
        await backend._txn_conn.execute(f"ROLLBACK TO SAVEPOINT sp_{backend._transaction_depth}")


async def _close_backend(backend: SQLiteBackend) -> None:
    """Close database connections."""
    if backend._txn_conn is not None:
        await backend._txn_conn.close()
        backend._txn_conn = None
    backend._transaction_depth = 0


# ---------------------------------------------------------------------------
# Connection lifecycle helpers (formerly async_sqlite_connections.py)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _backend_connection(backend: SQLiteBackend) -> AsyncIterator[aiosqlite.Connection]:
    """Public connection context for read/query helpers."""
    async with backend._get_connection() as conn:
        yield conn


@asynccontextmanager
async def _bulk_connection(backend: SQLiteBackend) -> AsyncIterator[None]:
    """Keep a single connection alive for many sequential operations."""
    await backend._ensure_schema_once()
    conn = await aiosqlite.connect(backend._db_path, timeout=DB_TIMEOUT)
    await configure_connection(conn)
    await conn.execute("BEGIN IMMEDIATE")
    backend._bulk_conn = conn
    backend._transaction_depth += 1
    try:
        yield
        await conn.commit()
    except BaseException:
        await conn.rollback()
        raise
    finally:
        backend._transaction_depth -= 1
        backend._bulk_conn = None
        await conn.close()


async def _bulk_flush(backend: SQLiteBackend) -> None:
    """Commit the current bulk transaction and start a new one."""
    if backend._bulk_conn is not None:
        await backend._bulk_conn.commit()
        await backend._bulk_conn.execute("BEGIN IMMEDIATE")


@asynccontextmanager
async def _read_pool(backend: SQLiteBackend, size: int = 4) -> AsyncIterator[None]:
    """Open a pool of reusable read connections for concurrent operations."""
    await backend._ensure_schema_once()
    pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue()
    connections: list[aiosqlite.Connection] = []

    for _ in range(size):
        conn = await aiosqlite.connect(backend._db_path, timeout=DB_TIMEOUT)
        await configure_connection(conn)
        connections.append(conn)
        pool.put_nowait(conn)

    backend._read_pool = pool
    try:
        yield
    finally:
        backend._read_pool = None
        for conn in connections:
            await conn.close()


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
        conn = await backend._read_pool.get()
        try:
            yield conn
        finally:
            backend._read_pool.put_nowait(conn)
        return

    async with aiosqlite.connect(backend._db_path, timeout=DB_TIMEOUT) as conn:
        await configure_connection(conn)
        yield conn


class SQLiteBackend(
    SQLiteArchiveMixin,
    SQLiteRawMixin,
):
    """Async SQLite storage backend implementation.

    This backend provides async/await API for database operations, enabling
    true concurrency for read operations while maintaining write safety.
    """

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

    def connection(self):
        """Public connection context for read/query helpers."""
        return _backend_connection(self)

    async def _ensure_schema_once(self) -> None:
        """Ensure schema is initialized exactly once (thread-safe via asyncio lock)."""
        await ensure_schema_once(self)

    def bulk_connection(self):
        """Keep a single connection alive for many sequential operations."""
        return _bulk_connection(self)

    async def bulk_flush(self) -> None:
        """Commit the current bulk transaction and start a new one."""
        await _bulk_flush(self)

    def read_pool(self, size: int = 4):
        """Open a pool of reusable read connections for concurrent operations."""
        return _read_pool(self, size=size)

    def _get_connection(self):
        """Get async database connection with schema ensured."""
        return _get_connection(self)

    async def _ensure_schema(self, conn) -> None:
        """Ensure database schema exists and is at the current schema version."""
        await ensure_current_schema(conn)

    # -- Transaction management ---------------------------------------------

    def transaction(self):
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

    # -- Derived stats (formerly SQLiteDerivedStatsMixin) --------------------

    async def upsert_conversation_stats(
        self,
        conversation_id: str,
        provider_name: str,
        messages: list[MessageRecord],
    ) -> None:
        """Upsert precomputed per-conversation aggregate stats."""
        async with self._get_connection() as conn:
            await stats_q.upsert_conversation_stats(
                conn,
                conversation_id,
                provider_name,
                messages,
                self._transaction_depth,
            )

    # -- Derived actions (formerly SQLiteDerivedActionsMixin) ----------------

    async def replace_action_events(
        self,
        conversation_id: str,
        records: list[ActionEventRecord],
    ) -> None:
        """Replace durable action-event rows for one conversation."""
        async with self._get_connection() as conn:
            await action_events_q.replace_action_events(
                conn,
                conversation_id,
                records,
                self._transaction_depth,
            )

    async def get_action_events(self, conversation_id: str) -> list[ActionEventRecord]:
        """Get durable action-event rows for one conversation."""
        return await self.queries.get_action_events(conversation_id)

    async def get_action_events_batch(
        self,
        conversation_ids: list[str],
    ) -> dict[str, list[ActionEventRecord]]:
        """Get durable action-event rows for multiple conversations."""
        return await self.queries.get_action_events_batch(conversation_ids)

    # -- Derived products (formerly SQLiteDerivedProductsMixin) --------------

    async def replace_session_profile(
        self,
        record: SessionProfileRecord,
    ) -> None:
        """Replace one durable session-profile row."""
        async with self._get_connection() as conn:
            await session_product_profiles_q.replace_session_profile(
                conn,
                record,
                self._transaction_depth,
            )

    async def replace_session_work_events(
        self,
        conversation_id: str,
        records: list[SessionWorkEventRecord],
    ) -> None:
        """Replace durable work-event rows for one conversation."""
        async with self._get_connection() as conn:
            await session_product_timelines_q.replace_session_work_events(
                conn,
                conversation_id,
                records,
                self._transaction_depth,
            )

    async def replace_session_phases(
        self,
        conversation_id: str,
        records: list[SessionPhaseRecord],
    ) -> None:
        """Replace durable phase rows for one conversation."""
        async with self._get_connection() as conn:
            await session_product_timelines_q.replace_session_phases(
                conn,
                conversation_id,
                records,
                self._transaction_depth,
            )

    async def replace_work_thread(
        self,
        thread_id: str,
        record: WorkThreadRecord | None,
    ) -> None:
        """Replace one durable work-thread row."""
        async with self._get_connection() as conn:
            await session_product_threads_q.replace_work_thread(
                conn,
                thread_id,
                record,
                self._transaction_depth,
            )


__all__ = [
    "SCHEMA_DDL",
    "SQLiteBackend",
    "configure_connection",
    "default_db_path",
]

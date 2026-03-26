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

from polylogue.storage.backends.async_sqlite_archive import SQLiteArchiveMixin
from polylogue.storage.backends.async_sqlite_derived_actions import (
    SQLiteDerivedActionsMixin,
)
from polylogue.storage.backends.async_sqlite_derived_maintenance import (
    SQLiteDerivedMaintenanceMixin,
)
from polylogue.storage.backends.async_sqlite_derived_products import (
    SQLiteDerivedProductsMixin,
)
from polylogue.storage.backends.async_sqlite_derived_stats import SQLiteDerivedStatsMixin
from polylogue.storage.backends.async_sqlite_raw import SQLiteRawMixin
from polylogue.storage.backends.async_sqlite_schema import ensure_schema
from polylogue.storage.backends.async_sqlite_support import (
    configure_connection,
    default_db_path,
)
from polylogue.storage.backends.connection import DB_TIMEOUT
from polylogue.storage.backends.query_store import SQLiteQueryStore


class SQLiteBackend(
    SQLiteArchiveMixin,
    SQLiteDerivedActionsMixin,
    SQLiteDerivedProductsMixin,
    SQLiteDerivedMaintenanceMixin,
    SQLiteDerivedStatsMixin,
    SQLiteRawMixin,
):
    """Async SQLite storage backend implementation.

    This backend provides async/await API for database operations, enabling
    true concurrency for read operations while maintaining write safety.

    Thread Safety:
        - Each async task gets its own connection via asynccontextmanager
        - Write operations still use exclusive locks (SQLite limitation)
        - Safe for concurrent async tasks

    Transaction Management:
        - Use async with backend.transaction(): ... for transactions
        - All write operations should be within a transaction
        - Nested transactions use SAVEPOINTs

    Example:
        backend = SQLiteBackend()

        # Concurrent reads
        conv1, conv2 = await asyncio.gather(
            backend.get_conversation("id1"),
            backend.get_conversation("id2")
        )

        # Transaction for writes
        async with backend.transaction():
            await backend.save_conversation_record(conv)
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize async SQLite backend.

        Args:
            db_path: Path to SQLite database file. If None, uses default path.
        """
        self._db_path = Path(db_path) if db_path is not None else default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Write lock for serializing write operations
        self._write_lock = asyncio.Lock()

        # Lock and flag for schema initialization (prevents race condition)
        self._schema_lock = asyncio.Lock()
        self._schema_ensured = False

        # Transaction depth tracking for savepoint nesting
        self._transaction_depth = 0

        # Persistent connection for explicit transaction control
        self._txn_conn: aiosqlite.Connection | None = None

        # Reusable connection for bulk operations (e.g., acquisition)
        self._bulk_conn: aiosqlite.Connection | None = None

        # Connection pool for concurrent read operations (e.g., rendering)
        self._read_pool: asyncio.Queue[aiosqlite.Connection] | None = None

        # Canonical low-level read/query surface.
        self.queries = SQLiteQueryStore(connection_factory=self._get_connection)

    @property
    def db_path(self) -> Path:
        """Return the backing SQLite database path."""
        return self._db_path

    @property
    def transaction_depth(self) -> int:
        """Return the active transaction nesting depth."""
        return self._transaction_depth

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Public connection context for read/query helpers."""
        async with self._get_connection() as conn:
            yield conn

    async def _ensure_schema_once(self) -> None:
        """Ensure schema is initialized exactly once (thread-safe via asyncio lock)."""
        if self._schema_ensured:
            return
        async with self._schema_lock:
            if self._schema_ensured:
                return
            async with aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT) as init_conn:
                await configure_connection(init_conn)
                await self._ensure_schema(init_conn)
            self._schema_ensured = True

    @asynccontextmanager
    async def bulk_connection(self) -> AsyncIterator[None]:
        """Keep a single connection alive for many sequential operations.

        When active, ``_get_connection()`` reuses this connection instead of
        opening a new one per call.  All writes are grouped in a single
        transaction — call ``bulk_flush()`` periodically for intermediate
        durability.  On exit the final transaction is committed.

        This avoids both connection-per-call overhead (fd/WAL exhaustion)
        and per-item fsync overhead (each commit forces a WAL flush).
        """
        await self._ensure_schema_once()
        conn = await aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT)
        await configure_connection(conn)
        await conn.execute("BEGIN IMMEDIATE")
        self._bulk_conn = conn
        # Suppress per-item commits in methods that check _transaction_depth
        self._transaction_depth += 1
        try:
            yield
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise
        finally:
            self._transaction_depth -= 1
            self._bulk_conn = None
            await conn.close()

    async def bulk_flush(self) -> None:
        """Commit the current bulk transaction and start a new one.

        Call periodically during long bulk operations for intermediate
        durability.  Safe to call outside ``bulk_connection()`` (no-op).
        """
        if self._bulk_conn is not None:
            await self._bulk_conn.commit()
            await self._bulk_conn.execute("BEGIN IMMEDIATE")

    @asynccontextmanager
    async def read_pool(self, size: int = 4) -> AsyncIterator[None]:
        """Open a pool of reusable read connections for concurrent operations.

        While active, ``_get_connection()`` borrows from the pool instead of
        opening a fresh connection per call.  This eliminates per-call
        overhead: thread spawn, PRAGMA negotiation, and schema checks.

        Use for read-heavy phases like rendering where many concurrent
        tasks each need short-lived DB access.

        Args:
            size: Number of connections in the pool (match worker count)
        """
        await self._ensure_schema_once()
        pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue()
        connections: list[aiosqlite.Connection] = []

        for _ in range(size):
            conn = await aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT)
            await configure_connection(conn)
            connections.append(conn)
            pool.put_nowait(conn)

        self._read_pool = pool
        try:
            yield
        finally:
            self._read_pool = None
            for conn in connections:
                await conn.close()

    @asynccontextmanager
    async def _get_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Get async database connection with schema ensured.

        Connection reuse priority:
        1. Transaction connection (_txn_conn) when inside begin/commit block
        2. Bulk connection (_bulk_conn) when inside bulk_connection() context
        3. Read pool connection when inside read_pool() context
        4. Fresh connection per call (fallback)
        """
        await self._ensure_schema_once()

        # Reuse transaction connection when inside begin/commit block
        if self._txn_conn is not None and self._transaction_depth > 0:
            yield self._txn_conn
            return

        # Reuse bulk connection when inside bulk_connection() context
        if self._bulk_conn is not None:
            yield self._bulk_conn
            return

        # Borrow from read pool when available
        if self._read_pool is not None:
            conn = await self._read_pool.get()
            try:
                yield conn
            finally:
                self._read_pool.put_nowait(conn)
            return

        async with aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT) as conn:
            await configure_connection(conn)
            yield conn

    async def _ensure_schema(self, conn: aiosqlite.Connection) -> None:
        """Ensure database schema exists and is at the current schema version."""
        await ensure_schema(conn)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Context manager for database transactions.

        Acquires write lock to serialize write operations and manages
        explicit transaction control with savepoint nesting.
        """
        async with self._write_lock:
            # Create persistent connection for explicit transaction if needed
            if self._txn_conn is None:
                self._txn_conn = await aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT)
                await configure_connection(self._txn_conn)

            await self.begin()
            try:
                yield
                await self.commit()
            except Exception:
                await self.rollback()
                raise

    async def begin(self) -> None:
        """Begin a transaction or nested savepoint."""
        await self._ensure_schema_once()
        if self._txn_conn is None:
            self._txn_conn = await aiosqlite.connect(self._db_path, timeout=DB_TIMEOUT)
            await configure_connection(self._txn_conn)

        if self._transaction_depth == 0:
            await self._txn_conn.execute("BEGIN IMMEDIATE")
        else:
            await self._txn_conn.execute(f"SAVEPOINT sp_{self._transaction_depth}")
        self._transaction_depth += 1

    async def commit(self) -> None:
        """Commit the current transaction or release savepoint."""
        if self._transaction_depth <= 0:
            from polylogue.errors import DatabaseError

            raise DatabaseError("No active transaction to commit")

        if self._txn_conn is None:
            from polylogue.errors import DatabaseError

            raise DatabaseError("No transaction connection")

        self._transaction_depth -= 1

        if self._transaction_depth == 0:
            await self._txn_conn.commit()
            await self._txn_conn.close()
            self._txn_conn = None
        else:
            await self._txn_conn.execute(f"RELEASE SAVEPOINT sp_{self._transaction_depth}")

    async def rollback(self) -> None:
        """Rollback to the last begin() or savepoint."""
        if self._transaction_depth <= 0:
            from polylogue.errors import DatabaseError

            raise DatabaseError("No active transaction to rollback")

        if self._txn_conn is None:
            from polylogue.errors import DatabaseError

            raise DatabaseError("No transaction connection")

        self._transaction_depth -= 1

        if self._transaction_depth == 0:
            await self._txn_conn.rollback()
            await self._txn_conn.close()
            self._txn_conn = None
        else:
            await self._txn_conn.execute(f"ROLLBACK TO SAVEPOINT sp_{self._transaction_depth}")

    async def close(self) -> None:
        """Close database connections."""
        if self._txn_conn is not None:
            await self._txn_conn.close()
            self._txn_conn = None
        self._transaction_depth = 0


__all__ = [
    "SQLiteBackend",
    "default_db_path",
]

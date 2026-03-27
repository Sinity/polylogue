"""Async SQLite storage backend implementation using aiosqlite.

This backend provides async/await API for all database operations, enabling
concurrent queries and parallel processing without blocking.

Performance characteristics:
- Parallel reads: 5-10x faster for batch operations
- Write serialization: Still uses exclusive locks (SQLite limitation)
- Connection pooling: Each async context gets its own connection
"""

from __future__ import annotations

from pathlib import Path

from polylogue.storage.backends.async_sqlite_archive import SQLiteArchiveMixin
from polylogue.storage.backends.async_sqlite_connections import (
    bulk_connection as backend_bulk_connection,
)
from polylogue.storage.backends.async_sqlite_connections import (
    bulk_flush as backend_bulk_flush,
)
from polylogue.storage.backends.async_sqlite_connections import (
    connection as backend_connection,
)
from polylogue.storage.backends.async_sqlite_connections import (
    get_connection as backend_get_connection,
)
from polylogue.storage.backends.async_sqlite_connections import (
    read_pool as backend_read_pool,
)
from polylogue.storage.backends.async_sqlite_derived_actions import (
    SQLiteDerivedActionsMixin,
)
from polylogue.storage.backends.async_sqlite_derived_products import (
    SQLiteDerivedProductsMixin,
)
from polylogue.storage.backends.async_sqlite_derived_stats import SQLiteDerivedStatsMixin
from polylogue.storage.backends.async_sqlite_raw import SQLiteRawMixin
from polylogue.storage.backends.async_sqlite_runtime import (
    ensure_schema as ensure_backend_schema,
)
from polylogue.storage.backends.async_sqlite_runtime import (
    ensure_schema_once,
    initialize_backend_state,
)
from polylogue.storage.backends.async_sqlite_support import default_db_path
from polylogue.storage.backends.async_sqlite_transactions import (
    begin as backend_begin,
)
from polylogue.storage.backends.async_sqlite_transactions import (
    close_backend,
)
from polylogue.storage.backends.async_sqlite_transactions import (
    commit as backend_commit,
)
from polylogue.storage.backends.async_sqlite_transactions import (
    rollback as backend_rollback,
)
from polylogue.storage.backends.async_sqlite_transactions import (
    transaction as backend_transaction,
)
from polylogue.storage.backends.schema import SCHEMA_DDL


class SQLiteBackend(
    SQLiteArchiveMixin,
    SQLiteDerivedActionsMixin,
    SQLiteDerivedProductsMixin,
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
        initialize_backend_state(self, db_path)

    @property
    def db_path(self) -> Path:
        """Return the backing SQLite database path."""
        return self._db_path

    @property
    def transaction_depth(self) -> int:
        """Return the active transaction nesting depth."""
        return self._transaction_depth

    def connection(self):
        """Public connection context for read/query helpers."""
        return backend_connection(self)

    async def _ensure_schema_once(self) -> None:
        """Ensure schema is initialized exactly once (thread-safe via asyncio lock)."""
        await ensure_schema_once(self)

    def bulk_connection(self):
        """Keep a single connection alive for many sequential operations.

        When active, ``_get_connection()`` reuses this connection instead of
        opening a new one per call.  All writes are grouped in a single
        transaction — call ``bulk_flush()`` periodically for intermediate
        durability.  On exit the final transaction is committed.

        This avoids both connection-per-call overhead (fd/WAL exhaustion)
        and per-item fsync overhead (each commit forces a WAL flush).
        """
        return backend_bulk_connection(self)

    async def bulk_flush(self) -> None:
        """Commit the current bulk transaction and start a new one.

        Call periodically during long bulk operations for intermediate
        durability.  Safe to call outside ``bulk_connection()`` (no-op).
        """
        await backend_bulk_flush(self)

    def read_pool(self, size: int = 4):
        """Open a pool of reusable read connections for concurrent operations.

        While active, ``_get_connection()`` borrows from the pool instead of
        opening a fresh connection per call.  This eliminates per-call
        overhead: thread spawn, PRAGMA negotiation, and schema checks.

        Use for read-heavy phases like rendering where many concurrent
        tasks each need short-lived DB access.

        Args:
            size: Number of connections in the pool (match worker count)
        """
        return backend_read_pool(self, size=size)

    def _get_connection(self):
        """Get async database connection with schema ensured.

        Connection reuse priority:
        1. Transaction connection (_txn_conn) when inside begin/commit block
        2. Bulk connection (_bulk_conn) when inside bulk_connection() context
        3. Read pool connection when inside read_pool() context
        4. Fresh connection per call (fallback)
        """
        return backend_get_connection(self)

    async def _ensure_schema(self, conn) -> None:
        """Ensure database schema exists and is at the current schema version."""
        await ensure_backend_schema(conn)

    def transaction(self):
        """Context manager for database transactions.

        Acquires write lock to serialize write operations and manages
        explicit transaction control with savepoint nesting.
        """
        return backend_transaction(self)

    async def begin(self) -> None:
        """Begin a transaction or nested savepoint."""
        await backend_begin(self)

    async def commit(self) -> None:
        """Commit the current transaction or release savepoint."""
        await backend_commit(self)

    async def rollback(self) -> None:
        """Rollback to the last begin() or savepoint."""
        await backend_rollback(self)

    async def close(self) -> None:
        """Close database connections."""
        await close_backend(self)


__all__ = [
    "SCHEMA_DDL",
    "SQLiteBackend",
    "default_db_path",
]

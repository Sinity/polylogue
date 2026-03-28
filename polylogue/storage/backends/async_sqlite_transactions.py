"""Transaction helpers for the async SQLite backend."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import aiosqlite

from polylogue.errors import DatabaseError
from polylogue.storage.backends.async_sqlite_support import configure_connection
from polylogue.storage.backends.connection import DB_TIMEOUT

if TYPE_CHECKING:
    from polylogue.storage.backends.async_sqlite import SQLiteBackend


@asynccontextmanager
async def transaction(backend: SQLiteBackend) -> AsyncIterator[None]:
    """Context manager for database transactions."""
    async with backend._write_lock:
        if backend._txn_conn is None:
            backend._txn_conn = await aiosqlite.connect(backend._db_path, timeout=DB_TIMEOUT)
            await configure_connection(backend._txn_conn)

        await begin(backend)
        try:
            yield
            await commit(backend)
        except Exception:
            await rollback(backend)
            raise


async def begin(backend: SQLiteBackend) -> None:
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


async def commit(backend: SQLiteBackend) -> None:
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


async def rollback(backend: SQLiteBackend) -> None:
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


async def close_backend(backend: SQLiteBackend) -> None:
    """Close database connections."""
    if backend._txn_conn is not None:
        await backend._txn_conn.close()
        backend._txn_conn = None
    backend._transaction_depth = 0


__all__ = [
    "begin",
    "close_backend",
    "commit",
    "rollback",
    "transaction",
]

"""Connection lifecycle helpers for the async SQLite backend."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import aiosqlite

from polylogue.storage.backends.async_sqlite_support import configure_connection
from polylogue.storage.backends.connection import DB_TIMEOUT

if TYPE_CHECKING:
    from polylogue.storage.backends.async_sqlite import SQLiteBackend


@asynccontextmanager
async def connection(backend: SQLiteBackend) -> AsyncIterator[aiosqlite.Connection]:
    """Public connection context for read/query helpers."""
    async with backend._get_connection() as conn:
        yield conn


@asynccontextmanager
async def bulk_connection(backend: SQLiteBackend) -> AsyncIterator[None]:
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


async def bulk_flush(backend: SQLiteBackend) -> None:
    """Commit the current bulk transaction and start a new one."""
    if backend._bulk_conn is not None:
        await backend._bulk_conn.commit()
        await backend._bulk_conn.execute("BEGIN IMMEDIATE")


@asynccontextmanager
async def read_pool(backend: SQLiteBackend, size: int = 4) -> AsyncIterator[None]:
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
async def get_connection(backend: SQLiteBackend) -> AsyncIterator[aiosqlite.Connection]:
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

__all__ = [
    "bulk_connection",
    "bulk_flush",
    "connection",
    "get_connection",
    "read_pool",
]

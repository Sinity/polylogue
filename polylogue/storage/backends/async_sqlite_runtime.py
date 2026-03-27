"""Runtime/state helpers for the async SQLite backend."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite

from polylogue.storage.backends.async_sqlite_schema import ensure_schema as ensure_current_schema
from polylogue.storage.backends.async_sqlite_support import (
    configure_connection,
    default_db_path,
)
from polylogue.storage.backends.connection import DB_TIMEOUT
from polylogue.storage.backends.query_store import SQLiteQueryStore
from polylogue.storage.backends.schema import SCHEMA_DDL

if TYPE_CHECKING:
    from polylogue.storage.backends.async_sqlite import SQLiteBackend


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


async def ensure_schema(conn: aiosqlite.Connection) -> None:
    """Ensure database schema exists and is current."""
    await ensure_current_schema(conn)


__all__ = [
    "SCHEMA_DDL",
    "ensure_schema",
    "ensure_schema_once",
    "initialize_backend_state",
]

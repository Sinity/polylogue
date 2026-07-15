"""Centralized table existence checks for SQLite connections.

Lives directly under `storage/` (not `storage/sqlite/`) deliberately:
`polylogue/storage/sqlite/__init__.py` eagerly imports the heavy async
backend stack, so a low-level module like `storage/blob_publication.py`
importing a sibling under `storage.sqlite` triggers that whole chain and
circles back through `insights/` -- a real circular import, not a
hypothetical one. `storage/__init__.py` itself only exposes lazy
`__getattr__`-based exports, so importing this module doesn't pull in that
chain. Raw SQL belongs in `storage/` per the package-placement rules in
`docs/architecture.md`; `core/` is reserved for no-I/O shared primitives.

@owner storage-root
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite


def table_exists(conn: sqlite3.Connection, name: str, *, schema: str = "main") -> bool:
    """Check if a table exists in the given schema (sync SQLite).

    `schema` selects an attached database by name (e.g. a source.db attached
    to an index.db connection) and must be a trusted internal literal, never
    user input -- SQLite has no way to bind a schema/database name as a query
    parameter, so it is interpolated into the query text.

    Args:
        conn: SQLite connection
        name: Table name to check
        schema: Schema name (default: "main")

    Returns:
        True if the table exists, False otherwise
    """
    cursor = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    )
    return cursor.fetchone() is not None


async def table_exists_async(conn: aiosqlite.Connection, name: str, *, schema: str = "main") -> bool:
    """Check if a table exists in the given schema (async SQLite).

    See `table_exists` for the `schema` trust requirement.

    Args:
        conn: aiosqlite connection
        name: Table name to check
        schema: Schema name (default: "main")

    Returns:
        True if the table exists, False otherwise
    """
    cursor = await conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    )
    row = await cursor.fetchone()
    return row is not None


__all__ = ["table_exists", "table_exists_async"]

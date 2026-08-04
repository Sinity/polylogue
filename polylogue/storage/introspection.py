"""Centralized SQLite schema-introspection primitives (table/column/index existence).

Lives directly under `storage/` (not `storage/sqlite/`) deliberately:
`polylogue/storage/sqlite/__init__.py` eagerly imports the heavy async
backend stack, so a low-level module like `storage/blob_publication.py`
importing a sibling under `storage.sqlite` triggers that whole chain and
circles back through `insights/` -- a real circular import, not a
hypothetical one. `storage/__init__.py` itself only exposes lazy
`__getattr__`-based exports, so importing this module doesn't pull in that
chain. Raw SQL belongs in `storage/` per the package-placement rules in
`docs/architecture.md`; `core/` is reserved for no-I/O shared primitives.

This module consolidates what were ~25 near-identical, independently
maintained `_table_exists`/`_column_exists`/`_index_exists` helpers scattered
across `cli/`, `daemon/`, `storage/`, `sources/`, `insights/`, and
`operations/` (polylogue-48h). Several of those copies checked
`type IN ('table', 'virtual table')` or `type IN ('table', 'shadow')` --
neither `'virtual table'` nor `'shadow'` is ever an actual `sqlite_master`
`type` value (virtual tables, including FTS5 shadow tables, register with
`type='table'`), so those extra alternatives were dead and the plain
`type='table'` check here is behavior-preserving.

@owner storage-root
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite


def _schema_prefix(*, schema: str) -> str:
    # Deliberately empty for the default "main" schema: an unqualified
    # `sqlite_master`/`PRAGMA table_info(...)` already resolves to `main`
    # only (SQLite never merges attached-database schemas into a bare
    # `sqlite_master` read), and every pre-consolidation call site issued
    # exactly that bare form -- some with test doubles that pattern-match
    # the query text. Qualifying only for a genuinely non-default schema
    # keeps 100% of existing call sites byte-identical while still letting
    # `schema=` target an attached database explicitly.
    return "" if schema == "main" else f"{schema}."


def table_exists(conn: sqlite3.Connection, name: str, *, schema: str = "main") -> bool:
    """Check if a table (or virtual table) exists in the given schema (sync SQLite).

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
        f"SELECT 1 FROM {_schema_prefix(schema=schema)}sqlite_master WHERE type='table' AND name=? LIMIT 1",
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
        f"SELECT 1 FROM {_schema_prefix(schema=schema)}sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    )
    row = await cursor.fetchone()
    return row is not None


def index_exists(conn: sqlite3.Connection, name: str, *, schema: str = "main") -> bool:
    """Check if an index exists in the given schema (sync SQLite).

    See `table_exists` for the `schema` trust requirement.
    """
    cursor = conn.execute(
        f"SELECT 1 FROM {_schema_prefix(schema=schema)}sqlite_master WHERE type='index' AND name=? LIMIT 1",
        (name,),
    )
    return cursor.fetchone() is not None


async def index_exists_async(conn: aiosqlite.Connection, name: str, *, schema: str = "main") -> bool:
    """Check if an index exists in the given schema (async SQLite).

    See `table_exists` for the `schema` trust requirement.
    """
    cursor = await conn.execute(
        f"SELECT 1 FROM {_schema_prefix(schema=schema)}sqlite_master WHERE type='index' AND name=? LIMIT 1",
        (name,),
    )
    row = await cursor.fetchone()
    return row is not None


def _table_info_pragma(table: str, *, schema: str) -> str:
    return f"PRAGMA {_schema_prefix(schema=schema)}table_info({table})"


def column_exists(conn: sqlite3.Connection, table: str, column: str, *, schema: str = "main") -> bool:
    """Check if `column` exists on `table` in the given schema (sync SQLite).

    Returns False (rather than raising) when `table` itself does not exist,
    matching `PRAGMA table_info` on a missing table returning zero rows.
    See `table_exists` for the `schema`/`table` trust requirement -- both are
    interpolated into `PRAGMA` text since SQLite cannot bind pragma targets.
    """
    if not table_exists(conn, table, schema=schema):
        return False
    rows = conn.execute(_table_info_pragma(table, schema=schema)).fetchall()
    return any(str(row[1]) == column for row in rows)


async def column_exists_async(conn: aiosqlite.Connection, table: str, column: str, *, schema: str = "main") -> bool:
    """Check if `column` exists on `table` in the given schema (async SQLite).

    See `column_exists` for behavior on a missing table and the trust
    requirement on `schema`/`table`.
    """
    if not await table_exists_async(conn, table, schema=schema):
        return False
    cursor = await conn.execute(_table_info_pragma(table, schema=schema))
    rows = await cursor.fetchall()
    return any(str(row[1]) == column for row in rows)


__all__ = [
    "table_exists",
    "table_exists_async",
    "index_exists",
    "index_exists_async",
    "column_exists",
    "column_exists_async",
]

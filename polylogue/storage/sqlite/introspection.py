"""Shared SQLite introspection helpers.

This module consolidates schema query primitives (table_exists, index_exists,
column_exists) that were previously duplicated across 40+ call sites.
Sync versions work with sqlite3.Connection; async versions work with aiosqlite.Connection.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite


# Sync versions (sqlite3.Connection)


def table_exists(
    conn: sqlite3.Connection,
    name: str,
    *,
    schema: str = "main",
    include_virtual: bool = False,
    include_views: bool = False,
) -> bool:
    """Check if a table exists in the specified schema.

    Args:
        conn: sqlite3 connection
        name: Table name to check
        schema: Schema to query (default 'main')
        include_virtual: If True, include 'virtual table' in type filter
        include_views: If True, include 'view' in type filter

    Returns:
        True if the table exists, False otherwise
    """
    types: list[str] = ["table"]
    if include_virtual:
        types.append("virtual table")
    if include_views:
        types.append("view")

    type_placeholders = ", ".join("?" * len(types))
    row = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type IN ({type_placeholders}) AND name = ? LIMIT 1",
        (*types, name),
    ).fetchone()
    return row is not None


def index_exists(
    conn: sqlite3.Connection,
    name: str,
    *,
    schema: str = "main",
) -> bool:
    """Check if an index exists in the specified schema.

    Args:
        conn: sqlite3 connection
        name: Index name to check
        schema: Schema to query (default 'main')

    Returns:
        True if the index exists, False otherwise
    """
    row = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type = 'index' AND name = ? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def column_exists(
    conn: sqlite3.Connection,
    table: str,
    column: str,
) -> bool:
    """Check if a column exists in a table.

    Args:
        conn: sqlite3 connection
        table: Table name
        column: Column name to check

    Returns:
        True if the column exists, False otherwise
    """
    if not table_exists(conn, table):
        return False

    # Use PRAGMA table_info to check for column
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(row[1]) == column for row in rows)


def schema_object_exists(
    conn: sqlite3.Connection,
    name: str,
    *,
    types: Sequence[str],
    schema: str = "main",
) -> bool:
    """Check if a schema object of any of the specified types exists.

    Args:
        conn: sqlite3 connection
        name: Object name to check
        types: Sequence of sqlite_master type values (e.g., ('table', 'view'))
        schema: Schema to query (default 'main')

    Returns:
        True if an object with the specified name and type exists, False otherwise
    """
    type_placeholders = ", ".join("?" * len(types))
    row = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type IN ({type_placeholders}) AND name = ? LIMIT 1",
        (*types, name),
    ).fetchone()
    return row is not None


def trigger_exists(
    conn: sqlite3.Connection,
    name: str,
    *,
    schema: str = "main",
) -> bool:
    """Check if a trigger exists in the specified schema.

    Args:
        conn: sqlite3 connection
        name: Trigger name to check
        schema: Schema to query (default 'main')

    Returns:
        True if the trigger exists, False otherwise
    """
    row = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type = 'trigger' AND name = ? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


# Async versions (aiosqlite.Connection)


async def table_exists_async(
    conn: aiosqlite.Connection,
    name: str,
    *,
    schema: str = "main",
    include_virtual: bool = False,
    include_views: bool = False,
) -> bool:
    """Check if a table exists in the specified schema (async version).

    Args:
        conn: aiosqlite connection
        name: Table name to check
        schema: Schema to query (default 'main')
        include_virtual: If True, include 'virtual table' in type filter
        include_views: If True, include 'view' in type filter

    Returns:
        True if the table exists, False otherwise
    """
    types: list[str] = ["table"]
    if include_virtual:
        types.append("virtual table")
    if include_views:
        types.append("view")

    type_placeholders = ", ".join("?" * len(types))
    cursor = await conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type IN ({type_placeholders}) AND name = ? LIMIT 1",
        (*types, name),
    )
    row = await cursor.fetchone()
    return row is not None


async def index_exists_async(
    conn: aiosqlite.Connection,
    name: str,
    *,
    schema: str = "main",
) -> bool:
    """Check if an index exists in the specified schema (async version).

    Args:
        conn: aiosqlite connection
        name: Index name to check
        schema: Schema to query (default 'main')

    Returns:
        True if the index exists, False otherwise
    """
    cursor = await conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type = 'index' AND name = ? LIMIT 1",
        (name,),
    )
    row = await cursor.fetchone()
    return row is not None


async def column_exists_async(
    conn: aiosqlite.Connection,
    table: str,
    column: str,
) -> bool:
    """Check if a column exists in a table (async version).

    Args:
        conn: aiosqlite connection
        table: Table name
        column: Column name to check

    Returns:
        True if the column exists, False otherwise
    """
    if not await table_exists_async(conn, table):
        return False

    # Use PRAGMA table_info to check for column
    cursor = await conn.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    return any(str(row[1]) == column for row in rows)


async def schema_object_exists_async(
    conn: aiosqlite.Connection,
    name: str,
    *,
    types: Sequence[str],
    schema: str = "main",
) -> bool:
    """Check if a schema object of any of the specified types exists (async version).

    Args:
        conn: aiosqlite connection
        name: Object name to check
        types: Sequence of sqlite_master type values (e.g., ('table', 'view'))
        schema: Schema to query (default 'main')

    Returns:
        True if an object with the specified name and type exists, False otherwise
    """
    type_placeholders = ", ".join("?" * len(types))
    cursor = await conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type IN ({type_placeholders}) AND name = ? LIMIT 1",
        (*types, name),
    )
    row = await cursor.fetchone()
    return row is not None


async def trigger_exists_async(
    conn: aiosqlite.Connection,
    name: str,
    *,
    schema: str = "main",
) -> bool:
    """Check if a trigger exists in the specified schema (async version).

    Args:
        conn: aiosqlite connection
        name: Trigger name to check
        schema: Schema to query (default 'main')

    Returns:
        True if the trigger exists, False otherwise
    """
    cursor = await conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type = 'trigger' AND name = ? LIMIT 1",
        (name,),
    )
    row = await cursor.fetchone()
    return row is not None

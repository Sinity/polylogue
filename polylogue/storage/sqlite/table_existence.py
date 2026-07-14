"""Centralized table existence checks for SQLite connections."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite


def table_exists(conn: sqlite3.Connection, name: str, *, schema: str = "main") -> bool:
    """Check if a table exists in the given schema (sync SQLite).

    Args:
        conn: SQLite connection
        name: Table name to check
        schema: Schema name (default: "main")

    Returns:
        True if the table exists, False otherwise
    """
    cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? AND db=?",
        (name, schema),
    )
    return cursor.fetchone() is not None


async def table_exists_async(conn: aiosqlite.Connection, name: str, *, schema: str = "main") -> bool:
    """Check if a table exists in the given schema (async SQLite).

    Args:
        conn: aiosqlite connection
        name: Table name to check
        schema: Schema name (default: "main")

    Returns:
        True if the table exists, False otherwise
    """
    cursor = await conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? AND db=?",
        (name, schema),
    )
    row = await cursor.fetchone()
    return row is not None


__all__ = ["table_exists", "table_exists_async"]

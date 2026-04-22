"""Shared sqlite-vec extension loading primitive."""

from __future__ import annotations

import sqlite3

import aiosqlite


def try_load_sqlite_vec(conn: sqlite3.Connection) -> tuple[bool, Exception | None]:
    """Attempt to load sqlite-vec and return the failure for caller policy."""
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
            return True, None
        finally:
            conn.enable_load_extension(False)
    except Exception as exc:
        return False, exc


async def try_load_sqlite_vec_async(conn: aiosqlite.Connection) -> tuple[bool, Exception | None]:
    """Attempt to load sqlite-vec on an async connection."""
    try:
        import sqlite_vec

        await conn.enable_load_extension(True)
        try:
            await conn.load_extension(sqlite_vec.loadable_path())
            return True, None
        finally:
            await conn.enable_load_extension(False)
    except Exception as exc:
        return False, exc


__all__ = ["try_load_sqlite_vec", "try_load_sqlite_vec_async"]

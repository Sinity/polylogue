"""tests for polylogue.storage.table_existence.

polylogue-a7xr.9: the canonical table_exists()/table_exists_async() this
module exports had never been exercised against a real connection --
`SELECT 1 FROM sqlite_master WHERE type='table' AND name=? AND db=?` binds
`db` as a query parameter, but sqlite_master has no `db` column, so every
call raised sqlite3.OperationalError. Regression-tested here so a future
edit can't silently reintroduce a query that only "works" against a mock.
"""

from __future__ import annotations

import sqlite3

import aiosqlite
import pytest

from polylogue.storage.table_existence import table_exists, table_exists_async


def test_table_exists_true_for_a_real_table() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE sessions (id INTEGER)")
        assert table_exists(conn, "sessions") is True
    finally:
        conn.close()


def test_table_exists_false_for_a_missing_table() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE sessions (id INTEGER)")
        assert table_exists(conn, "nonexistent_table") is False
    finally:
        conn.close()


def test_table_exists_checks_the_named_attached_schema(tmp_path: object) -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE main_only (id INTEGER)")
        conn.execute(f"ATTACH DATABASE '{tmp_path}/source.db' AS source")
        conn.execute("CREATE TABLE source.raw_sessions (id INTEGER)")

        assert table_exists(conn, "main_only", schema="main") is True
        assert table_exists(conn, "main_only", schema="source") is False
        assert table_exists(conn, "raw_sessions", schema="source") is True
        assert table_exists(conn, "raw_sessions", schema="main") is False
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_table_exists_async_true_for_a_real_table() -> None:
    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute("CREATE TABLE sessions (id INTEGER)")
        assert await table_exists_async(conn, "sessions") is True
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_table_exists_async_false_for_a_missing_table() -> None:
    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute("CREATE TABLE sessions (id INTEGER)")
        assert await table_exists_async(conn, "nonexistent_table") is False
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_table_exists_async_checks_the_named_attached_schema(tmp_path: object) -> None:
    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute("CREATE TABLE main_only (id INTEGER)")
        await conn.execute(f"ATTACH DATABASE '{tmp_path}/source.db' AS source")
        await conn.execute("CREATE TABLE source.raw_sessions (id INTEGER)")

        assert await table_exists_async(conn, "main_only", schema="main") is True
        assert await table_exists_async(conn, "main_only", schema="source") is False
        assert await table_exists_async(conn, "raw_sessions", schema="source") is True
    finally:
        await conn.close()

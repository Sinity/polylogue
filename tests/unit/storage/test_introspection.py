"""tests for polylogue.storage.introspection.

polylogue-a7xr.9: the canonical table_exists()/table_exists_async() this
module exports had never been exercised against a real connection --
`SELECT 1 FROM sqlite_master WHERE type='table' AND name=? AND db=?` binds
`db` as a query parameter, but sqlite_master has no `db` column, so every
call raised sqlite3.OperationalError. Regression-tested here so a future
edit can't silently reintroduce a query that only "works" against a mock.

polylogue-48h: column_exists()/index_exists() (+ async variants) were added
alongside table_exists() when ~25 duplicated `_table_exists`/`_column_exists`/
`_index_exists` copies across the codebase were consolidated into this module.
"""

from __future__ import annotations

import sqlite3

import aiosqlite
import pytest

from polylogue.storage.introspection import (
    column_exists,
    column_exists_async,
    index_exists,
    index_exists_async,
    table_exists,
    table_exists_async,
)


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


def test_column_exists_true_for_a_real_column() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE sessions (id INTEGER, title TEXT)")
        assert column_exists(conn, "sessions", "title") is True
    finally:
        conn.close()


def test_column_exists_false_for_a_missing_column() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE sessions (id INTEGER)")
        assert column_exists(conn, "sessions", "nonexistent_column") is False
    finally:
        conn.close()


def test_column_exists_false_for_a_missing_table() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        assert column_exists(conn, "nonexistent_table", "id") is False
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_column_exists_async_true_for_a_real_column() -> None:
    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute("CREATE TABLE sessions (id INTEGER, title TEXT)")
        assert await column_exists_async(conn, "sessions", "title") is True
        assert await column_exists_async(conn, "sessions", "nonexistent_column") is False
        assert await column_exists_async(conn, "nonexistent_table", "id") is False
    finally:
        await conn.close()


def test_index_exists_true_for_a_real_index() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE sessions (id INTEGER)")
        conn.execute("CREATE INDEX idx_sessions_id ON sessions (id)")
        assert index_exists(conn, "idx_sessions_id") is True
    finally:
        conn.close()


def test_index_exists_false_for_a_missing_index() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE sessions (id INTEGER)")
        assert index_exists(conn, "nonexistent_index") is False
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_index_exists_async_true_for_a_real_index() -> None:
    conn = await aiosqlite.connect(":memory:")
    try:
        await conn.execute("CREATE TABLE sessions (id INTEGER)")
        await conn.execute("CREATE INDEX idx_sessions_id ON sessions (id)")
        assert await index_exists_async(conn, "idx_sessions_id") is True
        assert await index_exists_async(conn, "nonexistent_index") is False
    finally:
        await conn.close()

"""Contracts for shared embedding-stats helpers."""

from __future__ import annotations

import sqlite3

import aiosqlite
import pytest

from polylogue.storage.embedding_stats import (
    read_embedding_stats_async,
    read_embedding_stats_sync,
)


def test_read_embedding_stats_sync_missing_tables_returns_zeroes() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        stats = read_embedding_stats_sync(conn)
    finally:
        conn.close()

    assert stats.embedded_conversations == 0
    assert stats.embedded_messages == 0
    assert stats.pending_conversations == 0


def test_read_embedding_stats_sync_counts_available_tables() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE embedding_status (conversation_id TEXT, needs_reindex INTEGER NOT NULL)")
        conn.execute("CREATE TABLE message_embeddings (message_id TEXT)")
        conn.executemany(
            "INSERT INTO embedding_status (conversation_id, needs_reindex) VALUES (?, ?)",
            [("conv-1", 0), ("conv-2", 0), ("conv-3", 1)],
        )
        conn.executemany(
            "INSERT INTO message_embeddings (message_id) VALUES (?)",
            [("msg-1",), ("msg-2",), ("msg-3",)],
        )
        conn.commit()

        stats = read_embedding_stats_sync(conn)
    finally:
        conn.close()

    assert stats.embedded_conversations == 2
    assert stats.embedded_messages == 3
    assert stats.pending_conversations == 1


def test_read_embedding_stats_sync_propagates_non_missing_operational_errors() -> None:
    class LockedConnection:
        def execute(self, sql: str):  # pragma: no cover - trivial stub
            raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        read_embedding_stats_sync(LockedConnection())  # type: ignore[arg-type]


def test_read_embedding_stats_sync_treats_missing_vec_module_as_optional() -> None:
    class VeclessConnection:
        def execute(self, sql: str):
            if "message_embeddings" in sql:
                raise sqlite3.OperationalError("no such module: vec0")
            raise sqlite3.OperationalError("no such table: embedding_status")

    stats = read_embedding_stats_sync(VeclessConnection())  # type: ignore[arg-type]

    assert stats.embedded_conversations == 0
    assert stats.embedded_messages == 0
    assert stats.pending_conversations == 0


@pytest.mark.asyncio
async def test_read_embedding_stats_async_counts_available_tables() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE embedding_status (conversation_id TEXT, needs_reindex INTEGER NOT NULL)")
        await conn.execute("CREATE TABLE message_embeddings (message_id TEXT)")
        await conn.executemany(
            "INSERT INTO embedding_status (conversation_id, needs_reindex) VALUES (?, ?)",
            [("conv-1", 0), ("conv-2", 1)],
        )
        await conn.executemany(
            "INSERT INTO message_embeddings (message_id) VALUES (?)",
            [("msg-1",), ("msg-2",)],
        )
        await conn.commit()

        stats = await read_embedding_stats_async(conn)

    assert stats.embedded_conversations == 1
    assert stats.embedded_messages == 2
    assert stats.pending_conversations == 1


@pytest.mark.asyncio
async def test_read_embedding_stats_async_missing_tables_returns_zeroes() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        stats = await read_embedding_stats_async(conn)

    assert stats.embedded_conversations == 0
    assert stats.embedded_messages == 0
    assert stats.pending_conversations == 0

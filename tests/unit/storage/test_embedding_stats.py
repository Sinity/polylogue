"""Contracts for shared embedding-stats helpers."""

from __future__ import annotations

import sqlite3

import aiosqlite
import pytest

from polylogue.storage import embedding_stats as embedding_stats_mod
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


def test_read_embedding_stats_sync_exposes_retrieval_bands_when_archive_tables_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE conversations (conversation_id TEXT)")
        conn.executemany(
            "INSERT INTO conversations (conversation_id) VALUES (?)",
            [("conv-1",), ("conv-2",)],
        )
        conn.commit()

        monkeypatch.setattr(
            embedding_stats_mod,
            "action_event_read_model_status_sync",
            lambda _conn: {
                "count": 2,
                "action_fts_count": 2,
                "action_fts_ready": True,
                "stale_count": 0,
            },
        )
        monkeypatch.setattr(
            embedding_stats_mod,
            "session_product_status_sync",
            lambda _conn: {
                "profile_row_count": 2,
                "profile_evidence_fts_count": 2,
                "profile_evidence_fts_ready": True,
                "profile_evidence_fts_duplicate_count": 0,
                "work_event_inference_count": 2,
                "work_event_inference_fts_count": 2,
                "work_event_inference_fts_ready": True,
                "work_event_inference_fts_duplicate_count": 0,
                "phase_inference_count": 2,
                "phase_inference_rows_ready": True,
                "expected_phase_inference_count": 2,
                "stale_work_event_inference_count": 0,
                "stale_phase_inference_count": 0,
                "profile_inference_fts_count": 2,
                "profile_inference_fts_ready": True,
                "profile_inference_fts_duplicate_count": 0,
                "profile_enrichment_fts_count": 2,
                "profile_enrichment_fts_ready": True,
                "profile_enrichment_fts_duplicate_count": 0,
            },
        )

        stats = read_embedding_stats_sync(conn)
    finally:
        conn.close()

    assert set(stats.retrieval_bands) == {
        "transcript_embeddings",
        "evidence_retrieval",
        "inference_retrieval",
        "enrichment_retrieval",
    }
    assert stats.pending_conversations == 2
    assert stats.retrieval_bands["transcript_embeddings"]["pending_documents"] == 2
    assert "pending 2" in str(stats.retrieval_bands["transcript_embeddings"]["detail"])
    assert stats.retrieval_bands["evidence_retrieval"]["ready"] is True
    assert stats.retrieval_bands["inference_retrieval"]["ready"] is True
    assert stats.retrieval_bands["enrichment_retrieval"]["ready"] is True


def test_read_embedding_stats_sync_can_skip_retrieval_band_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_action_status(_conn):
        raise AssertionError("action-event status should not be read")

    def fail_session_status(_conn):
        raise AssertionError("session-product status should not be read")

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE conversations (conversation_id TEXT)")
        conn.executemany(
            "INSERT INTO conversations (conversation_id) VALUES (?)",
            [("conv-1",), ("conv-2",)],
        )
        conn.commit()

        monkeypatch.setattr(embedding_stats_mod, "action_event_read_model_status_sync", fail_action_status)
        monkeypatch.setattr(embedding_stats_mod, "session_product_status_sync", fail_session_status)

        stats = read_embedding_stats_sync(conn, include_retrieval_bands=False)
    finally:
        conn.close()

    assert stats.pending_conversations == 2
    assert stats.retrieval_bands == {}


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


@pytest.mark.asyncio
async def test_read_embedding_stats_async_derives_pending_from_total_conversations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_action_status(_conn):
        return {
            "count": 0,
            "action_fts_count": 0,
            "action_fts_ready": True,
            "stale_count": 0,
        }

    async def fake_session_status(_conn):
        return {
            "profile_row_count": 0,
            "profile_evidence_fts_count": 0,
            "profile_evidence_fts_ready": True,
            "profile_evidence_fts_duplicate_count": 0,
            "work_event_inference_count": 0,
            "work_event_inference_fts_count": 0,
            "work_event_inference_fts_ready": True,
            "work_event_inference_fts_duplicate_count": 0,
            "phase_inference_count": 0,
            "phase_inference_rows_ready": True,
            "expected_phase_inference_count": 0,
            "stale_work_event_inference_count": 0,
            "stale_phase_inference_count": 0,
            "profile_inference_fts_count": 0,
            "profile_inference_fts_ready": True,
            "profile_inference_fts_duplicate_count": 0,
            "profile_enrichment_fts_count": 0,
            "profile_enrichment_fts_ready": True,
            "profile_enrichment_fts_duplicate_count": 0,
        }

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE conversations (conversation_id TEXT)")
        await conn.executemany(
            "INSERT INTO conversations (conversation_id) VALUES (?)",
            [("conv-1",), ("conv-2",), ("conv-3",)],
        )
        await conn.commit()

        monkeypatch.setattr(embedding_stats_mod, "action_event_read_model_status_async", fake_action_status)
        monkeypatch.setattr(embedding_stats_mod, "session_product_status_async", fake_session_status)

        stats = await read_embedding_stats_async(conn)

    assert stats.pending_conversations == 3
    assert stats.retrieval_bands["transcript_embeddings"]["pending_documents"] == 3
    assert "pending 3" in str(stats.retrieval_bands["transcript_embeddings"]["detail"])


@pytest.mark.asyncio
async def test_read_embedding_stats_async_can_skip_retrieval_band_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_action_status(_conn):
        raise AssertionError("action-event status should not be read")

    async def fail_session_status(_conn):
        raise AssertionError("session-product status should not be read")

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE conversations (conversation_id TEXT)")
        await conn.executemany(
            "INSERT INTO conversations (conversation_id) VALUES (?)",
            [("conv-1",), ("conv-2",)],
        )
        await conn.commit()

        monkeypatch.setattr(embedding_stats_mod, "action_event_read_model_status_async", fail_action_status)
        monkeypatch.setattr(embedding_stats_mod, "session_product_status_async", fail_session_status)

        stats = await read_embedding_stats_async(conn, include_retrieval_bands=False)

    assert stats.pending_conversations == 2
    assert stats.retrieval_bands == {}

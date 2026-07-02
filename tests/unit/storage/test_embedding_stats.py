"""Contracts for shared embedding-stats helpers."""

from __future__ import annotations

import sqlite3

import aiosqlite
import pytest

from polylogue.storage.embeddings import embedding_stats as embedding_stats_mod
from polylogue.storage.embeddings.embedding_stats import (
    read_embedding_stats_async,
    read_embedding_stats_sync,
)
from polylogue.storage.insights.session.runtime import SessionInsightStatusSnapshot


def _create_embedding_stats_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE embedding_status (
            session_id TEXT PRIMARY KEY,
            message_count_embedded INTEGER NOT NULL DEFAULT 0,
            needs_reindex INTEGER NOT NULL,
            error_message TEXT
        );
        CREATE TABLE message_embeddings (message_id TEXT);
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY);
        CREATE TABLE messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            message_type TEXT NOT NULL,
            material_origin TEXT NOT NULL,
            word_count INTEGER NOT NULL,
            content_hash BLOB
        );
        """
    )


def _insert_prose_message(conn: sqlite3.Connection, session_id: str, message_id: str) -> None:
    conn.execute("INSERT OR IGNORE INTO sessions (session_id) VALUES (?)", (session_id,))
    conn.execute(
        """
        INSERT INTO messages (
            message_id, session_id, role, message_type, material_origin, word_count, content_hash
        ) VALUES (?, ?, 'user', 'message', 'human_authored', 6, zeroblob(32))
        """,
        (message_id, session_id),
    )


async def _create_embedding_stats_tables_async(conn: aiosqlite.Connection) -> None:
    await conn.executescript(
        """
        CREATE TABLE embedding_status (
            session_id TEXT PRIMARY KEY,
            message_count_embedded INTEGER NOT NULL DEFAULT 0,
            needs_reindex INTEGER NOT NULL,
            error_message TEXT
        );
        CREATE TABLE message_embeddings (message_id TEXT);
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY);
        CREATE TABLE messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            message_type TEXT NOT NULL,
            material_origin TEXT NOT NULL,
            word_count INTEGER NOT NULL,
            content_hash BLOB
        );
        """
    )


async def _insert_prose_message_async(conn: aiosqlite.Connection, session_id: str, message_id: str) -> None:
    await conn.execute("INSERT OR IGNORE INTO sessions (session_id) VALUES (?)", (session_id,))
    await conn.execute(
        """
        INSERT INTO messages (
            message_id, session_id, role, message_type, material_origin, word_count, content_hash
        ) VALUES (?, ?, 'user', 'message', 'human_authored', 6, zeroblob(32))
        """,
        (message_id, session_id),
    )


def test_read_embedding_stats_sync_missing_tables_returns_zeroes() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        stats = read_embedding_stats_sync(conn, include_retrieval_bands=False)
    finally:
        conn.close()

    assert stats.embedded_sessions == 0
    assert stats.embedded_messages == 0
    assert stats.pending_sessions == 0


def test_read_embedding_stats_sync_counts_available_tables() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        _create_embedding_stats_tables(conn)
        _insert_prose_message(conn, "conv-1", "msg-1")
        _insert_prose_message(conn, "conv-2", "msg-2")
        _insert_prose_message(conn, "conv-3", "msg-3")
        conn.executemany(
            "INSERT INTO embedding_status (session_id, message_count_embedded, needs_reindex) VALUES (?, ?, ?)",
            [("conv-1", 1, 0), ("conv-2", 1, 0), ("conv-3", 0, 1)],
        )
        conn.executemany(
            "INSERT INTO message_embeddings (message_id) VALUES (?)",
            [("msg-1",), ("msg-2",), ("msg-3",)],
        )
        conn.commit()

        stats = read_embedding_stats_sync(conn, include_retrieval_bands=False)
    finally:
        conn.close()

    assert stats.embedded_sessions == 2
    assert stats.embedded_messages == 3
    assert stats.pending_sessions == 1


def test_read_embedding_stats_counts_only_authored_prose_candidates() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        _create_embedding_stats_tables(conn)
        _insert_prose_message(conn, "conv-prose", "msg-prose")
        conn.execute("INSERT OR IGNORE INTO sessions (session_id) VALUES ('conv-tool')")
        conn.executemany(
            """
            INSERT INTO messages (
                message_id, session_id, role, message_type, material_origin, word_count, content_hash
            ) VALUES (?, 'conv-tool', ?, ?, ?, ?, zeroblob(32))
            """,
            [
                ("tool-use", "assistant", "tool_use", "assistant_authored", 10),
                ("tool-result", "tool", "tool_result", "tool_generated", 2000),
                ("protocol-context", "user", "message", "runtime_generated", 2000),
            ],
        )
        conn.commit()

        stats = read_embedding_stats_sync(conn, include_retrieval_bands=False)
    finally:
        conn.close()

    assert stats.pending_sessions == 1
    assert stats.pending_messages == 1
    assert stats.total_estimated_cost_usd == 0.0


def test_read_embedding_stats_sync_propagates_non_missing_operational_errors() -> None:
    class LockedConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: object = (), /) -> sqlite3.Cursor:  # pragma: no cover - trivial stub
            del sql, parameters
            raise sqlite3.OperationalError("database is locked")

    conn = sqlite3.connect(":memory:", factory=LockedConnection)
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        read_embedding_stats_sync(conn)
    conn.close()


def test_read_embedding_stats_sync_treats_missing_vec_module_as_optional() -> None:
    class VeclessConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: object = (), /) -> sqlite3.Cursor:
            del parameters
            if "message_embeddings" in sql:
                raise sqlite3.OperationalError("no such module: vec0")
            raise sqlite3.OperationalError("no such table: embedding_status")

    conn = sqlite3.connect(":memory:", factory=VeclessConnection)
    try:
        stats = read_embedding_stats_sync(conn)
    finally:
        conn.close()

    assert stats.embedded_sessions == 0
    assert stats.embedded_messages == 0
    assert stats.pending_sessions == 0


def test_read_embedding_stats_sync_exposes_retrieval_bands_when_archive_tables_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE sessions (session_id TEXT)")
        conn.execute(
            """
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message_type TEXT NOT NULL,
                material_origin TEXT NOT NULL,
                word_count INTEGER NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO sessions (session_id) VALUES (?)",
            [("conv-1",), ("conv-2",)],
        )
        conn.executemany(
            """
            INSERT INTO messages (message_id, session_id, role, message_type, material_origin, word_count)
            VALUES (?, ?, 'user', 'message', 'human_authored', 6)
            """,
            [("msg-1", "conv-1"), ("msg-2", "conv-2")],
        )
        conn.commit()

        monkeypatch.setattr(
            embedding_stats_mod,
            "session_insight_status_sync",
            lambda _conn: SessionInsightStatusSnapshot(
                profile_row_count=2,
                profile_evidence_fts_count=2,
                profile_evidence_fts_ready=True,
                profile_evidence_fts_duplicate_count=0,
                work_event_inference_count=2,
                work_event_inference_fts_count=2,
                work_event_inference_fts_ready=True,
                work_event_inference_fts_duplicate_count=0,
                phase_inference_count=2,
                phase_inference_rows_ready=True,
                expected_phase_inference_count=2,
                stale_work_event_inference_count=0,
                stale_phase_inference_count=0,
                profile_inference_fts_count=2,
                profile_inference_fts_ready=True,
                profile_inference_fts_duplicate_count=0,
                profile_enrichment_fts_count=2,
                profile_enrichment_fts_ready=True,
                profile_enrichment_fts_duplicate_count=0,
            ),
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
    assert stats.pending_sessions == 2
    assert stats.retrieval_bands["transcript_embeddings"]["pending_documents"] == 2
    assert "pending 2" in str(stats.retrieval_bands["transcript_embeddings"]["detail"])
    assert stats.retrieval_bands["evidence_retrieval"]["ready"] is True
    assert stats.retrieval_bands["inference_retrieval"]["ready"] is True
    assert stats.retrieval_bands["inference_retrieval"]["source_rows"] == 4
    assert stats.retrieval_bands["inference_retrieval"]["materialized_rows"] == 4
    assert "phase" not in str(stats.retrieval_bands["inference_retrieval"]["detail"])
    assert stats.retrieval_bands["enrichment_retrieval"]["ready"] is True


def test_read_embedding_stats_sync_can_skip_retrieval_band_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_session_status(_conn: object) -> None:
        raise AssertionError("session-insight status should not be read")

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE sessions (session_id TEXT)")
        conn.execute(
            """
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message_type TEXT NOT NULL,
                material_origin TEXT NOT NULL,
                word_count INTEGER NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO sessions (session_id) VALUES (?)",
            [("conv-1",), ("conv-2",)],
        )
        conn.executemany(
            """
            INSERT INTO messages (message_id, session_id, role, message_type, material_origin, word_count)
            VALUES (?, ?, 'user', 'message', 'human_authored', 6)
            """,
            [("msg-1", "conv-1"), ("msg-2", "conv-2")],
        )
        conn.commit()

        monkeypatch.setattr(embedding_stats_mod, "session_insight_status_sync", fail_session_status)

        stats = read_embedding_stats_sync(conn, include_retrieval_bands=False)
    finally:
        conn.close()

    assert stats.pending_sessions == 2
    assert stats.retrieval_bands == {}


@pytest.mark.asyncio
async def test_read_embedding_stats_async_counts_available_tables() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        await _create_embedding_stats_tables_async(conn)
        await _insert_prose_message_async(conn, "conv-1", "msg-1")
        await _insert_prose_message_async(conn, "conv-2", "msg-2")
        await conn.executemany(
            "INSERT INTO embedding_status (session_id, message_count_embedded, needs_reindex) VALUES (?, ?, ?)",
            [("conv-1", 1, 0), ("conv-2", 0, 1)],
        )
        await conn.executemany(
            "INSERT INTO message_embeddings (message_id) VALUES (?)",
            [("msg-1",), ("msg-2",)],
        )
        await conn.commit()

        stats = await read_embedding_stats_async(conn, include_retrieval_bands=False)

    assert stats.embedded_sessions == 1
    assert stats.embedded_messages == 2
    assert stats.pending_sessions == 1


@pytest.mark.asyncio
async def test_read_embedding_stats_async_missing_tables_returns_zeroes() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        stats = await read_embedding_stats_async(conn)

    assert stats.embedded_sessions == 0
    assert stats.embedded_messages == 0
    assert stats.pending_sessions == 0


@pytest.mark.asyncio
async def test_read_embedding_stats_async_does_not_derive_pending_from_session_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_session_status(_conn: object) -> SessionInsightStatusSnapshot:
        return SessionInsightStatusSnapshot(
            profile_row_count=0,
            profile_evidence_fts_count=0,
            profile_evidence_fts_ready=True,
            profile_evidence_fts_duplicate_count=0,
            work_event_inference_count=0,
            work_event_inference_fts_count=0,
            work_event_inference_fts_ready=True,
            work_event_inference_fts_duplicate_count=0,
            phase_inference_count=0,
            phase_inference_rows_ready=True,
            expected_phase_inference_count=0,
            stale_work_event_inference_count=0,
            stale_phase_inference_count=0,
            profile_inference_fts_count=0,
            profile_inference_fts_ready=True,
            profile_inference_fts_duplicate_count=0,
            profile_enrichment_fts_count=0,
            profile_enrichment_fts_ready=True,
            profile_enrichment_fts_duplicate_count=0,
        )

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE sessions (session_id TEXT)")
        await conn.executemany(
            "INSERT INTO sessions (session_id) VALUES (?)",
            [("conv-1",), ("conv-2",), ("conv-3",)],
        )
        await conn.commit()

        monkeypatch.setattr(embedding_stats_mod, "session_insight_status_async", fake_session_status)

        stats = await read_embedding_stats_async(conn)

    assert stats.pending_sessions == 0
    assert stats.retrieval_bands["transcript_embeddings"]["pending_documents"] == 0
    assert "pending 0" in str(stats.retrieval_bands["transcript_embeddings"]["detail"])


@pytest.mark.asyncio
async def test_read_embedding_stats_async_can_skip_retrieval_band_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_session_status(_conn: object) -> None:
        raise AssertionError("session-insight status should not be read")

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("CREATE TABLE sessions (session_id TEXT)")
        await conn.execute(
            """
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message_type TEXT NOT NULL,
                material_origin TEXT NOT NULL,
                word_count INTEGER NOT NULL
            )
            """
        )
        await conn.executemany(
            "INSERT INTO sessions (session_id) VALUES (?)",
            [("conv-1",), ("conv-2",)],
        )
        await conn.executemany(
            """
            INSERT INTO messages (message_id, session_id, role, message_type, material_origin, word_count)
            VALUES (?, ?, 'user', 'message', 'human_authored', 6)
            """,
            [("msg-1", "conv-1"), ("msg-2", "conv-2")],
        )
        await conn.commit()

        monkeypatch.setattr(embedding_stats_mod, "session_insight_status_async", fail_session_status)

        stats = await read_embedding_stats_async(conn, include_retrieval_bands=False)

    assert stats.pending_sessions == 2
    assert stats.retrieval_bands == {}

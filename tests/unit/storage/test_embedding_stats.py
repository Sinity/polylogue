"""Contracts for shared embedding-stats helpers."""

from __future__ import annotations

import sqlite3

import aiosqlite
import pytest

from polylogue.storage.derived.session.runtime import SessionInsightStatusSnapshot
from polylogue.storage.embeddings import embedding_stats as embedding_stats_mod
from polylogue.storage.embeddings.embedding_stats import (
    read_embedding_stats_async,
    read_embedding_stats_sync,
)


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


def test_read_embedding_stats_sync_reports_missing_vec_module_as_unmeasurable() -> None:
    """An unloadable vec0 extension is *cannot tell*, never a measured zero.

    Anti-vacuity: restoring ``"no such module: vec0"`` to
    ``is_missing_table_error`` (or dropping the ``EmbeddingCoverageUnmeasurableError``
    branch in ``read_embedding_stats_sync``) makes the counts 0 again and every
    assertion below fails. ``embeddings.db`` is re-purchased from a paid
    provider, so a false zero prescribes a full paid re-embed of intact vectors.
    """

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

    assert stats.embedded_sessions is None
    assert stats.embedded_messages is None
    assert stats.pending_sessions is None
    assert stats.coverage_measurable is False
    assert "no such module: vec0" in (stats.coverage_unmeasurable_reason or "")


def test_archive_stats_never_renders_unmeasurable_coverage_as_none_status() -> None:
    """The published ArchiveStats readiness must say ``unknown``, not ``none``.

    Anti-vacuity: reverting ``embedding_readiness_status`` to test
    ``embedded_messages <= 0`` raises TypeError on ``None``, and restoring the
    old int-zero default makes the status ``none`` -- the value that prescribes
    a paid backfill.
    """

    from polylogue.archive.stats import ArchiveStats

    stats = ArchiveStats(
        total_sessions=10,
        total_messages=100,
        embedded_sessions=None,
        embedded_messages=None,
        pending_embedding_sessions=None,
        embedding_coverage_unmeasurable_reason="no such module: vec0",
    )

    assert stats.embedding_readiness_status == "unknown"
    assert stats.retrieval_ready is None
    assert stats.embedding_coverage is None
    payload = stats.to_dict()
    assert payload["embedding_coverage_percent"] is None
    assert payload["embedding_coverage_measurable"] is False
    assert payload["embedded_messages"] is None


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
                profile_evidence_fts_duplicate_count=0,
                profile_inference_fts_count=2,
                profile_inference_fts_duplicate_count=0,
                profile_enrichment_fts_count=2,
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
    # polylogue-cuxz.7: the inference band is now the profile rows alone; the
    # work-event FTS contribution that used to double it is gone.
    assert stats.retrieval_bands["inference_retrieval"]["source_rows"] == 2
    assert stats.retrieval_bands["inference_retrieval"]["materialized_rows"] == 2
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
            profile_evidence_fts_duplicate_count=0,
            profile_inference_fts_count=0,
            profile_inference_fts_duplicate_count=0,
            profile_enrichment_fts_count=0,
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


def test_payload_reports_unknown_coverage_when_no_session_is_eligible() -> None:
    """polylogue-xtcdz: sessions exist but none is embedded, pending or blocked.

    The eligible denominator is zero, so nothing was weighed. The payload must
    say so (``status='unknown'``, null coverage) rather than advertise a
    ``complete``/100.0% archive that was never measured, and an archive with no
    sessions at all must still report the honest ``empty``.

    Anti-vacuity: restoring ``_coverage_percent``'s
    ``return 100.0 if total_sessions > 0 else 0.0`` zero-denominator branch, or
    dropping ``_embedding_status``'s ``embedded_sessions <= 0`` guard, turns
    this red on ``embedding_coverage_percent``/``status`` respectively.
    """

    from polylogue.storage.embeddings.models import EmbeddingStatsSnapshot
    from polylogue.storage.embeddings.status_payload import (
        EmbeddingStatusSettings,
        _payload_from_stats,
    )

    settings = EmbeddingStatusSettings(
        config_enabled=True,
        has_voyage_api_key=True,
        configured_model="voyage-4",
        configured_dimension=1024,
        monthly_cost_cap_usd=None,
    )

    payload = _payload_from_stats(
        settings=settings,
        total_sessions=8,
        stats=EmbeddingStatsSnapshot(embedded_sessions=0, embedded_messages=0, pending_sessions=0),
        latest_catchup_run=None,
        latest_material_catchup_run=None,
        pending_messages_exact=True,
    )

    assert payload["coverage_measurable"] is True
    assert payload["embedding_coverage_percent"] is None
    assert payload["status"] == "unknown"

    empty = _payload_from_stats(
        settings=settings,
        total_sessions=0,
        stats=EmbeddingStatsSnapshot(embedded_sessions=0, embedded_messages=0, pending_sessions=0),
        latest_catchup_run=None,
        latest_material_catchup_run=None,
        pending_messages_exact=True,
    )

    assert empty["status"] == "empty"
    assert empty["embedding_coverage_percent"] is None

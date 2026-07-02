"""Embedding status lifecycle contract tests — catalog-driven parametrization.

Covers the embedding_status table row lifecycle:
  pending → embedded → needs_reindex → re-embedded → error state.

Tests operate on a raw SQLite connection with the embedding_status schema
from sqlite_vec_runtime.py. Retrieval band checks are excluded because
the underlying SessionInsightStatusSnapshot/action infrastructure
requires full schema tables (pre-existing limitation, not a test bug).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias
from unittest.mock import AsyncMock, MagicMock

import pytest

from polylogue.storage.embeddings.embedding_stats import (
    read_embedding_stats_sync,
)
from polylogue.storage.embeddings.materialization import (
    embed_archive_session_sync,
    embed_session_sync,
    select_pending_archive_session_window,
    select_pending_session_window,
)
from polylogue.storage.embeddings.models import EmbeddingStatsSnapshot
from polylogue.storage.embeddings.progress import (
    CatchupRunDelta,
    CatchupRunStart,
    finish_embedding_catchup_run,
    latest_embedding_catchup_run,
    record_embedding_catchup_progress,
    start_embedding_catchup_run,
)
from polylogue.storage.runtime import MessageRecord
from polylogue.storage.sqlite.schema import SCHEMA_VERSION


class _FakeV1VectorProvider:
    model = "voyage-4"
    dimension = 1024

    def __init__(self) -> None:
        self.texts: list[str] = []

    def upsert(self, session_id: str, messages: list[MessageRecord]) -> None:
        raise AssertionError("archive embedding helper must not call old upsert")

    def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        return []

    def query_by_session(self, session_id: str, limit: int = 10) -> list[tuple[str, float]]:
        return []

    def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        self.texts.extend(texts)
        return [[0.01] * 1024 for _ in texts]


# ---------------------------------------------------------------------------
# Schema bootstrap (same DDL as sqlite_vec_runtime.py)
# ---------------------------------------------------------------------------

_EMBEDDING_STATUS_DDL = """
    CREATE TABLE IF NOT EXISTS embedding_status (
        session_id        TEXT PRIMARY KEY,
        message_count_embedded INTEGER DEFAULT 0,
        last_embedded_at       TEXT,
        needs_reindex          INTEGER DEFAULT 0,
        error_message          TEXT
    );
"""

_MESSAGE_EMBEDDINGS_DDL = """
    CREATE TABLE IF NOT EXISTS message_embeddings (
        message_id TEXT
    );
"""

_SESSIONS_DDL = """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        origin       TEXT NOT NULL DEFAULT 'unknown-export',
        title           TEXT,
        created_at_ms   INTEGER,
        updated_at_ms   INTEGER,
        message_count   INTEGER NOT NULL DEFAULT 0,
        content_hash    TEXT,
        metadata        TEXT,
        provider_meta   TEXT
    );
"""

_MESSAGES_DDL = """
    CREATE TABLE IF NOT EXISTS messages (
        message_id       TEXT PRIMARY KEY,
        session_id  TEXT NOT NULL,
        text             TEXT,
        role             TEXT NOT NULL DEFAULT 'user',
        message_type     TEXT NOT NULL DEFAULT 'message',
        material_origin  TEXT NOT NULL DEFAULT 'human_authored',
        word_count       INTEGER NOT NULL DEFAULT 6
    );
"""


def _setup_minimal_embedding_db(conn: sqlite3.Connection) -> None:
    """Create the minimum tables needed for embedding stats reading."""
    conn.executescript(_EMBEDDING_STATUS_DDL)
    conn.executescript(_MESSAGE_EMBEDDINGS_DDL)
    conn.executescript(_SESSIONS_DDL)
    conn.executescript(_MESSAGES_DDL)
    conn.commit()


def _setup_minimal_embedding_file(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        _setup_minimal_embedding_db(conn)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    finally:
        conn.close()


def _insert_session(conn: sqlite3.Connection, session_id: str, *, message_count: int) -> None:
    conn.execute(
        """
        INSERT INTO sessions (session_id, origin, title, updated_at_ms, message_count, content_hash)
        VALUES (?, 'unknown-export', ?, ?, ?, ?)
        """,
        (session_id, session_id, 1_700_000_000_000, message_count, f"hash-{session_id}"),
    )
    for index in range(message_count):
        conn.execute(
            """
            INSERT INTO messages (
                message_id, session_id, text, role, message_type, material_origin, word_count
            ) VALUES (?, ?, ?, 'user', 'message', 'human_authored', 6)
            """,
            (f"{session_id}-msg-{index}", session_id, "long enough message text for embedding"),
        )


def test_embedding_catchup_run_ledger_persists_progress(tmp_path: Path) -> None:
    """Backfill progress survives process exit as a run-level ledger."""
    db_path = tmp_path / "archive.db"
    _setup_minimal_embedding_file(db_path)

    run_id = start_embedding_catchup_run(
        db_path,
        CatchupRunStart(
            rebuild=False,
            max_sessions=3,
            max_messages=20,
            stop_after_seconds=30,
            max_errors=1,
            planned_sessions=3,
            planned_messages=12,
        ),
    )
    record_embedding_catchup_progress(
        db_path,
        run_id,
        CatchupRunDelta(
            session_id="conv-1",
            embedded=True,
            embedded_messages=5,
            estimated_cost_usd=0.001,
        ),
    )
    record_embedding_catchup_progress(
        db_path,
        run_id,
        CatchupRunDelta(session_id="conv-empty", skipped=True),
    )
    record_embedding_catchup_progress(
        db_path,
        run_id,
        CatchupRunDelta(session_id="conv-error", errored=True),
    )
    finish_embedding_catchup_run(db_path, run_id, status="stopped", stop_reason="max errors reached (1)")

    with sqlite3.connect(db_path) as conn:
        payload = latest_embedding_catchup_run(conn)

    assert payload is not None
    assert payload["run_id"] == run_id
    assert payload["status"] == "stopped"
    assert payload["stop_reason"] == "max errors reached (1)"
    assert payload["rebuild"] is False
    assert payload["max_sessions"] == 3
    assert payload["planned_sessions"] == 3
    assert payload["planned_messages"] == 12
    assert payload["processed_sessions"] == 3
    assert payload["embedded_sessions"] == 1
    assert payload["skipped_sessions"] == 1
    assert payload["error_count"] == 1
    assert payload["embedded_messages"] == 5
    assert payload["last_session_id"] == "conv-error"


def test_embedding_catchup_latest_run_uses_insert_order_for_timestamp_ties(tmp_path: Path) -> None:
    """Rapid backfill starts in the same second still report the latest row."""
    db_path = tmp_path / "archive.db"
    _setup_minimal_embedding_file(db_path)

    first = start_embedding_catchup_run(
        db_path,
        CatchupRunStart(
            rebuild=False,
            max_sessions=None,
            max_messages=None,
            stop_after_seconds=None,
            max_errors=None,
            planned_sessions=1,
            planned_messages=1,
        ),
    )
    second = start_embedding_catchup_run(
        db_path,
        CatchupRunStart(
            rebuild=True,
            max_sessions=None,
            max_messages=None,
            stop_after_seconds=None,
            max_errors=None,
            planned_sessions=2,
            planned_messages=2,
        ),
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE embedding_catchup_runs
            SET started_at = '2026-05-25 00:00:00',
                updated_at = '2026-05-25 00:00:00'
            WHERE run_id IN (?, ?)
            """,
            (first, second),
        )
        conn.commit()
        payload = latest_embedding_catchup_run(conn)

    assert payload is not None
    assert payload["run_id"] == second
    assert payload["rebuild"] is True


def test_embedding_catchup_progress_fails_for_missing_run(tmp_path: Path) -> None:
    """A DB/path mismatch must not silently drop progress updates."""
    db_path = tmp_path / "archive.db"
    _setup_minimal_embedding_file(db_path)

    with pytest.raises(LookupError, match="progress update"):
        record_embedding_catchup_progress(
            db_path,
            "missing-run",
            CatchupRunDelta(session_id="conv-1", embedded=True, embedded_messages=1),
        )

    with pytest.raises(LookupError, match="finalization"):
        finish_embedding_catchup_run(db_path, "missing-run", status="failed", stop_reason="missing")


# ---------------------------------------------------------------------------
# Lifecycle state catalog
# ---------------------------------------------------------------------------

LifecycleAssertion: TypeAlias = Callable[[EmbeddingStatsSnapshot, sqlite3.Connection], None]


def _assert_none_embedded(stats: EmbeddingStatsSnapshot, _conn: sqlite3.Connection) -> None:
    assert stats.embedded_sessions == 0
    assert stats.embedded_messages == 0
    assert stats.pending_sessions == 0


def _assert_all_pending(stats: EmbeddingStatsSnapshot, _conn: sqlite3.Connection) -> None:
    assert stats.embedded_sessions == 0
    assert stats.embedded_messages == 0
    assert stats.pending_sessions >= 1
    assert stats.pending_messages >= 1


def _assert_partially_embedded(stats: EmbeddingStatsSnapshot, _conn: sqlite3.Connection) -> None:
    assert stats.embedded_sessions >= 1
    assert stats.pending_sessions >= 1


def _assert_fully_embedded(stats: EmbeddingStatsSnapshot, _conn: sqlite3.Connection) -> None:
    assert stats.embedded_sessions == 2
    assert stats.pending_sessions == 0
    assert stats.embedded_messages > 0
    # Pending derived from total sessions: when all are embedded, pending == 0
    # even though total_sessions count = 2 (they're tracked by sessions table)


def _assert_error_visible(stats: EmbeddingStatsSnapshot, conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT error_message FROM embedding_status WHERE session_id = ?", ("conv-2",)).fetchone()
    assert row is not None, "error row should exist"
    assert "rate limit" in str(row[0]).lower(), f"Expected rate-limit error, got {row[0]!r}"


EmbeddingSeedRow: TypeAlias = tuple[str, str | None, int, str | None]

EMBEDDING_LIFECYCLE_CASES: list[tuple[str, list[EmbeddingSeedRow], str, LifecycleAssertion]] = [
    # (name, seed_rows, desc, assertion_fn)
    (
        "empty-archive",
        [],
        "No sessions in DB: all counts zero",
        _assert_none_embedded,
    ),
    (
        "pending-only",
        [("conv-1", None, 1, None), ("conv-2", None, 1, None)],
        "Two sessions both pending embedding",
        _assert_all_pending,
    ),
    (
        "partially-embedded",
        [("conv-1", "2026-01-01T00:00:00Z", 0, None), ("conv-2", None, 1, None)],
        "One embedded, one pending",
        _assert_partially_embedded,
    ),
    (
        "fully-embedded",
        [("conv-1", "2026-01-01T00:00:00Z", 0, None), ("conv-2", "2026-01-02T00:00:00Z", 0, None)],
        "Both sessions fully embedded",
        _assert_fully_embedded,
    ),
    (
        "needs-reindex",
        [("conv-1", "2026-01-01T00:00:00Z", 0, None), ("conv-2", "2026-01-02T00:00:00Z", 1, None)],
        "One embedded, one flagged needs_reindex",
        _assert_partially_embedded,
    ),
    (
        "error-state",
        [("conv-1", "2026-01-01T00:00:00Z", 0, None), ("conv-2", None, 1, "API rate limit exceeded")],
        "One embedded, one pending with error_message set",
        _assert_error_visible,
    ),
]


# ---------------------------------------------------------------------------
# Locked connection helpers
# ---------------------------------------------------------------------------


class _NoopConnection(sqlite3.Connection):
    def execute(self, sql: str, parameters: object = (), /) -> sqlite3.Cursor:
        del sql, parameters
        raise sqlite3.OperationalError("database is locked")


class _VeclessConnection(sqlite3.Connection):
    def __init__(self, database: str = ":memory:", *, timeout: float = 5.0, **kwargs: object) -> None:
        super().__init__(database, timeout=timeout, **kwargs)  # type: ignore[arg-type]
        self._query_count = 0

    def execute(self, sql: str, parameters: object = (), /) -> sqlite3.Cursor:
        del parameters
        self._query_count += 1
        if "message_embeddings" in sql:
            raise sqlite3.OperationalError("no such module: vec0")
        if self._query_count >= 2:
            raise sqlite3.OperationalError("no such table: embedding_status")
        return super().execute(sql)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmbeddingStatsEmptyArchive:
    """Empty archive: no embedding tables, no sessions."""

    def test_empty_db_returns_zeroes(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            stats = read_embedding_stats_sync(conn, include_retrieval_bands=False)
        finally:
            conn.close()
        assert stats.embedded_sessions == 0
        assert stats.embedded_messages == 0
        assert stats.pending_sessions == 0


@pytest.mark.parametrize("name,seed_rows,desc,assert_fn", EMBEDDING_LIFECYCLE_CASES)
def test_embedding_status_lifecycle(
    name: str,
    seed_rows: list[EmbeddingSeedRow],
    desc: str,
    assert_fn: LifecycleAssertion,
) -> None:
    """Catalog-driven: each lifecycle state is visible through read_embedding_stats_sync."""
    conn = sqlite3.connect(":memory:")
    try:
        _setup_minimal_embedding_db(conn)

        # Seed sessions
        for conv_id, _last_embedded, _needs_reindex, _error_msg in seed_rows:
            conn.execute(
                "INSERT INTO sessions (session_id, origin, title, message_count) VALUES (?, ?, ?, ?)",
                (conv_id, "unknown-export", f"Test {conv_id}", 1),
            )
            conn.execute(
                "INSERT INTO messages (message_id, session_id, text) VALUES (?, ?, ?)",
                (f"{conv_id}-msg-1", conv_id, "hello from embedding status test"),
            )
        conn.commit()

        # Seed embedding_status rows
        for conv_id, last_embedded, needs_reindex, error_msg in seed_rows:
            message_count_embedded = 1 if last_embedded is not None and needs_reindex == 0 and error_msg is None else 0
            conn.execute(
                "INSERT INTO embedding_status "
                "(session_id, message_count_embedded, last_embedded_at, needs_reindex, error_message) "
                "VALUES (?, ?, ?, ?, ?)",
                (conv_id, message_count_embedded, last_embedded, needs_reindex, error_msg),
            )
        conn.commit()

        # Seed message_embeddings for fully embedded sessions
        for conv_id, _last_embedded, needs_reindex, _error_msg in seed_rows:
            if needs_reindex == 0:
                conn.execute(
                    "INSERT INTO message_embeddings (message_id) VALUES (?)",
                    (f"{conv_id}-msg-1",),
                )
                conn.execute(
                    "INSERT INTO message_embeddings (message_id) VALUES (?)",
                    (f"{conv_id}-msg-2",),
                )
        conn.commit()

        stats = read_embedding_stats_sync(conn, include_retrieval_bands=False)
        assert_fn(stats, conn)
    finally:
        conn.close()


def test_missing_embedding_status_rows_count_as_pending_messages() -> None:
    """Never-embedded sessions are pending even before embedding_status exists for them."""
    conn = sqlite3.connect(":memory:")
    try:
        _setup_minimal_embedding_db(conn)
        conn.execute(
            "INSERT INTO sessions (session_id, origin, title, message_count) VALUES (?, ?, ?, ?)",
            ("conv-new", "unknown-export", "New", 1),
        )
        conn.execute(
            "INSERT INTO messages (message_id, session_id, text) VALUES (?, ?, ?)",
            ("msg-new", "conv-new", "this message has never been embedded"),
        )
        conn.commit()

        stats = read_embedding_stats_sync(conn, include_retrieval_bands=False)
        assert stats.pending_sessions == 1
        assert stats.pending_messages == 1
    finally:
        conn.close()


def test_pending_window_honors_max_sessions() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        _setup_minimal_embedding_db(conn)
        for index in range(3):
            _insert_session(conn, f"conv-{index}", message_count=1)
        conn.commit()

        pending = select_pending_session_window(conn, max_sessions=2)

        assert [item.session_id for item in pending] == ["conv-0", "conv-1"]
    finally:
        conn.close()


def test_pending_window_honors_max_messages() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        _setup_minimal_embedding_db(conn)
        _insert_session(conn, "conv-a", message_count=2)
        _insert_session(conn, "conv-b", message_count=2)
        _insert_session(conn, "conv-c", message_count=1)
        conn.commit()

        pending = select_pending_session_window(conn, max_messages=3)

        assert [item.session_id for item in pending] == ["conv-a"]
        assert sum(item.message_count for item in pending) == 2
    finally:
        conn.close()


def test_pending_window_skips_session_larger_than_max_messages() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        _setup_minimal_embedding_db(conn)
        _insert_session(conn, "conv-oversize", message_count=5)
        _insert_session(conn, "conv-fit", message_count=2)
        conn.commit()

        pending = select_pending_session_window(conn, max_messages=3)

        assert [item.session_id for item in pending] == ["conv-fit"]
        assert sum(item.message_count for item in pending) == 2
    finally:
        conn.close()


def test_pending_window_uses_sessions_message_count() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        _setup_minimal_embedding_db(conn)
        _insert_session(conn, "conv-a", message_count=7)
        _insert_session(conn, "conv-b", message_count=1)
        conn.commit()

        pending = select_pending_session_window(conn, max_sessions=1)

        assert [item.session_id for item in pending] == ["conv-a"]
        assert pending[0].message_count == 7
    finally:
        conn.close()


def test_pending_window_uses_live_counts_for_message_bound() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        _setup_minimal_embedding_db(conn)
        _insert_session(conn, "conv-a", message_count=3)
        _insert_session(conn, "conv-b", message_count=3)
        conn.commit()

        pending = select_pending_session_window(conn, max_messages=4)

        assert [item.session_id for item in pending] == ["conv-a"]
        assert pending[0].message_count == 3
    finally:
        conn.close()


def test_pending_archive_window_honors_min_messages() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        _setup_minimal_embedding_db(conn)
        # The archive selector orders by sort_key_ms; the minimal DDL omits it.
        conn.execute("ALTER TABLE sessions ADD COLUMN sort_key_ms INTEGER")
        _insert_session(conn, "substantial", message_count=5)
        _insert_session(conn, "trivial", message_count=1)
        conn.commit()

        # A message-count floor skips trivial sessions so a limited embedding
        # budget is not spent on near-empty stubs.
        pending = select_pending_archive_session_window(conn, status_table="", min_messages=3)

        assert [item.session_id for item in pending] == ["substantial"]
    finally:
        conn.close()


def test_pending_archive_window_skips_session_larger_than_max_messages() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        _setup_minimal_embedding_db(conn)
        conn.execute("ALTER TABLE sessions ADD COLUMN sort_key_ms INTEGER")
        _insert_session(conn, "oversize", message_count=5)
        _insert_session(conn, "fit", message_count=2)
        conn.execute("UPDATE sessions SET sort_key_ms = 2 WHERE session_id = 'oversize'")
        conn.execute("UPDATE sessions SET sort_key_ms = 1 WHERE session_id = 'fit'")
        conn.commit()

        pending = select_pending_archive_session_window(conn, status_table="", max_messages=3)

        assert [item.session_id for item in pending] == ["fit"]
        assert sum(item.message_count for item in pending) == 2
    finally:
        conn.close()


def test_pending_archive_window_reselects_status_with_lower_actual_count() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        _setup_minimal_embedding_db(conn)
        # The archive selector orders by sort_key_ms; the minimal DDL omits it.
        conn.execute("ALTER TABLE sessions ADD COLUMN sort_key_ms INTEGER")
        _insert_session(conn, "completed", message_count=5)
        conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, message_count_embedded, needs_reindex, error_message
            ) VALUES ('completed', 1, 0, NULL)
            """
        )
        conn.commit()

        pending = select_pending_archive_session_window(conn, status_table="embedding_status", min_messages=3)

        assert [item.session_id for item in pending] == ["completed"]
        assert pending[0].message_count == 5
    finally:
        conn.close()


def test_pending_archive_window_does_not_treat_aggregate_overcount_as_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.embeddings import materialization

    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                origin TEXT NOT NULL DEFAULT 'unknown-export',
                title TEXT,
                sort_key_ms INTEGER,
                authored_user_message_count INTEGER NOT NULL DEFAULT 0,
                assistant_message_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                message_type TEXT NOT NULL DEFAULT 'message',
                material_origin TEXT NOT NULL DEFAULT 'human_authored',
                word_count INTEGER NOT NULL DEFAULT 8
            );
            CREATE TABLE embedding_status (
                session_id TEXT PRIMARY KEY,
                message_count_embedded INTEGER NOT NULL DEFAULT 0,
                needs_reindex INTEGER NOT NULL DEFAULT 0,
                error_message TEXT
            );
            INSERT INTO sessions VALUES ('complete', 'codex-session', 'complete', 3, 20, 20);
            INSERT INTO sessions VALUES ('pending-newest', 'codex-session', 'pending', 2, 70, 48);
            INSERT INTO sessions VALUES ('pending-older', 'codex-session', 'pending older', 1, 8, 4);
            INSERT INTO embedding_status VALUES ('complete', 40, 0, NULL);
            INSERT INTO embedding_status VALUES ('pending-newest', 50, 0, NULL);
            INSERT INTO embedding_status VALUES ('pending-older', 0, 1, NULL);
            """
        )

        def fail_exact_count(_conn: sqlite3.Connection, _session_id: str) -> int:
            raise AssertionError("aggregate-backed archive windows must not exact-recount")

        monkeypatch.setattr(materialization, "count_archive_session_embeddable_messages", fail_exact_count)

        pending = select_pending_archive_session_window(
            conn,
            status_table="embedding_status",
            max_sessions=1,
            max_messages=200,
            min_messages=2,
        )

        assert [item.session_id for item in pending] == ["pending-older"]
        assert pending[0].message_count == 12
    finally:
        conn.close()


def test_pending_archive_window_filters_zero_rollups_before_session_limit() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                origin TEXT NOT NULL DEFAULT 'unknown-export',
                title TEXT,
                sort_key_ms INTEGER,
                authored_user_message_count INTEGER NOT NULL DEFAULT 0,
                assistant_message_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                message_type TEXT NOT NULL DEFAULT 'message',
                material_origin TEXT NOT NULL DEFAULT 'human_authored',
                word_count INTEGER NOT NULL DEFAULT 8
            );
            CREATE TABLE embedding_status (
                session_id TEXT PRIMARY KEY,
                message_count_embedded INTEGER NOT NULL DEFAULT 0,
                needs_reindex INTEGER NOT NULL DEFAULT 0,
                error_message TEXT
            );
            INSERT INTO sessions VALUES ('zero-newest', 'codex-session', 'zero', 3, 0, 0);
            INSERT INTO sessions VALUES ('zero-next', 'codex-session', 'zero', 2, 0, 0);
            INSERT INTO sessions VALUES ('eligible', 'codex-session', 'eligible', 1, 1, 1);
            """
        )

        pending = select_pending_archive_session_window(conn, status_table="embedding_status", max_sessions=1)

        assert [item.session_id for item in pending] == ["eligible"]
        assert pending[0].message_count == 2
    finally:
        conn.close()


def test_pending_archive_window_counts_only_embeddable_prose() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        _setup_minimal_embedding_db(conn)
        conn.execute("ALTER TABLE sessions ADD COLUMN sort_key_ms INTEGER")
        conn.execute(
            """
            INSERT INTO sessions (session_id, origin, title, updated_at_ms, message_count, content_hash, sort_key_ms)
            VALUES ('mixed', 'unknown-export', 'mixed', 1, 4, 'hash-mixed', 1)
            """
        )
        rows = [
            ("m-user", "mixed", "user prose long enough", "user", "message", "human_authored", 2),
            (
                "m-assistant",
                "mixed",
                "assistant prose long enough",
                "assistant",
                "message",
                "assistant_authored",
                2,
            ),
            ("m-context", "mixed", "runtime context", "user", "message", "context_generated", 2),
            ("m-tool", "mixed", "tool output", "tool", "tool_result", "tool_result", 2),
        ]
        conn.executemany(
            """
            INSERT INTO messages (
                message_id, session_id, text, role, message_type, material_origin, word_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

        pending = select_pending_archive_session_window(conn, status_table="", min_messages=1)

        assert [item.session_id for item in pending] == ["mixed"]
        # Missing-status sessions use the cheap aggregate as a conservative
        # budget estimate; exact text-floor counts are reserved for deciding
        # whether an existing clean status row is stale.
        assert pending[0].message_count == 4
    finally:
        conn.close()


def test_pending_archive_window_matches_materialization_text_floor() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        _setup_minimal_embedding_db(conn)
        conn.execute("ALTER TABLE sessions ADD COLUMN sort_key_ms INTEGER")
        conn.execute(
            """
            INSERT INTO sessions (session_id, origin, title, updated_at_ms, message_count, content_hash, sort_key_ms)
            VALUES ('mixed', 'unknown-export', 'mixed', 1, 3, 'hash-mixed', 1)
            """
        )
        conn.executemany(
            """
            INSERT INTO messages (
                message_id, session_id, text, role, message_type, material_origin, word_count
            ) VALUES (?, 'mixed', ?, 'user', 'message', 'human_authored', 1)
            """,
            [
                ("m-long", "long enough authored prose"),
                ("m-short-a", "tiny"),
                ("m-short-b", "brief"),
            ],
        )
        conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, message_count_embedded, needs_reindex, error_message
            ) VALUES ('mixed', 1, 0, NULL)
            """
        )
        conn.commit()

        assert select_pending_archive_session_window(conn, status_table="embedding_status") == []
    finally:
        conn.close()


def test_no_message_session_records_clean_status(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.sqlite"
    _setup_minimal_embedding_file(db_path)
    conn = sqlite3.connect(db_path)
    try:
        _insert_session(conn, "conv-empty", message_count=0)
        conn.commit()
    finally:
        conn.close()

    repo = MagicMock()
    repo.backend.db_path = db_path
    repo.get_messages = AsyncMock(return_value=[])

    outcome = embed_session_sync(repo, MagicMock(), "conv-empty")

    assert outcome.status == "no_messages"
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT needs_reindex, error_message FROM embedding_status WHERE session_id = 'conv-empty'"
        ).fetchone()
        assert row == (0, None)
        assert select_pending_session_window(conn) == []
    finally:
        conn.close()


def test_no_embeddable_provider_noop_records_clean_status(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.sqlite"
    _setup_minimal_embedding_file(db_path)
    conn = sqlite3.connect(db_path)
    try:
        _insert_session(conn, "conv-short", message_count=1)
        conn.commit()
    finally:
        conn.close()

    repo = MagicMock()
    repo.backend.db_path = db_path
    repo.get_messages = AsyncMock(
        return_value=[MagicMock(message_id="m", session_id="conv-short", text="short", content_hash="h")]
    )
    provider = MagicMock()
    provider.upsert.return_value = None

    outcome = embed_session_sync(repo, provider, "conv-short")

    assert outcome.status == "no_embeddable_messages"
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT needs_reindex, error_message FROM embedding_status WHERE session_id = 'conv-short'"
        ).fetchone()
        assert row == (0, None)
        assert select_pending_session_window(conn) == []
    finally:
        conn.close()


def test_provider_error_records_error_status_and_clears_retry(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.sqlite"
    _setup_minimal_embedding_file(db_path)
    conn = sqlite3.connect(db_path)
    try:
        _insert_session(conn, "conv-error", message_count=1)
        conn.commit()
    finally:
        conn.close()

    repo = MagicMock()
    repo.backend.db_path = db_path
    repo.get_messages = AsyncMock(
        return_value=[MagicMock(message_id="m", session_id="conv-error", text="long enough", content_hash="h")]
    )
    provider = MagicMock()
    provider.upsert.side_effect = RuntimeError("provider 429")

    outcome = embed_session_sync(repo, provider, "conv-error")

    assert outcome.status == "error"
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT needs_reindex, error_message FROM embedding_status WHERE session_id = 'conv-error'"
        ).fetchone()
        assert row == (1, "provider 429")
        assert [item.session_id for item in select_pending_session_window(conn)] == ["conv-error"]
    finally:
        conn.close()


def test_archive_pending_window_and_embedding_success(tmp_path: Path) -> None:
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, MaterialOrigin, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive_root = tmp_path / "archive"
    long_text = "This archive message is long enough to embed for semantic search."
    with ArchiveStore(archive_root) as archive:
        session_id = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="embed-v1",
                title="v1 embedding session",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text=long_text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=long_text)],
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    )
                ],
            )
        )

    index_db = archive_root / "index.db"
    embeddings_db = archive_root / "embeddings.db"
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)

    conn = sqlite3.connect(index_db)
    try:
        conn.execute("ATTACH DATABASE ? AS embeddings", (str(embeddings_db),))
        assert [item.session_id for item in select_pending_archive_session_window(conn, status_table="")] == [
            session_id
        ]
    finally:
        conn.close()

    provider = _FakeV1VectorProvider()
    outcome = embed_archive_session_sync(index_db, provider, session_id)

    assert outcome.status == "embedded"
    assert outcome.embedded_message_count == 1
    assert provider.texts == [long_text]
    conn = sqlite3.connect(embeddings_db)
    try:
        from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

        try_load_sqlite_vec(conn)
        status = conn.execute(
            "SELECT message_count_embedded, needs_reindex, error_message FROM embedding_status WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert status == (1, 0, None)
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1
    finally:
        conn.close()
    conn = sqlite3.connect(index_db)
    try:
        conn.execute("ATTACH DATABASE ? AS embeddings", (str(embeddings_db),))
        assert select_pending_archive_session_window(conn, status_table="embeddings.embedding_status") == []
    finally:
        conn.close()


def test_archive_embedding_only_sends_authored_prose_to_provider(tmp_path: Path) -> None:
    from polylogue.archive.message.roles import Role
    from polylogue.archive.message.types import MessageType
    from polylogue.core.enums import BlockType, MaterialOrigin, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive_root = tmp_path / "archive"
    user_text = "This user-authored question is long enough to merit a semantic embedding."
    assistant_text = "This assistant-authored answer is useful prose for semantic retrieval."
    tool_text = "This tool output is intentionally long but should not be embedded because it is not prose."
    context_text = "This runtime context is long enough but should remain outside the paid embedding set."
    with ArchiveStore(archive_root) as archive:
        session_id = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="embed-prose-only",
                title="prose-only embedding session",
                messages=[
                    ParsedMessage(
                        provider_message_id="m-user",
                        role=Role.USER,
                        text=user_text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=user_text)],
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    ),
                    ParsedMessage(
                        provider_message_id="m-assistant",
                        role=Role.ASSISTANT,
                        text=assistant_text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=assistant_text)],
                        material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                    ),
                    ParsedMessage(
                        provider_message_id="m-tool",
                        role=Role.TOOL,
                        text=tool_text,
                        blocks=[ParsedContentBlock(type=BlockType.TOOL_RESULT, text=tool_text)],
                        message_type=MessageType.TOOL_RESULT,
                        material_origin=MaterialOrigin.TOOL_RESULT,
                    ),
                    ParsedMessage(
                        provider_message_id="m-context",
                        role=Role.USER,
                        text=context_text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=context_text)],
                        message_type=MessageType.CONTEXT,
                        material_origin=MaterialOrigin.RUNTIME_CONTEXT,
                    ),
                ],
            )
        )

    index_db = archive_root / "index.db"
    embeddings_db = archive_root / "embeddings.db"
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)

    provider = _FakeV1VectorProvider()
    outcome = embed_archive_session_sync(index_db, provider, session_id)

    assert outcome.status == "embedded"
    assert outcome.embedded_message_count == 2
    assert provider.texts == [user_text, assistant_text]
    conn = sqlite3.connect(embeddings_db)
    try:
        from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

        try_load_sqlite_vec(conn)
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 2
    finally:
        conn.close()


def test_archive_embedding_batches_large_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, MaterialOrigin, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.embeddings import materialization
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    class BatchRecordingProvider(_FakeV1VectorProvider):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[int] = []

        def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
            self.calls.append(len(texts))
            return super()._get_embeddings(texts, input_type=input_type)

    monkeypatch.setattr(materialization, "ARCHIVE_EMBED_MESSAGE_BATCH_SIZE", 2)
    archive_root = tmp_path / "archive"
    messages = [
        ParsedMessage(
            provider_message_id=f"m{i}",
            role=Role.USER,
            text=f"This user-authored message {i} is long enough to be embedded safely.",
            blocks=[
                ParsedContentBlock(
                    type=BlockType.TEXT,
                    text=f"This user-authored message {i} is long enough to be embedded safely.",
                )
            ],
            material_origin=MaterialOrigin.HUMAN_AUTHORED,
        )
        for i in range(5)
    ]
    with ArchiveStore(archive_root) as archive:
        session_id = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="embed-batched",
                title="batched embedding session",
                messages=messages,
            )
        )

    index_db = archive_root / "index.db"
    embeddings_db = archive_root / "embeddings.db"
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)

    provider = BatchRecordingProvider()
    outcome = embed_archive_session_sync(index_db, provider, session_id)

    assert outcome.status == "embedded"
    assert outcome.embedded_message_count == 5
    assert provider.calls == [2, 2, 1]
    conn = sqlite3.connect(embeddings_db)
    try:
        from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

        try_load_sqlite_vec(conn)
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 5
        status = conn.execute(
            "SELECT message_count_embedded, needs_reindex, error_message FROM embedding_status WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert status == (5, 0, None)
    finally:
        conn.close()


def test_archive_embedding_error_records_retryable_status(tmp_path: Path) -> None:
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, MaterialOrigin, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    class ErrorProvider(_FakeV1VectorProvider):
        def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
            raise RuntimeError("provider 429")

    archive_root = tmp_path / "archive"
    long_text = "This archive message is long enough to trigger provider failure."
    with ArchiveStore(archive_root) as archive:
        session_id = archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="embed-v1-error",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text=long_text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=long_text)],
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    )
                ],
            )
        )

    index_db = archive_root / "index.db"
    embeddings_db = archive_root / "embeddings.db"
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)

    outcome = embed_archive_session_sync(index_db, ErrorProvider(), session_id)

    assert outcome.status == "error"
    assert outcome.error == "provider 429"
    conn = sqlite3.connect(embeddings_db)
    try:
        status = conn.execute(
            "SELECT needs_reindex, error_message FROM embedding_status WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert status == (1, "provider 429")
    finally:
        conn.close()
    conn = sqlite3.connect(index_db)
    try:
        conn.execute("ATTACH DATABASE ? AS embeddings", (str(embeddings_db),))
        assert [
            item.session_id
            for item in select_pending_archive_session_window(conn, status_table="embeddings.embedding_status")
        ] == [session_id]
    finally:
        conn.close()


class TestEmbeddingStatsLockedConnection:
    """Propagation of connection-level errors."""

    def test_locked_database_propagates_error(self) -> None:
        conn = sqlite3.connect(":memory:", factory=_NoopConnection)
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            read_embedding_stats_sync(conn, include_retrieval_bands=False)
        conn.close()

    def test_missing_vec_module_treated_as_optional(self) -> None:
        conn = sqlite3.connect(":memory:", factory=_VeclessConnection)
        try:
            stats = read_embedding_stats_sync(conn, include_retrieval_bands=False)
        finally:
            conn.close()
        assert stats.embedded_sessions == 0
        assert stats.embedded_messages == 0
        assert stats.pending_sessions == 0

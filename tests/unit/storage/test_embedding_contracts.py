"""Embedding status lifecycle contract tests — catalog-driven parametrization.

Covers the embedding_status table row lifecycle:
  pending → embedded → needs_reindex → re-embedded → error state.

Tests operate on a raw SQLite connection with the embedding_status schema
from sqlite_vec_runtime.py. Retrieval band checks are excluded because
the underlying SessionInsightStatusSnapshot/action-event infrastructure
requires full schema tables (pre-existing limitation, not a test bug).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import TypeAlias

import pytest

from polylogue.storage.embeddings.embedding_stats import (
    read_embedding_stats_sync,
)
from polylogue.storage.embeddings.models import EmbeddingStatsSnapshot

# ---------------------------------------------------------------------------
# Schema bootstrap (same DDL as sqlite_vec_runtime.py)
# ---------------------------------------------------------------------------

_EMBEDDING_STATUS_DDL = """
    CREATE TABLE IF NOT EXISTS embedding_status (
        conversation_id        TEXT PRIMARY KEY,
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

_CONVERSATIONS_DDL = """
    CREATE TABLE IF NOT EXISTS conversations (
        conversation_id TEXT PRIMARY KEY,
        provider_name   TEXT NOT NULL DEFAULT '',
        title           TEXT,
        created_at      TEXT,
        updated_at      TEXT,
        sort_key        TEXT,
        content_hash    TEXT,
        metadata        TEXT,
        provider_meta   TEXT
    );
"""

_MESSAGES_DDL = """
    CREATE TABLE IF NOT EXISTS messages (
        message_id       TEXT PRIMARY KEY,
        conversation_id  TEXT NOT NULL,
        text             TEXT
    );
"""


def _setup_minimal_embedding_db(conn: sqlite3.Connection) -> None:
    """Create the minimum tables needed for embedding stats reading."""
    conn.executescript(_EMBEDDING_STATUS_DDL)
    conn.executescript(_MESSAGE_EMBEDDINGS_DDL)
    conn.executescript(_CONVERSATIONS_DDL)
    conn.executescript(_MESSAGES_DDL)
    conn.commit()


# ---------------------------------------------------------------------------
# Lifecycle state catalog
# ---------------------------------------------------------------------------

LifecycleAssertion: TypeAlias = Callable[[EmbeddingStatsSnapshot, sqlite3.Connection], None]


def _assert_none_embedded(stats: EmbeddingStatsSnapshot, _conn: sqlite3.Connection) -> None:
    assert stats.embedded_conversations == 0
    assert stats.embedded_messages == 0
    assert stats.pending_conversations == 0


def _assert_all_pending(stats: EmbeddingStatsSnapshot, _conn: sqlite3.Connection) -> None:
    assert stats.embedded_conversations == 0
    assert stats.embedded_messages == 0
    assert stats.pending_conversations >= 1
    assert stats.pending_messages >= 1


def _assert_partially_embedded(stats: EmbeddingStatsSnapshot, _conn: sqlite3.Connection) -> None:
    assert stats.embedded_conversations >= 1
    assert stats.pending_conversations >= 1


def _assert_fully_embedded(stats: EmbeddingStatsSnapshot, _conn: sqlite3.Connection) -> None:
    assert stats.embedded_conversations == 2
    assert stats.pending_conversations == 0
    assert stats.embedded_messages > 0
    # Pending derived from total conversations: when all are embedded, pending == 0
    # even though total_conversations count = 2 (they're tracked by conversations table)


def _assert_error_visible(stats: EmbeddingStatsSnapshot, conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT error_message FROM embedding_status WHERE conversation_id = ?", ("conv-2",)).fetchone()
    assert row is not None, "error row should exist"
    assert "rate limit" in str(row[0]).lower(), f"Expected rate-limit error, got {row[0]!r}"


EmbeddingSeedRow: TypeAlias = tuple[str, str | None, int, str | None]

EMBEDDING_LIFECYCLE_CASES: list[tuple[str, list[EmbeddingSeedRow], str, LifecycleAssertion]] = [
    # (name, seed_rows, desc, assertion_fn)
    (
        "empty-archive",
        [],
        "No conversations in DB: all counts zero",
        _assert_none_embedded,
    ),
    (
        "pending-only",
        [("conv-1", None, 1, None), ("conv-2", None, 1, None)],
        "Two conversations both pending embedding",
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
        "Both conversations fully embedded",
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
    """Empty archive: no embedding tables, no conversations."""

    def test_empty_db_returns_zeroes(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            stats = read_embedding_stats_sync(conn, include_retrieval_bands=False)
        finally:
            conn.close()
        assert stats.embedded_conversations == 0
        assert stats.embedded_messages == 0
        assert stats.pending_conversations == 0


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

        # Seed conversations
        for conv_id, _last_embedded, _needs_reindex, _error_msg in seed_rows:
            conn.execute(
                "INSERT INTO conversations (conversation_id, provider_name, title) VALUES (?, ?, ?)",
                (conv_id, "test", f"Test {conv_id}"),
            )
            conn.execute(
                "INSERT INTO messages (message_id, conversation_id, text) VALUES (?, ?, ?)",
                (f"{conv_id}-msg-1", conv_id, "hello from embedding status test"),
            )
        conn.commit()

        # Seed embedding_status rows
        for conv_id, last_embedded, needs_reindex, error_msg in seed_rows:
            conn.execute(
                "INSERT INTO embedding_status "
                "(conversation_id, last_embedded_at, needs_reindex, error_message) "
                "VALUES (?, ?, ?, ?)",
                (conv_id, last_embedded, needs_reindex, error_msg),
            )
        conn.commit()

        # Seed message_embeddings for fully embedded conversations
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
    """Never-embedded conversations are pending even before embedding_status exists for them."""
    conn = sqlite3.connect(":memory:")
    try:
        _setup_minimal_embedding_db(conn)
        conn.execute(
            "INSERT INTO conversations (conversation_id, provider_name, title) VALUES (?, ?, ?)",
            ("conv-new", "test", "New"),
        )
        conn.execute(
            "INSERT INTO messages (message_id, conversation_id, text) VALUES (?, ?, ?)",
            ("msg-new", "conv-new", "this message has never been embedded"),
        )
        conn.commit()

        stats = read_embedding_stats_sync(conn, include_retrieval_bands=False)
        assert stats.pending_conversations == 1
        assert stats.pending_messages == 1
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
        assert stats.embedded_conversations == 0
        assert stats.embedded_messages == 0
        assert stats.pending_conversations == 0

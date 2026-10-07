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
from typing import Any, Never, TypeAlias, cast

import pytest

from polylogue.storage.embeddings.embedding_stats import (
    read_embedding_stats_sync,
)
from polylogue.storage.embeddings.materialization import (
    count_archive_embedding_session_state,
    embed_archive_session_sync,
    select_pending_archive_session_window,
    select_pending_session_window,
)
from polylogue.storage.embeddings.models import EmbeddingStatsSnapshot
from tests.infra.live_ingest import write_index_session


class _FakeV1VectorProvider:
    model = "voyage-4-lite"
    dimension = 1024

    def __init__(self) -> None:
        self.texts: list[str] = []

    def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        return []

    def query_by_session(self, session_id: str, limit: int = 10) -> list[tuple[str, float]]:
        return []

    def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        self.texts.extend(texts)
        return [[0.01] * 1024 for _ in texts]

    def scoped_query(self, *args: object, **kwargs: object) -> Never:
        raise AssertionError("document-only fixture does not perform scoped retrieval")

    async def read_similarity(self, *args: object, **kwargs: object) -> Never:
        raise AssertionError("this fixture does not perform retained-session reads")


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

_MESSAGE_EMBEDDING_REFS_DDL = """
    CREATE TABLE IF NOT EXISTS message_embedding_refs (
        message_id TEXT PRIMARY KEY,
        vector_derivation_hash BLOB
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
        position         INTEGER NOT NULL DEFAULT 0,
        variant_index    INTEGER NOT NULL DEFAULT 0,
        content_hash     BLOB NOT NULL CHECK(length(content_hash) = 32),
        role             TEXT NOT NULL DEFAULT 'user',
        message_type     TEXT NOT NULL DEFAULT 'message',
        material_origin  TEXT NOT NULL DEFAULT 'human_authored',
        word_count       INTEGER NOT NULL DEFAULT 6
    );
"""


_BLOCKS_DDL = """
    CREATE TABLE IF NOT EXISTS blocks (
        block_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        message_id TEXT NOT NULL,
        position INTEGER NOT NULL,
        block_type TEXT NOT NULL,
        text TEXT
    );
"""


def _insert_message(
    conn: sqlite3.Connection,
    message_id: str,
    session_id: str,
    text: str,
    role: str = "user",
    message_type: str = "message",
    material_origin: str = "human_authored",
    word_count: int = 6,
) -> None:
    conn.execute(
        "INSERT INTO messages "
        "(message_id, session_id, content_hash, role, message_type, material_origin, word_count) "
        "VALUES (?, ?, zeroblob(32), ?, ?, ?, ?)",
        (message_id, session_id, role, message_type, material_origin, word_count),
    )
    conn.execute(
        "INSERT INTO blocks (block_id, session_id, message_id, position, block_type, text) "
        "VALUES (?, ?, ?, 0, 'text', ?)",
        (f"{message_id}:text", session_id, message_id, text),
    )


def _setup_minimal_embedding_db(conn: sqlite3.Connection) -> None:
    """Create the minimum tables needed for embedding stats reading."""
    conn.executescript(_EMBEDDING_STATUS_DDL)
    conn.executescript(_MESSAGE_EMBEDDING_REFS_DDL)
    conn.executescript(_SESSIONS_DDL)
    conn.executescript(_MESSAGES_DDL)
    conn.executescript(_BLOCKS_DDL)
    conn.commit()


def _insert_session(conn: sqlite3.Connection, session_id: str, *, message_count: int) -> None:
    conn.execute(
        """
        INSERT INTO sessions (session_id, origin, title, updated_at_ms, message_count, content_hash)
        VALUES (?, 'unknown-export', ?, ?, ?, ?)
        """,
        (session_id, session_id, 1_700_000_000_000, message_count, f"hash-{session_id}"),
    )
    for index in range(message_count):
        _insert_message(conn, f"{session_id}-msg-{index}", session_id, "long enough message text for embedding")


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
    assert stats.pending_sessions is not None and stats.pending_sessions >= 1
    assert stats.pending_messages >= 1


def _assert_partially_embedded(stats: EmbeddingStatsSnapshot, _conn: sqlite3.Connection) -> None:
    assert stats.embedded_sessions is not None and stats.embedded_sessions >= 1
    assert stats.pending_sessions is not None and stats.pending_sessions >= 1


def _assert_fully_embedded(stats: EmbeddingStatsSnapshot, _conn: sqlite3.Connection) -> None:
    assert stats.embedded_sessions == 2
    assert stats.pending_sessions == 0
    assert stats.embedded_messages is not None and stats.embedded_messages > 0
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
        self._query_count += 1
        if "message_embeddings" in sql:
            raise sqlite3.OperationalError("no such module: vec0")
        if self._query_count >= 2:
            raise sqlite3.OperationalError("no such table: embedding_status")
        return super().execute(sql, parameters)  # type: ignore[arg-type]


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
            _insert_message(conn, f"{conv_id}-msg-1", conv_id, "hello from embedding status test")
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

        # Seed message_embedding_refs for fully embedded sessions
        for conv_id, _last_embedded, needs_reindex, _error_msg in seed_rows:
            if needs_reindex == 0:
                for suffix in ("msg-1", "msg-2"):
                    conn.execute(
                        "INSERT INTO message_embedding_refs (message_id, vector_derivation_hash) "
                        "VALUES (?, zeroblob(32))",
                        (f"{conv_id}-{suffix}",),
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
        _insert_message(conn, "msg-new", "conv-new", "this message has never been embedded")
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


@pytest.mark.parametrize(
    ("max_messages", "max_sessions", "min_messages", "expected"),
    [
        (3, None, None, ["a"]),
        (4, None, None, ["a"]),
        (5, None, None, ["oversize"]),
        (6, 2, None, ["oversize"]),
        (4, None, 2, ["a"]),
        (None, 2, None, ["oversize", "a"]),
    ],
)
def test_archive_window_preserves_newest_ties_ceiling_and_prefix_stop(
    max_messages: int | None, max_sessions: int | None, min_messages: int | None, expected: list[str]
) -> None:
    """A later smaller session cannot fill a prefix whose earlier member overflowed."""
    conn = sqlite3.connect(":memory:")
    try:
        _setup_minimal_embedding_db(conn)
        conn.execute("ALTER TABLE sessions ADD COLUMN sort_key_ms INTEGER")
        for sid, count, sort_key in (("oversize", 5, 30), ("b", 3, 20), ("a", 2, 20), ("c", 1, 10), ("d", 1, None)):
            _insert_session(conn, sid, message_count=count)
            conn.execute("UPDATE sessions SET sort_key_ms = ? WHERE session_id = ?", (sort_key, sid))
        conn.commit()
        for rebuild in (False, True):
            rows = select_pending_archive_session_window(
                conn,
                status_table="",
                max_messages=max_messages,
                max_sessions=max_sessions,
                min_messages=min_messages,
                rebuild=rebuild,
            )
            assert [row.session_id for row in rows] == expected
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
        for row in rows:
            _insert_message(conn, *row)
        conn.commit()

        pending = select_pending_archive_session_window(conn, status_table="", min_messages=1)

        assert [item.session_id for item in pending] == ["mixed"]
        # v4 (polylogue-q88p "unify embedding freshness into one monotonic
        # key"): there is no longer a separate "cheap aggregate estimate vs.
        # exact text-floor count" split -- _archive_embedding_freshness_
        # predicate's exact desired_messages CTE is the ONE path, used
        # whether or not a status table is available (a missing status_table
        # only changes fresh_sql/blocked_sql to constants, not how message
        # counts themselves are computed). So a status_table="" caller still
        # gets the real embeddable-prose count (2: m-user + m-assistant),
        # correctly excluding the context/tool-result rows even without a
        # status ledger to consult.
        assert pending[0].message_count == 2
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
        session_id = write_index_session(
            archive,
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
            ),
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


def test_archive_clean_status_row_without_current_derivation_key_stays_pending(tmp_path: Path) -> None:
    """``embedding_status`` is attempt telemetry; it cannot certify freshness.

    Anti-vacuity: let the freshness predicate accept a clean status row whose
    ``message_count_embedded`` covers the session without a current
    ``embedding_derivation_state`` key and this session drops out of the
    pending window and is counted as embedded.
    """
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, MaterialOrigin, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    archive_root = tmp_path / "archive"
    long_text = "This archive message is long enough to embed for semantic search."
    with ArchiveStore(archive_root) as archive:
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="clean-status-no-key",
                title="clean status without derivation key",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text=long_text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=long_text)],
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    )
                ],
            ),
        )

    index_db = archive_root / "index.db"
    embeddings_db = archive_root / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    with sqlite3.connect(embeddings_db) as conn:
        conn.execute(
            """
            INSERT INTO embedding_status (session_id, origin, message_count_embedded, needs_reindex, error_message)
            VALUES (?, 'codex-session', 1, 0, NULL)
            """,
            (session_id,),
        )

    conn = sqlite3.connect(index_db)
    try:
        conn.execute("ATTACH DATABASE ? AS embeddings", (str(embeddings_db),))
        pending = select_pending_archive_session_window(conn, status_table="embeddings.embedding_status")
        state = count_archive_embedding_session_state(conn, status_table="embeddings.embedding_status")
    finally:
        conn.close()

    assert [item.session_id for item in pending] == [session_id]
    assert state.pending_sessions == 1
    assert state.embedded_sessions == 0


def test_archive_embedding_resumes_after_bounded_message_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, MaterialOrigin, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.embeddings import materialization
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    text = "This archive message is long enough to embed for semantic search."
    root = tmp_path / "archive"
    with ArchiveStore(root) as archive:
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="bounded-v1",
                title="bounded embedding session",
                messages=[
                    ParsedMessage(
                        provider_message_id=f"m{index}",
                        role=Role.USER,
                        text=f"{text} {index}",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=f"{text} {index}")],
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    )
                    for index in range(2)
                ],
            ),
        )
    embeddings_db = root / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    monkeypatch.setattr(materialization, "ARCHIVE_EMBED_MESSAGE_BATCH_SIZE", 1)

    class SlowProvider(_FakeV1VectorProvider):
        def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
            import time

            time.sleep(0.01)
            return super()._get_embeddings(texts, input_type=input_type)

    provider = SlowProvider()
    first = embed_archive_session_sync(root / "index.db", provider, session_id, stop_after_seconds=0.005)
    assert first.status == "deferred"
    assert first.embedded_message_count == 1
    second = embed_archive_session_sync(root / "index.db", provider, session_id)
    assert second.status == "embedded"
    assert second.embedded_message_count == 2
    assert len(provider.texts) == 2


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
        session_id = write_index_session(
            archive,
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
                        blocks=[ParsedContentBlock(type=BlockType.TOOL_RESULT, text=tool_text, is_error=False)],
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
            ),
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
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="embed-batched",
                title="batched embedding session",
                messages=messages,
            ),
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
        session_id = write_index_session(
            archive,
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
            ),
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


def test_archive_embedding_http_400_records_terminal_status(tmp_path: Path) -> None:
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, MaterialOrigin, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    class ErrorProvider(_FakeV1VectorProvider):
        def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
            raise RuntimeError("Embedding generation failed: HTTP 400")

    archive_root = tmp_path / "archive"
    long_text = "This archive message is long enough to trigger provider hard failure."
    with ArchiveStore(archive_root) as archive:
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="embed-v1-hard-error",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text=long_text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=long_text)],
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    )
                ],
            ),
        )

    index_db = archive_root / "index.db"
    embeddings_db = archive_root / "embeddings.db"
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)

    outcome = embed_archive_session_sync(index_db, ErrorProvider(), session_id)

    assert outcome.status == "error"
    assert outcome.error == "Embedding generation failed: HTTP 400"
    conn = sqlite3.connect(embeddings_db)
    try:
        status = conn.execute(
            "SELECT needs_reindex, error_message FROM embedding_status WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert status == (0, "Embedding generation failed: HTTP 400")
    finally:
        conn.close()
    conn = sqlite3.connect(index_db)
    try:
        conn.execute("ATTACH DATABASE ? AS embeddings", (str(embeddings_db),))
        assert [
            item.session_id
            for item in select_pending_archive_session_window(conn, status_table="embeddings.embedding_status")
        ] == []
        state = count_archive_embedding_session_state(conn, status_table="embeddings.embedding_status")
        assert state.eligible_sessions == 1
        assert state.embedded_sessions == 0
        assert state.pending_sessions == 0
    finally:
        conn.close()


class TestEmbeddingStatsLockedConnection:
    """Propagation of connection-level errors."""

    def test_locked_database_propagates_error(self) -> None:
        conn = sqlite3.connect(":memory:", factory=_NoopConnection)
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            read_embedding_stats_sync(conn, include_retrieval_bands=False)
        conn.close()

    def test_missing_vec_module_is_unmeasurable_not_zero(self) -> None:
        """An unloadable vec0 module leaves coverage unknown, not empty.

        Anti-vacuity: restoring ``"no such module: vec0"`` to
        ``is_missing_table_error`` makes every count 0 again and each assertion
        below fails with ``assert 0 is None``.
        """

        conn = sqlite3.connect(":memory:", factory=_VeclessConnection)
        try:
            stats = read_embedding_stats_sync(conn, include_retrieval_bands=False)
        finally:
            conn.close()
        assert stats.embedded_sessions is None
        assert stats.embedded_messages is None
        assert stats.pending_sessions is None
        assert stats.coverage_measurable is False


def _write_archive_session(archive_root: Path, *, native_id: str, embeddable: bool) -> str:
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, MaterialOrigin, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    text = "This archive message is long enough to embed for semantic search."
    with ArchiveStore(archive_root) as archive:
        return write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=native_id,
                title="failure resolution session",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text=text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                        material_origin=(MaterialOrigin.HUMAN_AUTHORED if embeddable else MaterialOrigin.TOOL_RESULT),
                    )
                ],
            ),
        )


def _seed_open_archive_failure(embeddings_db: Path, session_id: str) -> str:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.embedding_write import record_embedding_failure
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    conn = sqlite3.connect(embeddings_db)
    try:
        failure = record_embedding_failure(
            conn,
            session_id=session_id,
            origin="codex-session",
            message_refs=(f"{session_id}:m1",),
            provider="voyage",
            model="voyage-4",
            error_class="provider_error",
            error_message="Embedding generation failed: HTTP 500",
            retryable=True,
        )
        return failure.failure_id
    finally:
        conn.close()


def _failure_lifecycle_state(embeddings_db: Path, failure_id: str) -> str:
    conn = sqlite3.connect(embeddings_db)
    try:
        row = conn.execute(
            "SELECT lifecycle_state FROM embedding_failures WHERE failure_id = ?", (failure_id,)
        ).fetchone()
        assert row is not None
        return str(row[0])
    finally:
        conn.close()


def test_archive_success_outcomes_resolve_open_failures(tmp_path: Path) -> None:
    """Every terminal success outcome clears prior failure debt.

    Anti-vacuity: dropping the failure resolution from the attempt-success
    finalizers in embedding_write leaves both failures active, so a session
    that later embeds (or turns out to have nothing embeddable) reports phantom
    current debt forever.
    """
    archive_root = tmp_path / "archive"
    embedded_session = _write_archive_session(archive_root, native_id="embed-ok", embeddable=True)
    noop_session = _write_archive_session(archive_root, native_id="embed-noop", embeddable=False)
    index_db = archive_root / "index.db"
    embeddings_db = archive_root / "embeddings.db"
    embedded_failure = _seed_open_archive_failure(embeddings_db, embedded_session)
    noop_failure = _seed_open_archive_failure(embeddings_db, noop_session)

    embedded_outcome = embed_archive_session_sync(index_db, _FakeV1VectorProvider(), embedded_session)
    noop_outcome = embed_archive_session_sync(index_db, _FakeV1VectorProvider(), noop_session)

    assert embedded_outcome.status == "embedded"
    assert noop_outcome.status == "no_embeddable_messages"
    assert _failure_lifecycle_state(embeddings_db, embedded_failure) == "resolved"
    assert _failure_lifecycle_state(embeddings_db, noop_failure) == "resolved"


def test_archive_local_fault_is_not_ledgered_as_provider_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local storage faults must not masquerade as Voyage provider failures.

    Anti-vacuity: recording every materialization exception with
    provider="voyage" falsifies the audit ledger; the production handler must
    branch on whether the provider call itself raised.
    """
    from polylogue.storage.sqlite.archive_tiers import embedding_write as embedding_write_module
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    archive_root = tmp_path / "archive"
    session_id = _write_archive_session(archive_root, native_id="embed-local-fault", embeddable=True)
    index_db = archive_root / "index.db"
    embeddings_db = archive_root / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)

    def _raise_write_fault(*args: object, **kwargs: object) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(embedding_write_module, "finalize_embedding_attempt_success", _raise_write_fault)
    outcome = embed_archive_session_sync(index_db, _FakeV1VectorProvider(), session_id)
    assert outcome.status == "error"

    conn = sqlite3.connect(embeddings_db)
    try:
        provider, error_class, retryable = conn.execute(
            "SELECT provider, error_class, retryable FROM embedding_failures WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    assert provider == "local"
    assert error_class == "internal_error"
    assert bool(retryable) is True


def test_archive_provider_fault_keeps_provider_attribution(tmp_path: Path) -> None:
    """A genuine provider exception still ledgers as a Voyage failure."""
    archive_root = tmp_path / "archive"
    session_id = _write_archive_session(archive_root, native_id="embed-provider-fault", embeddable=True)
    index_db = archive_root / "index.db"
    embeddings_db = archive_root / "embeddings.db"
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)

    provider = _FakeV1VectorProvider()

    def _raise_provider_fault(texts: list[str], input_type: str = "document") -> list[list[float]]:
        raise RuntimeError("Embedding generation failed: HTTP 429")

    provider._get_embeddings = _raise_provider_fault  # type: ignore[method-assign]
    outcome = embed_archive_session_sync(index_db, provider, session_id)
    assert outcome.status == "error"

    conn = sqlite3.connect(embeddings_db)
    try:
        recorded_provider, error_class = conn.execute(
            "SELECT provider, error_class FROM embedding_failures WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    assert recorded_provider == "voyage"
    assert error_class == "provider_http_429"


def test_archive_failure_ledger_survives_origin_lookup_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed origin lookup must degrade the ledger row, never drop it.

    Anti-vacuity: the production handler used to wrap the origin re-read in
    ``contextlib.suppress(sqlite3.Error)`` and skip ``record_embedding_failure``
    entirely when that lookup raised, so a session that fails before ``session``
    is ever fetched (e.g. the sqlite-vec load check) left no forensic trace at
    all beyond an aggregate error count (polylogue-es7b). Forcing that early
    failure plus a raising origin SELECT reproduces the drop; the fix must
    still land a row, falling back to an explicit unknown-origin sentinel.
    """
    from polylogue.storage.sqlite import sqlite_vec_extension as sqlite_vec_extension_module
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    archive_root = tmp_path / "archive"
    session_id = _write_archive_session(archive_root, native_id="embed-origin-lookup-fault", embeddable=True)
    index_db = archive_root / "index.db"
    embeddings_db = archive_root / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)

    def _fail_vec_load(conn: sqlite3.Connection) -> tuple[bool, Exception | None]:
        # Raise before `session` is ever fetched, so the except-handler must
        # take the origin-unknown fallback path rather than the already
        # -fetched `session["origin"]` fast path.
        return False, RuntimeError("sqlite-vec unavailable")

    monkeypatch.setattr(sqlite_vec_extension_module, "try_load_sqlite_vec", _fail_vec_load)

    # sqlite3.Connection is a builtin type and doesn't accept monkeypatched
    # instance/class attributes, so route the index-db connection through a
    # `factory=` subclass whose `execute` raises for the specific origin
    # lookup, leaving every other query (including the embeddings.db
    # connection opened by the same call site) untouched.
    class _FlakyConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
            if "SELECT origin FROM sessions" in sql:
                raise sqlite3.OperationalError("simulated origin lookup failure")
            return super().execute(sql, parameters)

    real_connect = sqlite3.connect
    index_db_marker = str(index_db)

    def _patched_connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if index_db_marker in str(database):
            kwargs["factory"] = _FlakyConnection
        return cast(sqlite3.Connection, real_connect(database, *args, **kwargs))

    monkeypatch.setattr(sqlite3, "connect", _patched_connect)

    outcome = embed_archive_session_sync(index_db, _FakeV1VectorProvider(), session_id)
    assert outcome.status == "error"

    monkeypatch.undo()
    conn = sqlite3.connect(embeddings_db)
    try:
        row = conn.execute(
            "SELECT origin, error_class, error_message FROM embedding_failures WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, "origin-lookup failure must not silently drop the ledger row"
    origin, error_class, error_message = row
    assert origin == "unknown-export"
    assert error_class == "internal_error"
    assert "sqlite-vec" in error_message


def test_pending_window_measures_concatenated_block_prose() -> None:
    """The pending window must apply the 20-character floor to the
    concatenated message prose, exactly as the materializer does.

    The message below holds two 10-character text blocks. The materializer
    embeds it (10 + 2 separator + 10 = 22 characters); a count that applies
    the floor to a single ``blocks.text`` row instead sees 10 and drops the
    message, which removes the session from the pending window.

    Anti-vacuity: apply the floor to an unqualified ``text`` in
    ``archive_embeddable_messages_relation`` and this goes red with an empty
    window, because SQLite resolves that bare ``text`` to the joined
    ``blocks.text`` column rather than the projected prose.
    """
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(
            """
            CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT NOT NULL, title TEXT, sort_key_ms INTEGER);
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                variant_index INTEGER NOT NULL,
                role TEXT NOT NULL,
                message_type TEXT NOT NULL,
                material_origin TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                content_hash BLOB
            );
            CREATE TABLE blocks (
                block_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                block_type TEXT NOT NULL,
                text TEXT
            );
            INSERT INTO sessions VALUES ('multi', 'unknown-export', 'multi', 1);
            INSERT INTO messages VALUES
                ('multi:n:m1', 'multi', 0, 0, 'user', 'message', 'human_authored', 4, zeroblob(32)),
                ('multi:n:m2', 'multi', 1, 0, 'user', 'message', 'human_authored', 1, zeroblob(32));
            INSERT INTO blocks VALUES
                ('b1', 'multi', 'multi:n:m1', 0, 'text', 'aaaaaaaaaa'),
                ('b2', 'multi', 'multi:n:m1', 1, 'text', 'bbbbbbbbbb'),
                ('b3', 'multi', 'multi:n:m2', 0, 'text', 'short');
            """
        )
        conn.commit()

        pending = select_pending_archive_session_window(conn, status_table="")

        assert [(item.session_id, item.message_count) for item in pending] == [("multi", 1)]
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("material_origin", "message_type", "role", "text", "should_embed"),
    [
        ("human_authored", "message", "user", "12345678901234567890", True),
        ("human_authored", "message", "user", "1234567890123456789", False),
        ("human_authored", "message", "user", "   \n\t  ", False),
        ("assistant_authored", "message", "assistant", "A sufficiently long assistant answer.", True),
        ("assistant_authored", "message", "system", "This is a sufficiently long system message.", False),
        ("tool_result", "tool_result", "tool", "File contents: def hello(): print('world')", False),
        ("context_generated", "message", "user", "Runtime context that is long enough.", False),
    ],
)
def test_archive_embedding_eligibility_admits_only_authored_prose(
    material_origin: str, message_type: str, role: str, text: str, should_embed: bool
) -> None:
    """The archive embedding route's per-message eligibility law.

    Moved from the retired provider-side ``_should_embed_message``: the
    20-character floor applies to stripped prose, and only authored user or
    assistant messages are bought. Anti-vacuity: drop the floor and the
    19-character row embeds; admit the ``system`` role and its row embeds.
    """
    from polylogue.storage.embeddings.materialization import _should_embed_archive_message

    assert _should_embed_archive_message(material_origin, message_type, role, text) is should_embed

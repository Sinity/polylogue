"""Unit coverage for embedding readiness branches (issue #828).

These tests pin the readiness/reconciliation/cost-cap behaviour added by the
embedding substrate work so the branches called out in the issue reopen
comment have explicit, fast unit coverage:

1. configured readiness branch (api key present, dim matches stored)
2. unconfigured readiness branch (no api key)
3. embedding failure branch (status reports failures)
4. dimension-mismatch / model-mismatch triggers ``needs_reindex``
5. cost-cap exhaustion halts further embedding work

The production code is exercised directly: ``_reconcile_embedding_config_change``
operate on real (in-memory) SQLite handles
plus a mocked ``EmbedSessionOutcome`` stream for the embed loop, and
``embedding_readiness_info`` is exercised through the tiny ``cfg``/db seam.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import polylogue.daemon.convergence_stages as stages
from polylogue.daemon.convergence_stages import (
    _reconcile_embedding_config_change,
)
from polylogue.daemon.embedding_readiness import embedding_readiness_info
from polylogue.storage.search_providers.sqlite_vec_runtime import (
    _reconcile_vec0_dimension,
    _vec0_table_dimension,
)

# ── helpers ────────────────────────────────────────────────────────


class _FakeCfg:
    """Minimal stand-in for ``PolylogueConfig`` with attribute + ``.get``."""

    def __init__(
        self,
        *,
        embedding_enabled: bool = True,
        voyage_api_key: str | None = "test-key",
        embedding_model: str = "voyage-4",
        embedding_dimension: int = 1024,
        embedding_max_cost_usd: float = 0.0,
    ) -> None:
        self.embedding_enabled = embedding_enabled
        self.voyage_api_key = voyage_api_key
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.embedding_max_cost_usd = embedding_max_cost_usd

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _seed_embedding_tables(
    conn: sqlite3.Connection,
    *,
    model: str,
    dimension: int,
    session_ids: tuple[str, ...] = (),
) -> None:
    """Create embeddings_meta + embedding_status with seeded rows."""
    conn.execute(
        """
        CREATE TABLE embeddings_meta (
            target_id TEXT PRIMARY KEY,
            target_type TEXT NOT NULL,
            model TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            embedded_at TEXT NOT NULL,
            content_hash TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE embedding_status (
            session_id TEXT PRIMARY KEY,
            message_count_embedded INTEGER DEFAULT 0,
            last_embedded_at TEXT,
            needs_reindex INTEGER DEFAULT 0,
            error_message TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO embeddings_meta(target_id, target_type, model, dimension, embedded_at) "
        "VALUES (?, 'message', ?, ?, '2026-01-01T00:00:00Z')",
        ("msg-1", model, dimension),
    )
    for conv_id in session_ids:
        conn.execute(
            "INSERT INTO embedding_status(session_id, needs_reindex) VALUES (?, 0)",
            (conv_id,),
        )
    conn.commit()


def _create_vec0_table(conn: sqlite3.Connection, dimension: int) -> None:
    """Simulate the vec0 virtual table by faking its DDL signature.

    The real table requires the sqlite-vec extension. ``_vec0_table_dimension``
    parses ``float[N]`` out of the DDL string returned by ``sqlite_master.sql``
    and accesses the row by column name, so we need ``sqlite3.Row`` factory and
    a column literally named ``embedding_float_NNNN`` so the regex matches the
    stored DDL text.
    """
    conn.row_factory = sqlite3.Row
    # The substring ``float[N]`` must appear verbatim in the stored DDL. SQLite
    # only allows it inside identifiers if quoted. Quoted column name keeps the
    # exact text inside ``sqlite_master.sql``.
    conn.execute(f'CREATE TABLE message_embeddings (message_id TEXT, "embedding float[{dimension}]" TEXT)')
    conn.commit()


def _seed_archive_embedding_readiness_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_path = path.with_name("embeddings.db")
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                message_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                content_hash BLOB NOT NULL
            );
            INSERT INTO sessions VALUES ('codex-session:complete', 1);
            INSERT INTO sessions VALUES ('codex-session:pending', 2);
            INSERT INTO sessions VALUES ('codex-session:error', 1);
            INSERT INTO messages VALUES ('codex-session:complete:m1', 'codex-session:complete', x'01');
            INSERT INTO messages VALUES ('codex-session:pending:m1', 'codex-session:pending', x'02');
            INSERT INTO messages VALUES ('codex-session:pending:m2', 'codex-session:pending', x'03');
            INSERT INTO messages VALUES ('codex-session:error:m1', 'codex-session:error', x'04');
            """
        )
        conn.commit()
    with sqlite3.connect(embeddings_path) as conn:
        conn.executescript(
            """
            CREATE TABLE message_embeddings (
                message_id TEXT PRIMARY KEY
            );
            CREATE TABLE embeddings_meta (
                target_id TEXT PRIMARY KEY,
                target_type TEXT NOT NULL,
                model TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                embedded_at_ms INTEGER NOT NULL,
                content_hash BLOB
            );
            CREATE TABLE embedding_status (
                session_id TEXT PRIMARY KEY,
                origin TEXT NOT NULL,
                message_count_embedded INTEGER NOT NULL DEFAULT 0,
                needs_reindex INTEGER NOT NULL DEFAULT 0,
                error_message TEXT
            );
            INSERT INTO message_embeddings VALUES ('codex-session:complete:m1');
            INSERT INTO embeddings_meta VALUES (
                'codex-session:complete:m1', 'message', 'voyage-4', 1024, 1767225700000, x'01'
            );
            INSERT INTO embedding_status VALUES ('codex-session:complete', 'codex-session', 1, 0, NULL);
            INSERT INTO embedding_status VALUES ('codex-session:error', 'codex-session', 0, 1, 'voyage timeout');
            """
        )
        conn.commit()


# ── 1. configured readiness branch ─────────────────────────────────


def test_readiness_configured_reports_enabled_with_model_and_dimension(
    workspace_env: dict[str, Path],
) -> None:
    """When config is enabled and key is present, readiness reports the configured model/dim."""
    db = workspace_env["data_root"] / "polylogue" / "polylogue.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
        conn.commit()

    cfg = _FakeCfg(
        embedding_enabled=True,
        voyage_api_key="vk-live",
        embedding_model="voyage-4",
        embedding_dimension=1024,
    )
    with patch("polylogue.config.load_polylogue_config", return_value=cfg):
        info = embedding_readiness_info(db)

    assert info["embedding_enabled"] is True
    assert info["embedding_model"] == "voyage-4"
    assert info["embedding_dimension"] == 1024
    # No tables yet → counts default to zero, but the configured shape is exposed.
    assert info["embedding_pending_count"] == 0
    assert info["embedding_failure_count"] == 0


# ── 2. unconfigured readiness branch ───────────────────────────────


def test_readiness_unconfigured_reports_disabled_when_no_api_key(
    workspace_env: dict[str, Path],
) -> None:
    """When no API key is present, ``embedding_enabled`` is False and counts are zero."""
    db = workspace_env["data_root"] / "polylogue" / "polylogue.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    db.touch()

    cfg = _FakeCfg(embedding_enabled=True, voyage_api_key=None)
    with patch("polylogue.config.load_polylogue_config", return_value=cfg):
        info = embedding_readiness_info(db)

    assert info["embedding_enabled"] is False
    assert info["embedding_pending_count"] == 0
    assert info["embedding_stale_count"] == 0
    assert info["embedding_failure_count"] == 0
    assert info["embedding_estimated_cost_usd"] == 0.0


def test_readiness_unconfigured_when_enabled_flag_off(
    workspace_env: dict[str, Path],
) -> None:
    """Even disabled config still exposes the backlog instead of hiding it."""
    db = workspace_env["data_root"] / "polylogue" / "polylogue.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE messages (message_id TEXT PRIMARY KEY, session_id TEXT)")
        conn.execute("CREATE TABLE embedding_status (session_id TEXT PRIMARY KEY, needs_reindex INTEGER)")
        conn.execute("INSERT INTO sessions VALUES ('conv-1')")
        conn.execute("INSERT INTO messages VALUES ('msg-1', 'conv-1')")
        conn.commit()

    cfg = _FakeCfg(embedding_enabled=False, voyage_api_key="vk-live")
    with patch("polylogue.config.load_polylogue_config", return_value=cfg):
        info = embedding_readiness_info(db)

    assert info["embedding_enabled"] is False
    assert info["embedding_config_enabled"] is False
    assert info["embedding_has_voyage_key"] is True
    assert info["embedding_status"] == "none"
    assert info["embedding_freshness_status"] == "none"
    assert info["embedding_retrieval_ready"] is False
    assert info["embedding_pending_count"] == 1
    assert info["embedding_pending_message_count"] == 0
    assert info["embedding_pending_message_count_exact"] is False

    with patch("polylogue.config.load_polylogue_config", return_value=cfg):
        detailed = embedding_readiness_info(db, detail=True)

    assert detailed["embedding_pending_count"] == 1
    assert detailed["embedding_pending_message_count"] == 1
    assert detailed["embedding_pending_message_count_exact"] is True


def test_readiness_reads_archive_file_set_without_polylogue_db(tmp_path: Path) -> None:
    legacy_db = tmp_path / "polylogue.db"
    archive_db = tmp_path / "index.db"
    _seed_archive_embedding_readiness_db(archive_db)

    cfg = _FakeCfg(embedding_enabled=True, voyage_api_key="vk-live")
    with patch("polylogue.config.load_polylogue_config", return_value=cfg):
        info = embedding_readiness_info(legacy_db)

    assert info["embedding_enabled"] is True
    assert info["embedding_status"] == "partial"
    assert info["embedding_freshness_status"] == "partial"
    assert info["embedding_retrieval_ready"] is True
    assert info["embedding_pending_count"] == 2
    assert info["embedding_pending_message_count"] == 0
    assert info["embedding_pending_message_count_exact"] is False
    assert info["embedding_failure_count"] == 1
    assert info["embedding_coverage_percent"] == 33.3


def test_readiness_reads_archive_file_set_detail_counts_pending_messages(tmp_path: Path) -> None:
    archive_db = tmp_path / "index.db"
    _seed_archive_embedding_readiness_db(archive_db)

    cfg = _FakeCfg(embedding_enabled=True, voyage_api_key="vk-live")
    with patch("polylogue.config.load_polylogue_config", return_value=cfg):
        info = embedding_readiness_info(archive_db, detail=True)

    assert info["embedding_pending_count"] == 2
    assert info["embedding_pending_message_count"] == 3
    assert info["embedding_pending_message_count_exact"] is True
    assert info["embedding_stale_count"] == 0
    assert info["embedding_estimated_cost_usd"] == 0.0


def test_readiness_prefers_archive_archive_when_legacy_db_exists(tmp_path: Path) -> None:
    legacy_db = tmp_path / "polylogue.db"
    archive_db = tmp_path / "index.db"
    with sqlite3.connect(legacy_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO sessions VALUES ('legacy-pending')")
        conn.commit()
    _seed_archive_embedding_readiness_db(archive_db)

    cfg = _FakeCfg(embedding_enabled=True, voyage_api_key="vk-live")
    with patch("polylogue.config.load_polylogue_config", return_value=cfg):
        info = embedding_readiness_info(legacy_db)

    assert info["embedding_status"] == "partial"
    assert info["embedding_pending_count"] == 2


# ── 3. embedding failure branch ────────────────────────────────────


def test_readiness_failure_branch_counts_error_message_rows(
    workspace_env: dict[str, Path],
) -> None:
    """Rows with non-null ``error_message`` show up in ``embedding_failure_count``."""
    db = workspace_env["data_root"] / "polylogue" / "polylogue.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO sessions VALUES ('conv-1')")
        conn.execute("INSERT INTO sessions VALUES ('conv-2')")
        _seed_embedding_tables(conn, model="voyage-4", dimension=1024, session_ids=("conv-1", "conv-2"))
        conn.execute("UPDATE embedding_status SET error_message = 'voyage api 429' WHERE session_id = 'conv-1'")
        conn.execute("CREATE TABLE message_embeddings (message_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE messages (message_id TEXT PRIMARY KEY, session_id TEXT, content_hash TEXT)")
        conn.commit()

    cfg = _FakeCfg(embedding_enabled=True, voyage_api_key="vk-live")
    with patch("polylogue.config.load_polylogue_config", return_value=cfg):
        info = embedding_readiness_info(db)

    assert info["embedding_failure_count"] == 1


def test_reconcile_embedding_dimension_mismatch_marks_reindex_and_drops_vec0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dimension change marks every embedding_status row and drops vec0."""
    conn = sqlite3.connect(":memory:")
    try:
        _seed_embedding_tables(
            conn,
            model="voyage-4",
            dimension=1024,
            session_ids=("conv-a", "conv-b"),
        )
        _create_vec0_table(conn, dimension=1024)
        assert _vec0_table_dimension(conn) == 1024

        cfg = _FakeCfg(
            embedding_enabled=True,
            voyage_api_key="vk-live",
            embedding_model="voyage-4",
            embedding_dimension=512,  # changed
        )
        monkeypatch.setattr(stages, "load_polylogue_config", lambda: cfg)

        _reconcile_embedding_config_change(conn)

        rows = conn.execute(
            "SELECT session_id, needs_reindex, error_message FROM embedding_status ORDER BY session_id"
        ).fetchall()
        assert [(r[0], r[1], r[2]) for r in rows] == [("conv-a", 1, None), ("conv-b", 1, None)]
        # vec0 dropped because dimension differed.
        assert _vec0_table_dimension(conn) is None
    finally:
        conn.close()


def test_reconcile_embedding_model_mismatch_marks_reindex_without_dropping_vec0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model change triggers reindex but leaves vec0 table intact when dim matches."""
    conn = sqlite3.connect(":memory:")
    try:
        _seed_embedding_tables(
            conn,
            model="voyage-3",
            dimension=1024,
            session_ids=("conv-a",),
        )
        _create_vec0_table(conn, dimension=1024)

        cfg = _FakeCfg(
            embedding_enabled=True,
            voyage_api_key="vk-live",
            embedding_model="voyage-4",  # changed
            embedding_dimension=1024,
        )
        monkeypatch.setattr(stages, "load_polylogue_config", lambda: cfg)

        _reconcile_embedding_config_change(conn)

        (needs_reindex,) = conn.execute(
            "SELECT needs_reindex FROM embedding_status WHERE session_id='conv-a'"
        ).fetchone()
        assert needs_reindex == 1
        # vec0 untouched — dimension still matches configured.
        assert _vec0_table_dimension(conn) == 1024
    finally:
        conn.close()


def test_reconcile_embedding_no_change_keeps_status_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matching model+dim leaves ``needs_reindex`` unchanged (idempotent)."""
    conn = sqlite3.connect(":memory:")
    try:
        _seed_embedding_tables(
            conn,
            model="voyage-4",
            dimension=1024,
            session_ids=("conv-a",),
        )
        cfg = _FakeCfg(
            embedding_enabled=True,
            voyage_api_key="vk-live",
            embedding_model="voyage-4",
            embedding_dimension=1024,
        )
        monkeypatch.setattr(stages, "load_polylogue_config", lambda: cfg)

        _reconcile_embedding_config_change(conn)

        (needs_reindex,) = conn.execute(
            "SELECT needs_reindex FROM embedding_status WHERE session_id='conv-a'"
        ).fetchone()
        assert needs_reindex == 0
    finally:
        conn.close()


def test_reconcile_vec0_dimension_drop_helper() -> None:
    """``_reconcile_vec0_dimension`` drops the table only when configured differs from stored."""
    conn = sqlite3.connect(":memory:")
    try:
        _create_vec0_table(conn, dimension=1024)
        _reconcile_vec0_dimension(conn, configured_dimension=1024)
        assert _vec0_table_dimension(conn) == 1024  # match → kept

        _reconcile_vec0_dimension(conn, configured_dimension=2048)
        assert _vec0_table_dimension(conn) is None  # mismatch → dropped
    finally:
        conn.close()


# ── 5. cost-cap exhaustion ─────────────────────────────────────────

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
and ``_embed_conversations_sync`` operate on real (in-memory) SQLite handles
plus a mocked ``EmbedConversationOutcome`` stream for the embed loop, and
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
    _embed_conversations_sync,
    _reconcile_embedding_config_change,
)
from polylogue.daemon.embedding_readiness import embedding_readiness_info
from polylogue.storage.embeddings.materialization import EmbedConversationOutcome
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
    conversation_ids: tuple[str, ...] = (),
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
            conversation_id TEXT PRIMARY KEY,
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
    for conv_id in conversation_ids:
        conn.execute(
            "INSERT INTO embedding_status(conversation_id, needs_reindex) VALUES (?, 0)",
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


# ── 1. configured readiness branch ─────────────────────────────────


def test_readiness_configured_reports_enabled_with_model_and_dimension(
    workspace_env: dict[str, Path],
) -> None:
    """When config is enabled and key is present, readiness reports the configured model/dim."""
    db = workspace_env["data_root"] / "polylogue" / "polylogue.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE conversations (conversation_id TEXT PRIMARY KEY)")
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
        conn.execute("CREATE TABLE conversations (conversation_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE messages (message_id TEXT PRIMARY KEY, conversation_id TEXT)")
        conn.execute("CREATE TABLE embedding_status (conversation_id TEXT PRIMARY KEY, needs_reindex INTEGER)")
        conn.execute("INSERT INTO conversations VALUES ('conv-1')")
        conn.execute("INSERT INTO messages VALUES ('msg-1', 'conv-1')")
        conn.commit()

    cfg = _FakeCfg(embedding_enabled=False, voyage_api_key="vk-live")
    with patch("polylogue.config.load_polylogue_config", return_value=cfg):
        info = embedding_readiness_info(db)

    assert info["embedding_enabled"] is False
    assert info["embedding_config_enabled"] is False
    assert info["embedding_has_voyage_key"] is True
    assert info["embedding_pending_count"] == 1
    assert info["embedding_pending_message_count"] == 1


# ── 3. embedding failure branch ────────────────────────────────────


def test_readiness_failure_branch_counts_error_message_rows(
    workspace_env: dict[str, Path],
) -> None:
    """Rows with non-null ``error_message`` show up in ``embedding_failure_count``."""
    db = workspace_env["data_root"] / "polylogue" / "polylogue.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE conversations (conversation_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO conversations VALUES ('conv-1')")
        conn.execute("INSERT INTO conversations VALUES ('conv-2')")
        _seed_embedding_tables(conn, model="voyage-4", dimension=1024, conversation_ids=("conv-1", "conv-2"))
        conn.execute("UPDATE embedding_status SET error_message = 'voyage api 429' WHERE conversation_id = 'conv-1'")
        conn.execute("CREATE TABLE message_embeddings (message_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE messages (message_id TEXT PRIMARY KEY, conversation_id TEXT, content_hash TEXT)")
        conn.commit()

    cfg = _FakeCfg(embedding_enabled=True, voyage_api_key="vk-live")
    with patch("polylogue.config.load_polylogue_config", return_value=cfg):
        info = embedding_readiness_info(db)

    assert info["embedding_failure_count"] == 1


def test_embed_loop_counts_errors_and_returns_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A single error outcome from ``embed_conversation_sync`` causes ``_embed_conversations_sync`` to return False."""
    db_path = tmp_path / "archive.sqlite"
    db_path.touch()

    cfg = _FakeCfg(embedding_max_cost_usd=0.0)
    monkeypatch.setattr(stages, "load_polylogue_config", lambda: cfg)

    fake_provider = object()
    monkeypatch.setattr(
        "polylogue.storage.search_providers.create_vector_provider",
        lambda **kwargs: fake_provider,
    )

    class _Repo:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        def close(self) -> None:  # not async — keeps run_coroutine_sync stub trivial
            return None

    monkeypatch.setattr("polylogue.storage.repository.ConversationRepository", _Repo)
    monkeypatch.setattr("polylogue.storage.sqlite.async_sqlite.SQLiteBackend", lambda **kw: object())
    monkeypatch.setattr("polylogue.api.sync.bridge.run_coroutine_sync", lambda coro: None)

    outcomes = iter(
        [
            EmbedConversationOutcome(
                status="error", conversation_id="conv-a", embedded_message_count=0, error="429 rate limit"
            ),
            EmbedConversationOutcome(status="embedded", conversation_id="conv-b", embedded_message_count=10),
        ]
    )
    monkeypatch.setattr(
        "polylogue.storage.embeddings.materialization.embed_conversation_sync",
        lambda repo, vec_provider, conversation_id: next(outcomes),
    )

    result = _embed_conversations_sync(db_path, ["conv-a", "conv-b"])
    assert result is False  # one error → overall failure


# ── 4. dimension / model mismatch triggers needs_reindex ───────────


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
            conversation_ids=("conv-a", "conv-b"),
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
            "SELECT conversation_id, needs_reindex, error_message FROM embedding_status ORDER BY conversation_id"
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
            conversation_ids=("conv-a",),
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
            "SELECT needs_reindex FROM embedding_status WHERE conversation_id='conv-a'"
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
            conversation_ids=("conv-a",),
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
            "SELECT needs_reindex FROM embedding_status WHERE conversation_id='conv-a'"
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


def test_embed_loop_halts_when_cost_cap_exceeded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``_embed_conversations_sync`` stops calling embed_conversation_sync past the cost cap."""
    db_path = tmp_path / "archive.sqlite"
    db_path.touch()

    # Tiny cap: cost per 10 messages ≈ 10 * 500 * 0.10 / 1e6 = $0.0005
    # Set cap = $0.0003 → first batch already exceeds, loop should break after 1 conversation.
    cfg = _FakeCfg(embedding_max_cost_usd=0.0003)
    monkeypatch.setattr(stages, "load_polylogue_config", lambda: cfg)

    monkeypatch.setattr(
        "polylogue.storage.search_providers.create_vector_provider",
        lambda **kwargs: object(),
    )

    class _Repo:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        def close(self) -> None:  # not async — keeps run_coroutine_sync stub trivial
            return None

    monkeypatch.setattr("polylogue.storage.repository.ConversationRepository", _Repo)
    monkeypatch.setattr("polylogue.storage.sqlite.async_sqlite.SQLiteBackend", lambda **kw: object())
    monkeypatch.setattr("polylogue.api.sync.bridge.run_coroutine_sync", lambda coro: None)

    calls: list[str] = []

    def fake_embed(repo: object, vec_provider: object, conversation_id: str) -> EmbedConversationOutcome:
        calls.append(conversation_id)
        return EmbedConversationOutcome(status="embedded", conversation_id=conversation_id, embedded_message_count=10)

    monkeypatch.setattr(
        "polylogue.storage.embeddings.materialization.embed_conversation_sync",
        fake_embed,
    )

    result = _embed_conversations_sync(db_path, ["conv-a", "conv-b", "conv-c"])

    # First conversation is embedded; cost cap then halts the loop.
    assert calls == ["conv-a"]
    # No errors were observed → returns True even though loop short-circuited.
    assert result is True


def test_embed_loop_no_cost_cap_processes_all(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``embedding_max_cost_usd = 0`` means unlimited — every conversation is embedded."""
    db_path = tmp_path / "archive.sqlite"
    db_path.touch()

    cfg = _FakeCfg(embedding_max_cost_usd=0.0)
    monkeypatch.setattr(stages, "load_polylogue_config", lambda: cfg)
    monkeypatch.setattr(
        "polylogue.storage.search_providers.create_vector_provider",
        lambda **kwargs: object(),
    )

    class _Repo:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        def close(self) -> None:  # not async — keeps run_coroutine_sync stub trivial
            return None

    monkeypatch.setattr("polylogue.storage.repository.ConversationRepository", _Repo)
    monkeypatch.setattr("polylogue.storage.sqlite.async_sqlite.SQLiteBackend", lambda **kw: object())
    monkeypatch.setattr("polylogue.api.sync.bridge.run_coroutine_sync", lambda coro: None)

    calls: list[str] = []

    def fake_embed(repo: object, vec_provider: object, conversation_id: str) -> EmbedConversationOutcome:
        calls.append(conversation_id)
        return EmbedConversationOutcome(
            status="embedded",
            conversation_id=conversation_id,
            embedded_message_count=10_000,  # high message count, no cap
        )

    monkeypatch.setattr(
        "polylogue.storage.embeddings.materialization.embed_conversation_sync",
        fake_embed,
    )

    assert _embed_conversations_sync(db_path, ["conv-a", "conv-b", "conv-c"]) is True
    assert calls == ["conv-a", "conv-b", "conv-c"]

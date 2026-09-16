"""Unit coverage for embedding readiness branches (issue #828).

These tests pin readiness and cost-cap behaviour from the embedding substrate
so the relevant branches have explicit, fast unit coverage:

1. configured readiness branch (api key present, dim matches stored)
2. unconfigured readiness branch (no api key)
3. embedding failure branch (status reports failures)
4. stale or missing vector evidence remains pending
5. cost-cap exhaustion halts further embedding work

``embedding_readiness_info`` is exercised through the tiny ``cfg``/db seam.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from unittest.mock import patch

import polylogue.logging as plog
from polylogue.config import PolylogueConfig
from polylogue.daemon.embedding_readiness import embedding_readiness_info
from tests.infra.embedding_config import embedding_config

# ── helpers ────────────────────────────────────────────────────────


def _config(
    *,
    embedding_enabled: bool = True,
    voyage_api_key: str | None = "test-key",
    embedding_model: str = "voyage-4",
    embedding_dimension: int = 1024,
    embedding_max_cost_usd: float = 0.0,
) -> PolylogueConfig:
    """A real ``PolylogueConfig``, not a duck-typed stand-in.

    ``embedding_status_settings_from_config`` dispatches nominally over
    ``PolylogueConfig``/``Config`` and refuses anything else, so a local
    double here could only ever prove that the double matches itself --
    and would go on passing after the production type it imitates changed.
    """
    return embedding_config(
        embedding_enabled=embedding_enabled,
        voyage_api_key=voyage_api_key,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        embedding_max_cost_usd=embedding_max_cost_usd,
    )


def _seed_embedding_tables(
    conn: sqlite3.Connection,
    *,
    model: str,
    dimension: int,
    session_ids: tuple[str, ...] = (),
) -> None:
    """Create legacy attempt telemetry for the failure-count read branch."""
    from polylogue.storage.embeddings.identity import EmbeddingRecipe

    stored_recipe = EmbeddingRecipe.current(model=model, dimensions=dimension)
    conn.execute(
        """
        CREATE TABLE message_embeddings_meta (
            vector_derivation_hash BLOB PRIMARY KEY,
            model TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            embedded_at_ms INTEGER,
            recipe_hash BLOB,
            output_contract_hash BLOB
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
        """
        CREATE TABLE embedding_derivation_state (
            session_id TEXT PRIMARY KEY,
            origin TEXT NOT NULL DEFAULT '',
            generation INTEGER NOT NULL,
            derivation_key BLOB NOT NULL,
            source_hash BLOB NOT NULL,
            recipe_hash BLOB NOT NULL,
            output_contract_hash BLOB NOT NULL,
            attempt_state TEXT NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0,
            updated_at_ms INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO message_embeddings_meta(vector_derivation_hash, model, dimension, embedded_at_ms, recipe_hash, output_contract_hash) "
        "VALUES (?, ?, ?, 1767225600000, ?, ?)",
        (b"\x01" * 32, model, dimension, stored_recipe.recipe_hash, stored_recipe.output_contract_hash),
    )
    for conv_id in session_ids:
        conn.execute(
            "INSERT INTO embedding_status(session_id, needs_reindex) VALUES (?, 0)",
            (conv_id,),
        )
        conn.execute(
            """
            INSERT INTO embedding_derivation_state (
                session_id, origin, generation, derivation_key, source_hash, recipe_hash,
                output_contract_hash, attempt_state, message_count, updated_at_ms
            ) VALUES (?, 'codex-session', 1, ?, ?, ?, ?, 'succeeded', 1, 1767225600000)
            """,
            (
                conv_id,
                hashlib.sha256(f"{conv_id}:derivation".encode()).digest(),
                hashlib.sha256(f"{conv_id}:source".encode()).digest(),
                stored_recipe.recipe_hash,
                stored_recipe.output_contract_hash,
            ),
        )
    conn.commit()


# v4 (polylogue-q88p): distinct, >=20-char prose per message so each
# message's vector_derivation_hash -- computed from exactly this text -- is
# real and unique, matching what production actually sends to the embedder.
_READINESS_COMPLETE_TEXT = "authored prose for the complete readiness session message, long enough"
_READINESS_PENDING_TEXT_1 = "authored prose for the first pending readiness message, long enough"
_READINESS_PENDING_TEXT_2 = "authored prose for the second pending readiness message, long enough"
_READINESS_ERROR_TEXT = "authored prose for the failed readiness session message, long enough"


def _seed_archive_embedding_readiness_db(path: Path) -> None:
    """Build a real index.db + embeddings.db pair for readiness reads.

    ``codex-session:complete`` is embedded through the real
    begin_embedding_attempt/complete_embedding_attempt_success write path;
    ``codex-session:error`` gets a real *retryable* failure through
    begin_embedding_attempt/record_embedding_failure (matching this fixture's
    original needs_reindex=1 intent -- a retryable failure stays part of the
    pending backlog, unlike an acknowledged/terminal one, which is what
    status_payload.py's exact freshness predicate now keys "blocked" off of
    via embedding_derivation_state.attempt_state, not a bare embedding_status
    error_message column).
    """
    from polylogue.storage.embeddings.identity import (
        EmbeddingRecipe,
        EmbeddingSourceDigest,
        vector_derivation_hash,
    )
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.embedding_write import (
        ArchiveEmbeddingWrite,
        begin_embedding_attempt,
        complete_embedding_attempt_success,
        record_embedding_failure,
    )
    from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

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
                text TEXT,
                role TEXT NOT NULL DEFAULT 'user',
                message_type TEXT NOT NULL DEFAULT 'message',
                material_origin TEXT NOT NULL DEFAULT 'human_authored',
                word_count INTEGER NOT NULL DEFAULT 8,
                content_hash BLOB NOT NULL
            );
            INSERT INTO sessions VALUES ('codex-session:complete', 1);
            INSERT INTO sessions VALUES ('codex-session:pending', 2);
            INSERT INTO sessions VALUES ('codex-session:error', 1);
            """
        )
        conn.execute(
            "INSERT INTO messages (message_id, session_id, text, content_hash) VALUES (?, ?, ?, ?)",
            ("codex-session:complete:m1", "codex-session:complete", _READINESS_COMPLETE_TEXT, b"\x01" * 32),
        )
        conn.execute(
            "INSERT INTO messages (message_id, session_id, text, content_hash) VALUES (?, ?, ?, ?)",
            ("codex-session:pending:m1", "codex-session:pending", _READINESS_PENDING_TEXT_1, b"\x02" * 32),
        )
        conn.execute(
            "INSERT INTO messages (message_id, session_id, text, content_hash) VALUES (?, ?, ?, ?)",
            ("codex-session:pending:m2", "codex-session:pending", _READINESS_PENDING_TEXT_2, b"\x03" * 32),
        )
        conn.execute(
            "INSERT INTO messages (message_id, session_id, text, content_hash) VALUES (?, ?, ?, ?)",
            ("codex-session:error:m1", "codex-session:error", _READINESS_ERROR_TEXT, b"\x04" * 32),
        )
        conn.commit()

    recipe = EmbeddingRecipe.current(model="voyage-4", dimensions=EMBEDDING_DIMENSION)
    econn = sqlite3.connect(embeddings_path)
    try:
        initialize_archive_tier(econn, ArchiveTier.EMBEDDINGS)

        complete_hash = vector_derivation_hash(model="voyage-4", input_text=_READINESS_COMPLETE_TEXT)
        complete_source = EmbeddingSourceDigest()
        complete_source.update(complete_hash)
        complete_attempt = begin_embedding_attempt(
            econn,
            session_id="codex-session:complete",
            origin="codex-session",
            source_hash=complete_source.digest(),
            recipe=recipe,
            started_at_ms=1_767_225_700_000,
        )
        complete_embedding_attempt_success(
            econn,
            attempt=complete_attempt,
            writes=[
                ArchiveEmbeddingWrite(
                    message_id="codex-session:complete:m1",
                    session_id="codex-session:complete",
                    origin="codex-session",
                    message_content_hash=b"\x01" * 32,
                    embedding=[0.01] * EMBEDDING_DIMENSION,
                    model="voyage-4",
                    embedded_at_ms=1_767_225_700_000,
                    vector_derivation_hash=complete_hash,
                    recipe_hash=complete_attempt.recipe_hash,
                    generation=complete_attempt.generation,
                )
            ],
            completed_at_ms=1_767_225_700_000,
        )

        error_hash = vector_derivation_hash(model="voyage-4", input_text=_READINESS_ERROR_TEXT)
        error_source = EmbeddingSourceDigest()
        error_source.update(error_hash)
        error_attempt = begin_embedding_attempt(
            econn,
            session_id="codex-session:error",
            origin="codex-session",
            source_hash=error_source.digest(),
            recipe=recipe,
            started_at_ms=1_767_225_700_000,
        )
        record_embedding_failure(
            econn,
            session_id="codex-session:error",
            origin="codex-session",
            message_refs=("codex-session:error:m1",),
            provider="voyage",
            model="voyage-4",
            error_class="provider_timeout",
            error_message="voyage timeout",
            retryable=True,
            occurred_at_ms=1_767_225_700_000,
            attempt=error_attempt,
        )
    finally:
        econn.close()


# ── 1. configured readiness branch ─────────────────────────────────


def test_readiness_configured_reports_enabled_with_model_and_dimension(
    workspace_env: dict[str, Path],
) -> None:
    """When config is enabled and key is present, readiness reports the configured model/dim."""
    db = workspace_env["data_root"] / "polylogue" / "index.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
        conn.commit()

    cfg = _config(
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
    db = workspace_env["data_root"] / "polylogue" / "index.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    db.touch()

    cfg = _config(embedding_enabled=True, voyage_api_key=None)
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
    db = workspace_env["data_root"] / "polylogue" / "index.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
        conn.execute(
            """
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT NOT NULL DEFAULT 'user',
                message_type TEXT NOT NULL DEFAULT 'message',
                material_origin TEXT NOT NULL DEFAULT 'human_authored',
                word_count INTEGER NOT NULL DEFAULT 8
            )
            """
        )
        conn.execute("CREATE TABLE embedding_status (session_id TEXT PRIMARY KEY, needs_reindex INTEGER)")
        conn.execute("INSERT INTO sessions VALUES ('conv-1')")
        conn.execute("INSERT INTO messages (message_id, session_id) VALUES ('msg-1', 'conv-1')")
        conn.commit()

    cfg = _config(embedding_enabled=False, voyage_api_key="vk-live")
    with patch("polylogue.config.load_polylogue_config", return_value=cfg):
        info = embedding_readiness_info(db)

    assert info["embedding_enabled"] is False
    assert info["embedding_config_enabled"] is False
    assert info["embedding_has_voyage_key"] is True
    assert info["embedding_status"] == "none"
    assert info["embedding_freshness_status"] == "none"
    assert info["embedding_retrieval_ready"] is False
    assert info["embedding_pending_count"] == 1
    assert info["embedding_pending_message_count"] is None
    assert info["embedding_pending_message_count_exact"] is False

    with patch("polylogue.config.load_polylogue_config", return_value=cfg):
        detailed = embedding_readiness_info(db, detail=True)

    assert detailed["embedding_pending_count"] == 1
    assert detailed["embedding_pending_message_count"] == 1
    assert detailed["embedding_pending_message_count_exact"] is True


def test_readiness_reads_archive_index(tmp_path: Path) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    archive_db = tmp_path / "index.db"
    _seed_archive_embedding_readiness_db(archive_db)

    cfg = _config(embedding_enabled=True, voyage_api_key="vk-live")
    with patch("polylogue.config.load_polylogue_config", return_value=cfg):
        info = embedding_readiness_info(db_anchor)

    assert info["embedding_enabled"] is True
    assert info["embedding_status"] == "partial"
    assert info["embedding_freshness_status"] == "partial"
    assert info["embedding_retrieval_ready"] is True
    assert info["embedding_pending_count"] == 2
    assert info["embedding_pending_message_count"] is None
    assert info["embedding_pending_message_count_exact"] is False
    assert info["embedding_failure_count"] == 1
    assert info["embedding_coverage_percent"] == 33.3


def test_readiness_reads_archive_file_set_detail_counts_pending_messages(tmp_path: Path) -> None:
    archive_db = tmp_path / "index.db"
    _seed_archive_embedding_readiness_db(archive_db)

    cfg = _config(embedding_enabled=True, voyage_api_key="vk-live")
    with patch("polylogue.config.load_polylogue_config", return_value=cfg):
        info = embedding_readiness_info(archive_db, detail=True)

    assert info["embedding_pending_count"] == 2
    assert info["embedding_pending_message_count"] == 3
    assert info["embedding_pending_message_count_exact"] is True
    assert info["embedding_stale_count"] == 0
    assert info["embedding_estimated_cost_usd"] == 0.0


def test_readiness_reads_index_when_db_anchor_exists(tmp_path: Path) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    archive_db = tmp_path / "index.db"
    with sqlite3.connect(db_anchor) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO sessions VALUES ('anchor-pending')")
        conn.commit()
    _seed_archive_embedding_readiness_db(archive_db)

    cfg = _config(embedding_enabled=True, voyage_api_key="vk-live")
    with patch("polylogue.config.load_polylogue_config", return_value=cfg):
        info = embedding_readiness_info(db_anchor)

    assert info["embedding_status"] == "partial"
    assert info["embedding_pending_count"] == 2


# ── 3. embedding failure branch ────────────────────────────────────


def test_readiness_failure_branch_counts_error_message_rows(tmp_path: Path) -> None:
    """Rows with non-null ``error_message`` show up in ``embedding_failure_count``.

    This pins the *pre-``embedding_failures``* fallback branch of
    ``_archive_embedding_status_payload``: when the embeddings tier carries no
    ``embedding_failures`` relation, the failure counts come from
    ``embedding_status.error_message`` joined to the index tier's ``sessions``.
    The embedding relations therefore live in the archive's ``embeddings.db``
    beside the index -- the readiness payload resolves that tier from the
    archive root, never from the status anchor it is handed, so a single-file
    seed exercises nothing.

    Anti-vacuity: drop the ``error_message IS NOT NULL`` fallback COUNT (or
    point it at the wrong tier) and this reports 0.
    """
    archive_index = tmp_path / "index.db"
    embeddings_db = tmp_path / "embeddings.db"
    with sqlite3.connect(archive_index) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO sessions VALUES ('conv-1')")
        conn.execute("INSERT INTO sessions VALUES ('conv-2')")
        conn.execute(
            """
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT NOT NULL DEFAULT 'user',
                message_type TEXT NOT NULL DEFAULT 'message',
                material_origin TEXT NOT NULL DEFAULT 'human_authored',
                word_count INTEGER NOT NULL DEFAULT 8,
                content_hash TEXT
            )
            """
        )
        conn.commit()
    with sqlite3.connect(embeddings_db) as conn:
        _seed_embedding_tables(conn, model="voyage-4", dimension=1024, session_ids=("conv-1", "conv-2"))
        conn.execute("UPDATE embedding_status SET error_message = 'voyage api 429' WHERE session_id = 'conv-1'")
        conn.execute("CREATE TABLE message_embeddings (message_id TEXT PRIMARY KEY)")
        conn.commit()

    cfg = _config(embedding_enabled=True, voyage_api_key="vk-live")
    with patch("polylogue.config.load_polylogue_config", return_value=cfg):
        info = embedding_readiness_info(tmp_path / "status.sqlite")

    assert info["embedding_failure_count"] == 1
    assert info["embedding_terminal_failure_count"] == 1
    assert info["embedding_retryable_failure_count"] == 0


# ── degrade-loudly (polylogue-cpf.4): a query failure must not look like ──
# ── a clean, empty archive ─────────────────────────────────────────────


def test_readiness_query_failure_logs_instead_of_looking_like_a_clean_archive(
    tmp_path: Path,
) -> None:
    """A transient query failure returns the same shape as "nothing to embed
    yet", so the failure must be logged loudly — otherwise it is invisible
    that ``embedding_status="empty"`` came from an error, not a fresh archive.
    """
    db = tmp_path / "status.sqlite"
    db.parent.mkdir(parents=True, exist_ok=True)
    db.touch()

    cfg = _config(embedding_enabled=True, voyage_api_key="vk-live")

    def _boom(*args: object, **kwargs: object) -> object:
        raise sqlite3.OperationalError("database is locked")

    with (
        patch("polylogue.config.load_polylogue_config", return_value=cfg),
        patch("polylogue.daemon.embedding_readiness.embedding_status_payload", side_effect=_boom),
        plog.capture() as records,
    ):
        info = embedding_readiness_info(db)

    # Same shape as a genuinely empty archive -- this is exactly the
    # ambiguity the doctrine flags. The log line is what makes the two
    # distinguishable.
    assert info["embedding_status"] == "empty"
    assert info["embedding_retrieval_ready"] is False
    # Anti-vacuity: delete the emit() in embedding_readiness and this is red.
    failures = [r for r in records if r["event"] == "daemon.embed.readiness_query_failed"]
    assert len(failures) == 1
    assert failures[0]["outcome"] == "degraded"
    assert failures[0]["level"] == "warning"
    assert failures[0]["error_type"] == "OperationalError"
    assert "database is locked" in str(failures[0]["error_detail"])

"""Daemon-side wiring for polylogue-1dk1 orphan embedding reconciliation.

Exercises ``reconcile_embedding_orphans_once`` — the bounded sync helper the
periodic daemon loop (``periodic_embedding_orphan_reconcile_check``) invokes
through the write coordinator — against real ``index.db``/``embeddings.db``
fixtures, confirming the daemon-facing gating (embedding config enabled,
index db resolvable, embeddings db present) composes correctly with the
substrate reconciliation logic already covered directly in
``tests/unit/storage/test_embedding_orphan_reconcile.py``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.core.enums import Origin
from polylogue.daemon.embedding_backlog import reconcile_embedding_orphans_once
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.embedding_write import (
    ArchiveEmbeddingWrite,
    upsert_message_embeddings,
)
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_INDEX_DDL = """
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    origin TEXT NOT NULL
);
CREATE TABLE messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL
);
"""


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    session_id = "codex-session:daemon-fixture"
    orphan_message_id = f"{session_id}:orphaned"

    index_db = tmp_path / "index.db"
    conn = sqlite3.connect(index_db)
    conn.executescript(_INDEX_DDL)
    conn.execute("INSERT INTO sessions (session_id, origin) VALUES (?, 'codex-session')", (session_id,))
    conn.commit()
    conn.close()

    embeddings_db = tmp_path / "embeddings.db"
    econn = sqlite3.connect(embeddings_db)
    try:
        initialize_archive_tier(econn, ArchiveTier.EMBEDDINGS)
    except sqlite3.OperationalError as exc:
        if "vec0" in str(exc) or "sqlite-vec" in str(exc):
            pytest.skip("sqlite-vec extension is unavailable")
        raise
    upsert_message_embeddings(
        econn,
        [
            ArchiveEmbeddingWrite(
                message_id=orphan_message_id,
                session_id=session_id,
                origin=Origin.CODEX_SESSION,
                embedding=[0.01] * EMBEDDING_DIMENSION,
                model="voyage-4",
                embedded_at_ms=1_700_000_000_000,
                content_hash=b"z" * 32,
            )
        ],
    )
    econn.execute(
        """
        INSERT INTO embedding_status (
            session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
        ) VALUES (?, 'codex-session', 1, 1700000000000, 0, NULL)
        """,
        (session_id,),
    )
    econn.commit()
    econn.close()
    return index_db, embeddings_db, session_id, orphan_message_id


def test_reconcile_embedding_orphans_once_noop_when_config_disabled(tmp_path: Path) -> None:
    index_db, _embeddings_db, _session_id, _orphan_id = _build_fixture(tmp_path)

    with patch("polylogue.daemon.convergence_stages.load_polylogue_config") as mock_cfg:
        mock_cfg.return_value.embedding_enabled = False
        mock_cfg.return_value.voyage_api_key = "test-key"
        result = reconcile_embedding_orphans_once(index_db)

    assert result is None


def test_reconcile_embedding_orphans_once_noop_when_index_missing(tmp_path: Path) -> None:
    missing_index_db = tmp_path / "index.db"  # never created

    with patch("polylogue.daemon.convergence_stages.load_polylogue_config") as mock_cfg:
        mock_cfg.return_value.embedding_enabled = True
        mock_cfg.return_value.voyage_api_key = "test-key"
        result = reconcile_embedding_orphans_once(missing_index_db)

    assert result is None


def test_reconcile_embedding_orphans_once_removes_orphan_when_enabled(tmp_path: Path) -> None:
    index_db, embeddings_db, session_id, orphan_message_id = _build_fixture(tmp_path)

    with patch("polylogue.daemon.convergence_stages.load_polylogue_config") as mock_cfg:
        mock_cfg.return_value.embedding_enabled = True
        mock_cfg.return_value.voyage_api_key = "test-key"
        result = reconcile_embedding_orphans_once(index_db)

    assert result is not None
    assert result.dry_run is False
    assert result.removed_message_rows == 1
    assert result.removed_vector_rows == 1

    conn = sqlite3.connect(embeddings_db)
    try:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM message_embeddings_meta WHERE message_id = ?", (orphan_message_id,)
            ).fetchone()[0]
            == 0
        )
        status_row = conn.execute(
            "SELECT message_count_embedded FROM embedding_status WHERE session_id = ?", (session_id,)
        ).fetchone()
        assert status_row is not None
        assert status_row[0] == 0
    finally:
        conn.close()

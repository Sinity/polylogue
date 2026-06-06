from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Origin
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.embedding_write import (
    ArchiveEmbeddingMeta,
    ArchiveEmbeddingStatus,
    mark_session_embedding_error,
    read_embedding_status,
    upsert_message_embedding,
)
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        initialize_archive_tier(conn, ArchiveTier.EMBEDDINGS)
    except sqlite3.OperationalError as exc:
        if "vec0" in str(exc) or "sqlite-vec" in str(exc):
            pytest.skip("sqlite-vec extension is unavailable")
        raise
    return conn


def test_archive_tiers_embedding_writer_upserts_vector_meta_and_status(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "embeddings.db")
    session_id = "codex-session:codex-embedding"
    message_id = f"{session_id}:m1"
    content_hash = b"x" * 32

    meta = upsert_message_embedding(
        conn,
        message_id=message_id,
        session_id=session_id,
        origin=Origin.CODEX_SESSION,
        embedding=[0.0] * EMBEDDING_DIMENSION,
        model="voyage-4",
        embedded_at_ms=1_767_225_700_000,
        content_hash=content_hash,
    )
    status = read_embedding_status(conn, session_id)

    assert meta == ArchiveEmbeddingMeta(
        target_id=message_id,
        target_type="message",
        model="voyage-4",
        dimension=EMBEDDING_DIMENSION,
        embedded_at_ms=1_767_225_700_000,
        content_hash=content_hash,
        origin="codex-session",
    )
    assert status == ArchiveEmbeddingStatus(
        session_id=session_id,
        origin="codex-session",
        message_count_embedded=1,
        last_embedded_at_ms=1_767_225_700_000,
        needs_reindex=False,
        error_message=None,
    )
    assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 1


def test_archive_tiers_embedding_writer_records_reindexable_errors(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "embeddings.db")
    status = mark_session_embedding_error(
        conn,
        session_id="codex-session:missing",
        origin=Origin.CODEX_SESSION,
        error_message="provider timeout",
    )

    assert status == ArchiveEmbeddingStatus(
        session_id="codex-session:missing",
        origin="codex-session",
        message_count_embedded=0,
        last_embedded_at_ms=None,
        needs_reindex=True,
        error_message="provider timeout",
    )

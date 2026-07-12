from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Origin
from polylogue.storage.embeddings.materialization import _record_archive_embedding_success
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.embedding_write import (
    ArchiveEmbeddingFailure,
    ArchiveEmbeddingMeta,
    ArchiveEmbeddingStatus,
    ArchiveEmbeddingWrite,
    list_active_embedding_failures,
    mark_session_embedding_error,
    read_embedding_failure,
    read_embedding_status,
    record_embedding_failure,
    resolve_embedding_failure,
    upsert_message_embedding,
    upsert_message_embeddings,
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

    assert meta == ArchiveEmbeddingMeta(
        message_id=message_id,
        model="voyage-4",
        dimension=EMBEDDING_DIMENSION,
        content_hash=content_hash,
        embedded_at_ms=1_767_225_700_000,
        needs_reindex=False,
    )
    assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 1

    # The per-message vector upsert intentionally does NOT touch
    # ``embedding_status``; that row is materialized by the session-level
    # orchestrator after a session's messages are embedded. Exercise that
    # seam directly to confirm the status row reflects the embedded session.
    _record_archive_embedding_success(
        conn,
        session_id=session_id,
        origin="codex-session",
        message_count=1,
    )
    status = read_embedding_status(conn, session_id)
    assert status.session_id == session_id
    assert status.origin == "codex-session"
    assert status.message_count_embedded == 1
    assert status.needs_reindex is False
    assert status.error_message is None
    assert isinstance(status.last_embedded_at_ms, int)


def test_archive_tiers_embedding_writer_batches_message_upserts(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "embeddings.db")
    session_id = "codex-session:codex-embedding"
    trace: list[str] = []
    conn.set_trace_callback(trace.append)

    upsert_message_embeddings(
        conn,
        [
            ArchiveEmbeddingWrite(
                message_id=f"{session_id}:m{index}",
                session_id=session_id,
                origin=Origin.CODEX_SESSION,
                embedding=[0.01] * EMBEDDING_DIMENSION,
                model="voyage-4",
                embedded_at_ms=1_767_225_700_000,
                content_hash=bytes([index]) * 32,
            )
            for index in range(3)
        ],
    )

    transaction_events = [statement for statement in trace if statement.startswith("BEGIN") or statement == "COMMIT"]
    assert transaction_events == ["BEGIN ", "COMMIT"]
    assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 3


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


def test_archive_tiers_embedding_writer_records_terminal_errors(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "embeddings.db")
    status = mark_session_embedding_error(
        conn,
        session_id="codex-session:bad-input",
        origin=Origin.CODEX_SESSION,
        error_message="Embedding generation failed: HTTP 400",
        retryable=False,
    )

    assert status == ArchiveEmbeddingStatus(
        session_id="codex-session:bad-input",
        origin="codex-session",
        message_count_embedded=0,
        last_embedded_at_ms=None,
        needs_reindex=False,
        error_message="Embedding generation failed: HTTP 400",
    )
    [failure] = list_active_embedding_failures(conn)
    assert failure.session_id == "codex-session:bad-input"
    assert failure.lifecycle_state == "terminal"
    assert failure.message_refs == ()
    assert failure.provider == "unknown"
    assert failure.error_class == "embedding_error"


def test_embedding_failure_lifecycle_preserves_audit_and_requeues(tmp_path: Path) -> None:
    """Terminal rows are inspectable, acknowledgeable, and requeueable.

    Anti-vacuity: replacing the ledger with aggregate-only status rows loses
    message/provider/error identities; deleting on resolution loses the audit;
    omitting the requeue mutation leaves the session permanently excluded.
    """
    conn = _connect(tmp_path / "embeddings.db")
    terminal = record_embedding_failure(
        conn,
        session_id="aistudio-drive:poisoned",
        origin="aistudio-drive",
        message_refs=("aistudio-drive:poisoned:m1",),
        provider="voyage",
        model="voyage-4",
        error_class="provider_http_400",
        error_message="Embedding generation failed: HTTP 400",
        retryable=False,
        occurred_at_ms=1_800_000_000_000,
    )
    assert terminal == ArchiveEmbeddingFailure(
        failure_id=terminal.failure_id,
        session_id="aistudio-drive:poisoned",
        origin="aistudio-drive",
        message_refs=("aistudio-drive:poisoned:m1",),
        provider="voyage",
        model="voyage-4",
        error_class="provider_http_400",
        error_message="Embedding generation failed: HTTP 400",
        retryable=False,
        lifecycle_state="terminal",
        created_at_ms=1_800_000_000_000,
        updated_at_ms=1_800_000_000_000,
        resolved_at_ms=None,
        resolution_action=None,
        resolution_note=None,
        superseded_by=None,
    )
    assert list_active_embedding_failures(conn) == (terminal,)

    acknowledged = resolve_embedding_failure(
        conn,
        failure_id=terminal.failure_id,
        action="acknowledge",
        note="provider rejects this historical payload",
        resolved_at_ms=1_800_000_000_100,
    )
    assert acknowledged.lifecycle_state == "acknowledged"
    assert acknowledged.resolution_action == "acknowledge"
    assert list_active_embedding_failures(conn) == ()
    assert conn.execute("SELECT COUNT(*) FROM embedding_failures").fetchone()[0] == 1
    assert read_embedding_status(conn, "aistudio-drive:poisoned").needs_reindex is False

    requeued = record_embedding_failure(
        conn,
        session_id="codex-session:retry-me",
        origin=Origin.CODEX_SESSION,
        message_refs=("codex-session:retry-me:m1",),
        provider="voyage",
        model="voyage-4",
        error_class="provider_http_400",
        error_message="Embedding generation failed: HTTP 400",
        retryable=False,
        occurred_at_ms=1_800_000_000_200,
    )
    resolved = resolve_embedding_failure(
        conn,
        failure_id=requeued.failure_id,
        action="requeue",
        resolved_at_ms=1_800_000_000_300,
    )
    assert resolved.lifecycle_state == "resolved"
    assert resolved.resolution_action == "requeue"
    status = read_embedding_status(conn, "codex-session:retry-me")
    assert status.needs_reindex is True
    assert status.error_message is None


def test_new_failure_supersedes_prior_active_failure_for_same_session(tmp_path: Path) -> None:
    """A later attempt is current debt; earlier attempts remain durable evidence.

    Anti-vacuity: deleting the lifecycle transition leaves both rows active, so
    status and archive debt overcount a single repeatedly failing session.
    """
    conn = _connect(tmp_path / "embeddings.db")
    prior = record_embedding_failure(
        conn,
        session_id="codex-session:retry-loop",
        origin=Origin.CODEX_SESSION,
        message_refs=("codex-session:retry-loop:m1",),
        provider="voyage",
        model="voyage-4",
        error_class="provider_timeout",
        error_message="Embedding generation timed out",
        retryable=True,
        occurred_at_ms=1_800_000_000_000,
    )
    current = record_embedding_failure(
        conn,
        session_id="codex-session:retry-loop",
        origin=Origin.CODEX_SESSION,
        message_refs=("codex-session:retry-loop:m2",),
        provider="voyage",
        model="voyage-4",
        error_class="provider_http_400",
        error_message="Embedding generation failed: HTTP 400",
        retryable=False,
        occurred_at_ms=1_800_000_100_000,
    )

    superseded = read_embedding_failure(conn, prior.failure_id)
    assert superseded.lifecycle_state == "superseded"
    assert superseded.resolution_action == "superseded"
    assert superseded.superseded_by == current.failure_id
    assert superseded.resolved_at_ms == current.created_at_ms
    assert list_active_embedding_failures(conn) == (current,)
    status = read_embedding_status(conn, "codex-session:retry-loop")
    assert status.needs_reindex is False
    assert status.error_message == "Embedding generation failed: HTTP 400"

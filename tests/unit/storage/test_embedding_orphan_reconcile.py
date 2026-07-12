"""Focused tests for polylogue-1dk1: cross-tier orphan embedding reconciliation.

``embeddings.db`` (message_embeddings_meta / message_embeddings vec0 /
embedding_status) is not rebuilt in lockstep with ``index.db``. These tests
build a synthetic "post index-rebuild" scenario directly — a minimal
``index.db`` fixture with a real ``embeddings.db`` (via the real
``ArchiveTier.EMBEDDINGS`` DDL + sqlite-vec) — and assert
:func:`reconcile_embedding_orphans` removes only the rows the identity guard
condemns, never the ones the content-hash and quiet-window guards protect.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.core.enums import Origin
from polylogue.storage.embeddings.materialization import select_pending_archive_session_window
from polylogue.storage.embeddings.reconcile import (
    inspect_embedding_orphans,
    reconcile_embedding_orphans,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.embedding_write import (
    ArchiveEmbeddingWrite,
    upsert_message_embeddings,
)
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION
from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_INDEX_DDL = """
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    origin TEXT NOT NULL,
    title TEXT,
    sort_key_ms INTEGER DEFAULT 0,
    authored_user_message_count INTEGER NOT NULL DEFAULT 1,
    assistant_message_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    text TEXT DEFAULT 'authored prose long enough for embedding',
    role TEXT NOT NULL DEFAULT 'user',
    message_type TEXT NOT NULL DEFAULT 'message',
    material_origin TEXT NOT NULL DEFAULT 'human_authored',
    word_count INTEGER NOT NULL DEFAULT 8,
    content_hash BLOB DEFAULT x'01'
);
"""


def _connect_index(path: Path, *, sessions: list[str], messages: dict[str, list[str]]) -> None:
    """Build a minimal synthetic ``index.db`` with only the sessions/messages
    listed — standing in for a rebuilt index that dropped some identities."""

    conn = sqlite3.connect(path)
    try:
        conn.executescript(_INDEX_DDL)
        conn.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
        for session_id in sessions:
            conn.execute(
                "INSERT INTO sessions (session_id, origin) VALUES (?, ?)",
                (session_id, "codex-session"),
            )
        for session_id, message_ids in messages.items():
            for message_id in message_ids:
                conn.execute(
                    "INSERT INTO messages (message_id, session_id) VALUES (?, ?)",
                    (message_id, session_id),
                )
        conn.commit()
    finally:
        conn.close()


def _connect_embeddings(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        initialize_archive_tier(conn, ArchiveTier.EMBEDDINGS)
    except sqlite3.OperationalError as exc:
        if "vec0" in str(exc) or "sqlite-vec" in str(exc):
            pytest.skip("sqlite-vec extension is unavailable")
        raise
    return conn


def _write_embedding(
    conn: sqlite3.Connection,
    *,
    message_id: str,
    session_id: str,
    embedded_at_ms: int,
    content_hash: bytes | None = None,
) -> None:
    upsert_message_embeddings(
        conn,
        [
            ArchiveEmbeddingWrite(
                message_id=message_id,
                session_id=session_id,
                origin=Origin.CODEX_SESSION,
                embedding=[0.01] * EMBEDDING_DIMENSION,
                model="voyage-4",
                embedded_at_ms=embedded_at_ms,
                content_hash=content_hash if content_hash is not None else (bytes([len(message_id) % 255]) * 32),
            )
        ],
    )


def _write_status(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    message_count_embedded: int,
    last_embedded_at_ms: int,
    needs_reindex: int = 0,
) -> None:
    conn.execute(
        """
        INSERT INTO embedding_status (
            session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
        ) VALUES (?, 'codex-session', ?, ?, ?, NULL)
        """,
        (session_id, message_count_embedded, last_embedded_at_ms, needs_reindex),
    )
    conn.commit()


_OLD_MS = 1_700_000_000_000  # far outside any quiet window relative to _NOW_MS
_NOW_MS = 1_800_000_000_000


def test_reconcile_removes_message_orphaned_by_index_rebuild(tmp_path: Path) -> None:
    session_id = "codex-session:s1"
    m1, m2 = f"{session_id}:m1", f"{session_id}:m2"

    # index.db post-rebuild: m2 no longer exists (e.g. a full-replace shifted
    # positions), but the session and m1 remain.
    _connect_index(tmp_path / "index.db", sessions=[session_id], messages={session_id: [m1]})

    embeddings_db = tmp_path / "embeddings.db"
    conn = _connect_embeddings(embeddings_db)
    _write_embedding(conn, message_id=m1, session_id=session_id, embedded_at_ms=_OLD_MS)
    _write_embedding(conn, message_id=m2, session_id=session_id, embedded_at_ms=_OLD_MS)
    _write_status(conn, session_id=session_id, message_count_embedded=2, last_embedded_at_ms=_OLD_MS)
    conn.close()

    report = reconcile_embedding_orphans(
        tmp_path / "index.db",
        embeddings_db,
        dry_run=False,
        now_ms=_NOW_MS,
        mutation_authority="offline-exclusive",
    )

    assert report.orphan_message_rows == 1
    assert report.removed_message_rows == 1
    assert report.removed_vector_rows == 1
    assert report.sessions_recounted == 1
    assert report.more_pending is False

    conn = sqlite3.connect(embeddings_db)
    try:
        assert (
            conn.execute("SELECT COUNT(*) FROM message_embeddings_meta WHERE message_id = ?", (m1,)).fetchone()[0] == 1
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM message_embeddings_meta WHERE message_id = ?", (m2,)).fetchone()[0] == 0
        )
        status = conn.execute(
            "SELECT message_count_embedded, needs_reindex FROM embedding_status WHERE session_id = ?", (session_id,)
        ).fetchone()
        assert status[0] == 1, "message_count_embedded must be recounted after removing the orphan"
        assert status[1] == 1, "surviving affected sessions must enter the ambient reindex backlog"
    finally:
        conn.close()

    with sqlite3.connect(tmp_path / "index.db") as selector_conn:
        selector_conn.execute("ATTACH DATABASE ? AS embeddings", (str(embeddings_db),))
        pending = select_pending_archive_session_window(
            selector_conn,
            status_table="embeddings.embedding_status",
            include_stale_checks=False,
        )
    assert [item.session_id for item in pending] == [session_id]


def test_reconcile_preserves_content_hash_mismatch_when_identity_present(tmp_path: Path) -> None:
    """Content-hash guard: a message that still exists (identity present) is
    never removed merely because its stored content_hash is stale — that is
    0k6's re-embed territory, not this reconciler's."""

    session_id = "codex-session:s2"
    m1 = f"{session_id}:m1"
    _connect_index(tmp_path / "index.db", sessions=[session_id], messages={session_id: [m1]})

    embeddings_db = tmp_path / "embeddings.db"
    conn = _connect_embeddings(embeddings_db)
    old_hash = b"a" * 32
    _write_embedding(conn, message_id=m1, session_id=session_id, embedded_at_ms=_OLD_MS, content_hash=old_hash)
    _write_status(conn, session_id=session_id, message_count_embedded=1, last_embedded_at_ms=_OLD_MS)
    conn.close()

    report = reconcile_embedding_orphans(
        tmp_path / "index.db",
        embeddings_db,
        dry_run=False,
        now_ms=_NOW_MS,
        mutation_authority="offline-exclusive",
    )

    assert report.orphan_message_rows == 0
    assert report.removed_message_rows == 0

    conn = sqlite3.connect(embeddings_db)
    try:
        row = conn.execute("SELECT content_hash FROM message_embeddings_meta WHERE message_id = ?", (m1,)).fetchone()
        assert row is not None
        assert bytes(row[0]) == old_hash, "identity-present row must be preserved untouched, hash and all"
    finally:
        conn.close()


def test_reconcile_removes_orphan_status_for_deleted_session(tmp_path: Path) -> None:
    # index.db post-rebuild has no trace of session_id at all.
    _connect_index(tmp_path / "index.db", sessions=[], messages={})

    session_id = "codex-session:gone"
    m1 = f"{session_id}:m1"
    embeddings_db = tmp_path / "embeddings.db"
    conn = _connect_embeddings(embeddings_db)
    _write_embedding(conn, message_id=m1, session_id=session_id, embedded_at_ms=_OLD_MS)
    _write_status(conn, session_id=session_id, message_count_embedded=1, last_embedded_at_ms=_OLD_MS)
    conn.close()

    report = reconcile_embedding_orphans(
        tmp_path / "index.db",
        embeddings_db,
        dry_run=False,
        now_ms=_NOW_MS,
        mutation_authority="offline-exclusive",
    )

    assert report.orphan_status_rows == 1
    assert report.removed_status_rows == 1
    assert report.removed_message_rows == 1  # message-identity join also condemns m1

    conn = sqlite3.connect(embeddings_db)
    try:
        assert (
            conn.execute("SELECT COUNT(*) FROM embedding_status WHERE session_id = ?", (session_id,)).fetchone()[0] == 0
        )
    finally:
        conn.close()


def test_reconcile_respects_quiet_window(tmp_path: Path) -> None:
    """Quiet-window guard: a candidate whose embedded_at_ms is within the
    quiet window is left alone — it may be mid-write from an in-flight
    rebuild/replace, not truly orphaned yet."""

    session_id = "codex-session:s3"
    m1 = f"{session_id}:m1"
    _connect_index(tmp_path / "index.db", sessions=[session_id], messages={session_id: []})

    embeddings_db = tmp_path / "embeddings.db"
    conn = _connect_embeddings(embeddings_db)
    recent_ms = _NOW_MS - 1_000  # 1 second ago — well within the 5 minute default window
    _write_embedding(conn, message_id=m1, session_id=session_id, embedded_at_ms=recent_ms)
    _write_status(conn, session_id=session_id, message_count_embedded=1, last_embedded_at_ms=recent_ms)
    conn.close()

    guarded = reconcile_embedding_orphans(
        tmp_path / "index.db",
        embeddings_db,
        dry_run=False,
        now_ms=_NOW_MS,
        mutation_authority="offline-exclusive",
    )
    assert guarded.removed_message_rows == 0
    assert guarded.skipped_recent_message_rows == 1
    assert guarded.more_pending is True

    conn = sqlite3.connect(embeddings_db)
    try:
        assert (
            conn.execute("SELECT COUNT(*) FROM message_embeddings_meta WHERE message_id = ?", (m1,)).fetchone()[0] == 1
        )
    finally:
        conn.close()

    unguarded = reconcile_embedding_orphans(
        tmp_path / "index.db",
        embeddings_db,
        dry_run=False,
        quiet_window_ms=0,
        now_ms=_NOW_MS,
        mutation_authority="offline-exclusive",
    )
    assert unguarded.removed_message_rows == 1


def test_reconcile_is_bounded_and_resumable(tmp_path: Path) -> None:
    session_id = "codex-session:s4"
    message_ids = [f"{session_id}:m{i}" for i in range(5)]
    _connect_index(tmp_path / "index.db", sessions=[session_id], messages={session_id: []})

    embeddings_db = tmp_path / "embeddings.db"
    conn = _connect_embeddings(embeddings_db)
    for message_id in message_ids:
        _write_embedding(conn, message_id=message_id, session_id=session_id, embedded_at_ms=_OLD_MS)
    _write_status(conn, session_id=session_id, message_count_embedded=5, last_embedded_at_ms=_OLD_MS)
    conn.close()

    first = reconcile_embedding_orphans(
        tmp_path / "index.db",
        embeddings_db,
        dry_run=False,
        max_count=2,
        now_ms=_NOW_MS,
        mutation_authority="offline-exclusive",
    )
    assert first.removed_message_rows == 2
    assert first.more_pending is True

    second = reconcile_embedding_orphans(
        tmp_path / "index.db",
        embeddings_db,
        dry_run=False,
        max_count=2,
        now_ms=_NOW_MS,
        mutation_authority="offline-exclusive",
    )
    assert second.removed_message_rows == 2
    assert second.more_pending is True

    third = reconcile_embedding_orphans(
        tmp_path / "index.db",
        embeddings_db,
        dry_run=False,
        max_count=2,
        now_ms=_NOW_MS,
        mutation_authority="offline-exclusive",
    )
    assert third.removed_message_rows == 1
    assert third.more_pending is False
    assert third.ok is True

    # Idempotent: a clean archive reports zero orphans and mutates nothing.
    clean = reconcile_embedding_orphans(
        tmp_path / "index.db",
        embeddings_db,
        dry_run=False,
        now_ms=_NOW_MS,
        mutation_authority="offline-exclusive",
    )
    assert clean.orphan_message_rows == 0
    assert clean.removed_message_rows == 0
    assert clean.ok is True


def test_reconcile_dry_run_does_not_mutate(tmp_path: Path) -> None:
    session_id = "codex-session:s5"
    m1 = f"{session_id}:m1"
    _connect_index(tmp_path / "index.db", sessions=[session_id], messages={session_id: []})

    embeddings_db = tmp_path / "embeddings.db"
    conn = _connect_embeddings(embeddings_db)
    _write_embedding(conn, message_id=m1, session_id=session_id, embedded_at_ms=_OLD_MS)
    _write_status(conn, session_id=session_id, message_count_embedded=1, last_embedded_at_ms=_OLD_MS)
    conn.close()

    report = reconcile_embedding_orphans(tmp_path / "index.db", embeddings_db, dry_run=True, now_ms=_NOW_MS)

    assert report.orphan_message_rows == 1
    assert report.removed_message_rows == 0
    assert report.more_pending is True
    assert report.samples
    assert report.samples[0].action == "would_remove"

    conn = sqlite3.connect(embeddings_db)
    try:
        assert (
            conn.execute("SELECT COUNT(*) FROM message_embeddings_meta WHERE message_id = ?", (m1,)).fetchone()[0] == 1
        )
    finally:
        conn.close()


def test_inspect_embedding_orphans_is_read_only(tmp_path: Path) -> None:
    session_id = "codex-session:s6"
    m1 = f"{session_id}:m1"
    _connect_index(tmp_path / "index.db", sessions=[session_id], messages={session_id: []})

    embeddings_db = tmp_path / "embeddings.db"
    conn = _connect_embeddings(embeddings_db)
    _write_embedding(conn, message_id=m1, session_id=session_id, embedded_at_ms=_OLD_MS)
    _write_status(conn, session_id=session_id, message_count_embedded=1, last_embedded_at_ms=_OLD_MS)
    conn.close()

    report = inspect_embedding_orphans(tmp_path / "index.db", embeddings_db, now_ms=_NOW_MS)

    assert report.dry_run is True
    assert report.orphan_message_rows == 1
    assert report.removed_message_rows == 0

    conn = sqlite3.connect(embeddings_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1
    finally:
        conn.close()


def test_reconcile_missing_embeddings_db_is_noop(tmp_path: Path) -> None:
    _connect_index(tmp_path / "index.db", sessions=[], messages={})

    report = reconcile_embedding_orphans(
        tmp_path / "index.db",
        tmp_path / "embeddings.db",
        dry_run=False,
        now_ms=_NOW_MS,
        mutation_authority="offline-exclusive",
    )

    assert report.orphan_message_rows == 0
    assert report.orphan_status_rows == 0
    assert report.more_pending is False
    assert report.ok is True


def test_apply_requires_owned_writer_authority(tmp_path: Path) -> None:
    """Anti-vacuity: removing the mutation-authority floor makes this unsafe direct apply succeed."""
    _connect_index(tmp_path / "index.db", sessions=[], messages={})
    conn = _connect_embeddings(tmp_path / "embeddings.db")
    conn.close()

    with pytest.raises(RuntimeError, match="requires daemon-coordinator or offline-exclusive"):
        reconcile_embedding_orphans(
            tmp_path / "index.db",
            tmp_path / "embeddings.db",
            dry_run=False,
            now_ms=_NOW_MS,
        )


def test_apply_refuses_live_v32_index_and_preserves_orphans(tmp_path: Path) -> None:
    """A stale active generation cannot be authoritative deletion truth.

    This reproduces the live pre-rebuild shape observed while packaged schema
    v34 was active: index v32 with orphan candidates accumulated in
    embeddings.db. Removing the schema guard makes the real rows disappear.
    """
    session_id = "codex-session:stale-index"
    message_id = f"{session_id}:orphan"
    index_db = tmp_path / "index.db"
    _connect_index(index_db, sessions=[session_id], messages={session_id: []})
    with sqlite3.connect(index_db) as index_conn:
        index_conn.execute("PRAGMA user_version = 32")

    embeddings_db = tmp_path / "embeddings.db"
    conn = _connect_embeddings(embeddings_db)
    _write_embedding(conn, message_id=message_id, session_id=session_id, embedded_at_ms=_OLD_MS)
    _write_status(conn, session_id=session_id, message_count_embedded=1, last_embedded_at_ms=_OLD_MS)
    conn.close()

    inspected = inspect_embedding_orphans(index_db, embeddings_db, now_ms=_NOW_MS)
    assert inspected.orphan_message_rows == 1

    with pytest.raises(
        RuntimeError,
        match=rf"active index is v32, packaged index is v{INDEX_SCHEMA_VERSION}",
    ):
        reconcile_embedding_orphans(
            index_db,
            embeddings_db,
            dry_run=False,
            now_ms=_NOW_MS,
            mutation_authority="daemon-coordinator",
        )

    with _connect_embeddings(embeddings_db) as verify:
        assert verify.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1
        assert verify.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 1
        assert verify.execute("SELECT COUNT(*) FROM embedding_status").fetchone()[0] == 1


def test_reconcile_counts_and_removes_meta_only_and_vec_only_orphans(tmp_path: Path) -> None:
    """Real vec0/DDL proof: row-kind counts and removals must not be inferred from identity count."""
    meta_session = "codex-session:meta-only"
    vec_session = "codex-session:vec-only"
    meta_id = f"{meta_session}:m1"
    vec_id = f"{vec_session}:m1"
    _connect_index(
        tmp_path / "index.db",
        sessions=[meta_session, vec_session],
        messages={meta_session: [], vec_session: []},
    )
    embeddings_db = tmp_path / "embeddings.db"
    conn = _connect_embeddings(embeddings_db)
    _write_embedding(conn, message_id=meta_id, session_id=meta_session, embedded_at_ms=_OLD_MS)
    _write_embedding(conn, message_id=vec_id, session_id=vec_session, embedded_at_ms=_OLD_MS)
    _write_status(conn, session_id=meta_session, message_count_embedded=1, last_embedded_at_ms=_OLD_MS)
    _write_status(conn, session_id=vec_session, message_count_embedded=1, last_embedded_at_ms=_OLD_MS)
    conn.execute("DELETE FROM message_embeddings WHERE message_id = ?", (meta_id,))
    conn.execute("DELETE FROM message_embeddings_meta WHERE message_id = ?", (vec_id,))
    conn.commit()
    conn.close()

    inspected = inspect_embedding_orphans(tmp_path / "index.db", embeddings_db, now_ms=_NOW_MS)
    assert inspected.scanned_message_meta_rows == 1
    assert inspected.scanned_vector_rows == 1
    assert inspected.orphan_message_rows == 2
    assert inspected.orphan_message_meta_rows == 1
    assert inspected.orphan_vector_rows == 1

    applied = reconcile_embedding_orphans(
        tmp_path / "index.db",
        embeddings_db,
        dry_run=False,
        now_ms=_NOW_MS,
        mutation_authority="offline-exclusive",
    )
    assert applied.removed_message_rows == 1
    assert applied.removed_vector_rows == 1
    assert applied.sessions_recounted == 2

    with _connect_embeddings(embeddings_db) as verify:
        assert verify.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 0
        assert verify.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 0
        statuses = [
            tuple(row)
            for row in verify.execute(
                "SELECT session_id, message_count_embedded, needs_reindex FROM embedding_status ORDER BY session_id"
            ).fetchall()
        ]
    assert statuses == [(meta_session, 0, 1), (vec_session, 0, 1)]


def test_reconcile_rolls_back_delete_recount_and_status_cleanup_then_retries(tmp_path: Path) -> None:
    """Failure injection enters real DDL and production transaction.

    Anti-vacuity: committing message/vector deletion before the status recount,
    or removing the enclosing rollback, leaves either embedding table changed.
    """
    session_id = "codex-session:rollback"
    message_id = f"{session_id}:orphan"
    deleted_session_id = "codex-session:deleted"
    deleted_message_id = f"{deleted_session_id}:orphan"
    _connect_index(tmp_path / "index.db", sessions=[session_id], messages={session_id: []})
    embeddings_db = tmp_path / "embeddings.db"
    conn = _connect_embeddings(embeddings_db)
    _write_embedding(conn, message_id=message_id, session_id=session_id, embedded_at_ms=_OLD_MS)
    _write_embedding(
        conn,
        message_id=deleted_message_id,
        session_id=deleted_session_id,
        embedded_at_ms=_OLD_MS,
    )
    _write_status(conn, session_id=session_id, message_count_embedded=1, last_embedded_at_ms=_OLD_MS)
    _write_status(conn, session_id=deleted_session_id, message_count_embedded=1, last_embedded_at_ms=_OLD_MS)
    conn.execute(
        f"""
        CREATE TRIGGER inject_status_cleanup_failure
        BEFORE DELETE ON embedding_status
        WHEN OLD.session_id = '{deleted_session_id}'
        BEGIN SELECT RAISE(ABORT, 'injected status cleanup failure'); END
        """
    )
    conn.commit()
    conn.close()

    with pytest.raises(sqlite3.IntegrityError, match="injected status cleanup failure"):
        reconcile_embedding_orphans(
            tmp_path / "index.db",
            embeddings_db,
            dry_run=False,
            now_ms=_NOW_MS,
            mutation_authority="offline-exclusive",
        )

    with _connect_embeddings(embeddings_db) as verify:
        assert verify.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 2
        assert verify.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 2
        status = verify.execute(
            "SELECT message_count_embedded, needs_reindex FROM embedding_status WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert status is not None and tuple(status) == (1, 0)
        assert (
            verify.execute(
                "SELECT COUNT(*) FROM embedding_status WHERE session_id = ?", (deleted_session_id,)
            ).fetchone()[0]
            == 1
        )
        verify.execute("DROP TRIGGER inject_status_cleanup_failure")
        verify.commit()

    retry = reconcile_embedding_orphans(
        tmp_path / "index.db",
        embeddings_db,
        dry_run=False,
        now_ms=_NOW_MS,
        mutation_authority="offline-exclusive",
    )
    assert retry.removed_message_rows == 2
    assert retry.removed_vector_rows == 2
    assert retry.removed_status_rows == 1
    assert retry.sessions_recounted == 1
    assert retry.more_pending is False


def test_index_identity_is_revalidated_before_atomic_commit(tmp_path: Path) -> None:
    """Anti-vacuity: removing the final identity check commits against stale index truth."""
    session_id = "codex-session:generation-race"
    message_id = f"{session_id}:orphan"
    _connect_index(tmp_path / "index.db", sessions=[session_id], messages={session_id: []})
    embeddings_db = tmp_path / "embeddings.db"
    conn = _connect_embeddings(embeddings_db)
    _write_embedding(conn, message_id=message_id, session_id=session_id, embedded_at_ms=_OLD_MS)
    _write_status(conn, session_id=session_id, message_count_embedded=1, last_embedded_at_ms=_OLD_MS)
    conn.close()

    with (
        patch(
            "polylogue.storage.embeddings.reconcile._assert_index_identity",
            side_effect=[None, RuntimeError("active index generation changed")],
        ),
        pytest.raises(RuntimeError, match="generation changed"),
    ):
        reconcile_embedding_orphans(
            tmp_path / "index.db",
            embeddings_db,
            dry_run=False,
            now_ms=_NOW_MS,
            mutation_authority="daemon-coordinator",
        )

    with _connect_embeddings(embeddings_db) as verify:
        assert verify.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1
        assert verify.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 1
        status = verify.execute(
            "SELECT message_count_embedded, needs_reindex FROM embedding_status WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert status is not None and tuple(status) == (1, 0)

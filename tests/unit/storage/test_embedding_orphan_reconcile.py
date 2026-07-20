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

import hashlib
import json
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


def _connect_index(
    path: Path,
    *,
    sessions: list[str],
    messages: dict[str, list[str]],
    authoritative_generation: bool = True,
) -> None:
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
    if authoritative_generation:
        generations = path.parent / ".index-generations" / "gen-current"
        generations.mkdir(parents=True)
        (path.parent / ".index-active-pointer").write_text(str(path.resolve()), encoding="utf-8")
        (generations / "generation.json").write_text(
            json.dumps(
                {
                    "generation_id": "gen-current",
                    "owner_id": "test",
                    "archive_root": str(path.parent),
                    "index_path": str(path),
                    "state": "active",
                    "created_at_ms": _NOW_MS,
                    "source_snapshot": "source-at-rebuild-start",
                }
            ),
            encoding="utf-8",
        )


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
    embedding_input_hash: bytes | None = None,
) -> None:
    """Write one message's vector/meta/ref via the real v4 content-addressed writer.

    Defaults to a hash derived from ``message_id`` (unique per message) unless
    an explicit ``embedding_input_hash`` is supplied -- e.g. to simulate two
    messages whose current embedder input text is identical and therefore
    dedups onto one shared vector row.
    """
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
                embedding_input_hash=(
                    embedding_input_hash
                    if embedding_input_hash is not None
                    else hashlib.sha256(message_id.encode("utf-8")).digest()
                ),
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
    # v4: message_embeddings/message_embeddings_meta are content-addressed and
    # shared -- this reconciler only ever removes the orphan message_id ->
    # hash *ref*, never the vector/meta row itself (module docstring).
    assert report.removed_vector_rows == 0
    assert report.sessions_recounted == 1
    assert report.more_pending is False

    conn = sqlite3.connect(embeddings_db)
    try:
        assert (
            conn.execute("SELECT COUNT(*) FROM message_embedding_refs WHERE message_id = ?", (m1,)).fetchone()[0] == 1
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM message_embedding_refs WHERE message_id = ?", (m2,)).fetchone()[0] == 0
        )
        # The trivially-simplified deeper invariant behind the old (now
        # removed) content-hash-mismatch test: an identity-present message's
        # vector/meta row is never touched by this reconciler -- and neither
        # is an orphaned message's, which is the new, stronger form of that
        # guarantee (there is no content_hash column left to compare against).
        m2_hash = hashlib.sha256(m2.encode("utf-8")).digest()
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM message_embeddings_meta WHERE embedding_input_hash = ?", (m2_hash,)
            ).fetchone()[0]
            == 1
        ), "an orphaned ref's underlying vector/meta row survives untouched -- reference-counted GC is out of scope"
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
        )
    assert [item.session_id for item in pending] == [session_id]


def test_apply_accepts_generation_metadata_beside_external_pointer_anchor(tmp_path: Path) -> None:
    """A public archive symlink must use the anchor tier's generation metadata.

    Anti-vacuity: looking beside ``public_root/index.db`` rather than beside
    the active-pointer anchor makes this production-shaped layout refuse the
    otherwise authorized orphan deletion.
    """
    public_root = tmp_path / "archive"
    tier_root = tmp_path / "db-tier"
    public_root.mkdir()
    tier_root.mkdir()
    session_id = "codex-session:external-tier"
    message_id = f"{session_id}:orphan"
    tier_index = tier_root / "index.db"
    _connect_index(tier_index, sessions=[session_id], messages={session_id: []}, authoritative_generation=False)

    public_index = public_root / "index.db"
    public_index.symlink_to(tier_index)
    (public_root / ".index-active-pointer").write_text(str(tier_index), encoding="utf-8")
    generation_dir = tier_root / ".index-generations" / "gen-current"
    generation_dir.mkdir(parents=True)
    (generation_dir / "generation.json").write_text(
        json.dumps(
            {
                "generation_id": "gen-current",
                "owner_id": "test",
                "archive_root": str(public_root),
                "index_path": str(tier_index),
                "state": "active",
                "created_at_ms": _NOW_MS,
                "source_snapshot": "source-at-rebuild-start",
            }
        ),
        encoding="utf-8",
    )

    embeddings_db = public_root / "embeddings.db"
    conn = _connect_embeddings(embeddings_db)
    _write_embedding(conn, message_id=message_id, session_id=session_id, embedded_at_ms=_OLD_MS)
    conn.close()

    report = reconcile_embedding_orphans(
        public_index,
        embeddings_db,
        dry_run=False,
        now_ms=_NOW_MS,
        mutation_authority="offline-exclusive",
    )

    assert report.removed_message_rows == 1
    # v4: the vector/meta row is content-addressed and never deleted by this
    # reconciler -- only the message_id -> hash ref is removed.
    assert report.removed_vector_rows == 0
    with _connect_embeddings(embeddings_db) as verify:
        assert verify.execute("SELECT COUNT(*) FROM message_embedding_refs").fetchone()[0] == 0
        assert verify.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1
        assert verify.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 1


# NOTE: the old `test_reconcile_preserves_content_hash_mismatch_when_identity_
# present` test verified that a message which still exists in the index is
# never removed merely because its recorded content_hash had gone stale --
# that mismatch-tolerance mechanism doesn't exist under v4 (there is no
# content_hash column on message_embedding_refs/message_embeddings_meta left
# to compare; presence alone is the check). The deeper invariant it protected
# — an identity-present message's underlying row is never disturbed by this
# reconciler — is now folded into
# `test_reconcile_removes_orphan_ref_but_preserves_shared_vector_and_meta_rows`
# below, which additionally proves the *stronger* v4 claim: even an orphaned
# message's now-refless vector/meta row is left untouched, since reference-
# counted GC is explicitly out of scope (see reconcile.py's module docstring).


def test_reconcile_removes_orphan_ref_but_preserves_shared_vector_and_meta_rows(tmp_path: Path) -> None:
    """v4: message_embeddings/message_embeddings_meta are content-addressed and
    shared across messages/sessions -- removing one message's orphan ref must
    never touch the vector/meta row a still-live message legitimately shares
    the same hash with (dedup), nor the row belonging to an orphan alone.
    Reference-counted vector GC is explicitly out of scope (module docstring).
    """

    live_session = "codex-session:live"
    orphan_session = "codex-session:orphan"
    live_id = f"{live_session}:m1"
    orphan_id = f"{orphan_session}:m1"
    shared_hash = b"\x42" * 32

    # orphan_session/orphan_id is intentionally absent from index.db --
    # standing in for a rebuild that dropped it while live_session survived.
    _connect_index(tmp_path / "index.db", sessions=[live_session], messages={live_session: [live_id]})

    embeddings_db = tmp_path / "embeddings.db"
    conn = _connect_embeddings(embeddings_db)
    _write_embedding(
        conn, message_id=live_id, session_id=live_session, embedded_at_ms=_OLD_MS, embedding_input_hash=shared_hash
    )
    _write_embedding(
        conn,
        message_id=orphan_id,
        session_id=orphan_session,
        embedded_at_ms=_OLD_MS,
        embedding_input_hash=shared_hash,
    )
    _write_status(conn, session_id=live_session, message_count_embedded=1, last_embedded_at_ms=_OLD_MS)
    assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1, (
        "both messages share one hash -- the write-once vector/meta row is written exactly once"
    )
    assert conn.execute("SELECT COUNT(*) FROM message_embedding_refs").fetchone()[0] == 2
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
    assert report.removed_vector_rows == 0, "vectors are content-addressed/shared -- never deleted by this reconciler"

    with _connect_embeddings(embeddings_db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM message_embedding_refs WHERE message_id = ?", (orphan_id,)).fetchone()[0]
            == 0
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM message_embedding_refs WHERE message_id = ?", (live_id,)).fetchone()[0]
            == 1
        )
        # The shared vector/meta row survives untouched -- the live message
        # can still resolve its embedding through it.
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM message_embeddings_meta WHERE embedding_input_hash = ?", (shared_hash,)
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM message_embeddings WHERE embedding_input_hash = ?", (shared_hash.hex(),)
            ).fetchone()[0]
            == 1
        )


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
            conn.execute("SELECT COUNT(*) FROM message_embedding_refs WHERE message_id = ?", (m1,)).fetchone()[0] == 1
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
            conn.execute("SELECT COUNT(*) FROM message_embedding_refs WHERE message_id = ?", (m1,)).fetchone()[0] == 1
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


# NOTE: the old `test_reconcile_counts_and_removes_meta_only_and_vec_only_
# orphans` test manually deleted rows straight out of message_embeddings /
# message_embeddings_meta (bypassing the writer) to construct a "meta row
# exists but vector row doesn't" / vice-versa scenario, then asserted the
# reconciler's has_meta/has_vector row-kind counts and removals differed
# accordingly. Under v4 that distinction is gone on both sides: the tables
# are keyed by embedding_input_hash, not message_id (so a raw `DELETE ...
# WHERE message_id = ?` against them no longer targets anything meaningful),
# and `_orphan_message_rows` reports has_meta/has_vector as always-true from
# a ref's perspective since a ref only ever points at a hash written
# atomically with both (see reconcile.py's `_orphan_message_rows`
# docstring) -- this reconciler doesn't distinguish or delete either kind
# any more. The test is removed as vacuous under the new schema; its
# "removal is scoped to real DDL, not inferred" spirit lives on in
# `test_reconcile_removes_orphan_ref_but_preserves_shared_vector_and_meta_rows`
# above, which proves against real vec0 + STRICT-table DDL that a ref
# removal never touches the surviving content-addressed row.


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
    assert retry.removed_vector_rows == 0  # v4: vectors/meta are content-addressed, never deleted here
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


def test_apply_refuses_inactive_generation_even_when_schema_matches(tmp_path: Path) -> None:
    """Only the source-snapshotted active generation may authorize deletion.

    Anti-vacuity: removing the generation-authority guard lets an inactive
    v35 rebuild candidate delete this real orphan vector and metadata row.
    """
    session_id = "codex-session:inactive-generation"
    message_id = f"{session_id}:orphan"
    index_db = tmp_path / "index.db"
    _connect_index(
        index_db,
        sessions=[session_id],
        messages={session_id: []},
        authoritative_generation=False,
    )
    generations = tmp_path / ".index-generations" / "gen-inactive"
    generations.mkdir(parents=True)
    (tmp_path / ".index-active-pointer").write_text(str(index_db.resolve()), encoding="utf-8")
    (generations / "generation.json").write_text(
        json.dumps(
            {
                "generation_id": "gen-inactive",
                "owner_id": "test",
                "archive_root": str(tmp_path),
                "index_path": str(index_db),
                "state": "inactive",
                "created_at_ms": _NOW_MS,
                "source_snapshot": "source-at-rebuild-start",
            }
        ),
        encoding="utf-8",
    )
    embeddings_db = tmp_path / "embeddings.db"
    conn = _connect_embeddings(embeddings_db)
    _write_embedding(conn, message_id=message_id, session_id=session_id, embedded_at_ms=_OLD_MS)
    conn.close()

    with pytest.raises(RuntimeError, match="active source-snapshotted index generation"):
        reconcile_embedding_orphans(
            index_db,
            embeddings_db,
            dry_run=False,
            now_ms=_NOW_MS,
            mutation_authority="offline-exclusive",
        )

    with _connect_embeddings(embeddings_db) as verify:
        assert verify.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1
        assert verify.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 1


def test_apply_refuses_generationless_index_even_when_schema_matches(tmp_path: Path) -> None:
    """Deletion requires rebuild-readiness evidence, not schema version alone.

    Anti-vacuity: restoring the generation-less compatibility path lets a
    current-schema index with no active source snapshot delete this orphan.
    """
    session_id = "codex-session:generationless"
    message_id = f"{session_id}:orphan"
    index_db = tmp_path / "index.db"
    _connect_index(
        index_db,
        sessions=[session_id],
        messages={session_id: []},
        authoritative_generation=False,
    )
    embeddings_db = tmp_path / "embeddings.db"
    conn = _connect_embeddings(embeddings_db)
    _write_embedding(conn, message_id=message_id, session_id=session_id, embedded_at_ms=_OLD_MS)
    conn.close()

    with pytest.raises(RuntimeError, match="requires an active index generation pointer"):
        reconcile_embedding_orphans(
            index_db,
            embeddings_db,
            dry_run=False,
            now_ms=_NOW_MS,
            mutation_authority="offline-exclusive",
        )

    with _connect_embeddings(embeddings_db) as verify:
        assert verify.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1
        assert verify.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0] == 1

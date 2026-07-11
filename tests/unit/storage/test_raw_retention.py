"""Superseded live raw snapshot cleanup contracts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.blob_store import BlobStore
from polylogue.storage.raw_retention import (
    RawRetentionAuthority,
    RawRetentionSafetyError,
    active_raw_retention_authority,
    cleanup_superseded_raw_snapshots,
    protected_active_raw_revision_ids,
    superseded_raw_snapshot_candidates,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _write_blob(store: BlobStore, payload: bytes) -> tuple[str, int]:
    return store.write_from_bytes(payload)


def _ensure_archive_source_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE raw_sessions (
            raw_id TEXT PRIMARY KEY,
            origin TEXT NOT NULL,
            native_id TEXT,
            source_path TEXT NOT NULL,
            source_index INTEGER NOT NULL DEFAULT 0,
            blob_hash BLOB NOT NULL CHECK(length(blob_hash) = 32),
            blob_size INTEGER NOT NULL CHECK(blob_size >= 0),
            acquired_at_ms INTEGER NOT NULL
        ) STRICT"""
    )
    conn.execute(
        """CREATE TABLE blob_refs (
            blob_hash BLOB NOT NULL CHECK(length(blob_hash) = 32),
            ref_id TEXT NOT NULL,
            ref_type TEXT NOT NULL CHECK(ref_type IN ('raw_payload', 'attachment', 'sidecar')),
            source_path TEXT,
            size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
            acquired_at_ms INTEGER NOT NULL,
            PRIMARY KEY(blob_hash, ref_type, ref_id)
        ) STRICT"""
    )


def _insert_archive_raw_session(
    conn: sqlite3.Connection,
    *,
    raw_id: str,
    source_path: Path,
    source_index: int,
    blob_hash: str,
    blob_size: int,
    acquired_at_ms: int,
) -> None:
    conn.execute(
        """
        INSERT INTO raw_sessions (
            raw_id, origin, native_id, source_path, source_index,
            blob_hash, blob_size, acquired_at_ms
        ) VALUES (?, 'codex', ?, ?, ?, ?, ?, ?)
        """,
        (raw_id, raw_id, str(source_path), source_index, bytes.fromhex(blob_hash), blob_size, acquired_at_ms),
    )
    conn.execute(
        """
        INSERT INTO blob_refs (
            blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms
        ) VALUES (?, ?, 'raw_payload', ?, ?, ?)
        """,
        (bytes.fromhex(blob_hash), raw_id, str(source_path), blob_size, acquired_at_ms),
    )


def _insert_revision_raw(
    conn: sqlite3.Connection,
    *,
    raw_id: str,
    source_path: Path,
    acquired_at_ms: int,
    kind: str,
    source_revision: str,
    generation: int,
    blob_size: int,
    predecessor_raw_id: str | None = None,
    predecessor_revision: str | None = None,
    baseline_raw_id: str | None = None,
    append_start_offset: int | None = None,
    append_end_offset: int | None = None,
    authority: str = "byte_proven",
) -> None:
    conn.execute(
        """
        INSERT INTO raw_sessions (
            raw_id, origin, native_id, source_path, source_index, blob_hash,
            blob_size, acquired_at_ms, logical_source_key, revision_kind,
            source_revision, predecessor_source_revision, predecessor_raw_id,
            baseline_raw_id, append_start_offset, append_end_offset,
            acquisition_generation, revision_authority
        ) VALUES (?, 'codex-session', ?, ?, ?, ?, ?, ?, 'codex:session-1', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            raw_id,
            raw_id,
            str(source_path),
            -1 if kind == "append" else 0,
            acquired_at_ms.to_bytes(32, "big"),
            blob_size,
            acquired_at_ms,
            kind,
            source_revision,
            predecessor_revision,
            predecessor_raw_id,
            baseline_raw_id,
            append_start_offset,
            append_end_offset,
            generation,
            authority,
        ),
    )


def _seed_index_authority(
    index_db_path: Path,
    *,
    session_raw_id: str,
    accepted_raw_id: str,
    accepted_revision: str,
    generation: int,
    frontier: int,
) -> None:
    with sqlite3.connect(index_db_path) as conn:
        conn.execute(
            """
            INSERT INTO sessions (native_id, origin, raw_id, title, content_hash)
            VALUES ('session-1', 'codex-session', ?, 'session', ?)
            """,
            (session_raw_id, bytes(32)),
        )
        conn.execute(
            """
            INSERT INTO raw_revision_heads (
                logical_source_key, session_id, accepted_raw_id,
                accepted_source_revision, accepted_content_hash,
                accepted_frontier_kind, accepted_frontier,
                acquisition_generation, append_end_offset, decided_at_ms
            ) VALUES ('codex:session-1', 'codex-session:session-1', ?, ?, ?,
                      'byte', ?, ?, ?, 1)
            """,
            (accepted_raw_id, accepted_revision, bytes(32), frontier, generation, frontier),
        )


def _seed_superseded_application(
    index_db_path: Path,
    *,
    raw_id: str,
    source_revision: str,
    generation: int,
    accepted_raw_id: str,
    accepted_revision: str,
) -> None:
    with sqlite3.connect(index_db_path) as conn:
        conn.execute(
            """
            INSERT INTO raw_revision_applications (
                decision_id, raw_id, session_id, logical_source_key,
                source_revision, acquisition_generation, decision,
                accepted_raw_id, accepted_source_revision, accepted_content_hash,
                detail, decided_at_ms
            ) VALUES (?, ?, 'codex-session:session-1', 'codex:session-1', ?, ?,
                      'superseded', ?, ?, ?, 'superseded by accepted full', 2)
            """,
            (
                f"decision-{raw_id}",
                raw_id,
                source_revision,
                generation,
                accepted_raw_id,
                accepted_revision,
                bytes(32),
            ),
        )


def test_active_raw_protection_joins_index_seeds_to_transitive_source_chain(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    source_path = tmp_path / "session.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        _insert_revision_raw(
            conn,
            raw_id="raw-session-only",
            source_path=source_path,
            acquired_at_ms=1,
            kind="unknown",
            source_revision="legacy",
            generation=0,
            blob_size=5,
            authority="quarantined",
        )
        _insert_revision_raw(
            conn,
            raw_id="raw-baseline",
            source_path=source_path,
            acquired_at_ms=2,
            kind="full",
            source_revision="revision-0",
            generation=0,
            blob_size=10,
        )
        _insert_revision_raw(
            conn,
            raw_id="raw-append-1",
            source_path=source_path,
            acquired_at_ms=3,
            kind="append",
            source_revision="revision-1",
            generation=1,
            blob_size=5,
            predecessor_raw_id="raw-baseline",
            predecessor_revision="revision-0",
            baseline_raw_id="raw-baseline",
            append_start_offset=10,
            append_end_offset=15,
        )
        _insert_revision_raw(
            conn,
            raw_id="raw-append-2",
            source_path=source_path,
            acquired_at_ms=4,
            kind="append",
            source_revision="revision-2",
            generation=2,
            blob_size=5,
            predecessor_raw_id="raw-append-1",
            predecessor_revision="revision-1",
            baseline_raw_id="raw-baseline",
            append_start_offset=15,
            append_end_offset=20,
        )
        conn.commit()
    _seed_index_authority(
        index_db,
        session_raw_id="raw-session-only",
        accepted_raw_id="raw-append-2",
        accepted_revision="revision-2",
        generation=2,
        frontier=20,
    )

    with sqlite3.connect(source_db) as conn:
        protected = protected_active_raw_revision_ids(conn, index_db_path=index_db)

    # Anti-vacuity: removing either index seed query loses raw-session-only or
    # raw-append-2; removing predecessor traversal loses both earlier links.
    assert protected == frozenset({"raw-session-only", "raw-baseline", "raw-append-1", "raw-append-2"})


def test_active_full_head_resets_retention_chain(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    source_path = tmp_path / "session.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        _insert_revision_raw(
            conn,
            raw_id="raw-old-full",
            source_path=source_path,
            acquired_at_ms=1,
            kind="full",
            source_revision="revision-old",
            generation=0,
            blob_size=10,
        )
        _insert_revision_raw(
            conn,
            raw_id="raw-new-full",
            source_path=source_path,
            acquired_at_ms=2,
            kind="full",
            source_revision="revision-new",
            generation=1,
            blob_size=20,
            predecessor_raw_id="raw-old-full",
            baseline_raw_id="raw-old-full",
        )
        conn.commit()
    _seed_index_authority(
        index_db,
        session_raw_id="raw-new-full",
        accepted_raw_id="raw-new-full",
        accepted_revision="revision-new",
        generation=1,
        frontier=20,
    )
    _seed_superseded_application(
        index_db,
        raw_id="raw-old-full",
        source_revision="revision-old",
        generation=0,
        accepted_raw_id="raw-new-full",
        accepted_revision="revision-new",
    )

    with sqlite3.connect(source_db) as conn:
        authority = active_raw_retention_authority(conn, index_db_path=index_db)

    # Anti-vacuity: following a full raw's historical predecessor would retain
    # raw-old-full and defeat the self-contained full reset contract.
    assert authority == RawRetentionAuthority(
        protected_raw_ids=frozenset({"raw-new-full"}),
        eligible_raw_ids=frozenset({"raw-old-full"}),
    )


@pytest.mark.parametrize("index_kind", ["missing", "malformed"])
def test_active_raw_protection_rejects_unreadable_index(tmp_path: Path, index_kind: str) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    if index_kind == "malformed":
        index_db.write_bytes(b"not sqlite")

    with (
        sqlite3.connect(source_db) as conn,
        pytest.raises(
            RawRetentionSafetyError,
            match="unavailable|unreadable",
        ),
    ):
        protected_active_raw_revision_ids(conn, index_db_path=index_db)


def test_active_raw_protection_rejects_empty_index_over_retained_source(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    source_path = tmp_path / "session.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        _insert_revision_raw(
            conn,
            raw_id="raw-baseline",
            source_path=source_path,
            acquired_at_ms=1,
            kind="full",
            source_revision="revision-0",
            generation=0,
            blob_size=10,
        )
        _insert_revision_raw(
            conn,
            raw_id="raw-append",
            source_path=source_path,
            acquired_at_ms=2,
            kind="append",
            source_revision="revision-1",
            generation=1,
            blob_size=5,
            predecessor_raw_id="raw-baseline",
            predecessor_revision="revision-0",
            baseline_raw_id="raw-baseline",
            append_start_offset=10,
            append_end_offset=15,
        )

    with (
        sqlite3.connect(source_db) as conn,
        pytest.raises(RawRetentionSafetyError, match="index has no raw authority"),
    ):
        active_raw_retention_authority(conn, index_db_path=index_db)

    with sqlite3.connect(source_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (2,)


def test_active_raw_protection_rejects_incomplete_predecessor_chain(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    source_path = tmp_path / "session.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        _insert_revision_raw(
            conn,
            raw_id="raw-append",
            source_path=source_path,
            acquired_at_ms=1,
            kind="append",
            source_revision="revision-1",
            generation=1,
            blob_size=5,
            predecessor_raw_id="raw-missing",
            predecessor_revision="revision-0",
            baseline_raw_id="raw-missing",
            append_start_offset=10,
            append_end_offset=15,
        )
        conn.commit()
    _seed_index_authority(
        index_db,
        session_raw_id="raw-append",
        accepted_raw_id="raw-append",
        accepted_revision="revision-1",
        generation=1,
        frontier=15,
    )

    with (
        sqlite3.connect(source_db) as conn,
        pytest.raises(
            RawRetentionSafetyError,
            match="missing from source tier",
        ),
    ):
        protected_active_raw_revision_ids(conn, index_db_path=index_db)


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        ("logical_source", "crosses logical sources"),
        ("predecessor_revision", "predecessor revision does not match"),
        ("offset", "not byte-contiguous"),
        ("generation", "generation does not match"),
        ("baseline", "wrong baseline"),
    ],
)
def test_active_raw_protection_rejects_corrupt_chain_invariants(
    tmp_path: Path,
    mutation: str,
    error_match: str,
) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    source_path = tmp_path / "session.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        _insert_revision_raw(
            conn,
            raw_id="raw-baseline",
            source_path=source_path,
            acquired_at_ms=1,
            kind="full",
            source_revision="revision-0",
            generation=0,
            blob_size=10,
        )
        _insert_revision_raw(
            conn,
            raw_id="raw-append",
            source_path=source_path,
            acquired_at_ms=2,
            kind="append",
            source_revision="revision-1",
            generation=1,
            blob_size=5,
            predecessor_raw_id="raw-baseline",
            predecessor_revision="revision-0",
            baseline_raw_id="raw-baseline",
            append_start_offset=10,
            append_end_offset=15,
        )
        if mutation == "logical_source":
            conn.execute("UPDATE raw_sessions SET logical_source_key = 'codex:other' WHERE raw_id = 'raw-baseline'")
        elif mutation == "predecessor_revision":
            conn.execute("UPDATE raw_sessions SET predecessor_source_revision = 'wrong' WHERE raw_id = 'raw-append'")
        elif mutation == "offset":
            conn.execute("UPDATE raw_sessions SET append_start_offset = 9 WHERE raw_id = 'raw-append'")
        elif mutation == "generation":
            conn.execute("UPDATE raw_sessions SET acquisition_generation = 2 WHERE raw_id = 'raw-append'")
        elif mutation == "baseline":
            conn.execute("UPDATE raw_sessions SET baseline_raw_id = 'raw-other' WHERE raw_id = 'raw-append'")
        else:
            raise AssertionError(mutation)
        conn.commit()
    _seed_index_authority(
        index_db,
        session_raw_id="raw-append",
        accepted_raw_id="raw-append",
        accepted_revision="revision-1",
        generation=1,
        frontier=15,
    )

    with sqlite3.connect(source_db) as conn, pytest.raises(RawRetentionSafetyError, match=error_match):
        protected_active_raw_revision_ids(conn, index_db_path=index_db)

    with sqlite3.connect(source_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (2,)


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        ("revision", "revision disagrees"),
        ("generation", "generation disagrees"),
        ("source_end", "frontier disagrees"),
        ("index_frontier", "frontier disagrees"),
    ],
)
def test_active_raw_protection_rejects_index_head_mismatch(
    tmp_path: Path,
    mutation: str,
    error_match: str,
) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    source_path = tmp_path / "session.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        _insert_revision_raw(
            conn,
            raw_id="raw-baseline",
            source_path=source_path,
            acquired_at_ms=1,
            kind="full",
            source_revision="revision-0",
            generation=0,
            blob_size=10,
        )
        _insert_revision_raw(
            conn,
            raw_id="raw-append",
            source_path=source_path,
            acquired_at_ms=2,
            kind="append",
            source_revision="revision-1",
            generation=1,
            blob_size=5,
            predecessor_raw_id="raw-baseline",
            predecessor_revision="revision-0",
            baseline_raw_id="raw-baseline",
            append_start_offset=10,
            append_end_offset=15,
        )
    _seed_index_authority(
        index_db,
        session_raw_id="raw-append",
        accepted_raw_id="raw-append",
        accepted_revision="revision-1",
        generation=1,
        frontier=15,
    )
    if mutation == "source_end":
        with sqlite3.connect(source_db) as conn:
            conn.execute("UPDATE raw_sessions SET append_end_offset = 14 WHERE raw_id = 'raw-append'")
    else:
        with sqlite3.connect(index_db) as conn:
            if mutation == "revision":
                conn.execute("UPDATE raw_revision_heads SET accepted_source_revision = 'wrong'")
            elif mutation == "generation":
                conn.execute("UPDATE raw_revision_heads SET acquisition_generation = 2")
            elif mutation == "index_frontier":
                conn.execute("UPDATE raw_revision_heads SET accepted_frontier = 14")
            else:
                raise AssertionError(mutation)

    with sqlite3.connect(source_db) as conn, pytest.raises(RawRetentionSafetyError, match=error_match):
        active_raw_retention_authority(conn, index_db_path=index_db)


def test_superseded_raw_snapshot_cleanup_keeps_newest_per_source(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    source = tmp_path / "rollout.jsonl"
    missing_source = tmp_path / "missing.jsonl"
    source.write_text('{"type":"message"}\n', encoding="utf-8")
    blob_store = BlobStore(tmp_path / "blob")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_archive_source_schema(conn)

    full_old, full_old_size = _write_blob(blob_store, b"full-old")
    full_new, full_new_size = _write_blob(blob_store, b"full-new")
    append_old, append_old_size = _write_blob(blob_store, b"append-old")
    append_current, append_current_size = _write_blob(blob_store, b"append-current")
    leased_old, leased_old_size = _write_blob(blob_store, b"leased-old")
    missing_old, missing_old_size = _write_blob(blob_store, b"missing-old")
    missing_new, missing_new_size = _write_blob(blob_store, b"missing-new")

    # Archive file-set retention ranks snapshots by recency, but callers must
    # protect raw rows still referenced by index.db sessions before deleting.
    def _seed(raw_id: str, source_path: Path, source_index: int, blob_size: int, acquired_at_ms: int) -> None:
        _insert_archive_raw_session(
            conn,
            raw_id=raw_id,
            source_path=source_path,
            source_index=source_index,
            blob_hash=raw_id,
            blob_size=blob_size,
            acquired_at_ms=acquired_at_ms,
        )

    _seed(full_old, source, 0, full_old_size, 1_000)
    _seed(full_new, source, 0, full_new_size, 2_000)
    _seed(append_old, source, -1, append_old_size, 3_000)
    _seed(append_current, source, -1, append_current_size, 4_000)
    _seed(leased_old, source, -1, leased_old_size, 2_500)
    _seed(missing_old, missing_source, 0, missing_old_size, 1_000)
    _seed(missing_new, missing_source, 0, missing_new_size, 2_000)
    conn.commit()

    # full_old (superseded by full_new) and append_old + leased_old (superseded
    # by append_current). missing_old is superseded too, but its source file is
    # gone, so it is excluded from candidates.
    candidates = superseded_raw_snapshot_candidates(conn, limit=100)
    assert {candidate.raw_id for candidate in candidates} == {full_old, append_old, leased_old}

    dry_run = cleanup_superseded_raw_snapshots(conn, dry_run=True, blob_store=blob_store)
    assert dry_run.candidate_count == 3
    assert blob_store.exists(full_old)
    assert blob_store.exists(append_old)
    assert blob_store.exists(leased_old)

    result = cleanup_superseded_raw_snapshots(conn, dry_run=False, blob_store=blob_store)
    assert result.deleted_raw_count == 3
    assert result.deleted_blob_count == 3
    assert not blob_store.exists(full_old)
    assert not blob_store.exists(append_old)
    assert not blob_store.exists(leased_old)
    assert blob_store.exists(full_new)
    assert blob_store.exists(append_current)
    assert blob_store.exists(missing_old)
    assert blob_store.exists(missing_new)

    remaining_raw_ids = {
        str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions ORDER BY raw_id").fetchall()
    }
    assert remaining_raw_ids == {full_new, append_current, missing_old, missing_new}
    remaining_ref_ids = {str(row[0]) for row in conn.execute("SELECT ref_id FROM blob_refs ORDER BY ref_id").fetchall()}
    assert remaining_ref_ids == {full_new, append_current, missing_old, missing_new}


def test_superseded_raw_snapshot_cleanup_preserves_index_referenced_raws(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    source = tmp_path / "rollout.jsonl"
    source.write_text('{"type":"message"}\n', encoding="utf-8")
    blob_store = BlobStore(tmp_path / "blob")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_archive_source_schema(conn)

    old_raw, old_size = _write_blob(blob_store, b"old-but-index-referenced")
    current_raw, current_size = _write_blob(blob_store, b"current")
    _insert_archive_raw_session(
        conn,
        raw_id=old_raw,
        source_path=source,
        source_index=0,
        blob_hash=old_raw,
        blob_size=old_size,
        acquired_at_ms=1_000,
    )
    _insert_archive_raw_session(
        conn,
        raw_id=current_raw,
        source_path=source,
        source_index=0,
        blob_hash=current_raw,
        blob_size=current_size,
        acquired_at_ms=2_000,
    )
    conn.commit()

    dry_run = cleanup_superseded_raw_snapshots(
        conn,
        dry_run=True,
        blob_store=blob_store,
        protected_raw_ids={old_raw},
    )
    assert dry_run.candidate_count == 0
    assert dry_run.skipped_referenced_count == 1

    result = cleanup_superseded_raw_snapshots(
        conn,
        dry_run=False,
        blob_store=blob_store,
        protected_raw_ids={old_raw},
    )
    assert result.deleted_raw_count == 0
    assert result.skipped_referenced_count == 1
    assert blob_store.exists(old_raw)
    assert blob_store.exists(current_raw)
    assert conn.execute("SELECT 1 FROM raw_sessions WHERE raw_id = ?", (old_raw,)).fetchone() is not None
    assert conn.execute("SELECT 1 FROM raw_sessions WHERE raw_id = ?", (current_raw,)).fetchone() is not None


def test_superseded_raw_cleanup_keeps_blob_with_remaining_protected_reference(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    source = tmp_path / "rollout.jsonl"
    source.write_text('{"type":"message"}\n', encoding="utf-8")
    blob_store = BlobStore(tmp_path / "blob")
    shared_blob, shared_size = _write_blob(blob_store, b"shared-evidence")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_archive_source_schema(conn)
    _insert_archive_raw_session(
        conn,
        raw_id="raw-old",
        source_path=source,
        source_index=0,
        blob_hash=shared_blob,
        blob_size=shared_size,
        acquired_at_ms=1,
    )
    _insert_archive_raw_session(
        conn,
        raw_id="raw-active",
        source_path=source,
        source_index=0,
        blob_hash=shared_blob,
        blob_size=shared_size,
        acquired_at_ms=2,
    )
    conn.commit()

    result = cleanup_superseded_raw_snapshots(
        conn,
        dry_run=False,
        blob_store=blob_store,
        protected_raw_ids={"raw-active"},
        eligible_raw_ids={"raw-old"},
    )

    assert result.deleted_raw_count == 1
    assert result.deleted_blob_count == 0
    assert blob_store.exists(shared_blob)
    assert conn.execute("SELECT 1 FROM raw_sessions WHERE raw_id = 'raw-old'").fetchone() is None
    assert conn.execute("SELECT 1 FROM raw_sessions WHERE raw_id = 'raw-active'").fetchone() is not None
    assert (
        conn.execute(
            "SELECT 1 FROM blob_refs WHERE blob_hash = ? AND ref_id = 'raw-active'",
            (bytes.fromhex(shared_blob),),
        ).fetchone()
        is not None
    )


def test_archive_cleanup_compacts_append_snapshot_without_session_events(tmp_path: Path) -> None:
    # Archive file-set cleanup simply compacts the superseded append snapshot.
    db_path = tmp_path / "source.db"
    source = tmp_path / "rollout.jsonl"
    source.write_text('{"type":"message"}\n', encoding="utf-8")
    blob_store = BlobStore(tmp_path / "blob")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_archive_source_schema(conn)

    old_raw, old_size = _write_blob(blob_store, b"old")
    current_raw, current_size = _write_blob(blob_store, b"current")
    _insert_archive_raw_session(
        conn,
        raw_id=old_raw,
        source_path=source,
        source_index=-1,
        blob_hash=old_raw,
        blob_size=old_size,
        acquired_at_ms=1_000,
    )
    _insert_archive_raw_session(
        conn,
        raw_id=current_raw,
        source_path=source,
        source_index=-1,
        blob_hash=current_raw,
        blob_size=current_size,
        acquired_at_ms=2_000,
    )
    conn.commit()

    dry_run = cleanup_superseded_raw_snapshots(conn, dry_run=True, blob_store=blob_store)
    assert dry_run.candidate_count == 1

    result = cleanup_superseded_raw_snapshots(conn, dry_run=False, blob_store=blob_store)
    assert result.deleted_raw_count == 1
    assert not blob_store.exists(old_raw)
    assert blob_store.exists(current_raw)
    assert conn.execute("SELECT 1 FROM raw_sessions WHERE raw_id = ?", (old_raw,)).fetchone() is None
    assert conn.execute("SELECT 1 FROM blob_refs WHERE ref_id = ?", (old_raw,)).fetchone() is None
    assert conn.execute("SELECT 1 FROM raw_sessions WHERE raw_id = ?", (current_raw,)).fetchone() is not None
    assert conn.execute("SELECT 1 FROM blob_refs WHERE ref_id = ?", (current_raw,)).fetchone() is not None


def test_superseded_raw_snapshot_cleanup_uses_archive_blob_hashes(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    source = tmp_path / "rollout.jsonl"
    source.write_text('{"type":"message"}\n', encoding="utf-8")
    blob_store = BlobStore(tmp_path / "blob")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_archive_source_schema(conn)

    old_blob, old_size = _write_blob(blob_store, b"v1-old")
    current_blob, current_size = _write_blob(blob_store, b"v1-current")
    _insert_archive_raw_session(
        conn,
        raw_id="raw-old-not-a-blob-hash",
        source_path=source,
        source_index=0,
        blob_hash=old_blob,
        blob_size=old_size,
        acquired_at_ms=1_790_000_000_000,
    )
    _insert_archive_raw_session(
        conn,
        raw_id="raw-current-not-a-blob-hash",
        source_path=source,
        source_index=0,
        blob_hash=current_blob,
        blob_size=current_size,
        acquired_at_ms=1_790_000_060_000,
    )
    conn.commit()

    candidates = superseded_raw_snapshot_candidates(conn, limit=100)
    assert [(candidate.raw_id, candidate.blob_store_hash) for candidate in candidates] == [
        ("raw-old-not-a-blob-hash", old_blob)
    ]

    result = cleanup_superseded_raw_snapshots(conn, dry_run=False, blob_store=blob_store)

    assert result.deleted_raw_count == 1
    assert result.deleted_blob_count == 1
    assert not blob_store.exists(old_blob)
    assert blob_store.exists(current_blob)
    assert conn.execute("SELECT 1 FROM raw_sessions WHERE raw_id = 'raw-old-not-a-blob-hash'").fetchone() is None
    assert conn.execute("SELECT 1 FROM blob_refs WHERE ref_id = 'raw-old-not-a-blob-hash'").fetchone() is None

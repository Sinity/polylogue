"""Superseded live raw snapshot cleanup contracts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.raw_retention import (
    RawRetentionAuthority,
    RawRetentionSafetyError,
    active_raw_retention_authority,
    cleanup_superseded_raw_snapshots,
    plan_stale_supersession_reissue,
    protected_active_raw_revision_ids,
    raw_frontier_integrity_projection,
    raw_frontier_integrity_snapshot,
    raw_frontier_integrity_summary,
    reissue_stale_supersession_receipts,
    superseded_raw_snapshot_candidates,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.ops_write import upsert_ingest_cursor
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
    append_end_offset: int | None,
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
                      'byte', ?, ?, ?, 2)
            """,
            (accepted_raw_id, accepted_revision, bytes(32), frontier, generation, append_end_offset),
        )


def _seed_superseded_application(
    index_db_path: Path,
    *,
    raw_id: str,
    source_revision: str,
    accepted_generation: int,
    accepted_raw_id: str,
    accepted_revision: str,
    accepted_append_end_offset: int | None,
) -> None:
    with sqlite3.connect(index_db_path) as conn:
        conn.execute(
            """
            INSERT INTO raw_revision_applications (
                decision_id, raw_id, session_id, logical_source_key,
                source_revision, acquisition_generation, decision,
                accepted_raw_id, accepted_source_revision, accepted_content_hash,
                append_end_offset, detail, decided_at_ms
            ) VALUES (?, ?, 'codex-session:session-1', 'codex:session-1', ?, ?,
                      'superseded', ?, ?, ?, ?, 'superseded by accepted full', 2)
            """,
            (
                f"decision-{raw_id}",
                raw_id,
                source_revision,
                accepted_generation,
                accepted_raw_id,
                accepted_revision,
                bytes(32),
                accepted_append_end_offset,
            ),
        )


def _parsed_session(*message_ids: str) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="session-1",
        messages=[
            ParsedMessage(provider_message_id=message_id, role=Role.USER, text=message_id) for message_id in message_ids
        ],
    )


def _seed_real_full_supersession(root: Path) -> tuple[str, str]:
    source_path = root / "session.jsonl"
    source_path.write_bytes(b"a" * 20)
    initialize_active_archive_root(root)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        old_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"a" * 10,
            source_path=str(source_path),
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            old_raw_id,
            RawRevisionEnvelope(
                "codex:session-1",
                RawRevisionKind.FULL,
                "revision-old",
                0,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        archive.apply_raw_revision_replay(
            archive.raw_revision_replay_plan("codex:session-1"),
            {old_raw_id: _parsed_session("m0")},
            acquired_at_ms=1,
        )
        new_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"a" * 20,
            source_path=str(source_path),
            acquired_at_ms=2,
        )
        archive.bind_raw_revision(
            new_raw_id,
            RawRevisionEnvelope(
                "codex:session-1",
                RawRevisionKind.FULL,
                "revision-new",
                1,
                authority=RawRevisionAuthority.BYTE_PROVEN,
            ),
        )
        archive.apply_raw_revision_replay(
            archive.raw_revision_replay_plan("codex:session-1"),
            {new_raw_id: _parsed_session("m0", "m1")},
            acquired_at_ms=2,
        )
    return old_raw_id, new_raw_id


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
        append_end_offset=20,
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
        append_end_offset=None,
    )
    _seed_superseded_application(
        index_db,
        raw_id="raw-old-full",
        source_revision="revision-old",
        accepted_generation=1,
        accepted_raw_id="raw-new-full",
        accepted_revision="revision-new",
        accepted_append_end_offset=None,
    )

    with sqlite3.connect(source_db) as conn:
        authority = active_raw_retention_authority(conn, index_db_path=index_db)

    # Anti-vacuity: following a full raw's historical predecessor would retain
    # raw-old-full and defeat the self-contained full reset contract.
    assert authority == RawRetentionAuthority(
        protected_raw_ids=frozenset({"raw-new-full"}),
        eligible_raw_ids=frozenset({"raw-old-full"}),
    )


def test_real_revision_receipt_authorizes_only_current_byte_head_supersession(tmp_path: Path) -> None:
    old_raw_id, new_raw_id = _seed_real_full_supersession(tmp_path)

    with sqlite3.connect(tmp_path / "source.db") as conn:
        authority = active_raw_retention_authority(conn, index_db_path=tmp_path / "index.db")

    assert authority == RawRetentionAuthority(
        protected_raw_ids=frozenset({new_raw_id}),
        eligible_raw_ids=frozenset({old_raw_id}),
    )


def test_semantic_head_receipt_authorizes_no_raw_deletion(tmp_path: Path) -> None:
    old_raw_id, new_raw_id = _seed_real_full_supersession(tmp_path)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE raw_revision_heads SET accepted_frontier_kind = 'semantic', accepted_frontier = 2")

    with sqlite3.connect(tmp_path / "source.db") as conn:
        authority = active_raw_retention_authority(conn, index_db_path=tmp_path / "index.db")

    assert authority == RawRetentionAuthority(
        protected_raw_ids=frozenset({new_raw_id}),
        eligible_raw_ids=frozenset(),
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id = ?", (old_raw_id,)).fetchone() == (1,)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("session_id", "codex-session:other"),
        ("logical_source_key", "codex:other"),
        ("accepted_raw_id", "other-raw"),
        ("accepted_source_revision", "other-revision"),
        ("accepted_content_hash", bytes(32)),
        ("acquisition_generation", 99),
        ("append_end_offset", 99),
        ("decided_at_ms", 99),
    ],
)
def test_real_revision_receipt_binding_drift_authorizes_no_deletion(
    tmp_path: Path,
    column: str,
    value: object,
) -> None:
    old_raw_id, new_raw_id = _seed_real_full_supersession(tmp_path)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute(
            f"UPDATE raw_revision_applications SET {column} = ? WHERE decision = 'superseded'",
            (value,),
        )

    with sqlite3.connect(tmp_path / "source.db") as conn:
        authority = active_raw_retention_authority(conn, index_db_path=tmp_path / "index.db")

    assert authority == RawRetentionAuthority(
        protected_raw_ids=frozenset({new_raw_id}),
        eligible_raw_ids=frozenset(),
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE raw_id = ?", (old_raw_id,)).fetchone() == (1,)


@pytest.mark.parametrize(
    ("column", "value", "error_match"),
    [
        ("source_revision", "tampered", "revision disagrees"),
        ("baseline_raw_id", "tampered", "baseline disagrees"),
        ("predecessor_raw_id", "tampered", "predecessor disagrees"),
    ],
)
def test_real_revision_receipt_rejects_conflicting_source_evidence(
    tmp_path: Path,
    column: str,
    value: object,
    error_match: str,
) -> None:
    old_raw_id, _new_raw_id = _seed_real_full_supersession(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(f"UPDATE raw_sessions SET {column} = ? WHERE raw_id = ?", (value, old_raw_id))

    with (
        sqlite3.connect(tmp_path / "source.db") as conn,
        pytest.raises(RawRetentionSafetyError, match=error_match),
    ):
        active_raw_retention_authority(conn, index_db_path=tmp_path / "index.db")


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
        append_end_offset=15,
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
        append_end_offset=15,
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
        append_end_offset=15,
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


def test_superseded_raw_cleanup_prunes_orphaned_raw_authority_plans(tmp_path: Path) -> None:
    """polylogue-i3zo: retention's raw delete must not leak raw_authority_plans,
    but must never corrupt the immutable census ledger doing so.

    ``raw_authority_plans.input_raw_ids_json`` references raws by JSON string,
    not FK, so deleting a raw here does not cascade-clean a plan naming it. A
    plan whose every input raw is gone AND which no census still references is
    safe to prune outright. But a plan a census still references must be left
    fully alone: ``raw_authority_censuses.plan_count`` / ``post_plan_count``
    are immutable per-census totals ``read_raw_authority_census`` uses
    verbatim for its reported total AND to decide whether pagination has more
    pages (chatgpt-codex-connector's review on the first version of this fix,
    PR #3530, flagged that an earlier draft deleted census-membership rows
    without adjusting those counts, making the total lie and, once a census's
    rows were all removed, making ``next_query_handle`` reissue forever since
    the offset advanced by zero). This test proves both halves against the
    real production entry point, :func:`cleanup_superseded_raw_snapshots`:

    * a plan with zero census references and all-missing inputs is pruned;
    * a plan a census still references is left untouched even though its
      only input raw is also superseded-and-deleted, and the census's own
      ``plan_count`` / actual surviving ``raw_authority_census_plans`` row
      count for it stay mutually consistent before and after the purge (the
      ledger-consistency proof the review asked for);
    * the same holds for a plan referenced only by an unresolved blocker
      (the durable-obligation case ``prune_raw_authority_census_history``
      already protects, defended here too even though structurally a
      blocker should never exist without a paired census_plans row).
    """
    db_path = tmp_path / "source.db"
    source = tmp_path / "rollout.jsonl"
    source.write_text('{"type":"message"}\n', encoding="utf-8")
    blob_store = BlobStore(tmp_path / "blob")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_archive_source_schema(conn)
    conn.execute(
        """CREATE TABLE raw_authority_plans (
            plan_id TEXT PRIMARY KEY,
            input_raw_ids_json TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE raw_authority_censuses (
            census_id TEXT PRIMARY KEY,
            plan_count INTEGER NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE raw_authority_census_plans (
            census_id TEXT NOT NULL,
            plan_id TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE raw_authority_census_post_plans (
            census_id TEXT NOT NULL,
            plan_id TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE raw_authority_blockers (
            blocker_id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL,
            census_id TEXT NOT NULL,
            resolved_at_ms INTEGER
        )"""
    )

    # Three superseded (about-to-be-deleted) raws, plus one that survives.
    old_free, old_free_size = _write_blob(blob_store, b"old-free")
    old_census, old_census_size = _write_blob(blob_store, b"old-census")
    old_blocker, old_blocker_size = _write_blob(blob_store, b"old-blocker")
    full_new, full_new_size = _write_blob(blob_store, b"full-new")
    for raw_id, blob_hash, size, ts in (
        (old_free, old_free, old_free_size, 1_000),
        (old_census, old_census, old_census_size, 1_100),
        (old_blocker, old_blocker, old_blocker_size, 1_200),
        (full_new, full_new, full_new_size, 2_000),
    ):
        _insert_archive_raw_session(
            conn,
            raw_id=raw_id,
            source_path=source,
            source_index=0,
            blob_hash=blob_hash,
            blob_size=size,
            acquired_at_ms=ts,
        )

    # plan-free-orphan: all inputs gone, zero census/blocker references -> pruned.
    # plan-census-orphan: all inputs gone, but still a live census_plans member -> untouched.
    # plan-blocker-orphan: all inputs gone, no census_plans row but an unresolved
    #   blocker still names it -> untouched (defensive; structurally shouldn't
    #   happen, but the guard must hold regardless).
    # plan-kept: its input raw survives -> untouched (not even an orphan candidate).
    conn.execute(
        "INSERT INTO raw_authority_plans (plan_id, input_raw_ids_json) VALUES ('plan-free-orphan', ?)",
        (f'["{old_free}"]',),
    )
    conn.execute(
        "INSERT INTO raw_authority_plans (plan_id, input_raw_ids_json) VALUES ('plan-census-orphan', ?)",
        (f'["{old_census}"]',),
    )
    conn.execute(
        "INSERT INTO raw_authority_plans (plan_id, input_raw_ids_json) VALUES ('plan-blocker-orphan', ?)",
        (f'["{old_blocker}"]',),
    )
    conn.execute(
        "INSERT INTO raw_authority_plans (plan_id, input_raw_ids_json) VALUES ('plan-kept', ?)",
        (f'["{full_new}"]',),
    )
    conn.execute("INSERT INTO raw_authority_censuses (census_id, plan_count) VALUES ('census-a', 1)")
    conn.execute(
        "INSERT INTO raw_authority_census_plans (census_id, plan_id) VALUES ('census-a', 'plan-census-orphan')"
    )
    conn.execute(
        "INSERT INTO raw_authority_blockers (blocker_id, plan_id, census_id, resolved_at_ms) "
        "VALUES ('blk-1', 'plan-blocker-orphan', 'census-a', NULL)"
    )
    conn.commit()

    def _census_ledger_consistent() -> bool:
        header_count = int(
            conn.execute("SELECT plan_count FROM raw_authority_censuses WHERE census_id = 'census-a'").fetchone()[0]
        )
        actual_count = int(
            conn.execute("SELECT COUNT(*) FROM raw_authority_census_plans WHERE census_id = 'census-a'").fetchone()[0]
        )
        return header_count == actual_count

    assert _census_ledger_consistent()  # 1 header count == 1 actual row, before the purge

    candidates = superseded_raw_snapshot_candidates(conn, limit=100)
    assert {candidate.raw_id for candidate in candidates} == {old_free, old_census, old_blocker}

    result = cleanup_superseded_raw_snapshots(conn, dry_run=False, blob_store=blob_store)
    assert result.deleted_raw_count == 3
    assert result.deleted_orphaned_authority_plan_count == 1  # only plan-free-orphan

    remaining_plans = {str(row[0]) for row in conn.execute("SELECT plan_id FROM raw_authority_plans").fetchall()}
    assert remaining_plans == {"plan-census-orphan", "plan-blocker-orphan", "plan-kept"}

    # Ledger consistency (the review's explicit ask): plan_count still matches
    # the actual surviving raw_authority_census_plans rows for census-a --
    # nothing this purge did touched census/blocker rows, so pagination over
    # census-a would still terminate correctly and report a truthful total.
    assert _census_ledger_consistent()
    assert conn.execute("SELECT COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL").fetchone()[0] == 1


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


def _seed_ops_cursor(
    ops_db_path: Path,
    *,
    source_path: Path,
    byte_offset: int,
    deferred_end_offset: int | None = None,
) -> None:
    initialize_archive_database(ops_db_path, ArchiveTier.OPS)
    with sqlite3.connect(ops_db_path) as conn:
        upsert_ingest_cursor(
            conn,
            source_path=str(source_path),
            updated_at_ms=1,
            byte_offset=byte_offset,
            deferred_end_offset=deferred_end_offset,
        )
        conn.commit()


# ---------------------------------------------------------------------------
# raw_frontier_integrity_snapshot (polylogue-yla8.7)
# ---------------------------------------------------------------------------


def test_raw_frontier_integrity_snapshot_healthy_full_plus_three_appends(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
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
            raw_id="raw-append-1",
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
        _insert_revision_raw(
            conn,
            raw_id="raw-append-2",
            source_path=source_path,
            acquired_at_ms=3,
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
        _insert_revision_raw(
            conn,
            raw_id="raw-append-3",
            source_path=source_path,
            acquired_at_ms=4,
            kind="append",
            source_revision="revision-3",
            generation=3,
            blob_size=5,
            predecessor_raw_id="raw-append-2",
            predecessor_revision="revision-2",
            baseline_raw_id="raw-baseline",
            append_start_offset=20,
            append_end_offset=25,
        )
        conn.commit()
    _seed_index_authority(
        index_db,
        session_raw_id="raw-append-3",
        accepted_raw_id="raw-append-3",
        accepted_revision="revision-3",
        generation=3,
        frontier=25,
        append_end_offset=25,
    )
    _seed_ops_cursor(ops_db, source_path=source_path, byte_offset=25)

    with sqlite3.connect(source_db) as conn:
        snapshot = raw_frontier_integrity_snapshot(conn, index_db_path=index_db, ops_db_path=ops_db)

    assert snapshot.broken_head_status == "healthy"
    assert snapshot.broken_head_count == 0
    assert snapshot.broken_head_checked_count == 1
    assert snapshot.broken_head_samples == ()
    assert snapshot.cursor_ahead_status == "healthy"
    assert snapshot.cursor_ahead_count == 0
    assert snapshot.cursor_ahead_checked_count == 1
    assert snapshot.cursor_ahead_samples == ()
    assert snapshot.overall_status == "healthy"


def test_raw_frontier_integrity_snapshot_detects_missing_accepted_predecessor(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
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
        append_end_offset=15,
    )
    initialize_archive_database(ops_db, ArchiveTier.OPS)

    with sqlite3.connect(source_db) as conn:
        snapshot = raw_frontier_integrity_snapshot(conn, index_db_path=index_db, ops_db_path=ops_db)

    assert snapshot.broken_head_status == "violated"
    assert snapshot.broken_head_count == 1
    assert snapshot.broken_head_checked_count == 1
    assert len(snapshot.broken_head_samples) == 1
    sample = snapshot.broken_head_samples[0]
    assert sample.accepted_raw_id == "raw-append"
    assert sample.logical_source_key == "codex:session-1"
    assert "missing from source tier" in sample.reason
    assert "1 active index raw seed" in snapshot.broken_head_reason
    assert snapshot.overall_status == "violated"


def test_raw_frontier_integrity_snapshot_traverses_session_seed_without_head(tmp_path: Path) -> None:
    """Removing the sessions.raw_id retention seed makes this regression false-green."""

    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
    source_path = tmp_path / "session.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(source_db) as conn:
        _insert_revision_raw(
            conn,
            raw_id="raw-session-append",
            source_path=source_path,
            acquired_at_ms=1,
            kind="append",
            source_revision="revision-1",
            generation=1,
            blob_size=5,
            predecessor_raw_id="raw-session-missing",
            predecessor_revision="revision-0",
            baseline_raw_id="raw-session-missing",
            append_start_offset=10,
            append_end_offset=15,
        )
        conn.commit()
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (native_id, origin, raw_id, title, content_hash)
            VALUES ('session-only', 'codex-session', 'raw-session-append', 'session seed', ?)
            """,
            (bytes(32),),
        )
        conn.commit()

    with sqlite3.connect(source_db) as conn:
        with pytest.raises(RawRetentionSafetyError, match="raw-session-missing"):
            active_raw_retention_authority(conn, index_db_path=index_db)
        snapshot = raw_frontier_integrity_snapshot(conn, index_db_path=index_db, ops_db_path=ops_db)

    assert snapshot.broken_head_status == "violated"
    assert snapshot.broken_head_count == 1
    assert snapshot.broken_head_checked_count == 1
    assert snapshot.broken_head_samples[0].accepted_raw_id == "raw-session-append"
    assert "raw-session-missing" in snapshot.broken_head_samples[0].reason
    assert snapshot.cursor_ahead_status == "healthy"
    assert snapshot.overall_status == "violated"


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
def test_raw_frontier_integrity_snapshot_detects_corrupt_chain_invariants(
    tmp_path: Path,
    mutation: str,
    error_match: str,
) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
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
        generation=2 if mutation == "generation" else 1,
        frontier=15,
        append_end_offset=15,
    )
    initialize_archive_database(ops_db, ArchiveTier.OPS)

    with sqlite3.connect(source_db) as conn:
        snapshot = raw_frontier_integrity_snapshot(conn, index_db_path=index_db, ops_db_path=ops_db)

    assert snapshot.broken_head_status == "violated"
    assert snapshot.broken_head_count == 1
    assert len(snapshot.broken_head_samples) == 1
    assert error_match in snapshot.broken_head_samples[0].reason


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        ("revision", "revision disagrees"),
        ("generation", "generation disagrees"),
        ("frontier", "frontier disagrees"),
        ("append_end", "append end disagrees"),
    ],
)
def test_raw_frontier_integrity_snapshot_detects_index_head_metadata_drift(
    tmp_path: Path,
    mutation: str,
    error_match: str,
) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
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
        conn.commit()
    _seed_index_authority(
        index_db,
        session_raw_id="raw-append",
        accepted_raw_id="raw-append",
        accepted_revision="revision-1",
        generation=1,
        frontier=15,
        append_end_offset=15,
    )
    with sqlite3.connect(index_db) as conn:
        if mutation == "revision":
            conn.execute("UPDATE raw_revision_heads SET accepted_source_revision = 'wrong'")
        elif mutation == "generation":
            conn.execute("UPDATE raw_revision_heads SET acquisition_generation = 2")
        elif mutation == "frontier":
            conn.execute("UPDATE raw_revision_heads SET accepted_frontier = 14")
        elif mutation == "append_end":
            conn.execute("UPDATE raw_revision_heads SET append_end_offset = 14")
        else:
            raise AssertionError(mutation)
        conn.commit()
    _seed_ops_cursor(ops_db, source_path=source_path, byte_offset=15)

    with sqlite3.connect(source_db) as conn:
        snapshot = raw_frontier_integrity_snapshot(conn, index_db_path=index_db, ops_db_path=ops_db)

    assert snapshot.broken_head_status == "violated"
    assert snapshot.broken_head_count == 1
    assert len(snapshot.broken_head_samples) == 1
    assert error_match in snapshot.broken_head_samples[0].reason
    assert snapshot.overall_status == "violated"


def test_raw_frontier_integrity_projection_composes_real_missing_session_raw_authority(tmp_path: Path) -> None:
    """The sessions.raw_id seed reaches the canonical projection through its production census."""

    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(tmp_path / "ops.db", ArchiveTier.OPS)
    with sqlite3.connect(index_db) as conn:
        conn.executemany(
            """
            INSERT INTO sessions (native_id, origin, raw_id, title, content_hash)
            VALUES (?, 'codex-session', ?, 'lost', ?)
            """,
            [
                ("lost-session-a", "raw-missing-a", bytes(32)),
                ("lost-session-b", "raw-missing-b", bytes(32)),
            ],
        )
        conn.commit()

    readiness = raw_materialization_readiness_snapshot(tmp_path)
    projection = raw_frontier_integrity_projection(tmp_path, readiness, sample_limit=1)

    assert readiness["available"] is True
    assert readiness["lost_source_evidence_count"] == 2
    assert projection.missing_source_raw_status == "violated"
    assert projection.missing_source_raw_count == 2
    assert len(projection.missing_source_raw_samples) == 1
    sample = projection.missing_source_raw_samples[0]
    assert sample["session_id"] == "codex-session:lost-session-a"
    assert sample["missing_raw_id"] == "raw-missing-a"
    assert sample["evidence_status"] == "lost_source_evidence"
    assert projection.broken_head_status == "healthy"
    assert projection.cursor_ahead_status == "healthy"
    assert projection.overall_status == "violated"
    assert projection.available is True
    assert projection.summary == "2 indexed session(s) reference raw evidence missing from source tier"


def test_raw_frontier_integrity_summary_preserves_mixed_violation_and_unknown_reasons() -> None:
    summary = raw_frontier_integrity_summary(
        {
            "overall_status": "violated",
            "broken_head_reason": "1 accepted head is corrupt",
            "missing_source_raw_reason": "",
            "cursor_ahead_reason": "ops cursor authority is unavailable",
        }
    )

    assert summary == "1 accepted head is corrupt; ops cursor authority is unavailable"


def test_raw_frontier_integrity_snapshot_detects_cursor_ahead_of_accepted_material(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
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
        conn.commit()
    _seed_index_authority(
        index_db,
        session_raw_id="raw-append",
        accepted_raw_id="raw-append",
        accepted_revision="revision-1",
        generation=1,
        frontier=15,
        append_end_offset=15,
    )
    # The daemon has acquired further bytes into the source file (byte_offset
    # 30) than the index has actually accepted (frontier 15) — the exact
    # symptom yla8.6 found only via manual SQL.
    _seed_ops_cursor(ops_db, source_path=source_path, byte_offset=30)

    with sqlite3.connect(source_db) as conn:
        snapshot = raw_frontier_integrity_snapshot(conn, index_db_path=index_db, ops_db_path=ops_db)

    assert snapshot.broken_head_status == "healthy"
    assert snapshot.cursor_ahead_status == "violated"
    assert snapshot.cursor_ahead_count == 1
    assert snapshot.cursor_ahead_checked_count == 1
    assert snapshot.cursor_head_comparison_count == 1
    assert snapshot.cursor_ahead_comparison_count == 1
    assert len(snapshot.cursor_ahead_samples) == 1
    sample = snapshot.cursor_ahead_samples[0]
    assert sample.source_path == str(source_path)
    assert sample.cursor_byte_offset == 30
    assert sample.accepted_frontier == 15
    assert sample.affected_head_count == 1
    assert snapshot.overall_status == "violated"


def test_raw_frontier_integrity_counts_one_cursor_across_multiple_byte_head_comparisons(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
    source_path = tmp_path / "shared.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        _insert_revision_raw(
            conn,
            raw_id="raw-one",
            source_path=source_path,
            acquired_at_ms=1,
            kind="full",
            source_revision="revision-one",
            generation=0,
            blob_size=10,
        )
        _insert_revision_raw(
            conn,
            raw_id="raw-two",
            source_path=source_path,
            acquired_at_ms=2,
            kind="full",
            source_revision="revision-two",
            generation=0,
            blob_size=20,
        )
        conn.execute("UPDATE raw_sessions SET logical_source_key = 'codex:session-2' WHERE raw_id = 'raw-two'")
        conn.commit()
    _seed_index_authority(
        index_db,
        session_raw_id="raw-one",
        accepted_raw_id="raw-one",
        accepted_revision="revision-one",
        generation=0,
        frontier=10,
        append_end_offset=None,
    )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (native_id, origin, raw_id, title, content_hash)
            VALUES ('session-2', 'codex-session', 'raw-two', 'session two', ?)
            """,
            (bytes(32),),
        )
        conn.execute(
            """
            INSERT INTO raw_revision_heads (
                logical_source_key, session_id, accepted_raw_id,
                accepted_source_revision, accepted_content_hash,
                accepted_frontier_kind, accepted_frontier,
                acquisition_generation, append_end_offset, decided_at_ms
            ) VALUES ('codex:session-2', 'codex-session:session-2', 'raw-two',
                      'revision-two', ?, 'byte', 20, 0, NULL, 2)
            """,
            (bytes(32),),
        )
        conn.commit()
    _seed_ops_cursor(ops_db, source_path=source_path, byte_offset=30)

    with sqlite3.connect(source_db) as conn:
        snapshot = raw_frontier_integrity_snapshot(
            conn,
            index_db_path=index_db,
            ops_db_path=ops_db,
            sample_limit=1,
        )

    assert snapshot.cursor_ahead_status == "violated"
    assert snapshot.cursor_ahead_count == 1
    assert snapshot.cursor_ahead_checked_count == 1
    assert snapshot.cursor_head_comparison_count == 2
    assert snapshot.cursor_ahead_comparison_count == 2
    assert len(snapshot.cursor_ahead_samples) == 1
    assert snapshot.cursor_ahead_samples[0].source_path == str(source_path)
    assert snapshot.cursor_ahead_samples[0].accepted_frontier == 10
    assert snapshot.cursor_ahead_samples[0].affected_head_count == 2
    assert "1 ingest cursor row(s)" in snapshot.cursor_ahead_reason
    assert "2 cursor/head comparison(s)" in snapshot.cursor_ahead_reason


def test_raw_frontier_integrity_semantic_membership_cursor_is_intentionally_not_compared(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
    source_path = tmp_path / "membership-export.json"
    source_path.write_text("{}\n", encoding="utf-8")
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        _insert_revision_raw(
            conn,
            raw_id="raw-semantic",
            source_path=source_path,
            acquired_at_ms=1,
            kind="full",
            source_revision="semantic-revision",
            generation=0,
            blob_size=10,
        )
        conn.commit()
    _seed_index_authority(
        index_db,
        session_raw_id="raw-semantic",
        accepted_raw_id="raw-semantic",
        accepted_revision="semantic-revision",
        generation=0,
        frontier=10,
        append_end_offset=None,
    )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            UPDATE raw_revision_heads
            SET accepted_frontier_kind = 'semantic', accepted_frontier = 1, append_end_offset = NULL
            """
        )
        conn.commit()
    _seed_ops_cursor(ops_db, source_path=source_path, byte_offset=100)

    with sqlite3.connect(source_db) as conn:
        snapshot = raw_frontier_integrity_snapshot(conn, index_db_path=index_db, ops_db_path=ops_db)

    assert snapshot.broken_head_status == "healthy"
    assert snapshot.cursor_ahead_status == "healthy"
    assert snapshot.cursor_ahead_count == 0
    assert snapshot.cursor_ahead_checked_count == 0
    assert snapshot.cursor_head_comparison_count == 0
    assert snapshot.cursor_ahead_comparison_count == 0
    assert snapshot.cursor_authority_gap_count == 0
    assert snapshot.cursor_ahead_samples == ()
    assert snapshot.overall_status == "healthy"


def test_raw_frontier_integrity_snapshot_cursor_at_exact_accepted_frontier_is_healthy(tmp_path: Path) -> None:
    """A cursor sitting exactly at the accepted frontier (not past it) is healthy.

    Anti-vacuity: removing the strict ``>`` comparison in
    ``_check_cursor_ahead_of_accepted`` (e.g. replacing it with ``>=``) would
    make this test fail.
    """
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
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
            blob_size=15,
        )
        conn.commit()
    _seed_index_authority(
        index_db,
        session_raw_id="raw-baseline",
        accepted_raw_id="raw-baseline",
        accepted_revision="revision-0",
        generation=0,
        frontier=15,
        append_end_offset=None,
    )
    _seed_ops_cursor(ops_db, source_path=source_path, byte_offset=15)

    with sqlite3.connect(source_db) as conn:
        snapshot = raw_frontier_integrity_snapshot(conn, index_db_path=index_db, ops_db_path=ops_db)

    assert snapshot.cursor_ahead_status == "healthy"
    assert snapshot.cursor_ahead_count == 0
    assert snapshot.cursor_ahead_checked_count == 1
    assert snapshot.cursor_head_comparison_count == 1
    assert snapshot.cursor_ahead_comparison_count == 0


def test_raw_frontier_integrity_snapshot_reads_ops_with_zero_accepted_heads(tmp_path: Path) -> None:
    """Zero heads cannot bypass the ops authority check and false-green."""

    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(tmp_path / "ops.db", ArchiveTier.OPS)

    with sqlite3.connect(source_db) as conn:
        snapshot = raw_frontier_integrity_snapshot(
            conn,
            index_db_path=index_db,
            ops_db_path=tmp_path / "missing-ops.db",
        )

    assert snapshot.broken_head_status == "healthy"
    assert snapshot.cursor_ahead_status == "unknown"
    assert "ops tier is unavailable" in snapshot.cursor_ahead_reason
    assert snapshot.overall_status == "unknown"


def test_raw_frontier_integrity_snapshot_surfaces_cursor_without_accepted_head(tmp_path: Path) -> None:
    """A committed cursor with no comparable head is explicit unknown debt."""

    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
    source_path = tmp_path / "unmaterialized.jsonl"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        _insert_revision_raw(
            conn,
            raw_id="raw-unmaterialized",
            source_path=source_path,
            acquired_at_ms=1,
            kind="full",
            source_revision="revision-0",
            generation=0,
            blob_size=10,
        )
        conn.commit()
    _seed_ops_cursor(ops_db, source_path=source_path, byte_offset=42)

    with sqlite3.connect(source_db) as conn:
        snapshot = raw_frontier_integrity_snapshot(conn, index_db_path=index_db, ops_db_path=ops_db)

    assert snapshot.cursor_ahead_status == "unknown"
    assert snapshot.cursor_ahead_count == 0
    assert snapshot.cursor_authority_gap_count == 1
    assert snapshot.cursor_authority_gap_samples[0].source_path == str(source_path)
    assert snapshot.cursor_authority_gap_samples[0].cursor_byte_offset == 42
    assert snapshot.cursor_authority_gap_samples[0].state == "source_raws_without_accepted_head"
    assert "source tier has raw evidence" in snapshot.cursor_authority_gap_samples[0].reason
    assert snapshot.overall_status == "unknown"


def test_raw_frontier_integrity_snapshot_classifies_deferred_cursor_separately_from_ahead(
    tmp_path: Path,
) -> None:
    """A captured-but-unresolved cursor is blocking deferred authority, not a false violation."""

    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
    source_path = tmp_path / "deferred.jsonl"
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
        conn.commit()
    _seed_index_authority(
        index_db,
        session_raw_id="raw-baseline",
        accepted_raw_id="raw-baseline",
        accepted_revision="revision-0",
        generation=0,
        frontier=10,
        append_end_offset=None,
    )
    _seed_ops_cursor(ops_db, source_path=source_path, byte_offset=10, deferred_end_offset=20)

    with sqlite3.connect(source_db) as conn:
        snapshot = raw_frontier_integrity_snapshot(conn, index_db_path=index_db, ops_db_path=ops_db)

    assert snapshot.cursor_ahead_status == "unknown"
    assert snapshot.cursor_ahead_count == 0
    assert snapshot.cursor_head_comparison_count == 0
    assert snapshot.cursor_authority_gap_count == 1
    sample = snapshot.cursor_authority_gap_samples[0]
    assert sample.state == "deferred"
    assert sample.cursor_byte_offset == 10
    assert "awaiting authority resolution" in sample.reason
    assert snapshot.overall_status == "unknown"


def test_raw_frontier_integrity_projection_preserves_violation_when_sibling_is_unknown(tmp_path: Path) -> None:
    """Known corruption dominates unavailable cursor authority without claiming full availability."""

    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(tmp_path / "ops.db", ArchiveTier.OPS)
    _seed_index_authority(
        index_db,
        session_raw_id="raw-missing",
        accepted_raw_id="raw-missing",
        accepted_revision="revision-1",
        generation=1,
        frontier=15,
        append_end_offset=15,
    )

    projection = raw_frontier_integrity_projection(
        tmp_path,
        {"available": True, "lost_source_evidence_count": 0},
    )

    assert projection.broken_head_status == "violated"
    assert projection.cursor_ahead_status == "unknown"
    assert projection.cursor_authority_gap_count == 1
    assert projection.overall_status == "violated"
    assert projection.available is False


@pytest.mark.parametrize("index_kind", ["missing", "malformed"])
def test_raw_frontier_integrity_snapshot_unavailable_index_tier_is_unknown_never_healthy(
    tmp_path: Path,
    index_kind: str,
) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    if index_kind == "malformed":
        index_db.write_bytes(b"not sqlite")

    with sqlite3.connect(source_db) as conn:
        snapshot = raw_frontier_integrity_snapshot(conn, index_db_path=index_db, ops_db_path=ops_db)

    assert snapshot.broken_head_status == "unknown"
    assert snapshot.broken_head_count == 0
    assert snapshot.cursor_ahead_status == "unknown"
    assert snapshot.cursor_ahead_count == 0
    assert snapshot.overall_status == "unknown"
    assert "unavailable" in snapshot.broken_head_reason or "unreadable" in snapshot.broken_head_reason


def test_raw_frontier_integrity_snapshot_unavailable_source_tier_is_unknown_never_healthy(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
    source_path = tmp_path / "session.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    _seed_index_authority(
        index_db,
        session_raw_id="raw-baseline",
        accepted_raw_id="raw-baseline",
        accepted_revision="revision-0",
        generation=0,
        frontier=15,
        append_end_offset=None,
    )
    # source.db is created without the raw_sessions table at all.
    with sqlite3.connect(source_db) as conn:
        conn.execute("CREATE TABLE placeholder (id INTEGER PRIMARY KEY)")
        conn.commit()

    with sqlite3.connect(source_db) as conn:
        snapshot = raw_frontier_integrity_snapshot(conn, index_db_path=index_db, ops_db_path=ops_db)

    assert snapshot.broken_head_status == "unknown"
    assert snapshot.cursor_ahead_status == "unknown"
    assert snapshot.overall_status == "unknown"
    assert "unreadable" in snapshot.broken_head_reason


def test_raw_frontier_integrity_snapshot_partial_source_schema_is_unknown_not_violated(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    _seed_index_authority(
        index_db,
        session_raw_id="raw-baseline",
        accepted_raw_id="raw-baseline",
        accepted_revision="revision-0",
        generation=0,
        frontier=15,
        append_end_offset=None,
    )
    with sqlite3.connect(source_db) as conn:
        conn.execute("CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY) STRICT")
        conn.execute("INSERT INTO raw_sessions (raw_id) VALUES ('raw-baseline')")
        conn.commit()

    with sqlite3.connect(source_db) as conn:
        snapshot = raw_frontier_integrity_snapshot(conn, index_db_path=index_db, ops_db_path=ops_db)

    assert snapshot.broken_head_status == "unknown"
    assert snapshot.broken_head_count == 0
    assert snapshot.broken_head_checked_count == 0
    assert snapshot.cursor_ahead_status == "unknown"
    assert snapshot.overall_status == "unknown"
    assert "schema missing column(s)" in snapshot.broken_head_reason


def test_raw_frontier_integrity_snapshot_unavailable_ops_tier_is_unknown_never_healthy(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"  # never created
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
            blob_size=15,
        )
        conn.commit()
    _seed_index_authority(
        index_db,
        session_raw_id="raw-baseline",
        accepted_raw_id="raw-baseline",
        accepted_revision="revision-0",
        generation=0,
        frontier=15,
        append_end_offset=None,
    )

    with sqlite3.connect(source_db) as conn:
        snapshot = raw_frontier_integrity_snapshot(conn, index_db_path=index_db, ops_db_path=ops_db)

    # The broken-head check is independent of ops.db and stays healthy...
    assert snapshot.broken_head_status == "healthy"
    # ...but cursor-ahead cannot be proven healthy without a readable ops
    # tier, so it must degrade to unknown rather than a false healthy zero.
    assert snapshot.cursor_ahead_status == "unknown"
    assert snapshot.cursor_ahead_count == 0
    assert "unavailable" in snapshot.cursor_ahead_reason
    # Overall status must not render green when any sub-check is unknown.
    assert snapshot.overall_status == "unknown"


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


# ---------------------------------------------------------------------------
# Stale supersession receipt reissue (polylogue-ktwa)
# ---------------------------------------------------------------------------


def _seed_stale_full_reset_chain(source_db: Path, index_db: Path) -> None:
    """Three full-reset generations: old -> mid -> new.

    ``raw-old-full``'s receipt was recorded when ``raw-mid-full`` was the
    accepted head (generation 1). The head has since advanced again to
    ``raw-new-full`` (generation 2), so ``raw-old-full``'s receipt is stale.
    ``raw-mid-full`` carries its own receipt recorded against the *current*
    head, so it is already current (nothing to reissue).
    """
    source_path = source_db.parent / "session.jsonl"
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
            raw_id="raw-mid-full",
            source_path=source_path,
            acquired_at_ms=2,
            kind="full",
            source_revision="revision-mid",
            generation=1,
            blob_size=20,
        )
        _insert_revision_raw(
            conn,
            raw_id="raw-new-full",
            source_path=source_path,
            acquired_at_ms=3,
            kind="full",
            source_revision="revision-new",
            generation=2,
            blob_size=30,
            predecessor_raw_id="raw-mid-full",
            baseline_raw_id="raw-mid-full",
        )
        conn.commit()
    _seed_index_authority(
        index_db,
        session_raw_id="raw-new-full",
        accepted_raw_id="raw-new-full",
        accepted_revision="revision-new",
        generation=2,
        frontier=30,
        append_end_offset=None,
    )
    # Stale: recorded against the now-superseded mid-generation head.
    _seed_superseded_application(
        index_db,
        raw_id="raw-old-full",
        source_revision="revision-old",
        accepted_generation=1,
        accepted_raw_id="raw-mid-full",
        accepted_revision="revision-mid",
        accepted_append_end_offset=None,
    )
    # Already current: recorded against the live head exactly.
    _seed_superseded_application(
        index_db,
        raw_id="raw-mid-full",
        source_revision="revision-mid",
        accepted_generation=2,
        accepted_raw_id="raw-new-full",
        accepted_revision="revision-new",
        accepted_append_end_offset=None,
    )


def test_stale_supersession_reissue_plan_finds_only_the_stale_receipt(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    _seed_stale_full_reset_chain(source_db, index_db)

    with sqlite3.connect(source_db) as conn:
        plan = plan_stale_supersession_reissue(conn, index_db_path=index_db)

    assert plan.already_current_count == 1
    assert plan.stale_count == 1
    assert [item.raw_id for item in plan.eligible] == ["raw-old-full"]
    candidate = plan.eligible[0]
    assert candidate.session_id == "codex-session:session-1"
    assert candidate.logical_source_key == "codex:session-1"
    assert candidate.source_revision == "revision-old"
    assert candidate.baseline_raw_id is None
    assert candidate.predecessor_raw_id is None
    assert candidate.current_head.accepted_raw_id == "raw-new-full"
    assert not plan.ineligible_reason_counts


def test_stale_supersession_reissue_dry_run_writes_nothing(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    _seed_stale_full_reset_chain(source_db, index_db)

    with sqlite3.connect(source_db) as source_conn:
        result = reissue_stale_supersession_receipts(source_conn, source_conn, index_db_path=index_db, dry_run=True)

    assert result.eligible_count == 1
    assert result.reissued_count == 0
    assert result.errors == ()
    with sqlite3.connect(index_db) as index_conn:
        count = index_conn.execute(
            "SELECT COUNT(*) FROM raw_revision_applications WHERE raw_id = 'raw-old-full'"
        ).fetchone()[0]
    assert count == 1


def test_stale_supersession_reissue_authorizes_release_end_to_end(tmp_path: Path) -> None:
    """The killer integration proof: a reissued receipt must actually flip
    ``active_raw_retention_authority`` eligibility, exactly as a fresh
    production-written receipt would."""
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    _seed_stale_full_reset_chain(source_db, index_db)

    with sqlite3.connect(source_db) as conn:
        authority_before = active_raw_retention_authority(conn, index_db_path=index_db)
    # Before reissue: only the already-current mid receipt is eligible; the
    # stale old receipt authorizes nothing.
    assert authority_before.eligible_raw_ids == frozenset({"raw-mid-full"})
    assert authority_before.protected_raw_ids == frozenset({"raw-new-full"})

    with sqlite3.connect(source_db) as source_conn, sqlite3.connect(index_db) as index_conn:
        result = reissue_stale_supersession_receipts(source_conn, index_conn, index_db_path=index_db, dry_run=False)

    assert result.reissued_count == 1
    assert result.errors == ()

    with sqlite3.connect(source_db) as conn:
        authority_after = active_raw_retention_authority(conn, index_db_path=index_db)
    assert authority_after.eligible_raw_ids == frozenset({"raw-mid-full", "raw-old-full"})
    assert authority_after.protected_raw_ids == frozenset({"raw-new-full"})

    # The original stale receipt is untouched (immutable); a second, distinct
    # row now exists for the same raw.
    with sqlite3.connect(index_db) as index_conn:
        rows = index_conn.execute(
            "SELECT accepted_raw_id, acquisition_generation FROM raw_revision_applications "
            "WHERE raw_id = 'raw-old-full' ORDER BY acquisition_generation"
        ).fetchall()
    assert rows == [("raw-mid-full", 1), ("raw-new-full", 2)]


def test_stale_supersession_reissue_refuses_when_head_is_in_progress_append(tmp_path: Path) -> None:
    """Anti-vacuity: an in-progress append chain must never authorize a
    receipt for a *different* raw. Failure mode this guards: if the proof
    rule were loosened to "raw is an ancestor of the current head's chain"
    instead of "the current head is a self-contained full reset", this test
    would start asserting a receipt for ``raw-baseline`` -- which is still a
    required, protected ancestor of the live append chain, not something the
    current head has proven superseded."""
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
            source_revision="revision-baseline",
            generation=0,
            blob_size=10,
        )
        _insert_revision_raw(
            conn,
            raw_id="raw-append-1",
            source_path=source_path,
            acquired_at_ms=2,
            kind="append",
            source_revision="revision-1",
            generation=1,
            blob_size=5,
            predecessor_raw_id="raw-baseline",
            predecessor_revision="revision-baseline",
            baseline_raw_id="raw-baseline",
            append_start_offset=10,
            append_end_offset=15,
        )
        _insert_revision_raw(
            conn,
            raw_id="raw-append-2",
            source_path=source_path,
            acquired_at_ms=3,
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
        session_raw_id="raw-append-2",
        accepted_raw_id="raw-append-2",
        accepted_revision="revision-2",
        generation=2,
        frontier=20,
        append_end_offset=20,
    )
    # Stale: recorded when raw-append-1 was (momentarily) treated as an
    # accepted head; the live head has since advanced to raw-append-2. This
    # receipt should never have existed in real production (an ongoing
    # append chain's ancestors are never marked superseded), but the reissue
    # pass must refuse it defensively regardless of how it got here.
    _seed_superseded_application(
        index_db,
        raw_id="raw-baseline",
        source_revision="revision-baseline",
        accepted_generation=1,
        accepted_raw_id="raw-append-1",
        accepted_revision="revision-1",
        accepted_append_end_offset=15,
    )

    with sqlite3.connect(source_db) as conn:
        plan = plan_stale_supersession_reissue(conn, index_db_path=index_db)

    assert plan.eligible == ()
    assert plan.stale_count == 1
    assert any("not a byte-proven full reset" in reason for reason in plan.ineligible_reason_counts)

    with sqlite3.connect(source_db) as source_conn, sqlite3.connect(index_db) as index_conn:
        result = reissue_stale_supersession_receipts(source_conn, index_conn, index_db_path=index_db, dry_run=False)
    assert result.reissued_count == 0
    with sqlite3.connect(index_db) as index_conn:
        count = index_conn.execute(
            "SELECT COUNT(*) FROM raw_revision_applications WHERE raw_id = 'raw-baseline'"
        ).fetchone()[0]
    assert count == 1

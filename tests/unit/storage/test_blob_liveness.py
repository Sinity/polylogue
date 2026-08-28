"""Production-schema proofs for the canonical current blob-owner relation."""

from __future__ import annotations

import os
import re
import sqlite3
import time
from pathlib import Path

import pytest

import polylogue.storage.hook_payload_ref_reconciliation as hook_payload_ref_reconciliation
from polylogue.storage.blob_gc import run_blob_gc_report
from polylogue.storage.blob_integrity import referenced_blob_hashes
from polylogue.storage.blob_liveness import (
    LivenessState,
    inspect_blob_liveness,
    project_live_blob_hashes,
    validated_blob_ref_liveness_joins,
)
from polylogue.storage.blob_ref_liveness import classify_blob_ref_liveness
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import deterministic_raw_session_id


def _archive(tmp_path: Path) -> tuple[Path, bytes]:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    return root, b"x" * 32


@pytest.mark.parametrize(
    ("ref_type", "table", "identifier", "expected"),
    (
        ("raw_payload", "raw_sessions", "raw-live", "source.db.blob_refs"),
        ("attachment", "raw_sessions", "raw-live", "source.db.blob_refs"),
        ("hook_payload", "raw_hook_events", "hook-live", "source.db.blob_refs"),
        ("sidecar", "history_sidecars", "sidecar-live", "source.db.blob_refs"),
    ),
)
def test_joined_ledger_kinds_protect_only_live_referents(
    tmp_path: Path, ref_type: str, table: str, identifier: str, expected: str
) -> None:
    """Deleting the mapped referent is the negative twin for each ledger kind."""
    root, blob_hash = _archive(tmp_path)
    with sqlite3.connect(root / "source.db") as conn, sqlite3.connect(root / "index.db") as index:
        if table == "raw_sessions":
            conn.execute(
                """INSERT INTO raw_sessions
                (raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms)
                VALUES (?, 'codex-session', '/raw', 0, ?, 1, 1)""",
                (identifier, b"r" * 32),
            )
        elif table == "raw_hook_events":
            conn.execute(
                """INSERT INTO raw_hook_events
                (hook_event_id, origin, source_path, event_type, payload_json, observed_at_ms, blob_hash)
                VALUES (?, 'codex-session', '/hook', 'PostToolUse', '{}', 1, ?)""",
                (identifier, b"h" * 32),
            )
        else:
            conn.execute(
                """INSERT INTO history_sidecars
                (sidecar_id, origin, source_path, payload_json, observed_at_ms, content_hash)
                VALUES (?, 'codex-session', '/sidecar', '{}', 1, ?)""",
                (identifier, b"s" * 32),
            )
        conn.execute(
            """INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, ?, '/fixture', 1, 1)""",
            (blob_hash, identifier, ref_type),
        )

        assert expected in inspect_blob_liveness(conn, blob_hash.hex(), index_conn=index, require_index=True).surfaces
        if table == "raw_sessions":
            conn.execute("DELETE FROM raw_sessions WHERE raw_id = ?", (identifier,))
        elif table == "raw_hook_events":
            conn.execute("DELETE FROM raw_hook_events WHERE hook_event_id = ?", (identifier,))
        else:
            conn.execute("DELETE FROM history_sidecars WHERE sidecar_id = ?", (identifier,))
        # A bare ledger row is not authority. Replacing the relation with a
        # membership test makes this assertion fail.
        assert (
            inspect_blob_liveness(conn, blob_hash.hex(), index_conn=index, require_index=True).state
            is LivenessState.UNREFERENCED
        )


def test_direct_owners_and_reservation_survive_missing_or_dangling_ledger(tmp_path: Path) -> None:
    """Direct ownership, duplicate ownership, and publish reservations are independent protections."""
    root, blob_hash = _archive(tmp_path)
    with sqlite3.connect(root / "source.db") as conn, sqlite3.connect(root / "index.db") as index:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(
            """INSERT INTO raw_sessions
            (raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-direct', 'codex-session', '/raw', 0, ?, 1, 1)""",
            (blob_hash,),
        )
        index.execute(
            "INSERT INTO attachments (attachment_id, blob_hash, acquisition_status) VALUES ('att-direct', ?, 'acquired')",
            (blob_hash,),
        )
        conn.execute(
            """INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'gone', 'raw_payload', '/gone', 1, 1)""",
            (blob_hash,),
        )
        decision = inspect_blob_liveness(conn, blob_hash.hex(), index_conn=index, require_index=True)
        assert set(decision.surfaces) == {"source.db.raw_sessions", "index.db.attachments"}
        conn.execute("DELETE FROM raw_sessions")
        index.execute("DELETE FROM attachments")
        conn.execute(
            """INSERT INTO blob_publication_reservations
            (publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms)
            VALUES ('publication', ?, 1, 'publisher', 1)""",
            (blob_hash,),
        )
        assert (
            inspect_blob_liveness(conn, blob_hash.hex(), index_conn=index, require_index=True).state
            is LivenessState.UNREFERENCED
        )
        conn.execute("DELETE FROM blob_publication_reservations")
        assert (
            inspect_blob_liveness(conn, blob_hash.hex(), index_conn=index, require_index=True).state
            is LivenessState.UNREFERENCED
        )


def test_unknown_or_missing_owner_surface_is_an_observable_gc_blocker(tmp_path: Path) -> None:
    root, blob_hash = _archive(tmp_path)
    with sqlite3.connect(root / "source.db") as conn, sqlite3.connect(root / "index.db") as index:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(
            """INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'future', 'future-kind', '/future', 1, 1)""",
            (blob_hash,),
        )
        unknown = inspect_blob_liveness(conn, blob_hash.hex(), index_conn=index, require_index=True)
        assert unknown.state is LivenessState.BLOCKED and any("unknown" in blocker for blocker in unknown.blockers)
        index.execute("DROP TABLE attachments")
        unavailable = inspect_blob_liveness(conn, blob_hash.hex(), index_conn=index, require_index=True)
        assert unavailable.state is LivenessState.BLOCKED and "index.attachments is missing" in unavailable.blockers


@pytest.mark.uses_real_clock("backdates real GC candidates")
def test_unknown_ref_kind_on_another_hash_blocks_the_entire_destructive_pass(tmp_path: Path) -> None:
    """A future ledger kind blocks unlink of an otherwise orphaned hash."""
    root, _unused = _archive(tmp_path)
    store = BlobStore(root / "blob")
    unknown_hash, _size = store.write_from_bytes(b"unknown ledger payload")
    orphan_hash, _size = store.write_from_bytes(b"orphan candidate")
    for blob_hash in (unknown_hash, orphan_hash):
        os.utime(store.blob_path(blob_hash), (time.time() - 3600, time.time() - 3600))
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(
            """INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'future', 'future-kind', '/future', 1, 1)""",
            (bytes.fromhex(unknown_hash),),
        )

    report = run_blob_gc_report(root / "source.db", store.root)

    assert report.blocked_reason is not None
    assert "unknown blob_refs ref_type" in report.blocked_reason
    assert store.exists(orphan_hash)


def test_structured_decision_fails_closed_for_missing_required_source_schema(tmp_path: Path) -> None:
    root, blob_hash = _archive(tmp_path)
    with sqlite3.connect(root / "source.db") as source, sqlite3.connect(root / "index.db") as index:
        decision = inspect_blob_liveness(source, blob_hash.hex(), index_conn=index, require_index=True)
        assert decision.state is LivenessState.UNREFERENCED
        source.execute("DROP TABLE blob_refs")
        blocked = inspect_blob_liveness(source, blob_hash.hex(), index_conn=index, require_index=True)
        assert blocked.state is LivenessState.BLOCKED
        assert "source.blob_refs is missing" in blocked.blockers


def test_bulk_projection_excludes_dangling_ledger_rows_and_verification_receipts(tmp_path: Path) -> None:
    """Integrity and seal denominators count canonical owners only."""
    root, blob_hash = _archive(tmp_path)
    with sqlite3.connect(root / "source.db") as source:
        source.execute(
            """INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'gone', 'raw_payload', '/gone', 1, 1)""",
            (blob_hash,),
        )
        source.execute(
            """INSERT INTO verified_blob_receipts
            (blob_hash, st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns, verified_at_ms)
            VALUES (?, 1, 1, 1, 1, 1, 1)""",
            (blob_hash,),
        )
        projection = project_live_blob_hashes(source)

    assert projection.live_hashes == frozenset()


@pytest.mark.uses_real_clock("backdates legacy hook and dangling blobs for the production GC age gate")
def test_legacy_hook_rekey_proof_is_live_across_liveness_integrity_diagnostics_and_gc(tmp_path: Path) -> None:
    """Matched and ambiguous legacy hook refs both retain their payload bytes."""

    root, _unused = _archive(tmp_path)
    store = BlobStore(root / "blob")
    legacy_hash, legacy_size = store.write_from_bytes(b'{"event":"PostToolUse","legacy":true}')
    ambiguous_hash, ambiguous_size = store.write_from_bytes(b'{"event":"PostToolUse","ambiguous":true}')
    dangling_hash, dangling_size = store.write_from_bytes(b"arbitrary dangling raw-payload ledger row")
    old = time.time() - 3_600
    os.utime(store.blob_path(legacy_hash), (old, old))
    os.utime(store.blob_path(ambiguous_hash), (old, old))
    os.utime(store.blob_path(dangling_hash), (old, old))
    source_path = "/hooks/legacy.jsonl"
    native_id = "legacy-native-id"
    deterministic_raw_id = deterministic_raw_session_id(
        "codex-session", source_path, 0, bytes.fromhex(legacy_hash), native_id
    )
    ambiguous_path = "/hooks/ambiguous.jsonl"
    ambiguous_raw_id = deterministic_raw_session_id(
        "codex-session", ambiguous_path, 0, bytes.fromhex(ambiguous_hash), "ambiguous-native-a"
    )
    with sqlite3.connect(root / "source.db") as source, sqlite3.connect(root / "index.db") as index:
        # This is the pre-v22 shape produced by the historic hook writer: the
        # hook has no direct blob hash and the raw_payload ref has no raw row.
        source.execute(
            """INSERT INTO raw_hook_events
            (hook_event_id, origin, native_id, session_native_id, source_path, event_type, payload_json, observed_at_ms)
            VALUES ('legacy-hook', 'codex-session', ?, 'session-1', ?, 'PostToolUse', '{}', 1)""",
            (native_id, source_path),
        )
        # Two old hooks share a path and have no direct blob hashes. We cannot
        # safely rewrite the ref to either hook, but its bytes remain owned.
        source.executemany(
            """INSERT INTO raw_hook_events
            (hook_event_id, origin, native_id, session_native_id, source_path, event_type, payload_json, observed_at_ms)
            VALUES (?, 'codex-session', ?, 'session-1', ?, 'PostToolUse', '{}', 1)""",
            (
                ("ambiguous-hook-a", "ambiguous-native-a", ambiguous_path),
                ("ambiguous-hook-b", "ambiguous-native-b", ambiguous_path),
            ),
        )
        source.executemany(
            """INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, 'raw_payload', ?, ?, 1)""",
            (
                (bytes.fromhex(legacy_hash), deterministic_raw_id, source_path, legacy_size),
                (bytes.fromhex(ambiguous_hash), ambiguous_raw_id, ambiguous_path, ambiguous_size),
                (bytes.fromhex(dangling_hash), "not-a-deterministic-hook-id", "/hooks/dangling.jsonl", dangling_size),
            ),
        )

        live = inspect_blob_liveness(source, legacy_hash, index_conn=index, require_index=True)
        ambiguous = inspect_blob_liveness(source, ambiguous_hash, index_conn=index, require_index=True)
        dangling = inspect_blob_liveness(source, dangling_hash, index_conn=index, require_index=True)
        projection = project_live_blob_hashes(source, index_conn=index, require_index=True)
        diagnostics = classify_blob_ref_liveness(source)

    assert live.state is LivenessState.LIVE
    assert live.surfaces == ("source.db.rekeyable_hook_payload",)
    assert ambiguous.state is LivenessState.LIVE
    assert ambiguous.surfaces == ("source.db.rekeyable_hook_payload",)
    assert dangling.state is LivenessState.UNREFERENCED
    assert legacy_hash in projection.live_hashes
    assert ambiguous_hash in projection.live_hashes
    assert dangling_hash not in projection.live_hashes
    assert diagnostics.rekeyable_hook_payload_count == 2
    assert diagnostics.orphaned_by_ref_type == {"raw_payload": 1}
    assert legacy_hash in referenced_blob_hashes(root / "source.db")
    assert ambiguous_hash in referenced_blob_hashes(root / "source.db")
    assert dangling_hash not in referenced_blob_hashes(root / "source.db")

    report = run_blob_gc_report(root / "source.db", store.root)

    assert report.deleted_count == 1
    assert store.exists(legacy_hash)
    assert store.exists(ambiguous_hash)
    assert not store.exists(dangling_hash)


@pytest.mark.uses_real_clock("backdates a blob while legacy matcher evidence is unavailable")
def test_legacy_hook_rekey_schema_gap_blocks_liveness_and_gc(tmp_path: Path) -> None:
    root, _unused = _archive(tmp_path)
    store = BlobStore(root / "blob")
    blob_hash, _size = store.write_from_bytes(b"must remain when rekey evidence is incomplete")
    old = time.time() - 3_600
    os.utime(store.blob_path(blob_hash), (old, old))

    with sqlite3.connect(root / "source.db") as source, sqlite3.connect(root / "index.db") as index:
        source.execute("ALTER TABLE blob_refs DROP COLUMN acquired_at_ms")
        decision = inspect_blob_liveness(source, blob_hash, index_conn=index, require_index=True)

    assert decision.state is LivenessState.BLOCKED
    assert "source.blob_refs is missing columns: acquired_at_ms" in decision.blockers
    assert "source.legacy hook rekey evidence is unavailable" in decision.blockers

    report = run_blob_gc_report(root / "source.db", store.root)

    assert report.blocked_reason is not None
    assert "legacy hook rekey evidence is unavailable" in report.blocked_reason
    assert store.exists(blob_hash)


@pytest.mark.uses_real_clock("backdates a blob while the legacy matcher fails")
def test_legacy_hook_rekey_matcher_failure_blocks_liveness_and_gc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _unused = _archive(tmp_path)
    store = BlobStore(root / "blob")
    blob_hash, _size = store.write_from_bytes(b"must remain when rekey matcher fails")
    old = time.time() - 3_600
    os.utime(store.blob_path(blob_hash), (old, old))

    def fail_matcher(_conn: sqlite3.Connection) -> tuple[int, int, int, int]:
        raise RuntimeError("injected matcher integrity failure")

    monkeypatch.setattr(hook_payload_ref_reconciliation, "_create_match_stage", fail_matcher)
    with sqlite3.connect(root / "source.db") as source, sqlite3.connect(root / "index.db") as index:
        decision = inspect_blob_liveness(source, blob_hash, index_conn=index, require_index=True)
        projection = project_live_blob_hashes(source, index_conn=index, require_index=True)

    assert decision.state is LivenessState.BLOCKED
    assert "legacy hook rekey matcher failed" in decision.blockers[0]
    assert projection.blockers and "legacy hook rekey matcher failed" in projection.blockers[0]

    report = run_blob_gc_report(root / "source.db", store.root)

    assert report.blocked_reason is not None
    assert "legacy hook rekey matcher failed" in report.blocked_reason
    assert store.exists(blob_hash)


@pytest.mark.uses_real_clock("backdates the real blob mtime for GC's production age gate")
def test_gc_refuses_missing_current_owner_surface_before_unlink(tmp_path: Path) -> None:
    """Dropping index ownership is a controlled mutation of GC's safety gate."""
    root, _unused = _archive(tmp_path)
    store = BlobStore(root / "blob")
    blob_hash, _size = store.write_from_bytes(b"would be collected without the index owner")
    path = store.blob_path(blob_hash)
    old = time.time() - 3600
    os.utime(path, (old, old))
    with sqlite3.connect(root / "index.db") as index:
        index.execute("DROP TABLE attachments")

    report = run_blob_gc_report(root / "source.db", store.root)

    assert report.deleted_count == 0
    assert report.blocked_reason is not None
    assert "index.attachments is missing" in report.blocked_reason
    assert store.exists(blob_hash)


def test_blob_refs_check_vocabulary_and_liveness_join_map_agree(tmp_path: Path) -> None:
    """Every ref_type the source DDL admits must have a liveness referent join.

    A ref_type the schema accepts but the map cannot resolve makes liveness
    projection return BLOCKED for any archive holding such a row, stalling GC
    on data the writer was entitled to create. The two vocabularies live in
    different files, so nothing but this proof couples them.
    """

    root, _ = _archive(tmp_path)
    with sqlite3.connect(root / "source.db") as conn:
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'blob_refs'").fetchone()[0]
    match = re.search(r"ref_type\s+TEXT\s+NOT NULL\s+CHECK\(ref_type IN \(([^)]*)\)\)", sql)
    assert match is not None, f"could not read the blob_refs ref_type CHECK from: {sql}"
    admitted = {literal.strip().strip("'") for literal in match.group(1).split(",")}

    mapped = {ref_type for ref_type, _table, _column in validated_blob_ref_liveness_joins()}
    assert admitted == mapped, (
        f"blob_refs CHECK admits {sorted(admitted)} but the liveness join map resolves "
        f"{sorted(mapped)}; unresolved ref_types block liveness projection"
    )

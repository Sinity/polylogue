"""Read-only blob integrity scanner contracts (#1231)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.storage.blob_integrity import (
    classify_blob_reference_debt,
    plan_raw_backed_blob_reference_recovery,
    prune_orphan_blob_reference_debt,
    referenced_blob_hashes,
    restore_direct_blob_reference_debt,
    scan_blob_integrity,
    scan_blob_reference_debt,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE raw_sessions (
            raw_id TEXT PRIMARY KEY,
            source_name TEXT NOT NULL DEFAULT '',
            source_path TEXT NOT NULL DEFAULT '',
            blob_hash BLOB,
            blob_size INTEGER NOT NULL DEFAULT 0,
            acquired_at TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE blob_refs (
            blob_hash BLOB NOT NULL,
            raw_id TEXT NOT NULL,
            ref_type TEXT NOT NULL DEFAULT 'raw_payload'
        );
        CREATE TABLE pending_blob_refs (
            blob_hash TEXT NOT NULL,
            operation_id TEXT NOT NULL,
            acquired_at INTEGER NOT NULL,
            PRIMARY KEY (blob_hash, operation_id)
        );
        """
    )
    conn.commit()
    return conn


def test_scan_blob_integrity_classifies_missing_orphan_hash_mismatch_and_stale_lease(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    store = BlobStore(tmp_path / "blob")
    conn = _make_db(db_path)
    active_lease_time = 4_000_000_000

    referenced_ok, ok_size = store.write_from_bytes(b"referenced")
    orphan_hash, _ = store.write_from_bytes(b"orphan")
    corrupt_hash, corrupt_size = store.write_from_bytes(b"original")
    store.blob_path(corrupt_hash).write_bytes(b"corrupted")
    leased_hash, _ = store.write_from_bytes(b"leased but not committed")
    stale_lease_hash, _ = store.write_from_bytes(b"stale lease")
    missing_hash = "0" * 64

    for blob_hash, size in ((referenced_ok, ok_size), (missing_hash, 128), (corrupt_hash, corrupt_size)):
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, blob_size, acquired_at) VALUES (?, ?, ?)",
            (blob_hash, size, "2026-05-24T00:00:00+00:00"),
        )
    conn.execute(
        "INSERT INTO pending_blob_refs (blob_hash, operation_id, acquired_at) VALUES (?, ?, ?)",
        (leased_hash, "active-op", active_lease_time),
    )
    conn.execute(
        "INSERT INTO pending_blob_refs (blob_hash, operation_id, acquired_at) VALUES (?, ?, ?)",
        (stale_lease_hash, "stale-op", 1),
    )
    conn.commit()
    conn.close()

    report = scan_blob_integrity(db_path, store=store, full=True, stale_lease_s=3600)

    by_kind = {finding.kind: finding for finding in report.findings}
    assert by_kind["missing_referenced_blobs"].sample == (missing_hash,)
    assert by_kind["orphan_blobs"].sample == (orphan_hash,)
    assert by_kind["hash_mismatch"].sample == (corrupt_hash,)
    assert by_kind["stale_leases"].sample == (stale_lease_hash,)
    assert leased_hash not in by_kind["orphan_blobs"].sample
    assert stale_lease_hash not in by_kind["orphan_blobs"].sample


def test_scan_blob_integrity_bounds_default_probe_but_full_scans_everything(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    store = BlobStore(tmp_path / "blob")
    conn = _make_db(db_path)
    hashes = [store.write_from_bytes(f"payload-{idx}".encode())[0] for idx in range(3)]
    for blob_hash in hashes:
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, blob_size, acquired_at) VALUES (?, ?, ?)",
            (blob_hash, 9, "2026-05-24T00:00:00+00:00"),
        )
    conn.commit()
    conn.close()

    sampled = scan_blob_integrity(db_path, store=store, full=False, sample_size=1)
    full = scan_blob_integrity(db_path, store=store, full=True, sample_size=1)

    assert sampled.scanned_blobs == 1
    assert sampled.scanned_references == 1
    assert full.scanned_blobs == 3
    assert full.scanned_references == 3


def test_scan_blob_integrity_reads_source_tier_blob_refs(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    store = BlobStore(tmp_path / "blob")
    raw_hash, raw_size = store.write_from_bytes(b"raw payload")
    attachment_hash, attachment_size = store.write_from_bytes(b"attachment")
    orphan_hash, _ = store.write_from_bytes(b"orphan")
    with sqlite3.connect(source_db) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                blob_hash BLOB NOT NULL,
                blob_size INTEGER NOT NULL
            );
            CREATE TABLE blob_refs (
                blob_hash BLOB NOT NULL,
                raw_id TEXT NOT NULL,
                ref_type TEXT NOT NULL,
                source_path TEXT,
                size_bytes INTEGER NOT NULL,
                acquired_at_ms INTEGER NOT NULL,
                PRIMARY KEY(blob_hash, raw_id, ref_type)
            );
            """
        )
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, blob_hash, blob_size) VALUES (?, ?, ?)",
            ("raw-v1", bytes.fromhex(raw_hash), raw_size),
        )
        conn.execute(
            """
            INSERT INTO blob_refs (
                blob_hash, raw_id, ref_type, source_path, size_bytes, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (bytes.fromhex(attachment_hash), "raw-v1", "attachment", "/tmp/a.bin", attachment_size, 1),
        )

    report = scan_blob_integrity(source_db, store=store, full=True)

    by_kind = {finding.kind: finding for finding in report.findings}
    assert report.total_references_seen == 2
    assert by_kind["orphan_blobs"].sample == (orphan_hash,)
    assert raw_hash not in by_kind["orphan_blobs"].sample
    assert attachment_hash not in by_kind["orphan_blobs"].sample


def test_scan_blob_reference_debt_counts_all_missing_refs_with_bounded_sample(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    store = BlobStore(tmp_path / "blob")
    present_hash, present_size = store.write_from_bytes(b"present")
    missing_hashes = [f"{idx:064x}" for idx in range(5)]
    with sqlite3.connect(source_db) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                blob_hash BLOB NOT NULL,
                blob_size INTEGER NOT NULL
            );
            CREATE TABLE blob_refs (
                blob_hash BLOB NOT NULL,
                raw_id TEXT NOT NULL,
                ref_type TEXT NOT NULL,
                PRIMARY KEY(blob_hash, raw_id, ref_type)
            );
            """
        )
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, blob_hash, blob_size) VALUES (?, ?, ?)",
            ("raw-present", bytes.fromhex(present_hash), present_size),
        )
        for idx, blob_hash in enumerate(missing_hashes):
            conn.execute(
                "INSERT INTO blob_refs (blob_hash, raw_id, ref_type) VALUES (?, ?, ?)",
                (bytes.fromhex(blob_hash), f"raw-{idx}", "attachment"),
            )

    report = scan_blob_reference_debt(source_db, store=store, sample_size=2)

    assert report.ok is False
    assert report.total_references_seen == 6
    assert report.missing_referenced_blobs == 5
    assert report.sample == tuple(missing_hashes[:2])
    assert report.reference_sources == {"raw_sessions": 1, "blob_refs": 5}
    assert referenced_blob_hashes(source_db) == sorted([present_hash, *missing_hashes])


def test_scan_blob_reference_debt_reads_initialized_source_tier(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    store = BlobStore(tmp_path / "blob")
    present_hash, present_size = store.write_from_bytes(b"present")
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "raw-present",
                "codex-session",
                "native-1",
                "/tmp/session.jsonl",
                bytes.fromhex(present_hash),
                present_size,
                1,
            ),
        )

    report = scan_blob_reference_debt(source_db, store=store)

    assert report.ok is True
    assert report.total_references_seen == 1
    assert report.reference_sources == {"raw_sessions": 1}


def test_classify_blob_reference_debt_groups_recovery_evidence(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    store = BlobStore(tmp_path / "blob")
    source_file = tmp_path / "exports" / "chatgpt.json"
    source_file.parent.mkdir()
    source_file.write_text("{}", encoding="utf-8")
    present_hash, present_size = store.write_from_bytes(b"present")
    missing_raw_hash = "1" * 64
    missing_attachment_hash = "2" * 64
    orphan_ref_hash = "3" * 64

    with sqlite3.connect(source_db) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                native_id TEXT,
                source_path TEXT,
                blob_hash BLOB,
                blob_size INTEGER NOT NULL,
                parse_error TEXT,
                validation_status TEXT
            );
            CREATE TABLE blob_refs (
                blob_hash BLOB NOT NULL,
                raw_id TEXT NOT NULL,
                ref_type TEXT NOT NULL,
                source_path TEXT,
                size_bytes INTEGER NOT NULL,
                PRIMARY KEY(blob_hash, raw_id, ref_type)
            );
            """
        )
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, validation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "raw-present",
                "codex-session",
                "native-present",
                str(source_file),
                bytes.fromhex(present_hash),
                present_size,
                "passed",
            ),
        )
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, validation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "raw-missing",
                "chatgpt-export",
                "native-missing",
                str(source_file),
                bytes.fromhex(missing_raw_hash),
                123,
                "passed",
            ),
        )
        conn.execute(
            "INSERT INTO blob_refs (blob_hash, raw_id, ref_type, source_path, size_bytes) VALUES (?, ?, ?, ?, ?)",
            (bytes.fromhex(missing_attachment_hash), "raw-missing", "attachment", str(source_file), 456),
        )
        conn.execute(
            "INSERT INTO blob_refs (blob_hash, raw_id, ref_type, source_path, size_bytes) VALUES (?, ?, ?, ?, ?)",
            (bytes.fromhex(orphan_ref_hash), "raw-gone", "raw_payload", str(tmp_path / "missing.json"), 789),
        )

    report = classify_blob_reference_debt(source_db, store=store, sample_size=2, group_limit=3)

    assert report.ok is False
    assert report.distinct_referenced_blobs == 4
    assert report.reference_rows == 4
    assert report.missing_distinct_blobs == 3
    assert report.missing_by_table == {"blob_refs": 2, "raw_sessions": 1}
    assert report.missing_by_ref_type == {"attachment": 1, "raw_payload": 2}
    assert report.missing_by_origin == {"(none)": 1, "chatgpt-export": 2}
    assert report.missing_ref_id_join == {"ref_id_has_raw_session": 2, "ref_id_without_raw_session": 1}
    assert report.missing_source_path_presence == {
        "recoverable_source_path_exists": 2,
        "source_path_missing": 1,
    }
    assert report.missing_validation_status == {"(none)": 1, "passed": 2}
    assert report.missing_parse_error == {"no_parse_error": 3}
    assert len(report.samples) == 2
    payload = report.to_dict()
    samples = payload["samples"]
    assert isinstance(samples, list)
    first_sample = samples[0]
    assert isinstance(first_sample, dict)
    assert first_sample["sample_source_available"] is True


def test_restore_direct_blob_reference_debt_dry_run_and_apply(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    store = BlobStore(tmp_path / "blob")
    source = tmp_path / "session.jsonl"
    source.write_bytes(b"direct raw payload")
    expected_hash = "e6e9b015177c95ba07d4d2bd2cb423697aa06973606e922da27371e06dc0f457"
    container_hash = "4" * 64
    with sqlite3.connect(source_db) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                blob_hash BLOB NOT NULL,
                blob_size INTEGER NOT NULL,
                source_path TEXT
            );
            CREATE TABLE blob_refs (
                blob_hash BLOB NOT NULL,
                raw_id TEXT NOT NULL,
                ref_type TEXT NOT NULL,
                source_path TEXT,
                size_bytes INTEGER NOT NULL,
                PRIMARY KEY(blob_hash, raw_id, ref_type)
            );
            """
        )
        conn.execute(
            "INSERT INTO blob_refs (blob_hash, raw_id, ref_type, source_path, size_bytes) VALUES (?, ?, ?, ?, ?)",
            (bytes.fromhex(expected_hash), "raw-direct", "raw_payload", str(source), source.stat().st_size),
        )
        conn.execute(
            "INSERT INTO blob_refs (blob_hash, raw_id, ref_type, source_path, size_bytes) VALUES (?, ?, ?, ?, ?)",
            (bytes.fromhex(container_hash), "raw-container", "raw_payload", f"{source}:inner.json", 10),
        )

    dry_run = restore_direct_blob_reference_debt(source_db, store=store)

    assert dry_run.dry_run is True
    assert dry_run.missing_distinct_blobs == 2
    assert dry_run.candidate_count == 1
    assert dry_run.restored_count == 0
    assert dry_run.skipped_container_member == 1
    assert not store.exists(expected_hash)

    applied = restore_direct_blob_reference_debt(source_db, store=store, dry_run=False)

    assert applied.dry_run is False
    assert applied.candidate_count == 1
    assert applied.restored_count == 1
    assert applied.restored_bytes == source.stat().st_size
    assert applied.skipped_container_member == 1
    assert store.exists(expected_hash)
    assert not store.exists(container_hash)


def test_restore_direct_blob_reference_debt_rejects_hash_mismatch(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    store = BlobStore(tmp_path / "blob")
    source = tmp_path / "session.jsonl"
    source.write_bytes(b"different bytes")
    expected_hash = "5" * 64
    with sqlite3.connect(source_db) as conn:
        conn.executescript(
            """
            CREATE TABLE blob_refs (
                blob_hash BLOB NOT NULL,
                raw_id TEXT NOT NULL,
                ref_type TEXT NOT NULL,
                source_path TEXT,
                size_bytes INTEGER NOT NULL,
                PRIMARY KEY(blob_hash, raw_id, ref_type)
            );
            """
        )
        conn.execute(
            "INSERT INTO blob_refs (blob_hash, raw_id, ref_type, source_path, size_bytes) VALUES (?, ?, ?, ?, ?)",
            (bytes.fromhex(expected_hash), "raw-mismatch", "raw_payload", str(source), source.stat().st_size),
        )

    report = restore_direct_blob_reference_debt(source_db, store=store, dry_run=False)

    assert report.candidate_count == 1
    assert report.restored_count == 0
    assert report.skipped_hash_mismatch == 1
    assert not store.exists(expected_hash)
    assert [path for path in store.root.rglob("*") if path.is_file()] == []


def test_prune_orphan_blob_reference_debt_quarantines_before_delete(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    store = BlobStore(tmp_path / "blob")
    present_hash, _present_size = store.write_from_bytes(b"present blob")
    orphan_hash = "6" * 64
    raw_backed_hash = "7" * 64
    quarantine_path = tmp_path / "quarantine" / "orphan-refs.jsonl"
    with sqlite3.connect(source_db) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                blob_hash BLOB,
                blob_size INTEGER NOT NULL,
                source_path TEXT
            );
            CREATE TABLE blob_refs (
                blob_hash BLOB NOT NULL,
                ref_id TEXT NOT NULL,
                ref_type TEXT NOT NULL,
                source_path TEXT,
                size_bytes INTEGER NOT NULL,
                acquired_at_ms INTEGER NOT NULL,
                PRIMARY KEY(blob_hash, ref_type, ref_id)
            );
            """
        )
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, blob_hash, blob_size, source_path) VALUES (?, ?, ?, ?)",
            ("raw-present", bytes.fromhex(raw_backed_hash), 10, "recoverable.json"),
        )
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (bytes.fromhex(orphan_hash), "raw-gone", "raw_payload", "old.json", 12, 1),
        )
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (bytes.fromhex(raw_backed_hash), "raw-present", "raw_payload", "recoverable.json", 10, 1),
        )
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (bytes.fromhex(present_hash), "raw-gone-present", "raw_payload", "present.json", 12, 1),
        )

    dry_run = prune_orphan_blob_reference_debt(source_db, store=store, quarantine_path=quarantine_path)

    assert dry_run.dry_run is True
    assert dry_run.missing_orphan_refs == 1
    assert dry_run.pruned_refs == 0
    assert not quarantine_path.exists()

    applied = prune_orphan_blob_reference_debt(
        source_db,
        store=store,
        dry_run=False,
        quarantine_path=quarantine_path,
    )

    assert applied.dry_run is False
    assert applied.pruned_refs == 1
    assert applied.pruned_distinct_blobs == 1
    exported = [json.loads(line) for line in quarantine_path.read_text(encoding="utf-8").splitlines()]
    assert exported == [
        {
            "acquired_at": 1,
            "blob_hash": orphan_hash,
            "raw_session_present": 0,
            "ref_id": "raw-gone",
            "ref_type": "raw_payload",
            "size_bytes": 12,
            "source_path": "old.json",
        }
    ]
    with sqlite3.connect(source_db) as conn:
        remaining_refs = conn.execute("SELECT ref_id FROM blob_refs ORDER BY ref_id").fetchall()
    assert remaining_refs == [("raw-gone-present",), ("raw-present",)]


def test_plan_raw_backed_blob_reference_recovery_writes_manifest(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    store = BlobStore(tmp_path / "blob")
    exact_source = tmp_path / "exact.json"
    exact_source.write_bytes(b"exact payload")
    exact_hash = "0cfefcacfe03534dd908444efd6e4d0d1075fd8cf59ac79bf956312385679cfe"
    changed_source = tmp_path / "changed.json"
    changed_source.write_bytes(b"changed payload")
    container_outer = tmp_path / "export.zip"
    container_outer.write_bytes(b"zip bytes are not inspected by the planner")
    manifest_path = tmp_path / "manifest" / "raw-backed.jsonl"
    present_hash, present_size = store.write_from_bytes(b"present raw")
    with sqlite3.connect(source_db) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                native_id TEXT,
                source_path TEXT,
                source_index INTEGER,
                blob_hash BLOB,
                blob_size INTEGER NOT NULL
            );
            """
        )
        conn.executemany(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "raw-present",
                    "chatgpt-export",
                    "native-present",
                    str(tmp_path / "present.json"),
                    0,
                    bytes.fromhex(present_hash),
                    present_size,
                ),
                (
                    "raw-exact",
                    "chatgpt-export",
                    "native-exact",
                    str(exact_source),
                    0,
                    bytes.fromhex(exact_hash),
                    exact_source.stat().st_size,
                ),
                (
                    "raw-container",
                    "chatgpt-export",
                    "native-container",
                    f"{container_outer}:conversations.json",
                    2,
                    bytes.fromhex("8" * 64),
                    123,
                ),
                (
                    "raw-size-mismatch",
                    "claude-ai-export",
                    "native-size",
                    str(changed_source),
                    0,
                    bytes.fromhex("9" * 64),
                    changed_source.stat().st_size + 10,
                ),
                (
                    "raw-missing",
                    "claude-ai-export",
                    "native-missing",
                    str(tmp_path / "missing.json"),
                    0,
                    bytes.fromhex("a" * 64),
                    456,
                ),
            ],
        )

    report = plan_raw_backed_blob_reference_recovery(
        source_db,
        store=store,
        manifest_path=manifest_path,
        include_rows=True,
    )

    assert report.missing_raw_backed_blobs == 4
    assert report.by_action == {
        "container_member_reacquire_required": 1,
        "direct_exact_restore_candidate": 1,
        "direct_source_size_mismatch": 1,
        "source_missing": 1,
    }
    assert report.by_origin == {"chatgpt-export": 2, "claude-ai-export": 2}
    assert report.by_source_shape == {"container_member": 1, "direct_file": 3}
    assert {row.raw_id for row in report.rows} == {
        "raw-exact",
        "raw-container",
        "raw-size-mismatch",
        "raw-missing",
    }
    manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert {row["raw_id"] for row in manifest_rows} == {
        "raw-exact",
        "raw-container",
        "raw-size-mismatch",
        "raw-missing",
    }


def test_scan_blob_integrity_uses_sibling_archive_source_from_index_db(tmp_path: Path) -> None:
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    store = BlobStore(tmp_path / "blob")
    raw_hash, raw_size = store.write_from_bytes(b"raw payload")
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
    with sqlite3.connect(source_db) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                blob_hash BLOB NOT NULL,
                blob_size INTEGER NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, blob_hash, blob_size) VALUES (?, ?, ?)",
            ("raw-v1", bytes.fromhex(raw_hash), raw_size),
        )

    report = scan_blob_integrity(index_db, store=store, full=True)

    assert report.ok is True
    assert report.total_references_seen == 1
    assert report.scanned_references == 1

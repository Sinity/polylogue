"""Focused proofs for source-tier blob-ref liveness reconciliation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.maintenance.blob_ref_liveness_reconciliation import (
    BlobRefLivenessReconciliationError,
    reconcile_blob_ref_liveness,
)
from polylogue.storage.blob_ref_liveness import classify_blob_ref_liveness
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _source_archive(tmp_path: Path) -> Path:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-live', 'codex-session', '/live.jsonl', 0, ?, 10, 1)
            """,
            (b"l" * 32,),
        )
        conn.execute(
            """
            INSERT INTO raw_hook_events (
                hook_event_id, origin, source_path, event_type, payload_json, observed_at_ms
            ) VALUES ('hook-live', 'codex-session', '/hook.jsonl', 'PostToolUse', '{}', 1)
            """
        )
        conn.execute(
            """
            INSERT INTO history_sidecars (
                sidecar_id, origin, source_path, payload_json, observed_at_ms, content_hash
            ) VALUES ('sidecar-live', 'codex-session', '/sidecar.json', '{}', 1, ?)
            """,
            (b"s" * 32,),
        )
        refs = (
            (b"1" * 32, "raw-live", "raw_payload"),
            (b"2" * 32, "raw-gone", "raw_payload"),
            (b"3" * 32, "raw-live", "attachment"),
            (b"4" * 32, "raw-gone", "attachment"),
            (b"5" * 32, "hook-live", "hook_payload"),
            (b"6" * 32, "hook-gone", "hook_payload"),
            (b"7" * 32, "sidecar-live", "sidecar"),
            (b"8" * 32, "sidecar-gone", "sidecar"),
        )
        conn.executemany(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, ?, '/fixture', 1, 1)
            """,
            refs,
        )
    return archive_root


def test_classifier_proves_each_source_ref_type_with_actual_referent_join(tmp_path: Path) -> None:
    archive_root = _source_archive(tmp_path)
    with sqlite3.connect(archive_root / "source.db") as conn:
        classification = classify_blob_ref_liveness(conn)

    assert classification.scanned_count == 8
    assert classification.orphaned_by_ref_type == {
        "attachment": 1,
        "hook_payload": 1,
        "raw_payload": 1,
        "sidecar": 1,
    }
    assert {
        (candidate.ref_type, candidate.referent_table, candidate.referent_column)
        for candidate in classification.candidates
    } == {
        ("raw_payload", "raw_sessions", "raw_id"),
        ("attachment", "raw_sessions", "raw_id"),
        ("hook_payload", "raw_hook_events", "hook_event_id"),
        ("sidecar", "history_sidecars", "sidecar_id"),
    }
    assert classification.safe_to_apply is True


def test_dry_run_is_read_only_and_reports_attachment_parent_join(tmp_path: Path) -> None:
    archive_root = _source_archive(tmp_path)
    before = sqlite3.connect(archive_root / "source.db").execute("SELECT COUNT(*) FROM blob_refs").fetchone()[0]

    report = reconcile_blob_ref_liveness(archive_root)

    assert report.dry_run is True
    assert report.applied is False
    assert report.deleted_count == 0
    assert report.classification.orphaned_by_ref_type["attachment"] == 1
    after = sqlite3.connect(archive_root / "source.db").execute("SELECT COUNT(*) FROM blob_refs").fetchone()[0]
    assert after == before


def test_apply_requires_backup_and_receipt_before_mutation(tmp_path: Path) -> None:
    archive_root = _source_archive(tmp_path)
    with pytest.raises(BlobRefLivenessReconciliationError, match="backup manifest"):
        reconcile_blob_ref_liveness(archive_root, dry_run=False, receipt_path=tmp_path / "receipt.jsonl")
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_refs").fetchone() == (8,)


def test_apply_deletes_only_join_proven_orphans_and_persists_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = _source_archive(tmp_path)
    manifest = tmp_path / "backup" / "source.json"
    receipt = tmp_path / "receipts" / "liveness.jsonl"

    def fake_validate(path: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        assert connection.execute("SELECT 1").fetchone() == (1,)
        return path

    monkeypatch.setattr(
        "polylogue.maintenance.blob_ref_liveness_reconciliation.validate_migration_backup_manifest",
        fake_validate,
    )
    monkeypatch.setattr(
        "polylogue.maintenance.blob_ref_liveness_reconciliation.offline_maintenance_block_reason",
        lambda *args, **kwargs: None,
    )

    report = reconcile_blob_ref_liveness(
        archive_root,
        backup_manifest=manifest,
        receipt_path=receipt,
        dry_run=False,
    )

    assert report.applied is True
    assert report.deleted_count == 4
    with sqlite3.connect(archive_root / "source.db") as conn:
        remaining = conn.execute("SELECT ref_type, ref_id FROM blob_refs ORDER BY ref_type, ref_id").fetchall()
    assert remaining == [
        ("attachment", "raw-live"),
        ("hook_payload", "hook-live"),
        ("raw_payload", "raw-live"),
        ("sidecar", "sidecar-live"),
    ]
    receipt_rows = [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines()]
    assert receipt_rows[0]["phase"] == "prepared"
    assert receipt_rows[0]["candidate_count"] == 4
    assert receipt_rows[0]["candidate_digest"]
    assert receipt_rows[-1]["phase"] == "committed"
    assert receipt_rows[-1]["deleted_count"] == 4


def test_unknown_ref_type_blocks_apply_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_db = tmp_path / "source.db"
    with sqlite3.connect(source_db) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY);
            CREATE TABLE blob_refs (
                blob_hash BLOB NOT NULL, ref_id TEXT NOT NULL, ref_type TEXT NOT NULL,
                source_path TEXT, size_bytes INTEGER NOT NULL, acquired_at_ms INTEGER NOT NULL,
                PRIMARY KEY (blob_hash, ref_type, ref_id)
            );
            INSERT INTO blob_refs VALUES (X'1111111111111111111111111111111111111111111111111111111111111111', 'gone', 'future_type', '/future', 1, 1);
            """
        )
    archive_root = tmp_path
    with sqlite3.connect(source_db) as conn:
        classification = classify_blob_ref_liveness(conn)
    assert classification.unknown_ref_types == ("future_type",)
    assert classification.safe_to_apply is False

    monkeypatch.setattr(
        "polylogue.maintenance.blob_ref_liveness_reconciliation.validate_migration_backup_manifest",
        lambda *args, **kwargs: args[0],
    )
    monkeypatch.setattr(
        "polylogue.maintenance.blob_ref_liveness_reconciliation.offline_maintenance_block_reason",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(BlobRefLivenessReconciliationError, match="cannot be proven"):
        reconcile_blob_ref_liveness(
            archive_root,
            backup_manifest=tmp_path / "backup.json",
            receipt_path=tmp_path / "blocked.jsonl",
            dry_run=False,
        )
    assert not (tmp_path / "blocked.jsonl").exists()

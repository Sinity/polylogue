"""Focused proofs for source-tier blob-ref liveness reconciliation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import polylogue.maintenance.blob_ref_liveness_reconciliation as liveness_reconciliation
import polylogue.storage.hook_payload_ref_reconciliation as hook_payload_ref_reconciliation
from polylogue.daemon.backup import backup_archive
from polylogue.maintenance.blob_ref_liveness_reconciliation import (
    BlobRefLivenessReconciliationError,
    reconcile_blob_ref_liveness,
)
from polylogue.storage.blob_ref_liveness import (
    BlobRefLivenessStagedPlan,
    classify_blob_ref_liveness,
    stage_blob_ref_liveness,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import deterministic_blob_hash, deterministic_raw_session_id
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import (
    MigrationError,
    validate_migration_backup_live_fingerprint,
    validate_migration_backup_manifest,
)


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


def _real_source_archive(archive_root: Path) -> Path:
    initialize_active_archive_root(archive_root)
    blob_store = BlobStore(archive_root / "blob")
    live_payload = b"live-payload"
    gone_payloads = (b"gone-payload-1", b"gone-payload-2")
    live_hash = deterministic_blob_hash(live_payload)
    gone_hashes = tuple(deterministic_blob_hash(payload) for payload in gone_payloads)
    blob_store.write_from_bytes(live_payload)
    for payload in gone_payloads:
        blob_store.write_from_bytes(payload)
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms)
            VALUES ('raw-live', 'codex-session', '/live.jsonl', 0, ?, 1, 1)
            """,
            (live_hash,),
        )
        conn.executemany(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, 'raw_payload', ?, 1, 1)
            """,
            ((gone_hashes[index - 1], f"raw-gone-{index}", f"/gone-{index}.jsonl") for index in range(1, 3)),
        )
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'raw-live', 'raw_payload', '/live.jsonl', 1, 1)
            """,
            (live_hash,),
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


def test_legacy_hook_payload_ref_is_rekeyable_not_a_delete_candidate(tmp_path: Path) -> None:
    archive_root = _source_archive(tmp_path)
    blob_hash = b"h" * 32
    source_path = "/legacy-hook.jsonl"
    native_id = "tool-call-1"
    legacy_ref_id = deterministic_raw_session_id("codex-session", source_path, 0, blob_hash, native_id)
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_hook_events (
                hook_event_id, origin, native_id, source_path, event_type, payload_json, observed_at_ms
            ) VALUES ('hook-legacy', 'codex-session', ?, ?, 'PostToolUse', '{}', 1)
            """,
            (native_id, source_path),
        )
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, 'raw_payload', ?, 1, 1)
            """,
            (blob_hash, legacy_ref_id, source_path),
        )
        classification = classify_blob_ref_liveness(conn)

    assert classification.rekeyable_hook_payload_count == 1
    assert classification.safe_to_apply is False
    assert all(candidate.ref_id != legacy_ref_id for candidate in classification.candidates)


def test_apply_requires_backup_and_receipt_before_mutation(tmp_path: Path) -> None:
    archive_root = _source_archive(tmp_path)
    with pytest.raises(BlobRefLivenessReconciliationError, match="backup manifest"):
        reconcile_blob_ref_liveness(archive_root, dry_run=False, receipt_path=tmp_path / "receipt.jsonl")
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_refs").fetchone() == (8,)


def test_apply_refuses_a_running_daemon_before_receipt_or_blob_ref_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = _source_archive(tmp_path)
    receipt = tmp_path / "receipts" / "liveness.jsonl"
    monkeypatch.setattr(
        "polylogue.maintenance.blob_ref_liveness_reconciliation.running_daemon_pid",
        lambda _config: 1234,
    )

    with pytest.raises(BlobRefLivenessReconciliationError, match="while polylogued PID 1234 is running"):
        reconcile_blob_ref_liveness(
            archive_root,
            backup_manifest=tmp_path / "backup.json",
            receipt_path=receipt,
            dry_run=False,
        )

    assert not receipt.exists()
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_refs").fetchone() == (8,)


def test_restart_recovers_prepared_receipt_after_committed_delete(tmp_path: Path) -> None:
    archive_root = _source_archive(tmp_path)
    receipt = tmp_path / "receipts" / "interrupted.jsonl"
    with sqlite3.connect(archive_root / "source.db") as conn:
        classification = classify_blob_ref_liveness(conn)
    liveness_reconciliation._write_prepared_receipt(
        receipt, archive_root / "source.db", classification, tmp_path / "backup.json"
    )
    with sqlite3.connect(archive_root / "source.db") as conn:
        for candidate in classification.candidates:
            conn.execute(
                "DELETE FROM blob_refs WHERE blob_hash = ? AND ref_type = ? AND ref_id = ?",
                (bytes.fromhex(candidate.blob_hash), candidate.ref_type, candidate.ref_id),
            )

    with pytest.raises(BlobRefLivenessReconciliationError, match="recovered_committed"):
        reconcile_blob_ref_liveness(
            archive_root,
            backup_manifest=tmp_path / "backup.json",
            receipt_path=receipt,
            dry_run=False,
        )
    rows = [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["phase"] == "recovered_committed"


def test_restart_recovers_prepared_receipt_after_rollback(tmp_path: Path) -> None:
    archive_root = _source_archive(tmp_path)
    receipt = tmp_path / "receipts" / "rolled-back.jsonl"
    with sqlite3.connect(archive_root / "source.db") as conn:
        classification = classify_blob_ref_liveness(conn)
    liveness_reconciliation._write_prepared_receipt(
        receipt, archive_root / "source.db", classification, tmp_path / "backup.json"
    )

    with pytest.raises(BlobRefLivenessReconciliationError, match="recovered_rolled_back"):
        reconcile_blob_ref_liveness(
            archive_root,
            backup_manifest=tmp_path / "backup.json",
            receipt_path=receipt,
            dry_run=False,
        )
    rows = [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["phase"] == "recovered_rolled_back"


def test_restart_recovers_partial_prepared_plan_with_exact_counts(tmp_path: Path) -> None:
    archive_root = _source_archive(tmp_path)
    receipt = tmp_path / "receipts" / "partial.jsonl"
    with sqlite3.connect(archive_root / "source.db") as conn:
        classification = classify_blob_ref_liveness(conn)
    liveness_reconciliation._write_prepared_receipt(
        receipt, archive_root / "source.db", classification, tmp_path / "backup.json"
    )
    candidate = classification.candidates[0]
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            "DELETE FROM blob_refs WHERE blob_hash = ? AND ref_type = ? AND ref_id = ?",
            (bytes.fromhex(candidate.blob_hash), candidate.ref_type, candidate.ref_id),
        )

    with pytest.raises(BlobRefLivenessReconciliationError, match="recovered_partial"):
        reconcile_blob_ref_liveness(
            archive_root,
            backup_manifest=tmp_path / "backup.json",
            receipt_path=receipt,
            dry_run=False,
        )
    rows = [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["phase"] == "recovered_partial"
    assert rows[-1]["deleted_count"] == 1
    with pytest.raises(BlobRefLivenessReconciliationError, match="already terminal"):
        reconcile_blob_ref_liveness(
            archive_root,
            backup_manifest=tmp_path / "backup.json",
            receipt_path=receipt,
            dry_run=False,
        )


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
        "polylogue.maintenance.blob_ref_liveness_reconciliation.validate_migration_backup_live_fingerprint",
        fake_validate,
    )
    monkeypatch.setattr(
        "polylogue.maintenance.blob_ref_liveness_reconciliation.running_daemon_pid",
        lambda _config: None,
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


def test_real_backup_tamper_between_validations_refuses_before_delete(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = _real_source_archive(workspace_env["archive_root"])
    backup_result = backup_archive(output_dir=tmp_path / "backups", profile="rebuildable_cache_exclude", verify=True)
    assert backup_result.ok and backup_result.output_path is not None, backup_result.error
    manifest = Path(backup_result.output_path) / "manifest.json"
    receipt = tmp_path / "receipts" / "tampered.jsonl"
    calls: list[str] = []
    real_pre_validate = validate_migration_backup_manifest
    real_live_validate = validate_migration_backup_live_fingerprint
    real_stage = stage_blob_ref_liveness

    def record_pre(path: Path, tier: ArchiveTier, *, connection: sqlite3.Connection) -> Path:
        calls.append("pre")
        return real_pre_validate(path, tier, connection=connection)

    def record_live(path: Path, tier: ArchiveTier, *, connection: sqlite3.Connection) -> Path:
        calls.append("live")
        return real_live_validate(path, tier, connection=connection)

    def stage_then_tamper(conn: sqlite3.Connection) -> BlobRefLivenessStagedPlan:
        staged = real_stage(conn)
        with (manifest.parent / "source.db").open("ab") as handle:
            handle.write(b"tampered-after-precheck")
        return staged

    monkeypatch.setattr(liveness_reconciliation, "validate_migration_backup_manifest", record_pre)
    monkeypatch.setattr(liveness_reconciliation, "validate_migration_backup_live_fingerprint", record_live)
    monkeypatch.setattr(liveness_reconciliation, "stage_blob_ref_liveness", stage_then_tamper)

    with pytest.raises(MigrationError, match="tier artifact size mismatch"):
        reconcile_blob_ref_liveness(
            archive_root,
            backup_manifest=manifest,
            receipt_path=receipt,
            dry_run=False,
        )

    assert calls == ["pre", "live"]
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_refs").fetchone() == (3,)
    rows = [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["phase"] == "prepared"
    assert all(row.get("phase") != "committed" for row in rows)


def test_real_prepared_failure_rolls_back_and_recovery_records_exact_state(
    workspace_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = _real_source_archive(workspace_env["archive_root"])
    backup_result = backup_archive(output_dir=tmp_path / "backups", profile="rebuildable_cache_exclude", verify=True)
    assert backup_result.ok and backup_result.output_path is not None, backup_result.error
    manifest = Path(backup_result.output_path) / "manifest.json"
    receipt = tmp_path / "receipts" / "failed.jsonl"
    real_delete = liveness_reconciliation._delete_candidate_batch

    def delete_then_fail(conn: sqlite3.Connection, candidate_table: str, batch_table: str) -> int:
        deleted = real_delete(conn, candidate_table, batch_table)
        raise RuntimeError(f"injected after deleting {deleted} candidate(s) before commit")

    monkeypatch.setattr(liveness_reconciliation, "_delete_candidate_batch", delete_then_fail)
    with pytest.raises(RuntimeError, match="before commit"):
        reconcile_blob_ref_liveness(
            archive_root,
            backup_manifest=manifest,
            receipt_path=receipt,
            dry_run=False,
        )

    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_refs").fetchone() == (3,)
    prepared_rows = [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines()]
    assert prepared_rows[0]["phase"] == "prepared"
    assert prepared_rows[-1].get("phase") != "committed"

    with pytest.raises(BlobRefLivenessReconciliationError, match="recovered_rolled_back"):
        reconcile_blob_ref_liveness(
            archive_root,
            backup_manifest=manifest,
            receipt_path=receipt,
            dry_run=False,
        )
    recovered_rows = [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines()]
    assert recovered_rows[-1]["phase"] == "recovered_rolled_back"
    assert recovered_rows[-1]["deleted_count"] == 0


def test_apply_rejects_post_staging_duplicate_known_hash_hooks_before_deleting_any_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    orphan_count = liveness_reconciliation.BATCH_SIZE + 1
    source_path = "/hooks/post-staging.jsonl"
    duplicate_index = 0
    duplicate_hash = duplicate_index.to_bytes(4, "big") + b"o" * 28
    duplicate_native_id = "post-staging-native"
    duplicate_ref_id = deterministic_raw_session_id(
        "codex-session", source_path, 0, duplicate_hash, duplicate_native_id
    )
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.executemany(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, 'raw_payload', ?, 1, 1)
            """,
            (
                (
                    index.to_bytes(4, "big") + b"o" * 28,
                    duplicate_ref_id if index == duplicate_index else f"raw-orphan-{index}",
                    source_path,
                )
                for index in range(orphan_count)
            ),
        )

    real_stage = stage_blob_ref_liveness

    def stage_then_insert_hook(conn: sqlite3.Connection) -> BlobRefLivenessStagedPlan:
        staged = real_stage(conn)
        conn.executemany(
            """
            INSERT INTO raw_hook_events (
                hook_event_id, origin, native_id, source_path, event_type, payload_json, observed_at_ms, blob_hash
            ) VALUES (?, 'codex-session', ?, ?, 'PostToolUse', '{}', 1, ?)
            """,
            (
                ("post-staging-hook-1", duplicate_native_id, source_path, duplicate_hash),
                ("post-staging-hook-2", duplicate_native_id, source_path, duplicate_hash),
            ),
        )
        return staged

    def fake_validate(path: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        assert connection.execute("SELECT 1").fetchone() == (1,)
        return path

    monkeypatch.setattr(liveness_reconciliation, "stage_blob_ref_liveness", stage_then_insert_hook)
    monkeypatch.setattr(liveness_reconciliation, "validate_migration_backup_manifest", fake_validate)
    monkeypatch.setattr(liveness_reconciliation, "validate_migration_backup_live_fingerprint", fake_validate)
    monkeypatch.setattr(liveness_reconciliation, "running_daemon_pid", lambda _config: None)
    receipt = tmp_path / "receipts" / "post-staging-hook.jsonl"

    with pytest.raises(BlobRefLivenessReconciliationError, match="duplicate known-hash hook evidence"):
        reconcile_blob_ref_liveness(
            archive_root,
            backup_manifest=tmp_path / "backup" / "manifest.json",
            receipt_path=receipt,
            dry_run=False,
        )

    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_refs").fetchone() == (orphan_count,)
        assert conn.execute("SELECT COUNT(*) FROM raw_hook_events").fetchone() == (2,)


def test_apply_fails_closed_if_source_changes_between_bounded_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    orphan_count = liveness_reconciliation.BATCH_SIZE + 1
    source_path = "/hooks/inter-batch.jsonl"
    target_index = orphan_count - 1
    target_hash = target_index.to_bytes(4, "big") + b"o" * 28
    target_native_id = "inter-batch-native"
    target_ref_id = deterministic_raw_session_id("codex-session", source_path, 0, target_hash, target_native_id)
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.executemany(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, 'raw_payload', ?, 1, 1)
            """,
            (
                (
                    index.to_bytes(4, "big") + b"o" * 28,
                    target_ref_id if index == target_index else f"!orphan-{index:04d}",
                    source_path,
                )
                for index in range(orphan_count)
            ),
        )

    def fake_validate(path: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        assert connection.execute("SELECT 1").fetchone() == (1,)
        return path

    monkeypatch.setattr(liveness_reconciliation, "validate_migration_backup_manifest", fake_validate)
    monkeypatch.setattr(liveness_reconciliation, "validate_migration_backup_live_fingerprint", fake_validate)
    monkeypatch.setattr(liveness_reconciliation, "running_daemon_pid", lambda _config: None)
    real_append_footer = liveness_reconciliation._append_receipt_footer
    committed_batches = 0

    def append_footer_and_insert_hook(
        receipt_path: Path, *, phase: str, deleted_count: int | None = None, error: str | None = None
    ) -> None:
        nonlocal committed_batches
        real_append_footer(receipt_path, phase=phase, deleted_count=deleted_count, error=error)
        if phase == "batch_committed":
            committed_batches += 1
            if committed_batches == 1:
                with sqlite3.connect(archive_root / "source.db") as external_conn:
                    external_conn.executemany(
                        """
                        INSERT INTO raw_hook_events (
                            hook_event_id, origin, native_id, source_path, event_type, payload_json, observed_at_ms,
                            blob_hash
                        ) VALUES (?, 'codex-session', ?, ?, 'PostToolUse', '{}', 1, ?)
                        """,
                        (
                            ("inter-batch-hook-1", target_native_id, source_path, target_hash),
                            ("inter-batch-hook-2", target_native_id, source_path, target_hash),
                        ),
                    )

    monkeypatch.setattr(liveness_reconciliation, "_append_receipt_footer", append_footer_and_insert_hook)
    receipt = tmp_path / "receipts" / "inter-batch.jsonl"

    with pytest.raises(BlobRefLivenessReconciliationError, match="changed after locked hook snapshot"):
        reconcile_blob_ref_liveness(
            archive_root,
            backup_manifest=tmp_path / "backup" / "manifest.json",
            receipt_path=receipt,
            dry_run=False,
        )

    assert committed_batches == 1
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_refs").fetchone() == (1,)
        assert conn.execute(
            "SELECT COUNT(*) FROM blob_refs WHERE ref_type = 'raw_payload' AND ref_id = ?", (target_ref_id,)
        ).fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM raw_hook_events").fetchone() == (2,)


def test_locked_plan_rejects_referent_that_appears_after_staging(tmp_path: Path) -> None:
    archive_root = _source_archive(tmp_path)
    with sqlite3.connect(archive_root / "source.db") as conn:
        candidate = next(item for item in classify_blob_ref_liveness(conn).candidates if item.ref_type == "raw_payload")
        conn.execute(
            """
            CREATE TEMP TABLE locked_candidates (
                blob_hash BLOB NOT NULL, ref_type TEXT NOT NULL, ref_id TEXT NOT NULL,
                source_path TEXT, size_bytes INTEGER NOT NULL, acquired_at_ms INTEGER NOT NULL,
                referent_table TEXT NOT NULL, referent_column TEXT NOT NULL,
                PRIMARY KEY (blob_hash, ref_type, ref_id)
            ) STRICT
            """
        )
        conn.execute(
            """
            INSERT INTO locked_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bytes.fromhex(candidate.blob_hash),
                candidate.ref_type,
                candidate.ref_id,
                candidate.source_path,
                candidate.size_bytes,
                candidate.acquired_at_ms,
                candidate.referent_table,
                candidate.referent_column,
            ),
        )
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, 'codex-session', '/appeared.jsonl', 0, ?, 1, 1)
            """,
            (candidate.ref_id, bytes.fromhex(candidate.blob_hash)),
        )
        with pytest.raises(BlobRefLivenessReconciliationError, match="referents became live"):
            liveness_reconciliation._validate_locked_candidate_plan(conn, "locked_candidates", 1)


def test_scale_apply_streams_distinct_hook_paths_and_preserves_exact_survivors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    orphan_count = 10_000
    irrelevant_hook_count = orphan_count * 5
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
                hook_event_id, origin, native_id, source_path, event_type, payload_json, observed_at_ms
            ) VALUES ('hook-live', 'codex-session', 'native-live', '/hooks/live.jsonl', 'PostToolUse', '{}', 1)
            """
        )
        conn.executemany(
            """
            INSERT INTO raw_hook_events (
                hook_event_id, origin, native_id, source_path, event_type, payload_json, observed_at_ms
            ) VALUES (?, 'codex-session', ?, ?, 'PostToolUse', '{}', 1)
            """,
            ((f"hook-{index}", f"native-{index}", f"/hooks/{index}.jsonl") for index in range(orphan_count)),
        )
        conn.executemany(
            """
            INSERT INTO raw_hook_events (
                hook_event_id, origin, native_id, source_path, event_type, payload_json, observed_at_ms
            ) VALUES (?, 'codex-session', ?, ?, 'PostToolUse', '{}', 1)
            """,
            (
                (f"irrelevant-hook-{index}", f"irrelevant-native-{index}", f"/unrelated/{index}.jsonl")
                for index in range(irrelevant_hook_count)
            ),
        )
        conn.executemany(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, 'raw_payload', ?, 1, 1)
            """,
            (
                (index.to_bytes(4, "big") + b"o" * 28, f"raw-gone-{index}", f"/hooks/{index}.jsonl")
                for index in range(orphan_count)
            ),
        )
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'raw-live', 'raw_payload', '/live.jsonl', 10, 1)
            """,
            (b"l" * 32,),
        )
        assert conn.execute(
            "SELECT 1 FROM pragma_index_list('raw_hook_events') WHERE name = 'idx_raw_hook_events_source_hash'"
        ).fetchone() == (1,)
        query_plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT 1 FROM raw_hook_events WHERE source_path = ? AND blob_hash = ?",
            ("/hooks/0.jsonl", b"o" * 32),
        ).fetchall()
        assert any("idx_raw_hook_events_source_hash" in str(row[3]) for row in query_plan)
        conn.commit()

    calls = 0
    snapshot_calls = 0
    original_id = hook_payload_ref_reconciliation._deterministic_raw_session_id_udf

    def count_udf(*args: object) -> str | None:
        nonlocal calls
        calls += 1
        return original_id(*args)

    monkeypatch.setattr(
        "polylogue.storage.hook_payload_ref_reconciliation._deterministic_raw_session_id_udf", count_udf
    )
    monkeypatch.setattr(liveness_reconciliation, "_deterministic_raw_session_id_udf", count_udf)

    real_snapshot = liveness_reconciliation._stage_locked_hook_snapshot

    def count_snapshot(conn: sqlite3.Connection, candidate_table: str) -> str:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return real_snapshot(conn, candidate_table)

    monkeypatch.setattr(liveness_reconciliation, "_stage_locked_hook_snapshot", count_snapshot)

    def fake_validate(path: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        assert connection.execute("SELECT 1").fetchone() == (1,)
        return path

    monkeypatch.setattr(
        "polylogue.maintenance.blob_ref_liveness_reconciliation.validate_migration_backup_manifest",
        fake_validate,
    )
    monkeypatch.setattr(
        "polylogue.maintenance.blob_ref_liveness_reconciliation.validate_migration_backup_live_fingerprint",
        fake_validate,
    )
    monkeypatch.setattr(
        "polylogue.maintenance.blob_ref_liveness_reconciliation.running_daemon_pid", lambda _config: None
    )
    batches = 0
    real_delete = liveness_reconciliation._delete_candidate_batch

    def count_batch_delete(conn: sqlite3.Connection, candidate_table: str, batch_table: str) -> int:
        nonlocal batches
        batches += 1
        return real_delete(conn, candidate_table, batch_table)

    monkeypatch.setattr(liveness_reconciliation, "_delete_candidate_batch", count_batch_delete)
    receipt = tmp_path / "receipts" / "scale.jsonl"
    report = reconcile_blob_ref_liveness(
        archive_root,
        backup_manifest=tmp_path / "backup" / "manifest.json",
        receipt_path=receipt,
        dry_run=False,
    )

    assert report.deleted_count == orphan_count
    assert report.classification.orphaned_count == orphan_count
    # Initial staging, one whole-plan ownership validation, and subsequent
    # batch validations are all bounded by candidates, not all hook rows.
    assert calls <= orphan_count * 3
    assert calls < irrelevant_hook_count
    assert snapshot_calls == 1
    assert batches == orphan_count // liveness_reconciliation.BATCH_SIZE
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_refs").fetchone() == (1,)
        assert conn.execute("SELECT ref_type, ref_id FROM blob_refs").fetchone() == ("raw_payload", "raw-live")
    receipt_rows = [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines()]
    assert receipt_rows[0]["candidate_count"] == orphan_count
    assert sum(row.get("kind") == "candidate" for row in receipt_rows) == orphan_count
    assert receipt_rows[-1]["phase"] == "committed"
    assert receipt_rows[-1]["deleted_count"] == orphan_count


def test_failure_before_prepared_receipt_rolls_back_without_receipt_or_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = _source_archive(tmp_path)
    receipt = tmp_path / "receipts" / "injected.jsonl"

    def fake_validate(path: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        return path

    monkeypatch.setattr(
        "polylogue.maintenance.blob_ref_liveness_reconciliation.validate_migration_backup_manifest",
        fake_validate,
    )
    monkeypatch.setattr(
        "polylogue.maintenance.blob_ref_liveness_reconciliation.running_daemon_pid", lambda _config: None
    )
    monkeypatch.setattr(
        "polylogue.maintenance.blob_ref_liveness_reconciliation._write_prepared_receipt",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected before prepared receipt")),
    )

    with pytest.raises(RuntimeError, match="before prepared receipt"):
        reconcile_blob_ref_liveness(
            archive_root,
            backup_manifest=tmp_path / "backup" / "manifest.json",
            receipt_path=receipt,
            dry_run=False,
        )
    assert not receipt.exists()
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_refs").fetchone() == (8,)


def test_shared_legacy_hook_path_fails_closed_without_cross_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = _source_archive(tmp_path)
    row_count = 10_000
    source_path = "/hooks/shared-legacy.jsonl"
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.executemany(
            """
            INSERT INTO raw_hook_events (
                hook_event_id, origin, native_id, source_path, event_type, payload_json, observed_at_ms
            ) VALUES (?, 'codex-session', ?, ?, 'PostToolUse', '{}', 1)
            """,
            ((f"shared-hook-{index}", f"native-{index}", source_path) for index in range(row_count)),
        )
        conn.executemany(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, 'raw_payload', ?, 1, 1)
            """,
            (
                (index.to_bytes(4, "big") + b"s" * 28, f"shared-orphan-{index}", source_path)
                for index in range(row_count)
            ),
        )
        conn.commit()

    calls = 0
    original_id = hook_payload_ref_reconciliation._deterministic_raw_session_id_udf

    def count_udf(*args: object) -> str | None:
        nonlocal calls
        calls += 1
        return original_id(*args)

    monkeypatch.setattr(
        "polylogue.storage.hook_payload_ref_reconciliation._deterministic_raw_session_id_udf", count_udf
    )
    monkeypatch.setattr(liveness_reconciliation, "_deterministic_raw_session_id_udf", count_udf)
    with sqlite3.connect(archive_root / "source.db") as conn:
        classification = classify_blob_ref_liveness(conn)

    assert classification.orphaned_count == 4
    assert classification.rekeyable_hook_payload_count == row_count
    assert sum(candidate.source_path == source_path for candidate in classification.candidates) == 0
    assert calls == 0


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
        "polylogue.maintenance.blob_ref_liveness_reconciliation.running_daemon_pid",
        lambda _config: None,
    )
    with pytest.raises(BlobRefLivenessReconciliationError, match="cannot be proven"):
        reconcile_blob_ref_liveness(
            archive_root,
            backup_manifest=tmp_path / "backup.json",
            receipt_path=tmp_path / "blocked.jsonl",
            dry_run=False,
        )
    assert not (tmp_path / "blocked.jsonl").exists()

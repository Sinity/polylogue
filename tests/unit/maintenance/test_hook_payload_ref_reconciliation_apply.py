"""polylogue-tfzw0: apply the provable subset of orphaned hook-payload refs.

Builds the exact pre-v22 orphan shape (see
``tests/unit/storage/test_hook_payload_ref_reconciliation.py``) and proves:

* dry-run (the default) never mutates anything;
* --apply re-keys the confirmed match: deletes the stale 'raw_payload' ref,
  inserts the corrected 'hook_payload' ref keyed by the hook event's own id,
  and backfills ``raw_hook_events.blob_hash``;
* an unmatched orphan (no candidate hook event) is left untouched;
* applying without a backup manifest is refused before anything is touched.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.maintenance.hook_payload_ref_reconciliation_apply import (
    HookPayloadRefReconciliationApplyError,
    apply_hook_payload_ref_reconciliation,
)
from polylogue.maintenance.hook_payload_ref_reconciliation_receipt import (
    backup_manifest_identity,
    write_prepared_receipt,
)
from polylogue.storage.hook_payload_ref_reconciliation import plan_hook_payload_ref_reconciliation
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import (
    deterministic_blob_hash,
    deterministic_raw_session_id,
)


def _build_fixture_archive(tmp_path: Path) -> Path:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    matched_payload = b'{"event":"PostToolUse","n":1}'
    matched_blob_hash = deterministic_blob_hash(matched_payload)
    matched_source_path = "/hooks/matched.jsonl"
    matched_native_id = "native-matched"
    matched_ref_id = deterministic_raw_session_id(
        "codex-session", matched_source_path, 0, matched_blob_hash, matched_native_id
    )

    unmatched_blob_hash = deterministic_blob_hash(b"no-candidate-hook-event")

    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_hook_events (
                hook_event_id, origin, native_id, session_native_id, source_path, event_type,
                payload_json, observed_at_ms
            ) VALUES ('hook-matched', 'codex-session', ?, 'session-1', ?, 'PostToolUse', '{}', 1)
            """,
            (matched_native_id, matched_source_path),
        )
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, 'raw_payload', ?, ?, 1)
            """,
            (matched_blob_hash, matched_ref_id, matched_source_path, len(matched_payload)),
        )
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'raw-no-candidate', 'raw_payload', '/hooks/unmatched.jsonl', 7, 1)
            """,
            (unmatched_blob_hash,),
        )
        conn.commit()

    return archive_root


def _blob_refs_rows(archive_root: Path) -> set[tuple[str, str]]:
    with sqlite3.connect(archive_root / "source.db") as conn:
        return {(row[0], row[1]) for row in conn.execute("SELECT ref_type, ref_id FROM blob_refs")}


def _manifest(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"manifest":"test"}', encoding="utf-8")
    return path


def _accept_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_validate(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        assert connection.execute("SELECT 1").fetchone() == (1,)
        return manifest.with_name("verification-receipt.json")

    monkeypatch.setattr(
        "polylogue.maintenance.hook_payload_ref_reconciliation_apply.validate_migration_backup_manifest",
        _fake_validate,
    )


def test_dry_run_never_mutates(tmp_path: Path) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before = _blob_refs_rows(archive_root)

    report = apply_hook_payload_ref_reconciliation(archive_root, dry_run=True)

    assert report.applied is False
    assert report.scanned_count == 2
    assert report.matched_count == 1
    assert report.unmatched_count == 1
    assert report.reconciled_hook_event_ids == ("hook-matched",)
    assert _blob_refs_rows(archive_root) == before


def test_apply_rekeys_only_the_confirmed_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    _accept_manifest(monkeypatch)
    manifest = _manifest(tmp_path / "verified-backup" / "manifest.json")
    receipt_path = tmp_path / "receipts" / "apply.jsonl"
    report = apply_hook_payload_ref_reconciliation(
        archive_root,
        backup_manifest=manifest,
        receipt_path=receipt_path,
        dry_run=False,
    )

    assert report.applied is True
    assert report.matched_count == 1
    assert report.reconciled_hook_event_ids == ("hook-matched",)
    assert report.receipt_path == receipt_path
    assert report.post_classification == {
        "scanned_count": 1,
        "matched_count": 0,
        "matched_bytes": 0,
        "unmatched_count": 1,
    }

    rows = _blob_refs_rows(archive_root)
    assert ("hook_payload", "hook-matched") in rows
    assert ("raw_payload", "raw-no-candidate") in rows  # unmatched orphan untouched
    assert not any(ref_type == "raw_payload" and ref_id != "raw-no-candidate" for ref_type, ref_id in rows)

    with sqlite3.connect(archive_root / "source.db") as conn:
        blob_hash = conn.execute(
            "SELECT blob_hash FROM raw_hook_events WHERE hook_event_id = 'hook-matched'"
        ).fetchone()[0]
        assert blob_hash is not None

    receipt = [json.loads(line) for line in receipt_path.read_text(encoding="utf-8").splitlines()]
    assert receipt[0]["phase"] == "prepared"
    assert receipt[0]["tool_version"] == "hook-payload-ref-reconciliation-apply-v1"
    assert receipt[0]["backup_manifest"]["path"] == str(manifest.resolve())
    assert receipt[0]["pre_classification"] == {
        "scanned_count": 2,
        "matched_count": 1,
        "matched_bytes": len(b'{"event":"PostToolUse","n":1}'),
        "unmatched_count": 1,
    }
    assert receipt[-1]["terminal_state"] == "committed"
    assert receipt[-1]["post_classification"] == report.post_classification
    assert receipt[-1]["reconciled_hook_event_ids"] == ["hook-matched"]
    assert len(receipt[-1]["reconciled_ids_digest"]) == 64


def test_apply_leaves_duplicate_identity_with_existing_canonical_ref_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A canonical duplicate must remain evidence, not disappear from matching."""

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    payload = b'{"event":"PostToolUse","duplicate":true}'
    blob_hash = deterministic_blob_hash(payload)
    source_path = "/hooks/duplicate-canonical.jsonl"
    native_id = "duplicate-native"
    orphan_ref_id = deterministic_raw_session_id("codex-session", source_path, 0, blob_hash, native_id)

    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.executemany(
            """
            INSERT INTO raw_hook_events (
                hook_event_id, origin, native_id, session_native_id, source_path, event_type,
                payload_json, observed_at_ms, blob_hash
            ) VALUES (?, 'codex-session', ?, 'session-1', ?, 'PostToolUse', '{}', 1, ?)
            """,
            (
                ("hook-canonical", native_id, source_path, blob_hash),
                ("hook-missing", native_id, source_path, None),
            ),
        )
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'hook-canonical', 'hook_payload', ?, ?, 1)
            """,
            (blob_hash, source_path, len(payload)),
        )
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, 'raw_payload', ?, ?, 1)
            """,
            (blob_hash, orphan_ref_id, source_path, len(payload)),
        )
        conn.commit()

    _accept_manifest(monkeypatch)
    receipt_path = tmp_path / "receipts" / "duplicate-canonical.jsonl"
    manifest = _manifest(tmp_path / "verified-backup" / "manifest.json")
    report = apply_hook_payload_ref_reconciliation(
        archive_root,
        backup_manifest=manifest,
        receipt_path=receipt_path,
        dry_run=False,
    )

    assert report.applied is True
    assert report.matched_count == 0
    assert report.unmatched_count == 1
    assert report.reconciled_hook_event_ids == ()
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute(
            "SELECT ref_type, ref_id FROM blob_refs WHERE blob_hash = ? ORDER BY ref_type, ref_id",
            (blob_hash,),
        ).fetchall() == [
            ("hook_payload", "hook-canonical"),
            ("raw_payload", orphan_ref_id),
        ]
        assert conn.execute(
            "SELECT hook_event_id, blob_hash FROM raw_hook_events WHERE source_path = ? ORDER BY hook_event_id",
            (source_path,),
        ).fetchall() == [("hook-canonical", blob_hash), ("hook-missing", None)]


def test_apply_refuses_without_backup_manifest(tmp_path: Path) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before = _blob_refs_rows(archive_root)

    with pytest.raises(HookPayloadRefReconciliationApplyError, match="backup manifest"):
        apply_hook_payload_ref_reconciliation(
            archive_root,
            backup_manifest=None,
            receipt_path=tmp_path / "receipt.jsonl",
            dry_run=False,
        )

    assert _blob_refs_rows(archive_root) == before


def test_apply_refuses_without_explicit_receipt_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before = _blob_refs_rows(archive_root)
    _accept_manifest(monkeypatch)

    with pytest.raises(HookPayloadRefReconciliationApplyError, match="receipt output"):
        apply_hook_payload_ref_reconciliation(
            archive_root,
            backup_manifest=_manifest(tmp_path / "verified-backup" / "manifest.json"),
            dry_run=False,
        )

    assert _blob_refs_rows(archive_root) == before


def test_apply_refuses_when_manifest_validation_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before = _blob_refs_rows(archive_root)
    manifest = _manifest(tmp_path / "verified-backup" / "manifest.json")
    receipt_path = tmp_path / "receipts" / "apply.jsonl"

    def _reject_manifest(_manifest_path: Path, _tier: object, *, connection: sqlite3.Connection) -> Path:
        assert connection.execute("SELECT 1").fetchone() == (1,)
        raise ValueError("backup manifest does not match source.db")

    monkeypatch.setattr(
        "polylogue.maintenance.hook_payload_ref_reconciliation_apply.validate_migration_backup_manifest",
        _reject_manifest,
    )
    with pytest.raises(ValueError, match="does not match"):
        apply_hook_payload_ref_reconciliation(
            archive_root,
            backup_manifest=manifest,
            receipt_path=receipt_path,
            dry_run=False,
        )

    assert _blob_refs_rows(archive_root) == before
    assert not receipt_path.exists()


def test_apply_refuses_when_offline_guard_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before = _blob_refs_rows(archive_root)
    _accept_manifest(monkeypatch)
    monkeypatch.setattr(
        "polylogue.maintenance.hook_payload_ref_reconciliation_apply.offline_maintenance_block_reason",
        lambda *_args, **_kwargs: "source maintenance requires the daemon to be offline",
    )

    with pytest.raises(HookPayloadRefReconciliationApplyError, match="daemon to be offline"):
        apply_hook_payload_ref_reconciliation(
            archive_root,
            backup_manifest=_manifest(tmp_path / "verified-backup" / "manifest.json"),
            receipt_path=tmp_path / "receipts" / "apply.jsonl",
            dry_run=False,
        )

    assert _blob_refs_rows(archive_root) == before


def test_apply_leaves_same_path_non_exact_orphan_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    _accept_manifest(monkeypatch)
    wrong_blob_hash = deterministic_blob_hash(b"same source path, wrong deterministic identity")
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'not-the-exact-hook-id', 'raw_payload', '/hooks/matched.jsonl', 3, 1)
            """,
            (wrong_blob_hash,),
        )
        conn.commit()

    apply_hook_payload_ref_reconciliation(
        archive_root,
        backup_manifest=_manifest(tmp_path / "verified-backup" / "manifest.json"),
        receipt_path=tmp_path / "receipts" / "apply.jsonl",
        dry_run=False,
    )

    assert ("raw_payload", "not-the-exact-hook-id") in _blob_refs_rows(archive_root)


def test_existing_prepared_receipt_is_recovered_after_committed_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    _accept_manifest(monkeypatch)
    manifest = _manifest(tmp_path / "verified-backup" / "manifest.json")
    receipt_path = tmp_path / "receipts" / "prepared.jsonl"
    with sqlite3.connect(archive_root / "source.db") as conn:
        plan = plan_hook_payload_ref_reconciliation(conn)
        candidate = plan.matched[0]
        write_prepared_receipt(
            receipt_path,
            source_db=archive_root / "source.db",
            tool_version="hook-payload-ref-reconciliation-apply-v1",
            backup_manifest=backup_manifest_identity(manifest),
            plan=plan,
        )
        conn.execute(
            "DELETE FROM blob_refs WHERE blob_hash = ? AND ref_type = 'raw_payload' AND ref_id = ?",
            (candidate.blob_hash, candidate.orphaned_ref_id),
        )
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, 'hook_payload', ?, ?, ?)
            """,
            (
                candidate.blob_hash,
                candidate.hook_event_id,
                candidate.source_path,
                candidate.size_bytes,
                candidate.acquired_at_ms,
            ),
        )
        conn.execute(
            "UPDATE raw_hook_events SET blob_hash = ? WHERE hook_event_id = ?",
            (candidate.blob_hash, candidate.hook_event_id),
        )
        conn.commit()

    with pytest.raises(
        HookPayloadRefReconciliationApplyError, match="recovered existing prepared receipt as recovered_committed"
    ):
        apply_hook_payload_ref_reconciliation(
            archive_root,
            backup_manifest=manifest,
            receipt_path=receipt_path,
            dry_run=False,
        )

    receipt = [json.loads(line) for line in receipt_path.read_text(encoding="utf-8").splitlines()]
    assert receipt[-1]["terminal_state"] == "recovered_committed"
    assert receipt[-1]["post_classification"] == {
        "scanned_count": 1,
        "matched_count": 0,
        "matched_bytes": 0,
        "unmatched_count": 1,
    }


def test_apply_records_empty_reconciliation_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    _accept_manifest(monkeypatch)
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("DELETE FROM blob_refs WHERE ref_id = ?", ("raw-no-candidate",))
        conn.execute("DELETE FROM blob_refs WHERE ref_type = 'raw_payload' AND ref_id != ?", ("raw-no-candidate",))
        conn.commit()

    receipt_path = tmp_path / "receipts" / "empty.jsonl"
    report = apply_hook_payload_ref_reconciliation(
        archive_root,
        backup_manifest=_manifest(tmp_path / "verified-backup" / "manifest.json"),
        receipt_path=receipt_path,
        dry_run=False,
    )

    assert report.reconciled_hook_event_ids == ()
    receipt = [json.loads(line) for line in receipt_path.read_text(encoding="utf-8").splitlines()]
    assert receipt[-1]["terminal_state"] == "committed"
    assert receipt[-1]["reconciled_hook_event_ids"] == []
    assert len(receipt[-1]["reconciled_ids_digest"]) == 64

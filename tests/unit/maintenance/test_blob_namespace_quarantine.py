"""Contracts for offline invalid-namespace blob quarantine."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.maintenance.blob_namespace_quarantine import (
    BlobNamespaceQuarantineError,
    classify_blob_namespace_quarantine_recovery,
    quarantine_blob_namespace,
)
from polylogue.sources.source_parsing import iter_source_sessions_with_raw
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.unit.sources.test_parsers_local_agent import _write_hermes_state_db


def _archive(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    return root


def _verified_source_backup(archive_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    from polylogue.daemon.backup import backup_archive

    result = backup_archive(output_dir=tmp_path / "backups", verify=True)
    assert result.ok, result.error
    assert result.verified, result.verification
    assert result.output_path is not None
    return Path(result.output_path) / "manifest.json"


def _inject_invalid_entries(blob_root: Path) -> dict[str, bytes]:
    entries = {
        "state.db-wal": b"wal sidecar",
        "state.db-shm": b"shm sidecar",
        "ab/.blob.orphan": b"prepared but unpublished",
        "not-a-shard": b"root junk",
    }
    for relative, payload in entries.items():
        path = blob_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (blob_root / "cd" / "nested").mkdir(parents=True)
    (blob_root / "cd" / "nested" / "sidecar").write_bytes(b"nested invalid directory")
    (blob_root / "cd" / "link").symlink_to("nested/sidecar")
    return entries


def test_dry_run_classifies_mixed_namespace_without_mutating(tmp_path: Path) -> None:
    archive_root = _archive(tmp_path)
    store = BlobStore(archive_root / "blob")
    canonical_hash, _ = store.write_from_bytes(b"canonical payload")
    invalid = _inject_invalid_entries(store.root)

    report = quarantine_blob_namespace(archive_root)

    assert report.applied is False
    assert report.census.safe_to_apply
    assert [entry.hash_hex for entry in report.census.canonical] == [canonical_hash]
    assert {entry.relative_path for entry in report.census.candidates} == {
        *invalid,
        "cd/link",
        "cd/nested",
    }
    assert store.blob_path(canonical_hash).read_bytes() == b"canonical payload"
    assert all((store.root / relative).exists() for relative in invalid)
    assert (store.root / "cd" / "link").is_symlink()


def test_apply_moves_only_invalid_entries_and_writes_immutable_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = _archive(tmp_path)
    store = BlobStore(archive_root / "blob")
    canonical_payload = b"canonical payload"
    canonical_hash, _ = store.write_from_bytes(canonical_payload)
    canonical_path = store.blob_path(canonical_hash)
    canonical_before = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
    source_before = hashlib.sha256((archive_root / "source.db").read_bytes()).hexdigest()
    invalid = _inject_invalid_entries(store.root)
    manifest = _verified_source_backup(archive_root, tmp_path, monkeypatch)
    receipt_dir = tmp_path / "receipts" / "operation-one"
    receipt_dir.parent.mkdir()

    report = quarantine_blob_namespace(
        archive_root,
        backup_manifest=manifest,
        receipt_dir=receipt_dir,
        dry_run=False,
    )

    assert report.applied is True
    assert report.moved_count == 6
    assert report.quarantine_root is not None
    assert hashlib.sha256(canonical_path.read_bytes()).hexdigest() == canonical_before
    assert canonical_path.read_bytes() == canonical_payload
    assert hashlib.sha256((archive_root / "source.db").read_bytes()).hexdigest() == source_before
    assert store.verify_all().passed is True
    assert [(entry.relative_path, entry.hash_hex) for entry in store.iter_namespace()] == [
        (f"{canonical_hash[:2]}/{canonical_hash[2:]}", canonical_hash)
    ]
    for relative, payload in invalid.items():
        destination = report.quarantine_root / relative
        assert not os.path.lexists(store.root / relative)
        assert destination.read_bytes() == payload
    assert (report.quarantine_root / "cd" / "link").is_symlink()
    assert (report.quarantine_root / "cd" / "nested" / "sidecar").read_bytes() == b"nested invalid directory"
    before = receipt_dir / "before.json"
    after = receipt_dir / "after.json"
    assert before.exists() and after.exists()
    with pytest.raises(FileExistsError):
        before.open("x").close()
    after_payload = json.loads(after.read_text(encoding="utf-8"))
    assert (
        after_payload["quarantined_candidate_inventory_digest"]
        == json.loads(before.read_text(encoding="utf-8"))["candidate_inventory_digest"]
    )
    assert after_payload["full_blob_verification"] == {
        "canonical_checked": 1,
        "checked_bytes": len(canonical_payload),
        "hash_failures": 0,
        "invalid_namespace_entries": 0,
        "truncated": False,
    }


def test_canonical_hash_mismatch_and_backup_identity_mismatch_refuse_without_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = _archive(tmp_path)
    store = BlobStore(archive_root / "blob")
    canonical_hash, _ = store.write_from_bytes(b"original")
    invalid = _inject_invalid_entries(store.root)
    manifest = _verified_source_backup(archive_root, tmp_path, monkeypatch)
    store.blob_path(canonical_hash).write_bytes(b"tampered")
    receipt_parent = tmp_path / "receipts"
    receipt_parent.mkdir()

    with pytest.raises(BlobNamespaceQuarantineError, match="canonical-shaped hash mismatch"):
        quarantine_blob_namespace(
            archive_root,
            backup_manifest=manifest,
            receipt_dir=receipt_parent / "mismatch",
            dry_run=False,
        )
    assert all((store.root / relative).exists() for relative in invalid)

    # Restore the canonical bytes, then alter source.db after the attested
    # backup. Backup identity rejection happens before any namespace move.
    store.blob_path(canonical_hash).write_bytes(b"original")
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("CREATE TABLE backup_identity_mismatch (value TEXT)")
    with pytest.raises(Exception, match="live tier .* mismatch"):
        quarantine_blob_namespace(
            archive_root,
            backup_manifest=manifest,
            receipt_dir=receipt_parent / "stale-backup",
            dry_run=False,
        )
    assert all((store.root / relative).exists() for relative in invalid)


def test_apply_refuses_active_writer_and_read_only_recovery_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = _archive(tmp_path)
    store = BlobStore(archive_root / "blob")
    store.write_from_bytes(b"canonical")
    _inject_invalid_entries(store.root)
    manifest = _verified_source_backup(archive_root, tmp_path, monkeypatch)
    receipt_parent = tmp_path / "receipts"
    receipt_parent.mkdir()

    from polylogue.storage.index_generation import ActiveWriterLease

    lease = ActiveWriterLease(archive_root)
    lease.acquire()
    try:
        with pytest.raises(RuntimeError, match="lease"):
            quarantine_blob_namespace(
                archive_root,
                backup_manifest=manifest,
                receipt_dir=receipt_parent / "blocked-writer",
                dry_run=False,
            )
    finally:
        lease.close()

    report = quarantine_blob_namespace(
        archive_root,
        backup_manifest=manifest,
        receipt_dir=receipt_parent / "recovery",
        dry_run=False,
    )
    assert report.quarantine_root is not None
    committed = classify_blob_namespace_quarantine_recovery(receipt_parent / "recovery")
    assert committed.outcome == "committed"
    assert classify_blob_namespace_quarantine_recovery(receipt_parent / "recovery") == committed

    # Simulate the all-sources-present crash classification without asking the
    # recovery route to make any repair move itself.
    for candidate in report.census.candidates:
        destination = Path(candidate.destination)
        source = store.root / candidate.relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        os.replace(destination, source)
    rolled_back = classify_blob_namespace_quarantine_recovery(receipt_parent / "recovery")
    assert rolled_back.outcome == "rolled_back"
    first = report.census.candidates[0]
    os.replace(store.root / first.relative_path, Path(first.destination))
    assert classify_blob_namespace_quarantine_recovery(receipt_parent / "recovery").outcome == "indeterminate"


def test_root_level_interrupted_blob_is_quarantined_and_recovery_classified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise crash recovery through a real root-level ``.blob.*`` entry, never PreparedBlob cleanup."""
    archive_root = _archive(tmp_path)
    store = BlobStore(archive_root / "blob")
    store.write_from_bytes(b"canonical")
    interrupted = store.root / ".blob.crashed-before-publish"
    interrupted.write_bytes(b"interrupted payload")
    manifest = _verified_source_backup(archive_root, tmp_path, monkeypatch)
    receipt_dir = tmp_path / "receipts" / "root-level-interrupted"
    receipt_dir.parent.mkdir()

    report = quarantine_blob_namespace(
        archive_root,
        backup_manifest=manifest,
        receipt_dir=receipt_dir,
        dry_run=False,
    )

    assert report.quarantine_root is not None
    destination = report.quarantine_root / interrupted.name
    assert not interrupted.exists()
    assert destination.read_bytes() == b"interrupted payload"
    assert store.verify_all().passed
    assert classify_blob_namespace_quarantine_recovery(receipt_dir).outcome == "committed"

    os.replace(destination, interrupted)
    assert classify_blob_namespace_quarantine_recovery(receipt_dir).outcome == "rolled_back"


def test_apply_refuses_live_daemon_busy_checkpoint_and_symlinked_receipt_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = _archive(tmp_path)
    store = BlobStore(archive_root / "blob")
    store.write_from_bytes(b"canonical")
    invalid = _inject_invalid_entries(store.root)
    receipt_parent = tmp_path / "receipts"
    receipt_parent.mkdir()

    monkeypatch.setattr("polylogue.maintenance.blob_namespace_quarantine.running_daemon_pid", lambda _config: 123)
    with pytest.raises(BlobNamespaceQuarantineError, match="PID 123"):
        quarantine_blob_namespace(
            archive_root,
            backup_manifest=tmp_path / "not-needed-before-offline-gate.json",
            receipt_dir=receipt_parent / "daemon-running",
            dry_run=False,
        )
    assert all((store.root / relative).exists() for relative in invalid)

    monkeypatch.setattr("polylogue.maintenance.blob_namespace_quarantine.running_daemon_pid", lambda _config: None)
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(BlobNamespaceQuarantineError, match="non-symlink directory"):
        quarantine_blob_namespace(
            archive_root,
            backup_manifest=tmp_path / "not-needed-before-receipt-parent-gate.json",
            receipt_dir=symlink_parent / "escaped",
            dry_run=False,
        )
    assert all((store.root / relative).exists() for relative in invalid)


def test_busy_checkpoint_and_destination_conflict_leave_sources_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = _archive(tmp_path)
    store = BlobStore(archive_root / "blob")
    store.write_from_bytes(b"canonical")
    invalid = _inject_invalid_entries(store.root)
    manifest = _verified_source_backup(archive_root, tmp_path, monkeypatch)
    receipt_parent = tmp_path / "receipts"
    receipt_parent.mkdir()

    def _busy(_conn: sqlite3.Connection) -> None:
        raise BlobNamespaceQuarantineError("source.db WAL checkpoint was not clean")

    monkeypatch.setattr("polylogue.maintenance.blob_namespace_quarantine._checkpoint_source_db", _busy)
    with pytest.raises(BlobNamespaceQuarantineError, match="checkpoint was not clean"):
        quarantine_blob_namespace(
            archive_root,
            backup_manifest=manifest,
            receipt_dir=receipt_parent / "busy",
            dry_run=False,
        )
    assert all((store.root / relative).exists() for relative in invalid)

    from polylogue.maintenance import blob_namespace_quarantine as quarantine_mod

    monkeypatch.undo()
    manifest = _verified_source_backup(archive_root, tmp_path / "second-backup", monkeypatch)

    def _make_destination_conflict(quarantine_root: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"must not overwrite")

    monkeypatch.setattr(quarantine_mod, "_ensure_destination_parent", _make_destination_conflict)
    with pytest.raises(BlobNamespaceQuarantineError, match="destination already exists"):
        quarantine_blob_namespace(
            archive_root,
            backup_manifest=manifest,
            receipt_dir=receipt_parent / "conflict",
            dry_run=False,
        )
    assert all((store.root / relative).exists() for relative in invalid)


def test_real_hermes_sqlite_snapshot_stays_byte_identical_after_sidecar_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the same Hermes snapshot route that produces retained SQLite blobs."""
    archive_root = _archive(tmp_path)
    source_db = tmp_path / "state.db"
    _write_hermes_state_db(source_db)
    rows = list(
        iter_source_sessions_with_raw(
            Source(name="hermes", path=source_db),
            capture_raw=True,
            blob_root=archive_root / "blob",
        )
    )
    raw = rows[0][0]
    assert raw is not None and raw.blob_hash is not None
    store = BlobStore(archive_root / "blob")
    sqlite_blob = store.blob_path(raw.blob_hash)
    sqlite_before = hashlib.sha256(sqlite_blob.read_bytes()).hexdigest()
    _inject_invalid_entries(store.root)
    manifest = _verified_source_backup(archive_root, tmp_path, monkeypatch)
    receipt_parent = tmp_path / "receipts"
    receipt_parent.mkdir()

    report = quarantine_blob_namespace(
        archive_root,
        backup_manifest=manifest,
        receipt_dir=receipt_parent / "hermes",
        dry_run=False,
    )

    assert hashlib.sha256(sqlite_blob.read_bytes()).hexdigest() == sqlite_before
    with sqlite3.connect(f"file:{sqlite_blob}?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT id FROM sessions WHERE id = 'hermes-root'").fetchone() == ("hermes-root",)
    assert store.verify_all().passed
    assert report.quarantine_root is not None
    assert (report.quarantine_root / "state.db-wal").read_bytes() == b"wal sidecar"
    assert not os.path.lexists(store.root / "state.db-wal")

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from polylogue.daemon import whale_outbox


def _record(key: str = "receipt-1") -> dict[str, Any]:
    return {
        "kind": "whale.completed",
        "idempotency_key": key,
        "operation_id": "operation-1",
        "payload": {"status": "ok"},
    }


def test_enqueue_rejects_traversal_idempotency_key(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    with pytest.raises(ValueError, match="idempotency"):
        whale_outbox.enqueue(**_record("../../outside"), root=tmp_path)
    assert not (tmp_path.parent / "outside.json").exists()


def test_enqueue_rejects_symlinked_outbox_root(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside.chmod(0o700)
    (tmp_path / "whale-receipt-outbox").symlink_to(outside, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        whale_outbox.enqueue(**_record(), root=tmp_path)
    assert not (outside / "receipt-1.json").exists()


def test_enqueue_rejects_unsafe_permissions(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    (tmp_path / "whale-receipt-outbox").mkdir(mode=0o755)
    with pytest.raises((OSError, ValueError)):
        whale_outbox.enqueue(**_record(), root=tmp_path)


def test_list_pending_rejects_replaced_receipt_without_reading_target(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    target = whale_outbox.enqueue(**_record(), root=tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text('{"payload":{"status":"hostile"}}', encoding="utf-8")
    outside.chmod(0o600)
    target.unlink()
    target.symlink_to(outside)
    assert whale_outbox.list_pending(root=tmp_path) == []


def test_enqueue_and_acknowledge_are_idempotent(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    target = whale_outbox.enqueue(**_record(), root=tmp_path)
    assert target.exists()
    record = whale_outbox.list_pending(root=tmp_path)[0]
    whale_outbox.acknowledge(record)
    whale_outbox.acknowledge(record)
    assert not target.exists()


def test_duplicate_idempotency_key_requires_complete_identity(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    first = whale_outbox.enqueue(**_record(), root=tmp_path)
    original = first.read_bytes()
    same = whale_outbox.enqueue(**_record(), root=tmp_path)
    assert same == first
    assert first.read_bytes() == original
    assert list((tmp_path / "whale-receipt-outbox").glob(".*.tmp")) == []
    with pytest.raises(ValueError, match="conflicts"):
        whale_outbox.enqueue(
            kind="different.kind",
            idempotency_key="receipt-1",
            operation_id="operation-1",
            payload={"status": "different"},
            root=tmp_path,
        )
    assert first.read_bytes() == original


def test_intermediate_archive_symlink_is_rejected(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        whale_outbox.enqueue(**_record(), root=link / "archive")
    assert not (outside / "archive").exists()


def test_acknowledge_requires_pinned_identity(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    target = whale_outbox.enqueue(**_record(), root=tmp_path)
    record = {**_record(), "_path": target, "_name": target.name}
    whale_outbox.acknowledge(record)
    assert target.exists()


def test_acknowledge_rejects_regular_file_replacement(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    target = whale_outbox.enqueue(**_record(), root=tmp_path)
    record = whale_outbox.list_pending(root=tmp_path)[0]
    target.unlink()
    target.write_text('{"kind":"different","idempotency_key":"receipt-1","operation_id":"operation-1","payload":{}}')
    target.chmod(0o600)
    whale_outbox.acknowledge(record)
    assert target.exists()


def test_acknowledge_does_not_delete_replacement_at_quarantine_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path.chmod(0o700)
    target = whale_outbox.enqueue(**_record(), root=tmp_path)
    record = whale_outbox.list_pending(root=tmp_path)[0]
    replacement = b'{"kind":"replacement","idempotency_key":"receipt-1","operation_id":"operation-1","payload":{}}'
    real_rename = os.rename
    replaced = False

    def replace_before_move(source: str, destination: str, *, src_dir_fd: int = -1, dst_dir_fd: int = -1) -> None:
        nonlocal replaced
        if not replaced and source == target.name:
            target.unlink()
            target.write_bytes(replacement)
            target.chmod(0o600)
            replaced = True
        real_rename(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(os, "rename", replace_before_move)
    whale_outbox.acknowledge(record)
    assert target.read_bytes() == replacement


def test_acknowledge_makes_quarantine_race_recoverable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path.chmod(0o700)
    target = whale_outbox.enqueue(**_record(), root=tmp_path)
    record = whale_outbox.list_pending(root=tmp_path)[0]
    replacement = b"replacement"
    real_move = whale_outbox._rename_noreplace
    real_rename = os.rename
    replaced = False
    inserted = False

    def replace_before_move(source: str, destination: str, *, src_dir_fd: int = -1, dst_dir_fd: int = -1) -> None:
        nonlocal replaced
        if not replaced and source == target.name:
            target.unlink()
            target.write_bytes(b"first replacement")
            target.chmod(0o600)
            replaced = True
        real_rename(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    def insert_before_restore(source: str, destination: str, *, directory_fd: int) -> None:
        nonlocal inserted
        if not inserted and source.endswith(".ack"):
            target.write_bytes(replacement)
            target.chmod(0o600)
            inserted = True
        real_move(source, destination, directory_fd=directory_fd)

    monkeypatch.setattr(os, "rename", replace_before_move)
    monkeypatch.setattr(whale_outbox, "_rename_noreplace", insert_before_restore)
    whale_outbox.acknowledge(record)
    outbox = tmp_path / "whale-receipt-outbox"
    assert target.read_bytes() == replacement
    assert list(outbox.glob("*.ack")) == []
    assert len(list(outbox.glob("receipt-1.json.recovery.*.json"))) == 1


def test_acknowledge_rejects_archive_outbox_root_replacement(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    target = whale_outbox.enqueue(**_record(), root=tmp_path)
    record = whale_outbox.list_pending(root=tmp_path)[0]
    outbox = tmp_path / "whale-receipt-outbox"
    old_outbox = tmp_path / "old-outbox"
    original_bytes = target.read_bytes()
    outbox.rename(old_outbox)
    outbox.mkdir(mode=0o700)
    replacement = outbox / target.name
    replacement.write_bytes(original_bytes)
    replacement.chmod(0o600)
    whale_outbox.acknowledge(record)
    assert replacement.exists()


def test_list_pending_cross_validates_filename_and_identity_fields(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    outbox = tmp_path / "whale-receipt-outbox"
    outbox.mkdir(mode=0o700)
    (outbox / "receipt-1.json").write_text(
        '{"kind":"","idempotency_key":"other","operation_id":"operation-1","payload":{}}',
        encoding="utf-8",
    )
    (outbox / "receipt-1.json").chmod(0o600)
    assert whale_outbox.list_pending(root=tmp_path) == []


def test_acknowledge_rejects_tampered_body_identity(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    target = whale_outbox.enqueue(**_record(), root=tmp_path)
    record = whale_outbox.list_pending(root=tmp_path)[0]
    record["operation_id"] = "tampered"
    whale_outbox.acknowledge(record)
    assert target.exists()


def test_intermediate_component_replacement_does_not_escape(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    target = whale_outbox.enqueue(**_record(), root=tmp_path)
    record = whale_outbox.list_pending(root=tmp_path)[0]
    outbox = tmp_path / "whale-receipt-outbox"
    moved = tmp_path / "moved-outbox"
    outbox.rename(moved)
    outbox.symlink_to(moved, target_is_directory=True)
    whale_outbox.acknowledge(record)
    assert (moved / target.name).exists()
    assert outbox.is_symlink()
    assert whale_outbox.list_pending(root=tmp_path) == []


def test_enqueue_does_not_follow_replaced_archive_root(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    archive = tmp_path / "archive"
    archive.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    archive.rename(tmp_path / "old-archive")
    archive.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        whale_outbox.enqueue(**_record(), root=archive)
    assert not (outside / "whale-receipt-outbox").exists()

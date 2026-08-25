"""Failure atomicity for descriptor-pinned maintenance receipt directories."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from polylogue.maintenance.receipt_fs import (
    MaintenanceReceiptPathError,
    atomic_replace_receipt,
    maintenance_receipt_directory,
)


def test_fsync_failure_removes_only_the_new_empty_receipt_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed persistence barrier leaves no fresh child residue and preserves an existing child."""
    root = tmp_path / "archive"
    state = root / ".maintenance-state"
    state.mkdir(parents=True)
    existing = state / "existing"
    existing.mkdir()
    state_inode = state.stat().st_ino
    real_fsync = os.fsync

    def fail_state_fsync(descriptor: int) -> None:
        if os.fstat(descriptor).st_ino == state_inode:
            raise OSError("directory fsync failed")
        real_fsync(descriptor)

    monkeypatch.setattr("polylogue.maintenance.receipt_fs.os.fsync", fail_state_fsync)
    with pytest.raises(MaintenanceReceiptPathError, match="maintenance receipt directory"):
        with maintenance_receipt_directory(root, "new-child"):
            pytest.fail("the failed child directory must not be yielded")

    assert not (state / "new-child").exists()
    assert existing.is_dir()


def test_receipt_directory_wraps_mkdir_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Publication callers receive the typed path error when directory creation fails."""
    root = tmp_path / "archive"
    (root / ".maintenance-state").mkdir(parents=True)

    def deny_mkdir(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("read-only maintenance state")

    monkeypatch.setattr(os, "mkdir", deny_mkdir)
    with pytest.raises(MaintenanceReceiptPathError, match="cannot create maintenance receipt directory"):
        with maintenance_receipt_directory(root, "new-child"):
            pytest.fail("the failed child directory must not be yielded")


def test_receipt_directory_rejects_symlinked_state_without_touching_target(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (root / ".maintenance-state").symlink_to(external, target_is_directory=True)

    with pytest.raises(MaintenanceReceiptPathError, match="without following links"):
        with maintenance_receipt_directory(root, "receipts"):
            pytest.fail("a symlinked maintenance state must never be entered")

    assert list(external.iterdir()) == []


def test_atomic_receipt_publication_removes_partial_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    state = root / ".maintenance-state" / "receipts"
    state.mkdir(parents=True)
    with maintenance_receipt_directory(root, "receipts") as directory_fd:

        def fail_replace(*_args: object, **_kwargs: object) -> None:
            raise OSError("rename failed")

        monkeypatch.setattr("polylogue.maintenance.receipt_fs.os.replace", fail_replace)
        with pytest.raises(MaintenanceReceiptPathError, match="atomically publish"):
            atomic_replace_receipt(directory_fd, "result.json", b"partial")

    assert not (state / "result.json").exists()
    assert not tuple(state.glob("*.tmp"))

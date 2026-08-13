"""Failure atomicity for descriptor-pinned maintenance receipt directories."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from polylogue.operations._maintenance_receipt_fs import (
    MaintenanceReceiptPathError,
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

    monkeypatch.setattr("polylogue.operations._maintenance_receipt_fs.os.fsync", fail_state_fsync)
    with pytest.raises(MaintenanceReceiptPathError, match="maintenance receipt directory"):
        with maintenance_receipt_directory(root, "new-child"):
            pytest.fail("the failed child directory must not be yielded")

    assert not (state / "new-child").exists()
    assert existing.is_dir()

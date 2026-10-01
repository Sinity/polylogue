from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from polylogue.core.prepared_file import PreparedFileSeal

pytestmark = pytest.mark.uses_real_clock


def test_seal_binds_complete_bytes_and_survives_repeat_capture(tmp_path: Path) -> None:
    path = tmp_path / "closed-artifact"
    payload = b"synthetic material" * 1000
    path.write_bytes(payload)
    seal = PreparedFileSeal.capture(path)
    assert seal.sha256 == hashlib.sha256(payload).hexdigest()
    assert PreparedFileSeal.capture(path) == seal
    seal.verify(path, full=True)
    seal.verify(path, full=False)


def test_same_bytes_on_another_inode_refuse_publication(tmp_path: Path) -> None:
    path = tmp_path / "closed-artifact"
    path.write_bytes(b"synthetic")
    seal = PreparedFileSeal.capture(path)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"synthetic")
    replacement.replace(path)
    with pytest.raises(ValueError):
        seal.verify(path, full=False)


def test_mutation_with_restored_mtime_refuses_publication(tmp_path: Path) -> None:
    path = tmp_path / "closed-artifact"
    path.write_bytes(b"synthetic-one")
    seal = PreparedFileSeal.capture(path)
    path.chmod(0o600)
    path.write_bytes(b"synthetic-two")
    os.utime(path, ns=(seal.mtime_ns, seal.mtime_ns))
    with pytest.raises(ValueError):
        seal.verify(path, full=False)


def test_capture_refuses_a_redirect_without_changing_target_permissions(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"synthetic")
    target.chmod(0o600)
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    with pytest.raises(OSError):
        PreparedFileSeal.capture(alias)
    assert target.stat().st_mode & 0o777 == 0o600

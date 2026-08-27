"""Crash-consistency contracts for the shared durable filesystem primitives."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from polylogue.core.durable_fs import DurableFilesystemError, append_line, atomic_replace, write_once


@pytest.mark.parametrize("operation", ["write_once", "atomic_replace", "append_line"])
def test_durable_file_operations_sync_the_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Mutation: removing the directory fsync would make this assertion fail."""
    fsynced: list[bool] = []
    real_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        fsynced.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
        real_fsync(descriptor)

    monkeypatch.setattr("polylogue.core.durable_fs.os.fsync", record_fsync)
    path = tmp_path / "nested" / "receipt"
    if operation == "write_once":
        write_once(path, b"first")
    elif operation == "atomic_replace":
        atomic_replace(path, b"replacement")
    else:
        append_line(path, "line")

    assert path.read_bytes() in {b"first", b"replacement", b"line\n"}
    assert any(fsynced)


def test_write_once_refuses_existing_path(tmp_path: Path) -> None:
    path = tmp_path / "receipt"
    path.write_bytes(b"original")

    with pytest.raises(DurableFilesystemError):
        write_once(path, b"replacement")

    assert path.read_bytes() == b"original"

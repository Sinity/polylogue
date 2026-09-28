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


def test_clone_or_copy_replace_stages_a_maximal_component_name(tmp_path: Path) -> None:
    """A 255-byte member name stages; the temporary name stays short.

    Anti-vacuity: build the temporary from the destination name (as
    ``.<name>.XXXXXXXX``) and ``mkstemp`` fails with ``ENAMETOOLONG``.
    """
    from polylogue.core.durable_fs import clone_or_copy_replace

    source = tmp_path / "source.json"
    source.write_bytes(b'{"member": 1}')
    source.chmod(0o640)
    os.utime(source, ns=(1_000_000_000, 2_000_000_000))
    destination = tmp_path / "staged" / ("m" * 250 + ".json")

    clone_or_copy_replace(source, destination)

    assert destination.read_bytes() == b'{"member": 1}'
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640
    assert destination.stat().st_mtime_ns == 2_000_000_000
    assert [path.name for path in destination.parent.iterdir()] == [destination.name]


def test_clone_or_copy_replace_sets_metadata_before_its_file_barrier(tmp_path: Path) -> None:
    """Mode and timestamps are on the temporary before its ``fsync``.

    Anti-vacuity: apply them after the only file ``fsync`` and the barrier
    observes the ``mkstemp`` defaults instead of the source's mode and mtime.
    """
    from unittest.mock import patch

    from polylogue.core.durable_fs import clone_or_copy_replace

    source = tmp_path / "source.json"
    source.write_text("{}")
    source.chmod(0o640)
    os.utime(source, ns=(1_000_000_000, 2_000_000_000))
    observed: list[tuple[int, int]] = []
    real_fsync = os.fsync

    def observing_fsync(fd: int) -> None:
        info = os.fstat(fd)
        if stat.S_ISREG(info.st_mode):
            observed.append((stat.S_IMODE(info.st_mode), info.st_mtime_ns))
        real_fsync(fd)

    with patch("polylogue.core.durable_fs.os.fsync", side_effect=observing_fsync):
        clone_or_copy_replace(source, tmp_path / "staged" / "source.json")

    assert observed == [(0o640, 2_000_000_000)]

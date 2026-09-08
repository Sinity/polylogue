"""Atomic provider-tree publication and coherent reader snapshots."""

from __future__ import annotations

import ctypes
import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from polylogue.core.durable_fs import sync_directory


@contextmanager
def provider_tree_lock(root: Path, *, exclusive: bool = False) -> Iterator[None]:
    """Lock the stable parent inode while reading or replacing provider trees."""
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        os.close(descriptor)


def _read_provider_tree(provider_dir: Path) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for path in sorted(provider_dir.rglob("*")):
        relative = path.relative_to(provider_dir).as_posix()
        if path.is_dir():
            snapshot[f"{relative}/"] = b""
        elif path.is_file() and (path.suffix == ".json" or path.name.endswith(".json.gz")):
            snapshot[relative] = path.read_bytes()
    return snapshot


def read_provider_snapshot(provider_dir: Path) -> dict[str, bytes]:
    """Retain compressed bytes so a cached catalog never resolves newer files."""
    if not provider_dir.parent.exists():
        return {}
    with provider_tree_lock(provider_dir.parent):
        return _read_provider_tree(provider_dir) if provider_dir.is_dir() else {}


def publish_provider_tree(staged: Path, destination: Path, *, expected_snapshot: dict[str, bytes]) -> None:
    """Publish a complete staged tree; after exchange, staged holds the old tree."""
    for path in sorted(staged.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for path in sorted((path for path in staged.rglob("*") if path.is_dir()), reverse=True):
        sync_directory(path)
    sync_directory(staged)
    with provider_tree_lock(destination.parent, exclusive=True):
        current = _read_provider_tree(destination) if destination.exists() else {}
        if current != expected_snapshot:
            raise RuntimeError("Schema packages changed during preparation; retry from the current catalog")
        if destination.exists():
            libc = ctypes.CDLL(None, use_errno=True)
            rename = getattr(libc, "renameat2", None)
            if rename is None:
                raise NotImplementedError("Atomic schema replacement requires renameat2")
            rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            if rename(-100, os.fsencode(staged), -100, os.fsencode(destination), 2) != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error), str(destination))
        else:
            os.rename(staged, destination)
        sync_directory(destination.parent)
        sync_directory(staged.parent)

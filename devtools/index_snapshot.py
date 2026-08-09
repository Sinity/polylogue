"""Shared selected-index file-set observation for evidence reports."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SNAPSHOT_HASH_CHUNK_BYTES = 1024 * 1024


class IndexSnapshotRaceError(RuntimeError):
    """The selected index pathname no longer names the opened database."""


class IndexSnapshotUnsafeSidecarError(RuntimeError):
    """A selected SQLite sidecar is not a safe regular file."""


@dataclass(frozen=True, slots=True)
class OpenedIndexFileSet:
    """Descriptors retained while SQLite and evidence observe one index."""

    main_fd: int
    sidecar_fds: Mapping[str, int]


@contextmanager
def open_index_file_set(index_db: Path) -> Iterator[OpenedIndexFileSet]:
    """Open the selected database and existing sidecars without following links."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        try:
            main_fd = os.open(index_db, flags)
        except OSError as exc:
            raise IndexSnapshotRaceError(f"cannot open selected index safely: {index_db}") from exc
        descriptors.append(main_fd)
        sidecar_fds: dict[str, int] = {}
        for suffix in ("-wal", "-shm", "-journal"):
            path = Path(f"{index_db}{suffix}")
            try:
                descriptor = os.open(path, flags)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise IndexSnapshotUnsafeSidecarError(f"cannot open selected index sidecar safely: {path}") from exc
            descriptors.append(descriptor)
            sidecar_fds[suffix] = descriptor
        yield OpenedIndexFileSet(main_fd=main_fd, sidecar_fds=sidecar_fds)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


@contextmanager
def open_index_file_handle(index_db: Path) -> Iterator[int]:
    """Keep the selected main database inode open for evidence snapshots."""
    with open_index_file_set(index_db) as file_set:
        yield file_set.main_fd


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_SNAPSHOT_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, _SNAPSHOT_HASH_CHUNK_BYTES, offset)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        offset += len(chunk)


def snapshot_index_file_set(
    index_db: Path,
    *,
    opened_main_fd: int | None = None,
    opened_sidecar_fds: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Capture one selected index and its SQLite sidecars under one contract.

    Each present file is hashed between two metadata reads. A disappearing or
    changing file makes ``observation_complete`` false, while the file-set
    digest remains useful evidence for the caller's stability comparison.
    """
    paths = (index_db, Path(f"{index_db}-wal"), Path(f"{index_db}-shm"), Path(f"{index_db}-journal"))
    files: list[dict[str, Any]] = []
    complete = True
    for path in paths:
        suffix = "" if path == index_db else path.name.removeprefix(index_db.name)
        if path != index_db and opened_sidecar_fds is not None and suffix not in opened_sidecar_fds:
            try:
                sidecar_metadata = path.stat(follow_symlinks=False)
            except FileNotFoundError:
                files.append({"path": str(path), "present": False})
                continue
            if not stat.S_ISREG(sidecar_metadata.st_mode):
                raise IndexSnapshotUnsafeSidecarError(f"selected index sidecar is not a regular file: {path}")
            try:
                late_sidecar_fd = os.open(
                    path,
                    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                )
            except OSError as exc:
                raise IndexSnapshotUnsafeSidecarError(f"cannot open selected index sidecar safely: {path}") from exc
            try:
                late_sidecar_metadata = os.fstat(late_sidecar_fd)
                late_sidecar_digest = _file_sha256_descriptor(late_sidecar_fd)
            finally:
                os.close(late_sidecar_fd)
            complete = False
            files.append(
                {
                    "path": str(path),
                    "present": True,
                    "size": late_sidecar_metadata.st_size,
                    "mtime_ns": late_sidecar_metadata.st_mtime_ns,
                    "inode": late_sidecar_metadata.st_ino,
                    "sha256": late_sidecar_digest,
                    "changed_during_observation": True,
                }
            )
            continue
        opened_fd = opened_main_fd if path == index_db else (opened_sidecar_fds or {}).get(suffix)
        if opened_fd is not None:
            handle_metadata = os.fstat(opened_fd)
            try:
                path_metadata_before = path.stat(follow_symlinks=False)
            except FileNotFoundError:
                complete = False
            else:
                if not stat.S_ISREG(path_metadata_before.st_mode):
                    error = IndexSnapshotRaceError if path == index_db else IndexSnapshotUnsafeSidecarError
                    raise error(f"selected index file is not regular: {path}")
                if (path_metadata_before.st_dev, path_metadata_before.st_ino) != (
                    handle_metadata.st_dev,
                    handle_metadata.st_ino,
                ):
                    label = "selected index path" if path == index_db else "selected index sidecar"
                    error = IndexSnapshotRaceError if path == index_db else IndexSnapshotUnsafeSidecarError
                    raise error(f"{label} was replaced while its reader was open: {path}")
            digest = _file_sha256_descriptor(opened_fd)
            try:
                path_metadata_after = path.stat(follow_symlinks=False)
            except FileNotFoundError:
                complete = False
                path_present = False
            else:
                if not stat.S_ISREG(path_metadata_after.st_mode):
                    error = IndexSnapshotRaceError if path == index_db else IndexSnapshotUnsafeSidecarError
                    raise error(f"selected index file is not regular: {path}")
                if (path_metadata_after.st_dev, path_metadata_after.st_ino) != (
                    handle_metadata.st_dev,
                    handle_metadata.st_ino,
                ):
                    label = "selected index path" if path == index_db else "selected index sidecar"
                    error = IndexSnapshotRaceError if path == index_db else IndexSnapshotUnsafeSidecarError
                    raise error(f"{label} was replaced during snapshot observation: {path}")
                path_present = True
            files.append(
                {
                    "path": str(path),
                    "present": path_present,
                    "size": handle_metadata.st_size,
                    "mtime_ns": handle_metadata.st_mtime_ns,
                    "inode": handle_metadata.st_ino,
                    "sha256": digest,
                    "changed_during_observation": False,
                }
            )
            continue
        try:
            metadata_before = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            if path == index_db:
                # A missing sidecar is a normal quiescent SQLite state; a
                # missing selected database is not evidence of an observed
                # snapshot, even when an already-open connection can still
                # serve reads from the unlinked inode.
                complete = False
            files.append({"path": str(path), "present": False})
            continue
        if not stat.S_ISREG(metadata_before.st_mode):
            if path == index_db:
                raise IndexSnapshotRaceError(f"selected index is not a regular file: {path}")
            raise IndexSnapshotUnsafeSidecarError(f"selected index sidecar is not a regular file: {path}")
        try:
            digest = _file_sha256(path)
            metadata_after = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            complete = False
            files.append({"path": str(path), "present": False, "changed_during_observation": True})
            continue
        unchanged = (
            metadata_before.st_dev,
            metadata_before.st_ino,
            metadata_before.st_size,
            metadata_before.st_mtime_ns,
        ) == (
            metadata_after.st_dev,
            metadata_after.st_ino,
            metadata_after.st_size,
            metadata_after.st_mtime_ns,
        )
        complete = complete and unchanged
        files.append(
            {
                "path": str(path),
                "present": True,
                "size": metadata_after.st_size,
                "mtime_ns": metadata_after.st_mtime_ns,
                "inode": metadata_after.st_ino,
                "sha256": digest,
                "changed_during_observation": not unchanged,
            }
        )
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    main = files[0]
    return {
        "path": str(index_db),
        "index_db": str(index_db),
        "present": main["present"],
        "size": main.get("size"),
        "files": files,
        "observation_complete": complete,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }

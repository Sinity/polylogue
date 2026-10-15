"""Read a selected physical file without releasing this process's SQLite locks."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import select
import stat
import subprocess
import sys
import time
from builtins import BaseExceptionGroup
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_READ_CHUNK = 1024 * 1024
_RESULT_LIMIT = 4096
_CHILD = "from polylogue.storage.sqlite.physical_file import _physical_file_hash_child; _physical_file_hash_child()"


@dataclass(frozen=True, slots=True)
class PhysicalFileDigest:
    """Exact digest and physical metadata observed by the lock-isolated reader."""

    sha256: str
    size_bytes: int
    device: int
    inode: int
    mode: int
    mtime_ns: int
    ctime_ns: int


def open_physical_identity(path: str | Path, *, dir_fd: int | None = None) -> int:
    """Return a caller-owned Linux identity pin, never an ordinary data FD."""
    path_flag = getattr(os, "O_PATH", None)
    if path_flag is None:
        raise OSError(errno.ENOTSUP, "lock-preserving file identity requires O_PATH", str(path))
    return os.open(path, path_flag | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=dir_fd)


def require_physical_identity(descriptor: int) -> os.stat_result:
    """An ordinary descriptor's eventual close would drop native SQLite locks."""
    path_flag = getattr(os, "O_PATH", None)
    if path_flag is None or not fcntl.fcntl(descriptor, fcntl.F_GETFL) & path_flag:
        raise ValueError("SQLite identity requires an O_PATH descriptor")
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError(errno.EINVAL, "physical identity is not a regular file")
    return metadata


def validate_physical_identity_path(
    path: str | Path, descriptor: int, *, expected_parent: tuple[int, int] | None = None
) -> tuple[int, int]:
    """Recheck the actual native pathname and its parent around SQLite open."""
    metadata = require_physical_identity(descriptor)
    native_path = Path(path)
    selected = native_path.parent.resolve(strict=True) / native_path.name
    leaf = selected.stat(follow_symlinks=False)
    if not stat.S_ISREG(leaf.st_mode):
        raise OSError(errno.ESTALE, "physical identity is not a regular native leaf", str(path))
    _require_identity(leaf, metadata.st_dev, metadata.st_ino, selected)
    parent = _directory_identity_at(selected.parent)
    if expected_parent is not None and parent != expected_parent:
        raise OSError(errno.ESTALE, "physical identity parent changed during SQLite open", str(path))
    return parent


def physical_identity_operation(
    descriptor: int,
    *,
    copy_directory_fd: int | None = None,
    copy_name: str | None = None,
    chmod: int | None = None,
    progress: Callable[[], None] | None = None,
) -> PhysicalFileDigest | None:
    """Hash/copy or chmod a borrowed pin in the existing isolated child.

    Copy destinations are new private leaves in a caller-pinned directory.
    The caller owns partial destination cleanup and the original identity pin.
    """
    from polylogue.core.compute_cancel import check_compute_cancelled

    metadata = require_physical_identity(descriptor)
    request: dict[str, object] = {"identity_fd": descriptor}
    descriptors = [descriptor]
    if copy_directory_fd is not None:
        if not copy_name or copy_name in {".", ".."} or "/" in copy_name:
            raise ValueError("physical copy requires a single destination leaf")
        _directory_identity(copy_directory_fd)
        descriptors.append(copy_directory_fd)
        request.update(copy_directory_fd=copy_directory_fd, copy_name=copy_name)
    elif copy_name is not None:
        raise ValueError("physical copy requires a pinned destination directory")
    if chmod is not None:
        if copy_directory_fd is not None:
            raise ValueError("physical chmod cannot also copy")
        request["chmod"] = chmod

    def check() -> None:
        check_compute_cancelled()
        if progress is not None:
            progress()

    payload = _run_physical_child(request, tuple(descriptors), check)
    if chmod is not None:
        if json.loads(payload) != {"chmod": chmod}:
            raise OSError(errno.EPROTO, "physical chmod did not acknowledge its operation")
        return None
    return _decode_child_result(payload, expected_device=metadata.st_dev, expected_inode=metadata.st_ino)


def _run_physical_child(
    request: dict[str, object], descriptors: tuple[int, ...], check_cancelled: Callable[[], None]
) -> bytes:
    check_cancelled()
    command = f"import sys; sys.path[:] = {sys.path!r}; {_CHILD}"
    process = subprocess.Popen(
        [sys.executable, "-c", command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        pass_fds=descriptors,
    )
    try:
        assert process.stdin is not None
        process.stdin.write(json.dumps(request, separators=(",", ":")).encode("ascii"))
        process.stdin.close()
        payload = _read_child_result(process, check_cancelled)
        _wait_child(process, check_cancelled)
        if process.returncode != 0:
            raise OSError(errno.EIO, "physical file child failed")
        return payload
    finally:
        _settle_child(process)


def physical_file_sha256(path: Path, *, expected_device: int, expected_inode: int) -> PhysicalFileDigest:
    """Hash one selected regular leaf in a child process.

    The parent opens and retains only the containing directory. The child's
    ordinary data descriptor is therefore never closed in the process that
    may hold SQLite's process-wide POSIX locks.
    """
    from polylogue.core.compute_cancel import check_compute_cancelled

    check_compute_cancelled()
    selected = path.absolute()
    name = selected.name
    if not name or name in {".", ".."} or "/" in name:
        raise OSError(errno.EINVAL, "physical file path has no safe leaf name", str(path))

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_fd = os.open(selected.parent, directory_flags)
    try:
        directory_identity = _directory_identity(directory_fd)
        if _directory_identity_at(selected.parent) != directory_identity:
            raise OSError(errno.ESTALE, "physical file parent changed during selection", str(selected.parent))
        selected_metadata = _named_leaf_metadata(directory_fd, name)
        _require_identity(selected_metadata, expected_device, expected_inode, selected)
        result = {
            "directory_fd": directory_fd,
            "name": name,
            "device": expected_device,
            "inode": expected_inode,
        }
        payload = _run_physical_child(result, (directory_fd,), check_compute_cancelled)
        after_metadata = _named_leaf_metadata(directory_fd, name)
        _require_identity(after_metadata, expected_device, expected_inode, selected)
        if _directory_identity_at(selected.parent) != directory_identity:
            raise OSError(errno.ESTALE, "physical file parent changed during hash", str(selected.parent))
        digest = _decode_child_result(payload, expected_device=expected_device, expected_inode=expected_inode)
        if _metadata(after_metadata) != _digest_metadata(digest):
            raise OSError(errno.ESTALE, "physical file changed during hash", str(selected))
        return digest
    finally:
        os.close(directory_fd)


def _physical_file_hash_child() -> None:
    """Child entrypoint; this path imports standard-library modules only."""
    try:
        request = json.loads(sys.stdin.buffer.read(_RESULT_LIMIT + 1))
        if isinstance(request, dict) and "identity_fd" in request:
            _physical_identity_child(request)
            return
        if not isinstance(request, dict) or set(request) != {"directory_fd", "name", "device", "inode"}:
            raise OSError(errno.EPROTO, "invalid physical file hash request")
        directory_fd = request["directory_fd"]
        name = request["name"]
        device = request["device"]
        inode = request["inode"]
        if (
            type(directory_fd) is not int
            or not isinstance(name, str)
            or not name
            or "/" in name
            or type(device) is not int
            or type(inode) is not int
        ):
            raise OSError(errno.EPROTO, "invalid physical file hash selection")
        directory_metadata = os.fstat(directory_fd)
        if not stat.S_ISDIR(directory_metadata.st_mode):
            raise OSError(errno.ESTALE, "physical file parent descriptor changed")
        before = _named_leaf_metadata(directory_fd, name)
        _require_identity(before, device, inode, Path(name))
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        try:
            opened_before = os.fstat(descriptor)
            _require_identity(opened_before, device, inode, Path(name))
            if _metadata(before) != _metadata(opened_before):
                raise OSError(errno.ESTALE, "physical file changed while opening", name)
            digest = hashlib.sha256()
            size_bytes = 0
            while True:
                chunk = os.read(descriptor, _READ_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                size_bytes += len(chunk)
            after = os.fstat(descriptor)
            named_after = _named_leaf_metadata(directory_fd, name)
            if _metadata(opened_before) != _metadata(after) or _metadata(after) != _metadata(named_after):
                raise OSError(errno.ESTALE, "physical file changed while hashing", name)
            if size_bytes != after.st_size:
                raise OSError(errno.ESTALE, "physical file size changed while hashing", name)
            result = {
                "sha256": digest.hexdigest(),
                "size_bytes": size_bytes,
                "device": after.st_dev,
                "inode": after.st_ino,
                "mode": after.st_mode,
                "mtime_ns": after.st_mtime_ns,
                "ctime_ns": after.st_ctime_ns,
            }
            sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
            sys.stdout.flush()
        finally:
            os.close(descriptor)
    except BaseException:
        raise SystemExit(1) from None


def _physical_identity_child(request: dict[str, object]) -> None:
    """Only this child opens ordinary source descriptors for identity pins."""
    identity_fd = request["identity_fd"]
    if type(identity_fd) is not int:
        raise OSError(errno.EPROTO, "invalid physical identity")
    selected = require_physical_identity(identity_fd)
    descriptor = os.open(f"/proc/self/fd/{identity_fd}", os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
    destination = -1
    try:
        before = os.fstat(descriptor)
        _require_identity(before, selected.st_dev, selected.st_ino, Path(str(identity_fd)))
        if "chmod" in request:
            mode = request["chmod"]
            if set(request) != {"identity_fd", "chmod"} or type(mode) is not int:
                raise OSError(errno.EPROTO, "invalid physical chmod")
            os.fchmod(descriptor, mode)
            result: dict[str, object] = {"chmod": mode}
        else:
            if "copy_directory_fd" in request:
                directory = request["copy_directory_fd"]
                name = request.get("copy_name")
                if (
                    set(request) != {"identity_fd", "copy_directory_fd", "copy_name"}
                    or type(directory) is not int
                    or not isinstance(name, str)
                    or not name
                    or name in {".", ".."}
                    or "/" in name
                ):
                    raise OSError(errno.EPROTO, "invalid physical copy")
                _directory_identity(directory)
                destination = os.open(
                    name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory
                )
            elif set(request) != {"identity_fd"}:
                raise OSError(errno.EPROTO, "invalid physical read")
            digest = hashlib.sha256()
            size_bytes = 0
            while chunk := os.read(descriptor, _READ_CHUNK):
                digest.update(chunk)
                size_bytes += len(chunk)
                if destination >= 0:
                    remaining = memoryview(chunk)
                    while remaining:
                        written = os.write(destination, remaining)
                        if written <= 0:
                            raise OSError(errno.EIO, "physical copy made no write progress")
                        remaining = remaining[written:]
            after = os.fstat(descriptor)
            if _metadata(before) != _metadata(after) or size_bytes != after.st_size:
                raise OSError(errno.ESTALE, "physical identity changed while reading")
            if destination >= 0:
                os.fchmod(destination, stat.S_IMODE(before.st_mode))
            result = {
                "sha256": digest.hexdigest(),
                "size_bytes": size_bytes,
                "device": before.st_dev,
                "inode": before.st_ino,
                "mode": before.st_mode,
                "mtime_ns": before.st_mtime_ns,
                "ctime_ns": before.st_ctime_ns,
            }
        sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    finally:
        if destination >= 0:
            os.close(destination)
        os.close(descriptor)


def _read_child_result(process: subprocess.Popen[bytes], check_cancelled: Callable[[], None]) -> bytes:
    assert process.stdout is not None
    payload = bytearray()
    while True:
        check_cancelled()
        ready, _, _ = select.select((process.stdout,), (), (), 0.1)
        if not ready:
            if process.poll() is not None:
                continue
            continue
        chunk = os.read(process.stdout.fileno(), min(1024, _RESULT_LIMIT + 1 - len(payload)))
        if not chunk:
            return bytes(payload)
        payload.extend(chunk)
        if len(payload) > _RESULT_LIMIT:
            raise OSError(errno.EPROTO, "physical file hash result exceeds its bound")


def _wait_child(process: subprocess.Popen[bytes], check_cancelled: Callable[[], None]) -> None:
    while process.poll() is None:
        check_cancelled()
        time.sleep(0.05)
    process.wait()


def _settle_child(process: subprocess.Popen[bytes]) -> None:
    failures: list[BaseException] = []
    try:
        if process.poll() is None:
            process.kill()
        process.wait()
    except BaseException as exc:
        failures.append(exc)
    for stream in (process.stdin, process.stdout):
        if stream is None:
            continue
        try:
            stream.close()
        except BaseException as exc:
            failures.append(exc)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("physical file hash child settlement failed", failures)


def _decode_child_result(payload: bytes, *, expected_device: int, expected_inode: int) -> PhysicalFileDigest:
    try:
        result = json.loads(payload)
        if not isinstance(result, dict) or set(result) != {
            "sha256",
            "size_bytes",
            "device",
            "inode",
            "mode",
            "mtime_ns",
            "ctime_ns",
        }:
            raise ValueError("unexpected physical hash fields")
        digest = result["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or any(type(result[key]) is not int for key in result if key != "sha256")
            or result["size_bytes"] < 0
            or result["device"] != expected_device
            or result["inode"] != expected_inode
        ):
            raise ValueError("invalid physical hash result")
        return PhysicalFileDigest(
            sha256=digest,
            size_bytes=result["size_bytes"],
            device=result["device"],
            inode=result["inode"],
            mode=result["mode"],
            mtime_ns=result["mtime_ns"],
            ctime_ns=result["ctime_ns"],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OSError(errno.EPROTO, "invalid physical file hash result") from exc


def _named_leaf_metadata(directory_fd: int, name: str) -> os.stat_result:
    metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError(errno.ELOOP, "physical file is not a regular leaf", name)
    return metadata


def _require_identity(metadata: os.stat_result, device: int, inode: int, path: Path) -> None:
    if (metadata.st_dev, metadata.st_ino) != (device, inode):
        raise OSError(errno.ESTALE, "physical file identity changed", str(path))


def _directory_identity(directory_fd: int) -> tuple[int, int]:
    metadata = os.fstat(directory_fd)
    if not stat.S_ISDIR(metadata.st_mode):
        raise OSError(errno.ESTALE, "physical file parent is not a directory")
    return metadata.st_dev, metadata.st_ino


def _directory_identity_at(path: Path) -> tuple[int, int]:
    metadata = os.stat(path, follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise OSError(errno.ESTALE, "physical file parent is not a directory", str(path))
    return metadata.st_dev, metadata.st_ino


def _metadata(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _digest_metadata(value: PhysicalFileDigest) -> tuple[int, int, int, int, int, int]:
    return value.device, value.inode, value.mode, value.size_bytes, value.mtime_ns, value.ctime_ns


__all__ = [
    "PhysicalFileDigest",
    "open_physical_identity",
    "require_physical_identity",
    "validate_physical_identity_path",
    "physical_identity_operation",
    "physical_file_sha256",
]

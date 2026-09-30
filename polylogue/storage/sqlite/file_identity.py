"""Pinned SQLite identities whose lifecycle preserves the caller's POSIX locks."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import selectors
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from polylogue.core.compute_cancel import compute_cancel_requested


def _directory_path(descriptor: int, *, native: bool = False) -> Path:
    metadata = os.fstat(descriptor)
    request = getattr(fcntl, "F_GETPATH", None)
    candidates: list[Path] = []
    if isinstance(request, int):
        raw = fcntl.fcntl(descriptor, request, bytes(1024))
        if isinstance(raw, bytes):
            candidates.append(Path(os.fsdecode(raw.split(bytes(1), 1)[0])))
    for base in ("/dev/fd", "/proc/self/fd"):
        alias = Path(base) / str(descriptor)
        if native:
            try:
                candidates.append(Path(os.readlink(alias)))
            except OSError:
                continue
        else:
            candidates.append(alias)
    for candidate in candidates:
        try:
            current = candidate.stat()
        except OSError:
            continue
        if (current.st_dev, current.st_ino) == (metadata.st_dev, metadata.st_ino):
            return candidate
    raise OSError(errno.ENOTSUP, "no verified path for pinned SQLite directory")


def _decode_stat(payload: dict[str, Any]) -> os.stat_result:
    return os.stat_result(payload["fields"], payload["extra"])


class _Custodian:
    """A child owns every ordinary data descriptor, including its final close."""

    def __init__(self, directory: int, name: str, *, identity_fd: int | None = None) -> None:
        command = [sys.executable, "-m", "polylogue.storage.sqlite.identity_custodian", str(directory), name]
        inherited = [directory]
        if identity_fd is not None:
            command.append(str(identity_fd))
            inherited.append(identity_fd)
        self.child = subprocess.Popen(command, pass_fds=tuple(inherited), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._buffer = bytearray()
        self._lock = threading.Lock()
        try:
            self.metadata = _decode_stat(self._receive()["stat"])
        except BaseException:
            self.close()
            raise

    def _receive(self) -> dict[str, Any]:
        assert self.child.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(self.child.stdout, selectors.EVENT_READ)
            while True:
                if compute_cancel_requested():
                    self.close()
                    raise OSError(errno.ECANCELED, "SQLite identity operation cancelled")
                if b"\n" not in self._buffer:
                    if not selector.select(0.1):
                        continue
                    chunk = os.read(self.child.stdout.fileno(), 65536)
                    if not chunk:
                        raise OSError(errno.EIO, "SQLite identity custodian exited without a result")
                    self._buffer.extend(chunk)
                    continue
                raw, _, rest = self._buffer.partition(b"\n")
                self._buffer = bytearray(rest)
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise OSError(errno.EIO, "invalid SQLite identity result")
                if "error" in message:
                    raise OSError(message["errno"], message["error"])
                if "progress_bytes" not in message:
                    return message

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.child.poll() is not None:
                raise OSError(errno.EBADF, "SQLite identity custodian is closed")
            assert self.child.stdin is not None
            self.child.stdin.write(json.dumps(payload).encode() + b"\n")
            self.child.stdin.flush()
            return self._receive()

    def close(self) -> None:
        if self.child.poll() is None:
            self.child.kill()
        self.child.wait()
        for pipe in (self.child.stdin, self.child.stdout):
            if pipe is not None:
                pipe.close()


class SQLiteFileIdentity:
    """One inode and directory pin, with lock-safe metadata and byte custody.

    Linux retains an O_PATH inode descriptor. Where that facility is absent,
    the custodian child retains an ordinary no-follow descriptor; none of its
    data descriptors cross into this process. The parent retains only a safe
    directory descriptor and this identity handle. Path admission is checked
    against the pinned inode before and after SQLite opens it.
    """

    def __init__(self, path: str | Path, *, dir_fd: int | None = None, use_custodian: bool = False) -> None:
        selected = Path(path)
        self.name = selected.name
        self._directory = (
            os.dup(dir_fd)
            if dir_fd is not None
            else os.open(selected.absolute().parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        )
        self._descriptor: int | None = None
        self._custodian: _Custodian | None = None
        self._closed = False
        try:
            if dir_fd is not None and str(selected) != selected.name:
                raise ValueError("SQLite identity leaf must be relative to its pinned directory")
            expected = os.stat(self.name, dir_fd=self._directory, follow_symlinks=False)
            path_flag = getattr(os, "O_PATH", None)
            if path_flag is not None and not use_custodian:
                self._descriptor = os.open(self.name, path_flag | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=self._directory)
                metadata = os.fstat(self._descriptor)
            else:
                self._custodian = _Custodian(self._directory, self.name)
                metadata = self._custodian.metadata
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError(errno.EINVAL, "SQLite identity requires a regular file", str(path))
            self._identity = (metadata.st_dev, metadata.st_ino)
            if self._identity != (expected.st_dev, expected.st_ino):
                raise OSError(errno.ESTALE, "SQLite file changed during identity admission", str(path))
            self.assert_unchanged()
        except BaseException:
            self.close()
            raise

    def stat(self) -> os.stat_result:
        self._require_open()
        if self._descriptor is not None:
            return os.fstat(self._descriptor)
        assert self._custodian is not None
        return _decode_stat(self._custodian.request({"operation": "stat"})["stat"])

    def _require_open(self) -> None:
        if self._closed:
            raise OSError(errno.EBADF, "SQLite identity is closed")

    def assert_unchanged(self) -> None:
        self._require_open()
        current = os.stat(self.name, dir_fd=self._directory, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != self._identity:
            raise OSError(errno.ESTALE, "SQLite identity path was replaced", self.name)

    def sqlite_path(self) -> Path:
        self.assert_unchanged()
        if self._descriptor is not None:
            for base in ("/dev/fd", "/proc/self/fd"):
                alias = Path(base) / str(self._descriptor)
                try:
                    current = alias.stat()
                except OSError:
                    continue
                if (current.st_dev, current.st_ino) == self._identity:
                    return alias
        candidate = _directory_path(self._directory) / self.name
        current = candidate.stat(follow_symlinks=False)
        if (current.st_dev, current.st_ino) != self._identity:
            raise OSError(errno.ESTALE, "SQLite directory path was substituted", self.name)
        return candidate

    def physical_read(
        self, *, copy_to: Path | None, copy_exclusive: bool, copy_directory_fd: int | None
    ) -> dict[str, Any]:
        self._require_open()
        own_worker = self._custodian is None
        worker = self._custodian or _Custodian(self._directory, self.name, identity_fd=self._descriptor)
        destination_directory = copy_directory_fd
        owns_destination_directory = False
        try:
            payload: dict[str, Any] = {"operation": "read"}
            if copy_to is not None:
                # Existing custody children cannot inherit a later descriptor.
                # Resolve the pinned directory and verify its identity in the
                # child before admitting its named copy leaf.
                if destination_directory is None:
                    destination_directory = os.open(copy_to.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
                    owns_destination_directory = True
                metadata = os.fstat(destination_directory)
                payload.update(
                    destination=str(_directory_path(destination_directory, native=True)),
                    name=copy_to.name,
                    directory_identity=[metadata.st_dev, metadata.st_ino],
                    exclusive=copy_exclusive,
                )
            return worker.request(payload)
        finally:
            if own_worker:
                worker.close()
            if owns_destination_directory and destination_directory is not None:
                os.close(destination_directory)

    def chmod(self, mode: int) -> None:
        self._require_open()
        worker = self._custodian or _Custodian(self._directory, self.name, identity_fd=self._descriptor)
        try:
            worker.request({"operation": "chmod", "mode": mode})
        finally:
            if self._custodian is None:
                worker.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._custodian is not None:
                self._custodian.close()
            if self._descriptor is not None:
                os.close(self._descriptor)
        finally:
            os.close(self._directory)

    def __enter__(self) -> SQLiteFileIdentity:
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()


def open_sqlite_identity(path: str | Path, *, dir_fd: int | None = None) -> SQLiteFileIdentity:
    return SQLiteFileIdentity(path, dir_fd=dir_fd)


def require_sqlite_identity(identity: SQLiteFileIdentity) -> None:
    if not isinstance(identity, SQLiteFileIdentity):
        raise ValueError("descriptor-bound SQLite access requires a SQLiteFileIdentity")
    identity._require_open()

"""Child-only ordinary descriptors for SQLite inode metadata and physical bytes."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast


def _stat_payload(descriptor: int) -> dict[str, Any]:
    metadata = os.fstat(descriptor)
    # The stdlib reduction separates positional fields from named extras.
    # Sending all st_* names duplicates positional fields in Python 3.14.
    fields, extra = cast(tuple[tuple[int, ...], dict[str, Any]], metadata.__reduce__()[1])
    return {"fields": list(fields), "extra": extra}


def _send(payload: dict[str, Any]) -> None:
    print(json.dumps(payload), flush=True)


def _open_source(directory: int, name: str, identity_fd: int | None) -> int:
    if identity_fd is not None:
        expected = os.fstat(identity_fd)
        for base in ("/dev/fd", "/proc/self/fd"):
            try:
                descriptor = os.open(f"{base}/{identity_fd}", os.O_RDONLY | os.O_NONBLOCK)
            except OSError:
                continue
            if (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino) == (expected.st_dev, expected.st_ino):
                return descriptor
            os.close(descriptor)
        raise OSError(errno.ENOTSUP, "no byte alias for Linux identity descriptor")
    descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise OSError(errno.EINVAL, "SQLite identity requires a regular file")
    return descriptor


def _copy_metadata(source: int, destination: int) -> None:
    metadata = os.fstat(source)
    os.utime(destination, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    if hasattr(os, "listxattr"):
        try:
            names = os.listxattr(source)
        except OSError as exc:
            if exc.errno not in (errno.ENOTSUP, errno.ENODATA, errno.EINVAL):
                raise
        else:
            for name in names:
                try:
                    os.setxattr(destination, name, os.getxattr(source, name))
                except OSError as exc:
                    if exc.errno not in (errno.ENOTSUP, errno.ENODATA, errno.EINVAL, errno.EPERM):
                        raise
    os.fchmod(destination, stat.S_IMODE(metadata.st_mode))
    flags = getattr(metadata, "st_flags", None)
    if flags is not None and hasattr(os, "chflags"):
        import fcntl

        request = getattr(fcntl, "F_GETPATH", None)
        if request is None:
            raise OSError(errno.ENOTSUP, "cannot preserve physical copy flags")
        raw = fcntl.fcntl(destination, request, bytes(1024))
        path = os.fsdecode(raw.split(bytes(1), 1)[0])
        current = os.stat(path, follow_symlinks=False)
        pinned = os.fstat(destination)
        if (current.st_dev, current.st_ino) != (pinned.st_dev, pinned.st_ino):
            raise OSError(errno.ESTALE, "physical copy path was replaced")
        os.chflags(path, flags, follow_symlinks=False)


def _open_copy_destination(directory: int, name: str, *, exclusive: bool) -> int:
    """Create only a no-follow regular leaf under the verified copy directory."""
    if not name or Path(name).name != name or name in (".", ".."):
        raise OSError(errno.EINVAL, "invalid SQLite copy destination leaf")
    output = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | (os.O_EXCL if exclusive else os.O_TRUNC),
        0o600,
        dir_fd=directory,
    )
    try:
        if not stat.S_ISREG(os.fstat(output).st_mode):
            raise OSError(errno.EINVAL, "SQLite copy destination requires a regular file")
        return output
    except BaseException:
        os.close(output)
        raise


def _read(descriptor: int, request: dict[str, Any]) -> dict[str, Any]:
    with ExitStack() as cleanup:
        writer = None
        output: int | None = None
        if "destination" in request:
            name = request["name"]
            if not name or Path(name).name != name or name in (".", ".."):
                raise OSError(errno.EINVAL, "invalid SQLite copy destination leaf")
            directory = os.open(request["destination"], os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            cleanup.callback(os.close, directory)
            metadata = os.fstat(directory)
            if [metadata.st_dev, metadata.st_ino] != request["directory_identity"]:
                raise OSError(errno.ESTALE, "SQLite copy directory was replaced")
            output = _open_copy_destination(directory, name, exclusive=request["exclusive"])
            cleanup.callback(os.close, output)
            writer = os.fdopen(os.dup(output), "wb")
            cleanup.enter_context(writer)
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            if writer is not None:
                writer.write(chunk)
            _send({"progress_bytes": size})
        if writer is not None:
            writer.flush()
            assert output is not None
            _copy_metadata(descriptor, output)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def main() -> None:
    descriptor = _open_source(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) == 4 else None)
    try:
        _send({"stat": _stat_payload(descriptor)})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                result: dict[str, Any]
                if request["operation"] == "stat":
                    result = {"stat": _stat_payload(descriptor)}
                elif request["operation"] == "chmod":
                    os.fchmod(descriptor, request["mode"])
                    result = {"changed": True}
                elif request["operation"] == "read":
                    result = _read(descriptor, request)
                else:
                    raise OSError(errno.EINVAL, "unknown SQLite identity operation")
                _send(result)
            except OSError as exc:
                _send({"errno": exc.errno or errno.EIO, "error": str(exc)})
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    try:
        main()
    except OSError as exc:
        _send({"errno": exc.errno or errno.EIO, "error": str(exc)})
        raise SystemExit(1) from exc

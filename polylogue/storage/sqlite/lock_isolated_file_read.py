"""Physical SQLite reads in a process that cannot release the caller's locks."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import selectors
import shutil
import stat
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.compute_cancel import compute_cancel_requested
from polylogue.storage.sqlite.file_identity import open_sqlite_identity_descriptor, require_sqlite_identity_descriptor


@dataclass(frozen=True, slots=True)
class SQLiteFileRead:
    sha256: str
    size_bytes: int
    metadata: os.stat_result


def read_sqlite_file_in_lock_isolated_process(
    path: Path,
    *,
    copy_to: Path | None = None,
    opened_identity_fd: int | None = None,
    copy_directory_fd: int | None = None,
    copy_exclusive: bool = False,
) -> SQLiteFileRead:
    """Hash, and optionally copy, one pinned file without closing a data fd here.

    The caller owns any transaction or exclusion needed to stabilize these
    physical bytes. This helper establishes neither; observational callers
    must detect changes around the read before treating it as stable proof.
    Its O_PATH descriptor binds the child to that inode even if the pathname
    changes. Ordinary byte-reader descriptors belong solely to the child, whose
    close cannot release the parent's SQLite POSIX locks. Progress messages
    keep the pipe drained; the poll interval only admits owner cancellation.
    The caller owns cleanup of an incomplete copy destination after failure.
    """
    if compute_cancel_requested():
        raise OSError(errno.ECANCELED, "SQLite physical file read cancelled")
    descriptor = open_sqlite_identity_descriptor(path) if opened_identity_fd is None else opened_identity_fd
    destination_directory = copy_directory_fd
    owns_destination_directory = False
    try:
        require_sqlite_identity_descriptor(descriptor)
        metadata = os.fstat(descriptor)
        command = [sys.executable, "-m", __name__, str(descriptor)]
        inherited = [descriptor]
        if copy_to is not None:
            if destination_directory is None:
                destination_directory = os.open(copy_to.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
                owns_destination_directory = True
            inherited.append(destination_directory)
            command.extend((str(destination_directory), copy_to.name, "exclusive" if copy_exclusive else "replace"))
        with subprocess.Popen(command, pass_fds=tuple(inherited), stdout=subprocess.PIPE, text=True) as child:
            assert child.stdout is not None
            result: dict[str, object] | None = None
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(child.stdout, selectors.EVENT_READ)
                    while True:
                        if compute_cancel_requested():
                            raise OSError(errno.ECANCELED, "SQLite physical file read cancelled")
                        if not selector.select(0.1):
                            continue
                        line = child.stdout.readline()
                        if not line:
                            break
                        message = json.loads(line)
                        if not isinstance(message, dict):
                            raise OSError(errno.EIO, "invalid SQLite physical file read result")
                        if "progress_bytes" not in message:
                            result = message
                status = child.wait()
                if result is not None and "error" in result:
                    raise OSError(int(str(result["errno"])), str(result["error"]))
                if status != 0 or result is None:
                    raise OSError(errno.EIO, f"SQLite physical file reader exited without a fingerprint ({status})")
                digest = str(result["sha256"])
                size = int(str(result["size_bytes"]))
                return SQLiteFileRead(digest, size, metadata)
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()
    finally:
        if owns_destination_directory and destination_directory is not None:
            os.close(destination_directory)
        if opened_identity_fd is None:
            os.close(descriptor)


def _open_copy_destination(directory: int, name: str, *, exclusive: bool) -> int:
    """Admit only the named regular copy leaf in the caller's pinned directory."""
    if not name or Path(name).name != name or name in (".", ".."):
        raise OSError(errno.EINVAL, "invalid SQLite copy destination leaf")
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC) | os.O_NOFOLLOW | os.O_NONBLOCK,
        0o600,
        dir_fd=directory,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(errno.EINVAL, "SQLite copy destination requires a regular file")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_in_child(
    descriptor: int, destination_directory: int | None, destination_name: str | None, exclusive: bool
) -> None:
    """Stream one physical image. No SQLite connection is opened in this child."""
    digest = hashlib.sha256()
    size = 0
    source = Path("/proc/self/fd") / str(descriptor)
    destination = (
        Path("/proc/self/fd") / str(destination_directory) / destination_name
        if destination_directory is not None and destination_name is not None
        else None
    )
    with ExitStack() as cleanup:
        reader = cleanup.enter_context(source.open("rb"))
        if destination is not None and destination_name is not None and destination_directory is not None:
            destination_fd = _open_copy_destination(destination_directory, destination_name, exclusive=exclusive)
            writer = cleanup.enter_context(os.fdopen(destination_fd, "wb"))
        else:
            writer = None
        while chunk := reader.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            if writer is not None:
                writer.write(chunk)
            print(json.dumps({"progress_bytes": size}), flush=True)
    if destination is not None:
        shutil.copystat(source, destination)
    print(json.dumps({"sha256": digest.hexdigest(), "size_bytes": size}), flush=True)


if __name__ == "__main__":
    try:
        _read_in_child(
            int(sys.argv[1]),
            int(sys.argv[2]) if len(sys.argv) == 5 else None,
            sys.argv[3] if len(sys.argv) == 5 else None,
            len(sys.argv) == 5 and sys.argv[4] == "exclusive",
        )
    except OSError as exc:
        print(json.dumps({"errno": exc.errno or errno.EIO, "error": str(exc)}), flush=True)
        raise SystemExit(1) from exc

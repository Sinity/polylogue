"""File identity descriptors that leave SQLite's process-wide locks intact."""

from __future__ import annotations

import errno
import fcntl
import os
import stat
from pathlib import Path


def open_sqlite_identity_descriptor(path: str | Path, *, dir_fd: int | None = None) -> int:
    """Pin an inode without opening a second SQLite data-file description.

    Closing any ordinary descriptor for a SQLite inode releases this process's
    POSIX locks on that inode, including locks held by other SQLite connections.
    Linux O_PATH descriptors retain no-follow/fstat identity checks without that
    close side effect. A platform without this facility cannot provide this
    descriptor-bound contract.
    """
    path_flag = getattr(os, "O_PATH", None)
    if path_flag is None:
        raise OSError(errno.ENOTSUP, "SQLite inode verification requires O_PATH", str(path))
    descriptor = os.open(path, path_flag | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=dir_fd)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(errno.EINVAL, "SQLite identity descriptor requires a regular file", str(path))
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def require_sqlite_identity_descriptor(descriptor: int) -> None:
    """Do not admit a caller-owned descriptor whose later close drops locks."""
    path_flag = getattr(os, "O_PATH", None)
    if path_flag is None or not fcntl.fcntl(descriptor, fcntl.F_GETFL) & path_flag:
        raise ValueError("descriptor-bound SQLite access requires an O_PATH identity descriptor")

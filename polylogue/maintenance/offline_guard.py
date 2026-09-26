"""Guards for offline maintenance that must not race a live daemon."""

from __future__ import annotations

import fcntl
import os
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polylogue.config import Config

if TYPE_CHECKING:
    from polylogue.storage.archive_identity import OwnedArchiveLocation


class DaemonResidencyUndecidableError(RuntimeError):
    """This platform cannot prove whether a live daemon owns an archive.

    The residency probe below is the only fact the CLI write boundary arms
    on. A platform that cannot answer it must make the boundary *refuse*, not
    disappear: an unguarded durable write beside a live daemon is exactly the
    outcome the boundary exists to prevent, so "unknown" is raised rather than
    folded into "no daemon is running".
    """

    code = "daemon_residency_undecidable"


class ArchiveWriterOwnershipError(RuntimeError):
    """An archive write was refused because a resident daemon owns the archive.

    Raised by the CLI writer-ownership boundary
    (:mod:`polylogue.cli.write_authority`) and by the backup snapshot route
    (:func:`polylogue.daemon.backup.backup_archive`), which is a writer in its
    own right. It is defined here, beside the residency probe that decides it,
    so a non-CLI entry point can raise the same refusal without a
    ``daemon -> cli`` import edge (polylogue-8qm4k AC1).

    ``archive_root`` and ``resident_writer`` travel as attributes as well as
    inside the message. A ``--format json`` client that wants to route the
    write to the right daemon needs the archive it must reach as a field, not
    as prose it has to parse back out -- the same reason
    :class:`~polylogue.cli.shared.helper_support.DaemonRequiredError` carries
    its operation and archive root (polylogue-re6s3 AC4).
    """

    code = "archive_writer_ownership_unavailable"

    def __init__(
        self,
        message: str,
        *,
        archive_root: object = None,
        resident_writer: str | None = None,
    ) -> None:
        self.archive_root = None if archive_root is None else str(archive_root)
        self.resident_writer = resident_writer
        super().__init__(message)


class ArchiveWriterOwnershipUndecidableError(ArchiveWriterOwnershipError):
    """An archive write was refused because ownership could not be proven at all.

    The boundary fails **closed**. A platform that cannot answer "does a live
    daemon own this archive?" gets a loud refusal naming the reason, never a
    silently disarmed boundary. Before this existed the probe read
    ``/proc/<pid>/cmdline`` and swallowed ``OSError``, so on macOS -- a
    supported install target (``docs/installation.md``) -- every pid answered
    "no daemon" and the boundary armed nothing on every single invocation.
    """

    code = "archive_writer_ownership_undecidable"


def _pidfile_holder_is_live(pidfile: Path) -> bool:
    """Whether a live process holds the daemon's exclusive lock on ``pidfile``.

    ``polylogued run`` takes ``fcntl.flock(fd, LOCK_EX)`` on its pidfile in
    :func:`polylogue.daemon.cli._acquire_pidfile` and holds it for the whole
    run, so a *failed* non-blocking shared acquisition here is the daemon's own
    ownership token, observed rather than inferred. The kernel releases it when
    the holder's last descriptor closes, which process exit does, so a pidfile
    left behind by a crashed daemon answers "not held" without a liveness
    heuristic of its own.

    This replaced reading ``/proc/<pid>/cmdline``. That probe was not portable:
    ``/proc`` does not exist on macOS, a supported install target
    (``docs/installation.md``), so the read raised ``OSError`` for every pid
    and the boundary silently concluded that no daemon was running. It was also
    weaker where it did work -- a recycled pid whose command line happens to
    contain ``polylogued`` is not evidence that *this* archive is owned.
    """
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - POSIX-only build target
        raise DaemonResidencyUndecidableError(
            "this platform provides no fcntl.flock, so archive writer residency cannot be proven"
        ) from exc
    try:
        fd = os.open(pidfile, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise DaemonResidencyUndecidableError(
            f"cannot read the daemon pidfile to prove archive residency: {exc}"
        ) from exc
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError as exc:
            raise DaemonResidencyUndecidableError(f"cannot probe the daemon pidfile lock: {exc}") from exc
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def resident_daemon_pid(archive_root: Path) -> int | None:
    """Return a live polylogued PID owning ``archive_root``, if one is present.

    The root-shaped probe, so a caller that has only resolved an archive root
    -- the CLI writer-ownership boundary does exactly that, before any config
    object exists -- asks the same question as the config-shaped callers
    rather than reimplementing the pidfile contract.

    Ownership is decided by the pidfile lock, not by the recorded number: the
    pid is read only to *name* the holder. A pidfile whose contents are
    unreadable while the lock is held still reports a resident daemon (with
    pid ``-1``) rather than reporting none, because the lock is the fact that
    decides whether this process may write.

    Raises :class:`DaemonResidencyUndecidableError` where the question cannot be
    answered at all; a caller that arms a write boundary on the answer must
    refuse rather than treat the refusal as an absent daemon.
    """
    pidfile = archive_root / "daemon.pid"
    if not _pidfile_holder_is_live(pidfile):
        return None
    try:
        return int(pidfile.read_text().strip())
    except (OSError, ValueError):
        return -1


def running_daemon_pid(config: Config) -> int | None:
    """Return a live polylogued PID for this archive, if one is present."""
    return resident_daemon_pid(config.archive_root)


@contextmanager
def scoped_offline_archive_writer(archive_root: Path, *, owner_id: str) -> Iterator[OwnedArchiveLocation]:
    """Exclude daemon startup and other offline writers across one operation."""
    from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation

    root = archive_root.expanduser().resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(root / "daemon.pid", os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            pid = resident_daemon_pid(root)
            writer = f"polylogued PID {pid}" if pid is not None else "resident daemon"
            raise ArchiveWriterOwnershipError(
                f"{writer} owns {root}; submit the operation to that daemon",
                archive_root=root,
                resident_writer=writer,
            ) from exc
        with OwnedArchiveLocation.acquire(ArchiveLocation.resolve(root), owner_id=owner_id) as owner:
            yield owner
    finally:
        os.close(fd)


def offline_writer_block_reason(config: Config) -> str | None:
    """Return the concrete writer that makes a strictly offline operation unsafe."""
    from polylogue.daemon.write_coordinator import daemon_write_lease_active

    if daemon_write_lease_active():
        return "a daemon writer lease is active"
    daemon_pid = running_daemon_pid(config)
    if daemon_pid is not None:
        return f"live pidfile PID {daemon_pid} is running"
    return None


def offline_maintenance_block_reason(
    config: Config,
    *,
    active: bool,
    dry_run: bool,
) -> str | None:
    """Return a refusal reason when offline maintenance would race the daemon."""
    if dry_run or not active:
        return None
    # A daemon-owned writer is already serialized against every other archive
    # mutation.  Treat it as the online equivalent of the offline exclusion
    # boundary instead of rejecting the daemon's own convergence work.
    from polylogue.daemon.write_coordinator import daemon_write_lease_active

    if daemon_write_lease_active():
        return None
    daemon_pid = running_daemon_pid(config)
    if daemon_pid is None:
        return None
    return (
        f"Refusing offline maintenance while polylogued PID {daemon_pid} is running. "
        "Stop polylogued to run this operation offline, or let daemon convergence drain live work."
    )


_INTERCEPT_LOCK = threading.Lock()
_INTERCEPT_DEPTH = 0
_ORIGINAL_CONNECT: Callable[..., sqlite3.Connection] | None = None
_REFUSE: Callable[[Path], None] | None = None


def _residency_checked_connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
    original = _ORIGINAL_CONNECT
    assert original is not None
    refuse = _REFUSE
    if refuse is not None:
        from polylogue.storage.sqlite.write_guard import guarded_archive_tier_path

        path = guarded_archive_tier_path(database, uri=bool(kwargs.get("uri", False)))
        if path is not None:
            refuse(path)
    return original(database, *args, **kwargs)


@contextmanager
def refuse_writable_tier_opens(refuse: Callable[[Path], None]) -> Iterator[None]:
    """Call ``refuse`` before every writable archive-tier open in this process.

    The seam a one-shot writer needs to re-ask "does a daemon own this archive
    *now*?" at each mutation rather than once at entry. It lives here rather
    than in the CLI because deciding *which* opens are writable archive-tier
    opens is storage's definition
    (:func:`~polylogue.storage.sqlite.write_guard.guarded_archive_tier_path`),
    and restating it beside the caller would fork a load-bearing rule; the
    surface layering ratchet also forbids a fresh ``cli -> storage`` edge.

    Process-wide rather than thread-local, because the CLI writes from
    ``asyncio`` tasks and worker threads: a thread-local boundary would leave
    exactly those writes unchecked.

    This is *not* the write lease and does not replace
    :func:`~polylogue.storage.sqlite.write_guard.install_archive_write_guard`.
    It asks one question -- is someone else the archive's writer right now --
    and answers only with the caller's refusal.
    """
    global _INTERCEPT_DEPTH, _ORIGINAL_CONNECT, _REFUSE
    with _INTERCEPT_LOCK:
        if _INTERCEPT_DEPTH == 0:
            _ORIGINAL_CONNECT = sqlite3.connect
            _REFUSE = refuse
            sqlite3.connect = _residency_checked_connect  # type: ignore[assignment]
        _INTERCEPT_DEPTH += 1
    try:
        yield
    finally:
        with _INTERCEPT_LOCK:
            _INTERCEPT_DEPTH -= 1
            if _INTERCEPT_DEPTH == 0 and _ORIGINAL_CONNECT is not None:
                sqlite3.connect = _ORIGINAL_CONNECT  # type: ignore[assignment]
                _ORIGINAL_CONNECT = None
                _REFUSE = None


def writable_tier_opens_are_checked() -> bool:
    """Whether a later-arriving writer would be noticed at the next open."""
    return _INTERCEPT_DEPTH > 0


__all__ = [
    "DaemonResidencyUndecidableError",
    "offline_maintenance_block_reason",
    "offline_writer_block_reason",
    "refuse_writable_tier_opens",
    "resident_daemon_pid",
    "running_daemon_pid",
    "scoped_offline_archive_writer",
    "writable_tier_opens_are_checked",
]

"""The process's sole authority for opening a write-mode archive connection.

An outer root-bound lease holds cross-process archive custody before writable
SQL; this module also makes the in-process boundary structural. Where enforcement is armed, a
write-mode connection is obtainable only from a held :class:`WriteLease`, so an
unserialized writer raises instead of contending through the busy timeout.

Enforcement is armed by the owner of the process's writer discipline (the
daemon) and is off elsewhere. One-shot CLI and API mutation owners still acquire
the same physical custody through their root-bound write scope.

**Contexts are not a thread boundary** (polylogue-1oa7o). Whether a new
``threading.Thread`` starts with a copy of its creator's context depends on
the interpreter: ``sys.flags.thread_inherit_context`` is on by default on
free-threading builds and off on GIL builds, and either can be overridden
with ``-X thread_inherit_context``. Where it is on, every thread spawned while
a lease is held sees ``_ACTIVE`` -- the lease object itself, not a copy;
where it is off, the thread sees the default ``None``. ``ThreadPoolExecutor``
workers are started on demand and may be reused across submissions, so their
ambient context is not a reliable signal either way. Authority therefore
rests on explicit thread identity in both modes: ``WriteLease.bound_thread_ids``
(widened only by a single-use owner grant) and the ``owner_task`` check in
:func:`require_write_lease`, never on whether the context carried the lease.
Every place that widens ``bound_thread_ids`` or hands back an existing lease
re-checks that identity, because an inheriting thread would otherwise pass by
default. Physical-custody context is thread-local and additionally bound to
the exact owning task and thread; it cannot be borrowed merely because a
child execution inherited the ambient context.
"""

from __future__ import annotations

import asyncio
import contextvars
import errno
import fcntl
import os
import stat
import sys
import threading
import time
import weakref
from builtins import BaseExceptionGroup
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import BuiltinFunctionType
from typing import TYPE_CHECKING, Any, Protocol

from polylogue.core.sql_settlement import (
    SQLCustodyOwner,
    SQLSettlementRetry,
    capture_native_sql_owners,
    settle_native_sql,
)
from polylogue.logging import WARNING, emit, get_logger

if TYPE_CHECKING:
    import sqlite3


class KnownTierWriteAuthority(Protocol):
    @property
    def terminal_parent(self) -> SQLCustodyOwner: ...

    @property
    def tier(self) -> str: ...

    def configure_mutation_connection(self, connection: sqlite3.Connection, statements: tuple[str, ...]) -> None: ...

    def authorize_tier_sql(
        self,
        connection: sqlite3.Connection,
        action: int,
        first: str | None,
        second: str | None,
        schema: str | None,
        trigger: str | None,
    ) -> bool: ...


ARCHIVE_WRITE_CUSTODY_LOCK_NAME = ".archive-write-custody.lock"

__all__ = [
    "ARCHIVE_WRITE_CUSTODY_LOCK_NAME",
    "UnleasedWriteError",
    "WriteLease",
    "WriteLeaseDelegation",
    "WriteLeaseThreadGrant",
    "ArchiveWriteCustody",
    "ArchiveCustodySettlementError",
    "adopt_write_lease",
    "archive_write_custody",
    "async_write_lease",
    "arm_write_lease_enforcement",
    "bind_write_lease_thread",
    "current_write_lease",
    "delegate_write_lease",
    "grant_write_lease_thread",
    "require_write_lease",
    "write_lease",
    "write_lease_enforced",
]

logger = get_logger(__name__)
_CUSTODY_REGISTRY_LOCK = threading.RLock()
_CUSTODIES: set[ArchiveWriteCustody] = set()
_FORK_ABANDONED_CUSTODIES: list[ArchiveWriteCustody] = []


def _finish_cleanup(
    message: str,
    actions: tuple[Callable[[], object], ...],
    primary: BaseException | None = None,
) -> None:
    """Attempt each owned terminal action once and preserve its actual error."""
    failures: list[BaseException] = []
    for action in actions:
        try:
            action()
        except BaseException as error:
            failures.append(error)
    if not failures:
        return
    if primary is not None:
        raise BaseExceptionGroup(message, [primary, *failures]) from primary
    if len(failures) == 1:
        raise failures[0]
    raise BaseExceptionGroup(message, failures)


class ArchiveWriteCustody:
    """One process-shared, cross-process exclusive hold for an archive root.

    The descriptor is reference-counted because adopted daemon work may still
    be settling after the lease owner has stopped accepting new work.  The
    separate lock inode does not replace the active-store shared rebuild lock.
    """

    __slots__ = (
        "archive_root",
        "path",
        "_fd",
        "_directory_fd",
        "_directory_identity",
        "_file_identity",
        "_guard",
        "_refs",
        "_owner_open",
        "_locked",
        "_sql_owners",
        "_known_tier_mutation",
        "owner_pid",
        "owner_thread",
        "owner_task",
        "_abandoned_after_fork",
        "_fork_cleanup_error",
        "_pending_descriptor_closes",
        "_descriptor_cleanup_thread",
        "_descriptor_cleanup_task",
        "settlement_retry",
        "_authorized_removals",
        "__weakref__",
    )

    def __init__(
        self,
        archive_root: Path,
        path: Path,
        fd: int,
        directory_fd: int,
        *,
        settlement_retry: SQLSettlementRetry | None = None,
    ) -> None:
        self.archive_root = archive_root
        self.path = path
        self._fd = fd
        self._directory_fd = directory_fd
        self._directory_identity: tuple[int, int] | None = None
        self._file_identity: tuple[int, int] | None = None
        self._guard = threading.Lock()
        self._refs = 1
        self._owner_open = True
        self._locked = False
        self._sql_owners: dict[int, tuple[SQLCustodyOwner, threading.Thread, asyncio.Task[Any] | None]] = {}
        self._known_tier_mutation: KnownTierWriteAuthority | None = None
        self.owner_pid = os.getpid()
        self.owner_thread: threading.Thread | None = None
        self.owner_task: asyncio.Task[Any] | None = None
        self._abandoned_after_fork = False
        self._fork_cleanup_error: BaseException | None = None
        self._pending_descriptor_closes: dict[int, BaseException] = {}
        self._descriptor_cleanup_thread: threading.Thread | None = None
        self._descriptor_cleanup_task: asyncio.Task[Any] | None = None
        self.settlement_retry = settlement_retry
        self._authorized_removals: list[tuple[str, frozenset[str], threading.Thread, object | None, bool]] = []
        _CUSTODIES.add(self)
        try:
            directory = os.fstat(directory_fd)
            self._directory_identity = (directory.st_dev, directory.st_ino)
            if fd >= 0:
                metadata = os.fstat(fd)
                self._file_identity = (metadata.st_dev, metadata.st_ino)
        except BaseException as primary:
            try:
                self.abandon_failed_acquisition()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "Archive custody acquisition and cleanup failed", [primary, cleanup_error]
                ) from primary
            raise

    @contextmanager
    def known_tier_mutation(self, permit: KnownTierWriteAuthority) -> Iterator[None]:
        """Keep one exact Source effect bound through its observer acceptance."""
        require_write_lease("known Source mutation", archive_root=self.archive_root)
        if current_sql_custody() is not self:
            raise UnleasedWriteError("known Source mutation does not own the current physical custody")
        if self._known_tier_mutation is not None:
            raise UnleasedWriteError("physical custody already holds a known Source mutation")
        self._known_tier_mutation = permit
        try:
            yield
        finally:
            if self._known_tier_mutation is not permit:
                raise UnleasedWriteError("known Source mutation authority changed during settlement")
            self._known_tier_mutation = None

    @property
    def known_tier_authority(self) -> KnownTierWriteAuthority | None:
        return self._known_tier_mutation

    def bind_owner_context(self) -> None:
        self._check_process()
        with self._guard:
            thread = threading.current_thread()
            task = _current_task()
            if not self._owner_open or self._fd < 0 or self._refs < 1:
                raise UnleasedWriteError("cannot bind an inactive archive custody owner")
            if self.owner_thread is not None and (self.owner_thread is not thread or self.owner_task is not task):
                raise UnleasedWriteError("archive write custody is already bound to another owner")
            self.owner_thread = thread
            self.owner_task = task

    def require_owner_context(self, purpose: str) -> None:
        self._check_process()
        with self._guard:
            if self.owner_thread is not threading.current_thread() or self.owner_task is not _current_task():
                raise UnleasedWriteError(f"{purpose} inherited archive custody from a different task or thread")

    def retain(self) -> None:
        self._check_process()
        with _CUSTODY_REGISTRY_LOCK, self._guard:
            if self._fd < 0 or self._refs < 1:
                raise UnleasedWriteError("archive write custody is no longer available")
            self._refs += 1

    def retain_sql_owner(self, owner: SQLCustodyOwner) -> None:
        """Keep actual unsettled SQL handles and their exclusion recoverable."""
        self._check_process()
        with _CUSTODY_REGISTRY_LOCK, self._guard:
            if self._fd < 0 or self._refs < 1:
                raise UnleasedWriteError("cannot retain SQL after archive custody settled")
            if id(owner) not in self._sql_owners:
                self._sql_owners[id(owner)] = (owner, threading.current_thread(), _current_task())
                self._refs += 1

    def retained_sql_owners_on_current_thread(self) -> tuple[SQLCustodyOwner, ...]:
        """Retain the actual handles that a terminal worker must settle."""
        self._check_process()
        with _CUSTODY_REGISTRY_LOCK, self._guard:
            return tuple(
                owner for owner, thread, _task in self._sql_owners.values() if thread is threading.current_thread()
            )

    def require_sql_owner_context(self, owner: object) -> None:
        """Authorize only the original SQL owner to settle retained handles."""
        self._check_process()
        with _CUSTODY_REGISTRY_LOCK, self._guard:
            retained = self._sql_owners.get(id(owner))
            if (
                retained is None
                or retained[0] is not owner
                or retained[1] is not threading.current_thread()
                or retained[2] is not _current_task()
            ):
                raise UnleasedWriteError("retained archive SQL belongs to another task or thread")
            if self._fd < 0 or self._refs < 1:
                raise UnleasedWriteError("retained archive SQL has lost its physical custody")

    def release_sql_owner(self, owner: object) -> None:
        self._check_process()
        with _CUSTODY_REGISTRY_LOCK, self._guard:
            if self._sql_owners.pop(id(owner), None) is None:
                if self._refs == 0:
                    self._close_terminal_descriptors()
                return
        self.release()

    def retain_owner_scope(self) -> None:
        self._check_process()
        with _CUSTODY_REGISTRY_LOCK, self._guard:
            if self._fd < 0 or self._refs < 1 or not self._owner_open:
                raise UnleasedWriteError("archive write custody owner scope is no longer available")
            if self.owner_thread is not threading.current_thread() or self.owner_task is not _current_task():
                raise UnleasedWriteError("archive write custody cannot be borrowed by an inherited task or thread")
            self._refs += 1

    def _descriptor_binding_retired(self, descriptor: int, identity: tuple[int, int] | None) -> bool:
        try:
            metadata = os.fstat(descriptor)
        except OSError as error:
            return error.errno == errno.EBADF
        # A different file proves replacement. Equal inode metadata cannot
        # establish whether the original open-file-description survived.
        return identity is not None and identity != (metadata.st_dev, metadata.st_ino)

    def _close_descriptor(self, attribute: str, identity: tuple[int, int] | None) -> None:
        descriptor = getattr(self, attribute)
        if descriptor < 0:
            return
        pending = self._pending_descriptor_closes.get(descriptor)
        if pending is not None:
            if not self._descriptor_binding_retired(descriptor, identity):
                raise ArchiveCustodySettlementError(self, pending) from pending
            self._pending_descriptor_closes.pop(descriptor)
            setattr(self, attribute, -1)
            if attribute == "_fd":
                self._locked = False
            return
        closer = os.close
        native_linux_close = (
            sys.platform == "linux"
            and isinstance(closer, BuiltinFunctionType)
            and closer.__module__ == "posix"
            and closer.__name__ == "close"
        )
        try:
            closer(descriptor)
        except BaseException as error:
            if (native_linux_close and isinstance(error, OSError)) or self._descriptor_binding_retired(
                descriptor, identity
            ):
                setattr(self, attribute, -1)
                if attribute == "_fd":
                    self._locked = False
                raise
            self._pending_descriptor_closes[descriptor] = error
            self._descriptor_cleanup_thread = threading.current_thread()
            self._descriptor_cleanup_task = _current_task()
            raise ArchiveCustodySettlementError(self, error) from error
        else:
            setattr(self, attribute, -1)
            if attribute == "_fd":
                self._locked = False

    def _close_terminal_descriptors(self) -> None:
        """Settle both bindings once, retaining every ambiguous close."""
        if self._pending_descriptor_closes and (
            self._descriptor_cleanup_thread is not threading.current_thread()
            or self._descriptor_cleanup_task is not _current_task()
        ):
            raise UnleasedWriteError("failed archive descriptor cleanup belongs to its original execution unit")
        try:
            # Closing the actual lock descriptor releases flock. Explicitly
            # unlocking first would surrender exclusion on a pre-effect close
            # failure. Fork cleanup likewise must never unlock the parent.
            _finish_cleanup(
                "Archive custody descriptor cleanup failed",
                (
                    lambda: self._close_descriptor("_fd", self._file_identity),
                    self._close_directory_descriptor,
                ),
            )
        finally:
            if self._fd < 0 and self._directory_fd < 0:
                _CUSTODIES.discard(self)
                self._descriptor_cleanup_thread = None
                self._descriptor_cleanup_task = None

    def close(self) -> None:
        """Existing custody's terminal callback for its creator census."""
        self.close_owner()

    def request_sql_settlement(self) -> None:
        """Request cleanup on this existing async acquisition's physical owner."""
        retry = self.settlement_retry
        if retry is None:
            raise UnleasedWriteError("this custody has no asynchronous acquisition task")
        retry.request()

    def abandon_failed_acquisition(self) -> None:
        """Retire a failed acquisition without admitting SQL work."""
        self._check_process()
        with _CUSTODY_REGISTRY_LOCK, self._guard:
            self._refs = 0
            self._owner_open = False
            self._close_terminal_descriptors()

    def release(self) -> None:
        self._check_process()
        with _CUSTODY_REGISTRY_LOCK, self._guard:
            if self._abandoned_after_fork:
                return
            if self._refs < 1:
                raise RuntimeError("archive write custody reference underflow")
            self._refs -= 1
            if self._refs == 0:
                self._close_terminal_descriptors()

    @property
    def directory_identity(self) -> tuple[int, int]:
        """The directory incarnation selected before physical acquisition."""
        identity = self._directory_identity
        if identity is None:
            raise UnleasedWriteError("archive custody directory identity was not captured")
        return identity

    @property
    def held(self) -> bool:
        if self.owner_pid != os.getpid():
            return False
        with _CUSTODY_REGISTRY_LOCK, self._guard:
            return self._fd >= 0 and self._locked

    def close_owner(self) -> None:
        self._check_process()
        if self._pending_descriptor_closes and (
            self._descriptor_cleanup_thread is not threading.current_thread()
            or self._descriptor_cleanup_task is not _current_task()
        ):
            raise UnleasedWriteError("failed archive descriptor cleanup belongs to its original execution unit")
        from polylogue.storage.sqlite.connection_profile import settle_cached_connections_on_current_thread

        def release_owner() -> None:
            release = False
            with _CUSTODY_REGISTRY_LOCK, self._guard:
                if not self._owner_open:
                    if self._refs == 0:
                        self._close_terminal_descriptors()
                else:
                    self._owner_open = False
                    release = True
            if release:
                self.release()

        _finish_cleanup(
            "Archive cache and custody cleanup failed",
            (lambda: settle_cached_connections_on_current_thread(self), release_owner),
        )

    def _check_process(self) -> None:
        if self.owner_pid != os.getpid():
            raise UnleasedWriteError("archive write custody cannot be used from a forked process")

    def _close_directory_descriptor(self) -> None:
        self._close_descriptor("_directory_fd", self._directory_identity)

    def assert_namespace(self) -> None:
        """Require the owned lock namespace to retain its selected inodes."""
        self._check_process()
        with _CUSTODY_REGISTRY_LOCK, self._guard:
            directory = os.stat(self.archive_root, follow_symlinks=False)
            _validate_custody_directory(directory, self.archive_root)
            if (directory.st_dev, directory.st_ino) != self._directory_identity:
                raise UnleasedWriteError("archive write custody directory was replaced")
            metadata = os.stat(self.path.name, dir_fd=self._directory_fd, follow_symlinks=False)
            _validate_custody_lock(metadata, self.path)
            if (metadata.st_dev, metadata.st_ino) != self._file_identity:
                raise UnleasedWriteError("archive write custody lock was replaced")

    def abandon_after_fork(self) -> None:
        """Close copied bindings without unlocking the parent's flock."""
        self._refs = 0
        self._owner_open = False
        self._abandoned_after_fork = True
        # The child owns its copied descriptor cleanup, never parent SQL.
        self._descriptor_cleanup_thread = threading.current_thread()
        self._descriptor_cleanup_task = _current_task()
        self._close_terminal_descriptors()


class ArchiveCustodySettlementError(RuntimeError):
    """An exact custody binding has not been proven physically retired."""

    code = "archive_custody_unsettled"
    retryable = True

    def __init__(self, owner: ArchiveWriteCustody, failure: BaseException) -> None:
        self.owner = owner
        self.failure = failure
        super().__init__("archive custody descriptor cleanup remains unresolved")


def retained_custody_settlement_owners_on_current_thread() -> tuple[ArchiveWriteCustody, ...]:
    """Extend the existing custody census with its failed terminal bindings."""
    with _CUSTODY_REGISTRY_LOCK:
        return tuple(
            custody
            for custody in _CUSTODIES
            if custody.owner_pid == os.getpid()
            and custody._pending_descriptor_closes
            and custody._descriptor_cleanup_thread is threading.current_thread()
        )


def retained_sql_owners_on_current_thread() -> tuple[SQLCustodyOwner, ...]:
    """Return actual unsettled SQL from this process's original worker thread."""
    with _CUSTODY_REGISTRY_LOCK:
        return tuple(
            owner
            for custody in tuple(_CUSTODIES)
            if custody.owner_pid == os.getpid()
            for owner in custody.retained_sql_owners_on_current_thread()
        )


def _before_fork() -> None:
    _CUSTODY_REGISTRY_LOCK.acquire()


def _after_fork_parent() -> None:
    _CUSTODY_REGISTRY_LOCK.release()


def _after_fork_child() -> None:
    try:
        for custody in tuple(_CUSTODIES):
            try:
                custody.abandon_after_fork()
            except BaseException as exc:
                # Keep the failed owner, not a second error ledger. A child
                # with uncertain inherited cleanup cannot admit new writers.
                custody._fork_cleanup_error = exc
            else:
                # Keep copied SQL owners unreachable for use, without letting
                # their SQLite finalizers run as descriptor-hook cleanup.
                _FORK_ABANDONED_CUSTODIES.append(custody)
                _CUSTODIES.discard(custody)
    finally:
        _CUSTODY_REGISTRY_LOCK.release()
        _ACTIVE.set(None)
        if hasattr(_CUSTODY_CONTEXT, "custody"):
            del _CUSTODY_CONTEXT.custody
        if hasattr(_THREAD_GRANT_CONTEXT, "grant"):
            del _THREAD_GRANT_CONTEXT.grant


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_before_fork, after_in_parent=_after_fork_parent, after_in_child=_after_fork_child)


def _validate_custody_directory(metadata: os.stat_result, path: Path) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise UnleasedWriteError(f"archive custody directory must be owned and privately writable: {path}")


def _validate_custody_lock(metadata: os.stat_result, path: Path) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise UnleasedWriteError(f"archive custody lock must be owned, single-link and privately writable: {path}")


def _acquire_archive_write_custody(
    archive_root: str | Path,
    *,
    settlement_retry: SQLSettlementRetry | None = None,
) -> ArchiveWriteCustody:
    with _CUSTODY_REGISTRY_LOCK:
        for inherited in _CUSTODIES:
            if inherited._pending_descriptor_closes and inherited.archive_root == Path(archive_root).resolve():
                raise UnleasedWriteError("archive custody requires original-owner descriptor settlement")
            if inherited._fork_cleanup_error is not None:
                raise UnleasedWriteError(
                    "inherited archive descriptor cleanup failed; a fresh process is required for writes"
                ) from inherited._fork_cleanup_error
    root = Path(archive_root).resolve()
    if not root.is_dir():
        raise UnleasedWriteError(f"archive root is not an existing directory: {root}")
    path = root / ARCHIVE_WRITE_CUSTODY_LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    before: tuple[int, int] | None
    try:
        metadata = path.stat(follow_symlinks=False)
        before = (metadata.st_dev, metadata.st_ino)
        _validate_custody_lock(metadata, path)
    except FileNotFoundError:
        before = None
    fd = -1
    directory_fd = -1
    custody: ArchiveWriteCustody | None = None
    try:
        with _CUSTODY_REGISTRY_LOCK:
            # Opening and registration are atomic with respect to fork. The
            # guard is released before flock, which may wait for another
            # process to finish its archive mutation.
            directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
            selected_directory_fd, directory_fd = directory_fd, -1
            custody = ArchiveWriteCustody(root, path, -1, selected_directory_fd, settlement_retry=settlement_retry)
            directory = os.fstat(selected_directory_fd)
            _validate_custody_directory(directory, root)
            fd = os.open(path.name, flags | os.O_CLOEXEC, 0o600, dir_fd=selected_directory_fd)
            custody._fd = fd
            opened = os.fstat(fd)
            custody._file_identity = (opened.st_dev, opened.st_ino)
            _validate_custody_lock(opened, path)
            if before is not None and (opened.st_dev, opened.st_ino) != before:
                raise UnleasedWriteError(f"archive write custody path changed while opening: {path}")
            linked = os.stat(path.name, dir_fd=selected_directory_fd, follow_symlinks=False)
            if (linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino):
                raise UnleasedWriteError(f"archive write custody path changed while opening: {path}")
        # Never hold the registry guard while waiting on another process's
        # archive lease: fork must remain possible while a writer is queued.
        fcntl.flock(fd, fcntl.LOCK_EX)
        custody._locked = True
        custody.assert_namespace()
        return custody
    except BaseException as primary:
        try:
            if custody is not None:
                custody.abandon_failed_acquisition()
            else:
                _finish_cleanup(
                    "Unregistered custody acquisition cleanup failed",
                    tuple(partial(os.close, value) for value in (fd, directory_fd) if value >= 0),
                )
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "Archive custody acquisition and cleanup failed", [primary, cleanup_error]
            ) from primary
        raise


def _acquire_async_archive_custody(
    archive_root: str | Path,
    retry: SQLSettlementRetry,
    observed_generation: int,
) -> ArchiveWriteCustody:
    from polylogue.storage.sqlite.connection_profile import retained_native_settlement_owners_on_current_thread

    # Loading the sole provider precedes both entry capture and acquisition.
    # No new registry or physical executor owns this cleanup.
    retained_native_settlement_owners_on_current_thread(None)
    entry = capture_native_sql_owners()
    try:
        return _acquire_archive_write_custody(archive_root, settlement_retry=retry)
    except BaseException as primary:
        cleanup = settle_native_sql(
            retry=retry,
            initial_observed_generation=observed_generation,
            preserved_native_owners=entry,
            on_pending=lambda evidence: emit(
                "storage.write_custody.acquisition_unsettled",
                level=WARNING,
                owner_count=evidence.owner_count,
                failure_types=evidence.failure_types,
            ),
            on_settled=lambda: None,
        )
        if cleanup is not None:
            raise BaseExceptionGroup(
                "Custody acquisition and physical settlement failed", [primary, cleanup]
            ) from primary
        raise


async def _close_async_archive_custody(
    custody: ArchiveWriteCustody, *, borrowed: bool = False, initial_observed_generation: int | None = None
) -> None:
    """Keep transferred loop-task ownership alive through failed cleanup."""
    retry = custody.settlement_retry
    observed = (
        (retry.generation() if retry is not None else 0)
        if initial_observed_generation is None
        else initial_observed_generation
    )
    cleanup_failure: BaseException | None = None
    cancellation: asyncio.CancelledError | None = None
    wait_failure: BaseException | None = None
    first = True
    while True:
        try:
            if borrowed and first:
                custody.release()
            else:
                custody.close_owner()
        except BaseException as error:
            # Preserve the first complete terminal attempt. Repeated requests
            # only recheck the same retained ambiguity, never accumulate a
            # transcript of fresh wrappers around that identical failure.
            cleanup_failure = cleanup_failure or error
        first = False
        if not custody._pending_descriptor_closes:
            break
        if retry is None:
            # Borrowed custody is physically owned by its existing outer
            # executor. Preserve that owner rather than inventing an executor.
            break
        wait = asyncio.create_task(
            asyncio.to_thread(retry.wait_after, observed), name="polylogue-writer-custody:settlement-wait"
        )
        value, interrupted, failure = await _settle_task(wait, retry=retry)
        cancellation = cancellation or interrupted
        if failure is not None:
            wait_failure = failure
            break
        observed = int(value)
    failures = tuple(error for error in (cleanup_failure, cancellation, wait_failure) if error is not None)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("Async custody cleanup failed", failures)


@contextmanager
def archive_write_custody(archive_root: str | Path) -> Iterator[ArchiveWriteCustody]:
    """Hold physical archive custody for direct synchronous mutation owners."""
    root = Path(archive_root).resolve()
    inherited = _context_custody()
    if inherited is not None:
        if inherited.archive_root != root:
            raise UnleasedWriteError("nested archive custody requested for a different archive root")
        inherited.require_owner_context("nested archive custody")
        inherited.retain_owner_scope()
        primary: BaseException | None = None
        try:
            yield inherited
        except BaseException as error:
            primary = error
            raise
        finally:
            _finish_cleanup("Borrowed archive custody and cleanup failed", (inherited.release,), primary)
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise UnleasedWriteError("synchronous archive custody cannot block an event loop; use async_write_lease")
    custody = _acquire_archive_write_custody(archive_root)
    custody.bind_owner_context()
    previous = _set_context_custody(custody)
    owned_primary: BaseException | None = None
    try:
        yield custody
    except BaseException as error:
        owned_primary = error
        raise
    finally:
        _finish_cleanup(
            "Archive custody and cleanup failed",
            (lambda: _restore_context_custody(previous), custody.close_owner),
            owned_primary,
        )


class UnleasedWriteError(RuntimeError):
    """A write-mode connection was requested without holding the write lease."""

    code = "unleased_write"


@dataclass(eq=False, slots=True)
class WriteLeaseDelegation:
    """One explicit, revocable authorization to execute work under a lease.

    A lease's ambient identity -- the ``ContextVar`` plus the bound thread set
    -- cannot survive a hand-off to an arbitrary worker thread running a
    freshly created event loop, which is exactly the shape the daemon's HTTP
    write gate uses. Widening the ambient rules until it did survive would
    authorize every thread that happens to inherit the context. A delegation
    instead carries ownership as a *value*: the holder mints one and hands it
    to the unit of work that will actually write, and only a holder of that
    object can adopt the lease.

    The sole-writer guarantee is preserved by three properties:

    * it can only be minted by a context that already passes
      :func:`require_write_lease`, so it is never an escalation;
    * at most one execution may adopt it at a time, so it cannot fan a single
      admission out into concurrent writers;
    * it is revoked when its lease is released, so a stashed delegation
      authorizes nothing afterwards.

    Revocation refuses *future* adoptions; it cannot withdraw an execution
    that already adopted the lease and may be inside a SQLite transaction.
    :meth:`retire` therefore reports whether such an execution is still in
    flight, and :attr:`settled` tells the admitting holder when it has really
    finished, so the holder can keep the single-writer gate until then
    instead of releasing it because a *caller* stopped waiting
    (polylogue-8r4zq).
    """

    actor: str
    lease: WriteLease
    _guard: threading.Lock = field(default_factory=threading.Lock)
    _adopted_by: int | None = None
    _revoked: bool = False
    _settled: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        # Never adopted is trivially settled; adoption clears it.
        self._settled.set()

    def _require_process(self) -> None:
        if self.lease.owner_pid != os.getpid():
            raise UnleasedWriteError("cannot use a writer delegation inherited across fork")

    @property
    def live(self) -> bool:
        """Whether this delegation still authorizes an adoption."""
        self._require_process()
        with self.lease._lifecycle_guard, self._guard:
            return self.lease._active and not self._revoked

    @property
    def adopted(self) -> bool:
        """Whether an execution currently holds this authorization."""
        self._require_process()
        with self._guard:
            return self._adopted_by is not None

    @property
    def settled(self) -> bool:
        """Whether no execution is currently running under this authorization."""
        self._require_process()
        return self._settled.is_set()

    def revoke(self) -> None:
        """Retire this delegation; further adoptions are refused."""
        self._require_process()
        with self._guard:
            self._revoked = True

    def retire(self) -> bool:
        """Revoke, and report whether an adopted execution is still running.

        Atomic against :func:`adopt_write_lease`: a ``False`` answer means no
        execution can ever adopt this delegation again, so the holder may
        release its gate. ``True`` means one is in flight and the holder must
        wait for :attr:`settled` before another writer is admitted.
        """
        self._require_process()
        with self._guard:
            self._revoked = True
            return self._adopted_by is not None

    def wait_until_settled(self) -> None:
        """Wait without allowing repeated interrupts to abandon live work."""
        self._require_process()
        interruption: BaseException | None = None
        while not self._settled.is_set():
            try:
                self._settled.wait()
            except BaseException as exc:
                if interruption is None:
                    interruption = exc
        if interruption is not None:
            raise interruption


@dataclass(slots=True)
class WriteLease:
    """A held authorization to open write-mode connections in this context."""

    actor: str
    acquired_at: float
    max_hold_seconds: float | None
    archive_root: Path | None = None
    custody: ArchiveWriteCustody | None = None
    owns_custody: bool = False
    owns_custody_ref: bool = False
    _custody_guard: threading.Lock = field(default_factory=threading.Lock)
    _lifecycle_guard: threading.RLock = field(default_factory=threading.RLock)
    _active: bool = True
    owner_pid: int = field(default_factory=os.getpid)
    thread_grants: list[WriteLeaseThreadGrant] = field(default_factory=list)
    coordinator: object | None = None
    owner_task: asyncio.Task[Any] | None = None
    owner_thread_id: int = 0
    owner_thread: threading.Thread = field(default_factory=threading.current_thread)
    bound_thread_ids: set[int] | None = None
    delegations: list[WriteLeaseDelegation] = field(default_factory=list)
    #: Guards ``bound_thread_ids``. On a free-threading build several
    #: inheriting threads can reach ``bind_write_lease_thread`` concurrently,
    #: and the previous check-then-assign on a bare ``set`` was unsynchronized
    #: (polylogue-1oa7o residual 4).
    _bind_guard: threading.Lock = field(default_factory=threading.Lock)
    #: The ``threading.Thread`` behind each authorized ident. OS thread idents
    #: are reused once a thread exits, so an ident alone would admit a later
    #: inheriting thread that happened to receive a retired worker's ident
    #: (polylogue-1oa7o residual 3). Authority is the thread object; the ident
    #: is only its lookup key.
    _thread_refs: dict[int, weakref.ReferenceType[threading.Thread]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Leases are constructed on their owning thread.
        if self.owner_thread_id == threading.get_ident():
            self._thread_refs[self.owner_thread_id] = weakref.ref(threading.current_thread())

    def authorize_thread(self, thread_id: int) -> None:
        """Authorize the calling thread, which ``thread_id`` must identify."""
        if self.owner_pid != os.getpid():
            raise UnleasedWriteError("cannot authorize an inherited write lease")
        if thread_id != threading.get_ident():
            raise UnleasedWriteError("a write lease thread authorizes only itself")
        with self._bind_guard:
            if self.bound_thread_ids is None:
                self.bound_thread_ids = {self.owner_thread_id}
            self.bound_thread_ids.add(thread_id)
            self._thread_refs[thread_id] = weakref.ref(threading.current_thread())

    def authorized_threads(self) -> frozenset[int]:
        if self.owner_pid != os.getpid():
            raise UnleasedWriteError("cannot inspect an inherited write lease")
        with self._bind_guard:
            return frozenset(self.bound_thread_ids or {self.owner_thread_id})

    def current_thread_is_authorized(self) -> bool:
        """Whether the calling thread -- this thread object, not a reused ident -- is bound."""
        if self.owner_pid != os.getpid():
            return False
        thread_id = threading.get_ident()
        with self._bind_guard:
            if thread_id not in (self.bound_thread_ids or {self.owner_thread_id}):
                return False
            bound = self._thread_refs.get(thread_id)
        return bound is not None and bound() is threading.current_thread()

    @property
    def held_seconds(self) -> float:
        return time.perf_counter() - self.acquired_at

    @property
    def over_budget(self) -> bool:
        return self.max_hold_seconds is not None and self.held_seconds > self.max_hold_seconds

    @property
    def active(self) -> bool:
        if self.owner_pid != os.getpid():
            return False
        with self._lifecycle_guard:
            return self._active

    def require_active(self, purpose: str) -> None:
        if self.owner_pid != os.getpid():
            raise UnleasedWriteError(f"{purpose} uses a write lease inherited across fork")
        with self._lifecycle_guard:
            delegated_thread = _current_thread_grant(self)
            if not self._active and not delegated_thread:
                raise UnleasedWriteError(f"{purpose} uses a released write lease for {self.actor}")
            if self.custody is None or not self.custody.held:
                raise UnleasedWriteError(f"{purpose} uses a write lease without live archive custody")
            self.custody.assert_namespace()

    def retire(self) -> None:
        if self.owner_pid != os.getpid():
            raise UnleasedWriteError("cannot retire a write lease inherited across fork")
        with self._lifecycle_guard:
            if not self._active:
                return
            self._active = False
            grants = tuple(self.thread_grants)
            delegations = tuple(self.delegations)
        _finish_cleanup(
            "Writer grant and delegation retirement failed",
            (*tuple(grant.revoke for grant in grants), *tuple(item.revoke for item in delegations)),
        )


_ACTIVE: contextvars.ContextVar[WriteLease | None] = contextvars.ContextVar(
    "polylogue_active_write_lease", default=None
)
_CUSTODY_CONTEXT = threading.local()
_THREAD_GRANT_CONTEXT = threading.local()
_ENFORCEMENT = threading.local()
_ENFORCEMENT_DEFAULT = False
_PROCESS_ENFORCEMENT = False
_PROCESS_ENFORCEMENT_LOCK = threading.RLock()
_PROCESS_ENFORCEMENT_USERS = 0
_PROCESS_ENFORCEMENT_SUPPRESSORS = 0


def _current_task() -> asyncio.Task[Any] | None:
    """Return the exact task object, distinguishing inherited child contexts."""
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def _current_thread_grant(lease: WriteLease) -> bool:
    grant = getattr(_THREAD_GRANT_CONTEXT, "grant", None)
    return isinstance(grant, WriteLeaseThreadGrant) and grant.lease is lease and grant.live_for_current_thread()


_NO_CONTEXT_CUSTODY = object()


def _context_custody() -> ArchiveWriteCustody | None:
    custody = getattr(_CUSTODY_CONTEXT, "custody", None)
    return custody if isinstance(custody, ArchiveWriteCustody) else None


def current_sql_custody() -> ArchiveWriteCustody | None:
    """Observe this execution's admitted physical owner without minting authority."""
    lease = _ACTIVE.get()
    if lease is not None:
        if (
            lease.owner_pid == os.getpid()
            and lease._active
            and (
                _current_task() is lease.owner_task
                if _current_task() is not None
                else lease.current_thread_is_authorized()
            )
        ):
            return lease.custody
        return None
    custody = _context_custody()
    if custody is not None:
        if (
            custody.owner_pid != os.getpid()
            or custody.owner_thread is not threading.current_thread()
            or custody.owner_task is not _current_task()
        ):
            return None
        custody.require_owner_context("native SQLite custody")
    return custody


@contextmanager
def authorized_session_removal(
    *, archive_root: Path, plan_hash: str, session_ids: tuple[str, ...], excise_assertions: bool = False
) -> Iterator[None]:
    """Bind validated deletion intent to the existing physical apply custody."""
    require_write_lease("authorized session removal", archive_root=archive_root)
    custody = current_sql_custody()
    if custody is None or custody.archive_root.resolve() != archive_root.resolve():
        raise UnleasedWriteError("authorized removal requires matching physical archive custody")
    custody.assert_namespace()
    frame = (plan_hash, frozenset(session_ids), threading.current_thread(), _current_task(), excise_assertions)
    custody._authorized_removals.append(frame)
    try:
        yield
    finally:
        # Exact identity prevents nested or inherited execution from retiring
        # another physical apply's permission.
        for position in range(len(custody._authorized_removals) - 1, -1, -1):
            if custody._authorized_removals[position] is frame:
                del custody._authorized_removals[position]
                break


def permitted_session_removals(
    *, archive_root: Path, assertion_content: bool = False, plan_hash: str | None = None
) -> frozenset[str]:
    require_write_lease("observe authorized session removal", archive_root=archive_root)
    custody = current_sql_custody()
    if custody is None or custody.archive_root.resolve() != archive_root.resolve():
        return frozenset()
    custody.assert_namespace()
    for bound_hash, session_ids, thread, task, excise_assertions in reversed(custody._authorized_removals):
        if (
            thread is threading.current_thread()
            and task is _current_task()
            and (plan_hash is None or bound_hash == plan_hash)
        ):
            return session_ids if not assertion_content or excise_assertions else frozenset()
    return frozenset()


def _set_context_custody(custody: ArchiveWriteCustody) -> object:
    previous = getattr(_CUSTODY_CONTEXT, "custody", _NO_CONTEXT_CUSTODY)
    _CUSTODY_CONTEXT.custody = custody
    return previous


def _restore_context_custody(previous: object) -> None:
    if previous is _NO_CONTEXT_CUSTODY:
        if hasattr(_CUSTODY_CONTEXT, "custody"):
            del _CUSTODY_CONTEXT.custody
    else:
        _CUSTODY_CONTEXT.custody = previous


def write_lease_enforced() -> bool:
    """Whether an unleased write-mode open is an error in this thread."""
    return _PROCESS_ENFORCEMENT or bool(getattr(_ENFORCEMENT, "armed", _ENFORCEMENT_DEFAULT))


@contextmanager
def arm_write_lease_enforcement(*, armed: bool = True, process_wide: bool = False) -> Iterator[None]:
    """Make unleased write-mode opens raise for the duration of the block.

    Thread-local rather than process-global because the daemon runs its writer
    on dedicated threads while pytest and embedded callers share the process.
    The daemon opts into ``process_wide=True`` for its process-lifetime writer
    boundary; tests and one-shot callers keep the default local scope.
    """
    global _PROCESS_ENFORCEMENT, _PROCESS_ENFORCEMENT_USERS, _PROCESS_ENFORCEMENT_SUPPRESSORS
    # A process-wide arming is carried by the counters alone. Also saving and
    # restoring the thread-local flag made overlapping invocations that exit
    # out of order restore a stale ``armed=True`` and leave the thread armed.
    previous = getattr(_ENFORCEMENT, "armed", _ENFORCEMENT_DEFAULT)
    if process_wide:
        with _PROCESS_ENFORCEMENT_LOCK:
            if armed:
                _PROCESS_ENFORCEMENT_USERS += 1
            else:
                _PROCESS_ENFORCEMENT_SUPPRESSORS += 1
            _PROCESS_ENFORCEMENT = _PROCESS_ENFORCEMENT_USERS > 0 and _PROCESS_ENFORCEMENT_SUPPRESSORS == 0
    else:
        _ENFORCEMENT.armed = armed
    try:
        yield
    finally:
        if not process_wide:
            _ENFORCEMENT.armed = previous
        else:
            with _PROCESS_ENFORCEMENT_LOCK:
                if armed:
                    _PROCESS_ENFORCEMENT_USERS -= 1
                else:
                    _PROCESS_ENFORCEMENT_SUPPRESSORS -= 1
                _PROCESS_ENFORCEMENT = _PROCESS_ENFORCEMENT_USERS > 0 and _PROCESS_ENFORCEMENT_SUPPRESSORS == 0


def current_write_lease() -> WriteLease | None:
    """Return the lease held by this context, if any."""
    return _ACTIVE.get()


def coordinator_write_lease_active() -> bool:
    """Require actual task/thread and archive custody, not inherited context."""
    lease = current_write_lease()
    if lease is None or lease.coordinator is None:
        return False
    try:
        require_write_lease("coordinator lease observation", archive_root=lease.archive_root)
    except UnleasedWriteError:
        return False
    return True


def require_write_lease(purpose: str, *, archive_root: str | Path | None = None) -> WriteLease | None:
    """Assert the caller may open a write-mode connection for ``purpose``.

    Returns the held lease, or ``None`` where enforcement is not armed. Raising
    here is what makes "the daemon is the sole writer" an exception rather than
    a review rule: every write-mode factory calls this before connecting.
    """
    lease = _ACTIVE.get()
    if lease is not None:
        lease.require_active(purpose)
        task = _current_task()
        if task is not None:
            if task is not lease.owner_task:
                raise UnleasedWriteError(f"{purpose} uses a write lease inherited by a child task")
        elif not lease.current_thread_is_authorized():
            raise UnleasedWriteError(f"{purpose} uses a write lease from an unauthorized thread")
        if lease.archive_root is not None and archive_root is None:
            raise UnleasedWriteError(
                f"{purpose} omitted archive identity for writer {lease.actor} bound to {lease.archive_root}"
            )
        if archive_root is not None and lease.archive_root is not None and Path(archive_root) != lease.archive_root:
            expected = Path(archive_root).resolve()
            actual = lease.archive_root.resolve()
            if expected != actual:
                raise UnleasedWriteError(
                    f"{purpose} is outside the archive bound to writer {lease.actor}: {expected} != {actual}"
                )
        if archive_root is not None and lease.archive_root is None:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise UnleasedWriteError(
                    f"{purpose} reached an unbound synchronous lease on an event loop; "
                    "enter async_write_lease with the configured archive root first"
                )
            expected = Path(archive_root).resolve()
            with lease._custody_guard:
                if lease.archive_root is not None and lease.archive_root != expected:
                    raise UnleasedWriteError(
                        f"{purpose} is outside the archive bound to writer {lease.actor}: "
                        f"{expected} != {lease.archive_root}"
                    )
                if lease.custody is None:
                    inherited_custody = _context_custody()
                    if inherited_custody is not None and inherited_custody.archive_root == expected:
                        inherited_custody.retain_owner_scope()
                        lease.custody = inherited_custody
                        lease.owns_custody_ref = True
                    else:
                        lease.custody = _acquire_archive_write_custody(expected)
                        lease.owns_custody = True
                    lease.archive_root = expected
        return lease
    if not write_lease_enforced():
        return None
    raise UnleasedWriteError(
        f"{purpose} requires the daemon write lease; open it inside write_lease(...) "
        "so the single-writer boundary is serialized in-process"
    )


@dataclass(eq=False, slots=True)
class WriteLeaseThreadGrant:
    """One single-use authorization for *one* thread to join a held lease.

    polylogue-1oa7o residual 1: ``bind_write_lease_thread()`` used to read the
    ambient lease and add ``threading.get_ident()`` to it. On a free-threading
    build the ambient lease is inherited by *every* thread spawned during the
    hold, so that call was self-authorization -- any inheriting thread could
    bind itself into the daemon's live lease. Binding is now delegated by the
    owner rather than self-served: only a context that already passes
    :func:`require_write_lease` can mint a grant, and the spawned thread must
    present it.
    """

    lease: WriteLease
    _guard: threading.Lock = field(default_factory=threading.Lock)
    _used: bool = False
    _revoked: bool = False
    _custody_held: bool = True
    _settled: threading.Event = field(default_factory=threading.Event)
    _owner_pid: int = field(default_factory=os.getpid)
    _bound_thread: threading.Thread | None = None

    def __post_init__(self) -> None:
        self._settled.set()

    def _require_process(self) -> None:
        if self._owner_pid != os.getpid() or self.lease.owner_pid != os.getpid():
            raise UnleasedWriteError("cannot use a writer grant inherited across fork")

    def _claim(self) -> None:
        self._require_process()
        with self.lease._lifecycle_guard, self._guard:
            if (
                self._owner_pid != os.getpid()
                or self.lease.owner_pid != os.getpid()
                or not self.lease._active
                or self._revoked
                or not self._custody_held
            ):
                raise UnleasedWriteError(f"write lease thread grant for {self.lease.actor} is no longer live")
            if self._used:
                raise UnleasedWriteError(
                    f"write lease thread grant for {self.lease.actor} was already used; one grant authorizes one thread"
                )
            self._used = True
            self._bound_thread = threading.current_thread()
            self._settled.clear()

    def live_for_current_thread(self) -> bool:
        if self._owner_pid != os.getpid() or self.lease.owner_pid != os.getpid():
            return False
        with self._guard:
            return (
                self._owner_pid == os.getpid()
                and self._used
                and self._custody_held
                and self._bound_thread is threading.current_thread()
            )

    def revoke(self) -> None:
        self._require_process()
        release_custody = False
        with self._guard:
            self._revoked = True
            if not self._used and self._custody_held:
                self._custody_held = False
                release_custody = True
        if release_custody and self.lease.custody is not None:
            self.lease.custody.release()

    @property
    def custody_retired(self) -> bool:
        """Whether this grant returned its reference to the existing custody."""
        self._require_process()
        with self._guard:
            return not self._custody_held

    def complete(self) -> None:
        self._require_process()
        release_custody = False
        with self._guard:
            if self._custody_held:
                self._custody_held = False
                release_custody = True
            self._settled.set()
        if release_custody and self.lease.custody is not None:
            self.lease.custody.release()

    def wait_until_settled(self) -> None:
        self._require_process()
        interruption: BaseException | None = None
        while not self._settled.is_set():
            try:
                self._settled.wait()
            except BaseException as exc:
                if interruption is None:
                    interruption = exc
        if interruption is not None:
            raise interruption


def grant_write_lease_thread() -> WriteLeaseThreadGrant:
    """Mint a single-use authorization for one spawned thread to join this lease.

    Callable only from a context that already holds the lease -- the check runs
    through :func:`require_write_lease` -- so it can never manufacture
    authority the caller does not have. The owner mints this *before* starting
    the worker thread; the worker calls :func:`bind_write_lease_thread` with it.
    """
    active_lease = _ACTIVE.get()
    lease = require_write_lease(
        "granting a write lease thread binding",
        archive_root=active_lease.archive_root if active_lease is not None else None,
    )
    if lease is None:
        raise UnleasedWriteError(
            "cannot grant a write lease thread binding without holding the lease; "
            "mint the grant inside write_lease(...)"
        )
    if lease.custody is None or lease.archive_root is None:
        raise UnleasedWriteError("cannot grant a writer thread before archive custody is bound")
    with lease._lifecycle_guard:
        if not lease._active:
            raise UnleasedWriteError("cannot grant a writer thread from a released lease")
        lease.custody.retain()
        grant = WriteLeaseThreadGrant(lease=lease)
        lease.thread_grants.append(grant)
        return grant


def bind_write_lease_thread(grant: WriteLeaseThreadGrant) -> None:
    """Authorize the current thread for the lease ``grant`` was minted from.

    Refuses a grant whose lease is not the one this thread inherited: an
    inheriting thread must not be able to present a stale grant and join a
    different, later hold.
    """
    lease = _ACTIVE.get()
    if lease is not None and grant.lease is not lease:
        raise UnleasedWriteError(
            f"write lease thread grant for {grant.lease.actor} does not authorize the lease "
            f"held by {lease.actor} in this thread"
        )
    grant._claim()
    lease = grant.lease
    _THREAD_GRANT_CONTEXT.grant = grant
    if _ACTIVE.get() is None:
        # Without thread_inherit_context a new worker thread starts with an
        # empty context. The single-use owner grant carries the exact lease
        # and is the authority for installing it here.
        _ACTIVE.set(lease)
    lease.authorize_thread(threading.get_ident())


def delegate_write_lease() -> WriteLeaseDelegation:
    """Mint an explicit authorization for another execution unit to write.

    Callable only from a context that itself holds the lease: the ownership
    check runs through :func:`require_write_lease`, so delegation can never
    manufacture authority that the caller does not already have.
    """
    active_lease = _ACTIVE.get()
    lease = require_write_lease(
        "delegating the daemon write lease",
        archive_root=active_lease.archive_root if active_lease is not None else None,
    )
    if lease is None:
        raise UnleasedWriteError(
            "cannot delegate the write lease without holding it; mint the delegation "
            "inside write_lease(...) so the delegated work stays behind one admission"
        )
    if lease.custody is None or lease.archive_root is None:
        raise UnleasedWriteError("cannot delegate a writer before archive custody is bound")
    with lease._lifecycle_guard:
        if not lease._active:
            raise UnleasedWriteError("cannot delegate new work from a released lease")
        delegation = WriteLeaseDelegation(actor=lease.actor, lease=lease)
        lease.delegations.append(delegation)
    return delegation


@contextmanager
def adopt_write_lease(delegation: WriteLeaseDelegation) -> Iterator[WriteLease]:
    """Execute this block under the lease the ``delegation`` authorizes.

    Binds the adopting task *and* thread, so the adopted view is authorized
    exactly where it is presented and nowhere else. The hold budget stays with
    the minting lease, which is the hold that is actually being measured.
    """
    delegation._require_process()
    source = delegation.lease
    with source._lifecycle_guard, delegation._guard:
        if not source._active or delegation._revoked:
            raise UnleasedWriteError(
                f"write lease delegation for {delegation.actor} was revoked when its lease was released"
            )
        if delegation._adopted_by is not None:
            raise UnleasedWriteError(
                f"write lease delegation for {delegation.actor} is already executing on thread "
                f"{delegation._adopted_by}; one admission authorizes one writer at a time"
            )
        custody = source.custody
        if custody is not None:
            custody.retain()
        delegation._adopted_by = threading.get_ident()
        delegation._settled.clear()
    adopted = WriteLease(
        actor=source.actor,
        acquired_at=source.acquired_at,
        max_hold_seconds=None,
        archive_root=source.archive_root,
        custody=custody,
        coordinator=source.coordinator,
        owner_task=_current_task(),
        owner_thread_id=threading.get_ident(),
        bound_thread_ids={threading.get_ident()},
    )
    token = _ACTIVE.set(adopted)
    primary: BaseException | None = None
    try:
        yield adopted
    except BaseException as error:
        primary = error
        raise
    finally:

        def retire_delegation() -> None:
            with delegation._guard:
                delegation._adopted_by = None
                delegation._settled.set()

        actions: tuple[Callable[[], object], ...] = (
            lambda: _ACTIVE.reset(token),
            adopted.retire,
            retire_delegation,
        )
        if custody is not None:
            actions += (custody.release,)
        _finish_cleanup("Adopted writer and cleanup failed", actions, primary)


def _restore_active_lease(token: contextvars.Token[WriteLease | None], lease: WriteLease) -> None:
    """Retire a completed task's manual scope without resetting another context."""
    if lease.owner_pid != os.getpid() or lease.owner_thread is not threading.current_thread():
        raise UnleasedWriteError("write lease cleanup belongs to its original process and thread")
    owner = lease.owner_task
    current = _current_task()
    if current is not owner:
        if owner is None or not owner.done():
            raise UnleasedWriteError("a live writer task owns its lease cleanup")
        # The original task's Context is no longer executing. Its token cannot
        # be reset from this cleanup task; retiring authority below is global.
        if _ACTIVE.get() is lease:
            _ACTIVE.set(None)
        return
    _ACTIVE.reset(token)


@contextmanager
def write_lease(
    actor: str,
    *,
    max_hold_seconds: float | None = None,
    archive_root: str | Path | None = None,
    coordinator: object | None = None,
    _custody: ArchiveWriteCustody | None = None,
    _sql_owner: object | None = None,
) -> Iterator[WriteLease]:
    """Hold the write lease for ``actor``, authorizing write-mode opens.

    An outer acquisition names its archive root. Re-entrant within one context:
    nested acquisitions return the outer lease
    rather than a second authority, so a publish nested inside a batch does not
    reset the outer hold's budget.
    """
    held = _ACTIVE.get()
    if held is not None:
        # polylogue-1oa7o residual 2: the re-entrant branch used to hand back
        # the inherited lease with no ownership check, so a thread that
        # inherited it through free-threading contextvar propagation got the
        # parent's authority simply by asking for a nested lease. Re-run the
        # same identity check every write-mode open runs.
        # Re-entry names no new archive: without an explicit root it is the
        # held lease's own identity, and an explicit root must match it. Each
        # write-mode open inside the hold still names its archive itself.
        require_write_lease(
            f"nested write lease for {actor}",
            archive_root=archive_root if archive_root is not None else held.archive_root,
        )
        if coordinator is not None and held.coordinator is not None and coordinator is not held.coordinator:
            raise UnleasedWriteError("nested write lease requested by a different coordinator")
        yield held
        return
    if archive_root is None:
        raise UnleasedWriteError("an outer write lease must name its archive root")
    owns_custody = _custody is None
    owns_custody_ref = False
    custody = _custody
    if custody is not None:
        if _sql_owner is None:
            custody.require_owner_context(f"write lease for {actor}")
        else:
            custody.require_sql_owner_context(_sql_owner)
    elif _sql_owner is not None:
        raise UnleasedWriteError("retained SQL cleanup requires its original custody")
    if custody is None:
        inherited_custody = _context_custody()
        if inherited_custody is not None and inherited_custody.archive_root == Path(archive_root).resolve():
            inherited_custody.retain_owner_scope()
            custody = inherited_custody
            owns_custody_ref = True
            owns_custody = False
        else:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise UnleasedWriteError("synchronous write_lease cannot block an event loop; use async_write_lease")
            custody = _acquire_archive_write_custody(archive_root)
            custody.bind_owner_context()
    if Path(archive_root).resolve() != custody.archive_root:
        if owns_custody:
            custody.close_owner()
        raise UnleasedWriteError("archive write custody does not match the requested archive root")
    lease = WriteLease(
        actor=actor,
        acquired_at=time.perf_counter(),
        max_hold_seconds=max_hold_seconds,
        archive_root=Path(archive_root).resolve(),
        custody=custody,
        owns_custody=owns_custody,
        owns_custody_ref=owns_custody_ref,
        coordinator=coordinator,
        owner_task=_current_task(),
        owner_thread_id=threading.get_ident(),
        bound_thread_ids={threading.get_ident()},
    )
    token = _ACTIVE.set(lease)
    primary: BaseException | None = None
    try:
        yield lease
    except BaseException as error:
        primary = error
        raise
    finally:

        def report_hold() -> None:
            if lease.over_budget and lease.max_hold_seconds is not None:
                emit(
                    "storage.write_lease.hold_exceeded",
                    level=WARNING,
                    actor=actor,
                    hold_ms=lease.held_seconds * 1000,
                    budget_ms=lease.max_hold_seconds * 1000,
                    outcome="failed" if primary is not None else "committed",
                )

        actions: tuple[Callable[[], object], ...] = (
            lambda: _restore_active_lease(token, lease),
            lease.retire,
            report_hold,
        )
        if lease.owns_custody_ref:
            actions += (custody.release,)
        elif lease.owns_custody:
            actions += (custody.close_owner,)
        _finish_cleanup("Writer and lease cleanup failed", actions, primary)


@asynccontextmanager
async def async_write_lease(
    actor: str,
    *,
    max_hold_seconds: float | None = None,
    archive_root: str | Path,
    coordinator: object | None = None,
) -> AsyncIterator[WriteLease]:
    """Acquire physical custody without blocking the owning event loop."""
    held = _ACTIVE.get()
    if held is not None:
        with write_lease(
            actor, max_hold_seconds=max_hold_seconds, archive_root=archive_root, coordinator=coordinator
        ) as lease:
            yield lease
        return

    # An async scope is an offline executor only when no process-wide daemon
    # boundary has armed. In daemon mode the coordinator must provide the
    # authority; this helper never turns an unowned daemon call into a writer.
    coordinator_authorized = False
    if coordinator is not None:
        from polylogue.core.write_admission import active_write_admission

        coordinator_lease = active_write_admission.get()
        coordinator_authorized = coordinator_lease is not None and coordinator_lease.admits(coordinator, archive_root)
    if not coordinator_authorized:
        require_write_lease(f"async writer admission({actor})", archive_root=archive_root)
    inherited_custody = _context_custody()
    borrowed_custody = inherited_custody is not None and inherited_custody.archive_root == Path(archive_root).resolve()
    if borrowed_custody:
        assert inherited_custody is not None
        inherited_custody.retain_owner_scope()
        custody = inherited_custody
    else:
        retry = SQLSettlementRetry()
        observed_generation = retry.generation()
        acquire_task = asyncio.create_task(
            asyncio.to_thread(_acquire_async_archive_custody, archive_root, retry, observed_generation),
            name=f"polylogue-writer-custody:{actor}",
        )
        custody, cancellation, failure = await _settle_task(acquire_task, retry=retry)
        if cancellation is not None:
            if failure is not None:
                raise BaseExceptionGroup("Custody acquisition failed during cancellation", [cancellation, failure])
            if custody is not None:
                try:
                    await _close_async_archive_custody(custody)
                except BaseException as cleanup:
                    raise BaseExceptionGroup(
                        "Cancelled custody acquisition and cleanup failed", [cancellation, cleanup]
                    ) from cancellation
            raise cancellation
        if failure is not None:
            raise failure
        assert custody is not None
    primary: BaseException | None = None
    try:
        if not borrowed_custody:
            custody.bind_owner_context()
        with write_lease(
            actor,
            max_hold_seconds=max_hold_seconds,
            archive_root=archive_root,
            coordinator=coordinator,
            _custody=custody,
        ) as lease:
            yield lease
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            await _close_async_archive_custody(custody, borrowed=borrowed_custody)
        except BaseException as cleanup:
            if primary is not None:
                raise BaseExceptionGroup("Async writer and custody cleanup failed", [primary, cleanup]) from primary
            raise


async def _settle_task(
    task: asyncio.Future[Any],
    *,
    retry: SQLSettlementRetry | None = None,
) -> tuple[Any, asyncio.CancelledError | None, BaseException | None]:
    """Drain the retained task without forwarding waiter cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
            if retry is not None:
                retry.request()
        except BaseException:
            # The task is done with its own exception; collect it below.
            break
    try:
        return task.result(), cancellation, None
    except BaseException as exc:
        return None, cancellation, exc

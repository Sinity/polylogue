"""Canonical SQLite connection profiles and factory functions shared by sync and async backends.

Factories
---------
``open_connection(path)`` returns a read-write connection with write pragmas applied.
``open_daemon_connection(path)`` returns a read-write connection with a smaller
daemon/ops cache profile.
``open_readonly_connection(path)`` returns a uri=ro connection with read pragmas applied.
Pass ``validate_schema=False`` only for diagnostic readers that must inspect a
stale tier and report its version.
``connection_context(path)`` is a context manager for a single-use read-write connection.

These are lightweight one-shot wrappers around ``sqlite3.connect()``.  For the
thread-local cached connection used by the async runtime, use the factories in
``connection.py`` instead.
"""

from __future__ import annotations

import asyncio
import errno
import math
import os
import re
import sqlite3
import sys
import tempfile
import threading
import time
from builtins import BaseExceptionGroup
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import BuiltinFunctionType, TracebackType
from typing import TYPE_CHECKING, Literal, Self
from urllib.parse import parse_qs, quote, urlsplit

from polylogue.core.sql_settlement import SQLCustodyOwner, current_native_sql_lifetimes, register_native_sql_census
from polylogue.storage.io_phase_metrics import (
    close_connection_cursor,
    connect_measured,
    live_connection_cursors,
    native_connection_physically_closed,
    settle_connection_cursors,
)
from polylogue.storage.sqlite.write_lease import (
    KnownTierWriteAuthority,
    UnleasedWriteError,
    current_sql_custody,
    require_write_lease,
)

if TYPE_CHECKING:
    from polylogue.logging import BoundLoggerLike
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.audit_leaf import VerifiedAuditLeaf
    from polylogue.storage.sqlite.write_lease import ArchiveWriteCustody


class NativeConnectionSettlementError(RuntimeError):
    """A native connection remains owned until its original thread closes it."""

    code = "native_sql_unsettled"
    retryable = True

    def __init__(self, owner: NativeSQLCustodyOwner, failure: BaseException) -> None:
        super().__init__("native SQLite connection cleanup remains unsettled")
        self.owner = owner
        self.failure = failure


def _native_owner_task() -> asyncio.Task[object] | None:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


# Reload cannot discard live native handles or their creator-thread census.
if "_LIVE_NATIVE_SQL_OWNERS" not in globals():
    _LIVE_NATIVE_SQL_OWNERS_LOCK = threading.RLock()
    _LIVE_NATIVE_SQL_OWNERS: dict[int, NativeSQLCustodyOwner] = {}
    _FORK_ABANDONED_NATIVE_SQL_OWNERS: list[NativeSQLCustodyOwner] = []


def _before_native_owner_fork() -> None:
    _LIVE_NATIVE_SQL_OWNERS_LOCK.acquire()


def _after_native_owner_fork_parent() -> None:
    _LIVE_NATIVE_SQL_OWNERS_LOCK.release()


def _after_native_owner_fork_child() -> None:
    global _LIVE_NATIVE_SQL_OWNERS_LOCK, _LIVE_NATIVE_SQL_OWNERS
    # Quarantine copied handles without calling SQLite or a parent finalizer.
    _FORK_ABANDONED_NATIVE_SQL_OWNERS.extend(_LIVE_NATIVE_SQL_OWNERS.values())
    _LIVE_NATIVE_SQL_OWNERS = {}
    _LIVE_NATIVE_SQL_OWNERS_LOCK = threading.RLock()


if hasattr(os, "register_at_fork") and not globals().get("_NATIVE_FORK_REGISTERED", False):
    _NATIVE_FORK_REGISTERED = True
    os.register_at_fork(
        before=_before_native_owner_fork,
        after_in_parent=_after_native_owner_fork_parent,
        after_in_child=_after_native_owner_fork_child,
    )


def retained_native_sql_owners_on_current_thread() -> tuple[NativeSQLCustodyOwner, ...]:
    """Return actual handles awaiting cleanup by their creating thread."""
    pid, thread = os.getpid(), threading.current_thread()
    with _LIVE_NATIVE_SQL_OWNERS_LOCK:
        return tuple(owner for owner in _LIVE_NATIVE_SQL_OWNERS.values() if owner.pid == pid and owner.thread is thread)


def retained_native_settlement_owners_on_current_thread(
    preserved_native_owners: tuple[SQLCustodyOwner, ...] | None = (),
) -> tuple[SQLCustodyOwner, ...]:
    """Settle complete parents, preserving exact outer handles during nesting."""
    from polylogue.storage.sqlite.write_lease import retained_custody_settlement_owners_on_current_thread

    physical = retained_native_sql_owners_on_current_thread()
    custodies = tuple(
        custody
        for custody in retained_custody_settlement_owners_on_current_thread()
        if not any(owner.custody is custody for owner in physical)
    )
    if preserved_native_owners is None:
        return (*physical, *custodies)
    preserved_ids = {id(owner) for owner in preserved_native_owners}
    terminal_parents = {
        id(owner._terminal_parent)
        for owner in physical
        if owner._terminal_parent is not None
        and (owner._parent_cleanup_requested or (id(owner) in preserved_ids and owner.close_required))
    }
    protected_parents = {
        id(owner._terminal_parent)
        for owner in physical
        if id(owner) in preserved_ids
        and owner._terminal_parent is not None
        and id(owner._terminal_parent) not in terminal_parents
    }
    result: dict[int, SQLCustodyOwner] = {}
    for owner in physical:
        if id(owner) in preserved_ids and not (
            owner.close_required
            or owner._parent_cleanup_requested
            or (owner._terminal_parent is not None and id(owner._terminal_parent) in terminal_parents)
        ):
            continue
        parent = owner._terminal_parent
        if parent is not None and id(parent) in protected_parents:
            # A nested unit cannot close its outer parent's existing SQL or
            # artifacts. Close its new child only; keep the verified closed
            # binding available for the outer parent's eventual field cleanup.
            if owner._settled:
                continue
            terminal: SQLCustodyOwner = owner
        else:
            terminal = parent if parent is not None else owner
        result[id(terminal)] = terminal
    # Failed terminal bindings are selected even if an outer unit captured
    # their healthy owner on entry. They use the same custody registry.
    result.update((id(custody), custody) for custody in custodies)
    return tuple(result.values())


def settle_cached_connections_on_current_thread(custody: object) -> None:
    """Settle admitted cache entries through their existing physical owners."""
    owners = tuple(
        owner
        for owner in retained_native_sql_owners_on_current_thread()
        if owner.custody is custody and owner.cache_entry is not None
    )
    failures: list[BaseException] = []
    for owner in owners:
        if owner.connection is not None and (owner.close_required or owner.connection.in_transaction):
            failures.append(
                NativeConnectionSettlementError(
                    owner, RuntimeError("cached SQLite transaction or close remains unsettled")
                )
            )
    if not failures:
        for owner in owners:
            try:
                owner.close()
            except BaseException as error:
                failures.append(error)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("Cached native connection settlement failed", failures)


def retained_native_sql_owners_for_lifetime(dependency: object) -> tuple[NativeSQLCustodyOwner, ...]:
    """Protect artifact cleanup while any actual native owner retains it."""
    pid = os.getpid()
    with _LIVE_NATIVE_SQL_OWNERS_LOCK:
        return tuple(
            owner
            for owner in _LIVE_NATIVE_SQL_OWNERS.values()
            if owner.pid == pid
            and (not owner._settled or owner._terminal_parent is not None)
            and any(item is dependency for item in owner._lifetime_dependencies)
        )


def native_sql_children(parent: SQLCustodyOwner) -> tuple[NativeSQLCustodyOwner, ...]:
    return tuple(owner for owner in retained_native_sql_owners_on_current_thread() if owner._terminal_parent is parent)


def native_sql_owner_for_connection(connection: sqlite3.Connection) -> NativeSQLCustodyOwner | None:
    """Find the exact registered physical handle before any caller capture."""
    with _LIVE_NATIVE_SQL_OWNERS_LOCK:
        owners = tuple(owner for owner in _LIVE_NATIVE_SQL_OWNERS.values() if owner.connection is connection)
    if len(owners) > 1:
        raise RuntimeError("native SQLite handle has duplicate physical owners")
    if not owners:
        return None
    owner = owners[0]
    owner._require_owner()
    return owner


def native_sql_parent_for_connection(connection: sqlite3.Connection) -> SQLCustodyOwner | None:
    """Resolve an exact creator-owned handle's existing terminal parent."""
    for owner in retained_native_sql_owners_on_current_thread():
        if owner.connection is connection:
            owner._require_owner()
            return owner._terminal_parent
    return None


def request_native_sql_parent_cleanup(parent: SQLCustodyOwner) -> None:
    """Keep all existing siblings selected once their parent begins retirement."""
    for owner in native_sql_children(parent):
        owner._parent_cleanup_requested = True


def close_parent_native_connection(parent: SQLCustodyOwner, connection: sqlite3.Connection) -> None:
    owner = next((owner for owner in native_sql_children(parent) if owner._connection_identity == id(connection)), None)
    if owner is None:
        raise RuntimeError("native connection has no matching terminal parent")
    owner.close()


def retire_native_sql_parent(parent: SQLCustodyOwner) -> None:
    for owner in native_sql_children(parent):
        owner.retire_terminal_parent(parent)


register_native_sql_census(retained_native_settlement_owners_on_current_thread)


class NativeSQLCustodyOwner:
    """Pin an actual native handle during construction or one-shot execution.

    Successful construction hands its idle handle to the caller. Failed close
    keeps the handle in the existing physical custody's terminal owner census.
    """

    def __init__(
        self,
        connection: sqlite3.Connection | None,
        *,
        leaf: VerifiedAuditLeaf | None = None,
        cache_entry: tuple[dict[str, NativeSQLCustodyOwner], str] | None = None,
        frame: ReadFrame | None = None,
        anchored_descriptors: tuple[int, ...] = (),
        terminal_parent: SQLCustodyOwner | None = None,
        scratch_directory: tempfile.TemporaryDirectory[str] | None = None,
        lifetime_dependencies: tuple[object, ...] = (),
    ) -> None:
        self.close_required = False
        self._parent_cleanup_requested = False
        self._settled = False
        self._terminal_parent = terminal_parent
        self.scratch_directory = scratch_directory
        self._connection_identity = id(connection)
        # A terminal registration cannot hand off its handle. Capture the
        # existing artifact context there; temporary idle constructors keep
        # their current handoff semantics and capture only on failure.
        terminal_registration = (
            terminal_parent is not None
            or scratch_directory is not None
            or leaf is not None
            or cache_entry is not None
            or frame is not None
            or bool(anchored_descriptors)
        )
        dependencies = (
            (*current_native_sql_lifetimes(), *lifetime_dependencies)
            if terminal_registration
            else lifetime_dependencies
        )
        self._lifetime_dependencies: list[object] = list({id(item): item for item in dependencies}.values())
        self._settlement_callbacks: list[Callable[[], None]] = []
        self._incremental_blobs: list[sqlite3.Blob] = []
        self.leaf = leaf
        self.cache_entry = cache_entry
        self.anchored_descriptors = anchored_descriptors
        self._descriptor_bindings: dict[int, tuple[int, int]] = {}
        self._pending_descriptor_closes: dict[int, BaseException] = {}
        self.frame = frame
        if frame is not None:
            frame._sql_owner = self
        self.connection: sqlite3.Connection | None = connection
        self.pid = os.getpid()
        self.thread = threading.current_thread()
        self.task = _native_owner_task()
        self.custody: ArchiveWriteCustody | None = None
        with _LIVE_NATIVE_SQL_OWNERS_LOCK:
            _LIVE_NATIVE_SQL_OWNERS[id(self)] = self
        try:
            # Persistent parent handles retain creator-thread cleanup here;
            # their parent owns transaction admission and archive custody.
            # Pinning an idle reader/writer to its construction lease would
            # prevent that lease from retiring after a successful commit.
            self.custody = current_sql_custody() if terminal_parent is None else None
            if self.custody is not None:
                self.custody.retain_sql_owner(self)
                self.custody.assert_namespace()
            for descriptor in anchored_descriptors:
                metadata = os.fstat(descriptor)
                self._descriptor_bindings[descriptor] = (metadata.st_dev, metadata.st_ino)
        except BaseException as primary:
            _close_failed_native_construction(self, primary)
            raise

    def retain_anchored_descriptor(self, descriptor: int) -> None:
        """Attach a constructor's selected descriptor before returning its failure."""
        self._require_owner()
        self.anchored_descriptors += (descriptor,)
        try:
            metadata = os.fstat(descriptor)
        except BaseException as error:
            raise NativeConnectionSettlementError(self, error) from error
        self._descriptor_bindings[descriptor] = (metadata.st_dev, metadata.st_ino)

    def _descriptor_binding_retired(self, descriptor: int) -> bool:
        try:
            metadata = os.fstat(descriptor)
        except OSError as error:
            return error.errno == errno.EBADF
        previous = self._descriptor_bindings.get(descriptor)
        # A different file proves this numeric slot was replaced. An equal
        # inode does NOT prove the original open-file-description survives.
        return previous is not None and previous != (metadata.st_dev, metadata.st_ino)

    def _forget_descriptor(self, descriptor: int) -> None:
        self.anchored_descriptors = tuple(value for value in self.anchored_descriptors if value != descriptor)
        self._descriptor_bindings.pop(descriptor, None)
        self._pending_descriptor_closes.pop(descriptor, None)

    def _require_owner(self) -> None:
        if self.pid != os.getpid():
            raise RuntimeError("native SQLite connection belongs to another process")
        if self.thread is not threading.current_thread() or (
            self.task is not _native_owner_task() and (self.task is None or not self.task.done())
        ):
            raise RuntimeError("native SQLite connection belongs to another task or thread")

    def require_connection(self) -> sqlite3.Connection:
        """Admit new native work only on the exact current creator task."""
        connection = self._require_incremental_read()
        from polylogue.core.compute_cancel import compute_cancel_requested

        if compute_cancel_requested():
            raise asyncio.CancelledError("native SQLite work was cancelled")
        return connection

    def _require_incremental_read(self) -> sqlite3.Connection:
        """Check exact native custody, including committed settlement reads."""
        self._require_owner()
        if self.task is not _native_owner_task():
            raise RuntimeError("native SQLite work belongs to another task")
        if self.connection is None or self.close_required or self._parent_cleanup_requested:
            raise RuntimeError("native SQLite connection requires terminal cleanup")
        return self.connection

    def handoff(self) -> sqlite3.Connection:
        """Retire temporary construction custody without closing the idle handle."""
        self._require_owner()
        if self.close_required or self._parent_cleanup_requested:
            raise RuntimeError("native SQLite connection requires terminal cleanup")
        if (
            self._terminal_parent is not None
            or self.scratch_directory is not None
            or self._lifetime_dependencies
            or self._settlement_callbacks
            or self._incremental_blobs
        ):
            raise RuntimeError("native SQLite handle with terminal obligations cannot be handed off")
        connection = self.connection
        if connection is None:
            raise RuntimeError("native SQLite connection has already settled")
        if self.anchored_descriptors or self.leaf is not None or self.frame is not None or self.cache_entry is not None:
            primary = RuntimeError("native SQLite handle has retained lifetime obligations")
            _close_failed_native_construction(self, primary)
            raise primary
        if connection.in_transaction:
            primary = RuntimeError("native SQLite construction left an active transaction")
            _close_failed_native_construction(self, primary)
            raise primary
        if self.custody is not None:
            try:
                self.custody.release_sql_owner(self)
            except BaseException as primary:
                _close_failed_native_construction(self, primary)
                raise
        self.connection = None
        self.custody = None
        self.leaf = None
        self.cache_entry = None
        self._settled = True
        self._terminal_parent = None
        self._lifetime_dependencies.clear()
        with _LIVE_NATIVE_SQL_OWNERS_LOCK:
            _LIVE_NATIVE_SQL_OWNERS.pop(id(self), None)
        return connection

    def admit_cached_reuse(self) -> sqlite3.Connection:
        """Reuse only inside the same actual admitted operation and task."""
        self._require_owner()
        if self.task is not _native_owner_task():
            raise RuntimeError("cached SQLite connection belongs to an earlier task")
        if self.connection is None or self.close_required or self._parent_cleanup_requested:
            raise RuntimeError("cached SQLite connection requires terminal cleanup")
        custody = current_sql_custody()
        if self.custody is None or custody is not self.custody:
            raise RuntimeError("cached SQLite connection belongs to an earlier admitted operation")
        custody.assert_namespace()
        return self.connection

    def retain_lifetime(self, dependency: object) -> None:
        """Keep an artifact alive until this actual native handle settles."""
        self._require_owner()
        if self._settled:
            raise RuntimeError("a settled native owner cannot retain an artifact")
        with _LIVE_NATIVE_SQL_OWNERS_LOCK:
            if all(retained is not dependency for retained in self._lifetime_dependencies):
                self._lifetime_dependencies.append(dependency)

    def idle_handoff_ready(self, borrowed: tuple[object, ...]) -> bool:
        """Whether only ``borrowed`` lifetimes stand between this idle handle and handoff.

        Checked before :meth:`handoff`, whose refusal of a live obligation
        closes the handle; a borrower returning an unowned caller connection
        must never close it.
        """
        if self.pid != os.getpid() or self.thread is not threading.current_thread():
            return False
        if self.task is not _native_owner_task() and (self.task is None or not self.task.done()):
            return False
        connection = self.connection
        return (
            connection is not None
            and not self._settled
            and not self.close_required
            and not self._parent_cleanup_requested
            and self._terminal_parent is None
            and self.scratch_directory is None
            and not self._settlement_callbacks
            and not self._incremental_blobs
            and not self.anchored_descriptors
            and self.leaf is None
            and self.frame is None
            and self.cache_entry is None
            and not connection.in_transaction
            and all(any(dependency is item for item in borrowed) for dependency in self._lifetime_dependencies)
        )

    def release_lifetime(self, dependency: object) -> None:
        """Drop one artifact retained by :meth:`retain_lifetime` once it settled."""
        self._require_owner()
        with _LIVE_NATIVE_SQL_OWNERS_LOCK:
            if all(retained is not dependency for retained in self._lifetime_dependencies):
                raise RuntimeError("native owner does not retain this artifact")
            self._lifetime_dependencies[:] = [
                retained for retained in self._lifetime_dependencies if retained is not dependency
            ]

    def retain_incremental_blob(self, blob: sqlite3.Blob) -> None:
        """Retain the actual readonly incremental handle on its SQL creator."""
        self._require_owner()
        if self._settled or self.close_required or self._parent_cleanup_requested:
            raise RuntimeError("incremental handle requires its existing live native owner")
        if all(retained is not blob for retained in self._incremental_blobs):
            self._incremental_blobs.append(blob)

    def close_incremental_blob(self, blob: sqlite3.Blob) -> None:
        self._require_owner()
        if not any(retained is blob for retained in self._incremental_blobs):
            raise RuntimeError("incremental close requires its original native owner")
        if self.connection is not None and native_connection_physically_closed(self.connection):
            # The supported measured parent's successful native close retires
            # its actual Blob handles too. Their Python methods now refuse,
            # so only this existing physical-close proof authorizes retirement.
            self._incremental_blobs[:] = [retained for retained in self._incremental_blobs if retained is not blob]
            return
        # Native close is idempotent for this exact child while its original
        # parent remains live. A length probe refuses already closed children
        # and cannot prove physical settlement.
        blob.close()
        self._incremental_blobs[:] = [retained for retained in self._incremental_blobs if retained is not blob]

    @contextmanager
    def readonly_blob(self, table: str, column: str, row: int, *, settlement: bool = False) -> Iterator[sqlite3.Blob]:
        connection = self._require_incremental_read() if settlement else self.require_connection()
        blob = connection.blobopen(table, column, row, readonly=True)
        self.retain_incremental_blob(blob)
        primary: BaseException | None = None
        try:
            yield blob
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                # This context registered this exact child before yielding.
                # An explicit owner.close inside the body may already have
                # retired it before completing the original parent connection.
                if any(retained is blob for retained in self._incremental_blobs):
                    self.close_incremental_blob(blob)
            except BaseException as cleanup:
                self.close_required = True
                failure = (
                    cleanup
                    if primary is None
                    else BaseExceptionGroup("Incremental read and close failed", [primary, cleanup])
                )
                raise NativeConnectionSettlementError(self, failure) from cleanup

    def retain_settlement_callback(self, callback: Callable[[], None]) -> None:
        """Notify this actual handle's successful creator-owned settlement."""
        self._require_owner()
        if self._settled or self.close_required or self._parent_cleanup_requested:
            raise RuntimeError("settlement callback requires its existing live native owner")
        self._settlement_callbacks.append(callback)

    def release_settled_parent_lifetimes(self, parent: SQLCustodyOwner) -> None:
        """Release artifact dependencies while retaining the unsettled parent."""
        self._require_owner()
        if not self._settled or self._terminal_parent is not parent:
            raise RuntimeError("native artifact release requires its physically settled original parent")
        self._lifetime_dependencies.clear()

    def retire_terminal_parent(self, parent: SQLCustodyOwner) -> None:
        """Retire only after the actual handle and its parent's obligations settle."""
        self._require_owner()
        if not self._settled or self._terminal_parent is not parent:
            raise RuntimeError("native terminal parent has unsettled obligations")
        self._terminal_parent = None
        with _LIVE_NATIVE_SQL_OWNERS_LOCK:
            _LIVE_NATIVE_SQL_OWNERS.pop(id(self), None)
        self._lifetime_dependencies.clear()

    def physical_resources_settled(self) -> bool:
        """Prove native retirement independently of Python cleanup callbacks."""
        self._require_owner()
        return (
            self.connection is None
            and self.frame is None
            and self.leaf is None
            and self.scratch_directory is None
            and not self.anchored_descriptors
            and not self._incremental_blobs
            and self.custody is None
        )

    def close(self) -> None:
        self._require_owner()
        if self._settled:
            return
        self.close_required = True
        connection = self.connection
        failures: list[BaseException] = []
        if connection is not None:
            try:
                settle_connection_cursors(connection)
            except BaseException as error:
                failures.append(error)
            for blob in tuple(self._incremental_blobs):
                try:
                    self.close_incremental_blob(blob)
                except BaseException as error:
                    failures.append(error)
            if failures:
                if self.frame is not None:
                    live_ids = {id(cursor) for cursor in live_connection_cursors(connection)}
                    self.frame._cursors = {cursor for cursor in self.frame._cursors if id(cursor) in live_ids}
                    with _LIVE_READ_FRAMES_LOCK:
                        _LIVE_READ_FRAMES.add(self.frame)
                # Native children settle before rollback/connection close.
                failure = (
                    failures[0]
                    if len(failures) == 1
                    else BaseExceptionGroup("Native SQL children remain unsettled", failures)
                )
                raise NativeConnectionSettlementError(self, failure) from failure
            if self.frame is not None:
                self.frame._cursors.clear()
            try:
                # Cancellation interrupts work, never original-owner cleanup.
                if not native_connection_physically_closed(connection):
                    connection.set_progress_handler(None, 0)
                    if connection.in_transaction:
                        connection.rollback()
            except BaseException as error:
                failures.append(error)
            try:
                connection.close()
            except BaseException as error:
                if self.frame is not None:
                    with _LIVE_READ_FRAMES_LOCK:
                        _LIVE_READ_FRAMES.add(self.frame)
                failures.append(error)
                failure = (
                    failures[0] if len(failures) == 1 else BaseExceptionGroup("Native SQL cleanup failed", failures)
                )
                raise NativeConnectionSettlementError(self, failure) from error
            self.connection = None
        frame, self.frame = self.frame, None
        if frame is not None:
            frame._cursors.clear()
            with _LIVE_READ_FRAMES_LOCK:
                _LIVE_READ_FRAMES.discard(frame)
        cache_entry, self.cache_entry = self.cache_entry, None
        if cache_entry is not None:
            cache, key = cache_entry
            if cache.get(key) is self:
                del cache[key]
        for descriptor in tuple(self.anchored_descriptors):
            pending = self._pending_descriptor_closes.get(descriptor)
            if pending is not None:
                if self._descriptor_binding_retired(descriptor):
                    self._forget_descriptor(descriptor)
                else:
                    # An ambiguous close is never retried by numeric slot.
                    # Keep creator custody until original settlement is proven.
                    failures.append(pending)
                continue
            close_descriptor = os.close
            native_linux_close = (
                sys.platform == "linux"
                and isinstance(close_descriptor, BuiltinFunctionType)
                and close_descriptor.__module__ == "posix"
                and close_descriptor.__name__ == "close"
            )
            try:
                close_descriptor(descriptor)
            except BaseException as error:
                failures.append(error)
                # Linux's actual close syscall releases the descriptor before
                # reporting an OSError. A substituted/non-Linux closer has no
                # such guarantee; preserve ambiguity without an inode-only
                # claim about open-file-description identity.
                self._pending_descriptor_closes[descriptor] = error
                if (native_linux_close and isinstance(error, OSError)) or self._descriptor_binding_retired(descriptor):
                    self._forget_descriptor(descriptor)
            else:
                self._forget_descriptor(descriptor)
        if self.leaf is not None and not self.anchored_descriptors:
            try:
                self.leaf.close()
            except BaseException as error:
                failures.append(error)
            else:
                self.leaf = None
        if self.scratch_directory is not None and self.leaf is None and not self.anchored_descriptors:
            try:
                self.scratch_directory.cleanup()
            except BaseException as error:
                failures.append(error)
            else:
                self.scratch_directory = None
        resources_settled = (
            self.leaf is None
            and self.scratch_directory is None
            and not self.anchored_descriptors
            and not self._incremental_blobs
        )
        if resources_settled and self.custody is not None:
            try:
                self.custody.release_sql_owner(self)
            except BaseException as error:
                failures.append(error)
            else:
                self.custody = None
        if self.physical_resources_settled() and not failures:
            for callback in tuple(self._settlement_callbacks):
                try:
                    callback()
                except BaseException as error:
                    failures.append(error)
                else:
                    self._settlement_callbacks.remove(callback)
        callbacks_settled = not self._settlement_callbacks
        self._settled = self.physical_resources_settled() and callbacks_settled
        if self._settled and self._terminal_parent is None:
            with _LIVE_NATIVE_SQL_OWNERS_LOCK:
                _LIVE_NATIVE_SQL_OWNERS.pop(id(self), None)
            self._lifetime_dependencies.clear()
        if failures:
            failure = failures[0] if len(failures) == 1 else BaseExceptionGroup("Native SQL cleanup failed", failures)
            if not self._settled:
                raise NativeConnectionSettlementError(self, failure) from failure
            raise failure


def _close_failed_native_construction(owner: NativeSQLCustodyOwner, primary: BaseException) -> None:
    # Construction may fail before the receiving parent registers the idle
    # handle. Preserve its original artifact context on this same creator
    # before cleanup can fail, without attaching obligations to a handoff.
    if not owner._settled:
        for dependency in current_native_sql_lifetimes():
            owner.retain_lifetime(dependency)
    try:
        owner.close()
    except NativeConnectionSettlementError as cleanup:
        raise cleanup from primary
    except BaseException as cleanup:
        raise cleanup from primary


SCRATCH_SYNCHRONOUS_ENV = "POLYLOGUE_SQLITE_SYNCHRONOUS"


def scratch_synchronous_override() -> str | None:
    """``POLYLOGUE_SQLITE_SYNCHRONOUS=OFF`` drops fsync for throwaway archives.

    The test harness sets it: every archive under a pytest scratch tree is
    deleted seconds after it is written, and on a copy-on-write filesystem
    its fsyncs are the dominant disk load of a run. Only ``OFF`` is honoured
    and it applies to every profile that syncs at all; the value is read
    when statements are built, so it must be set before this module loads.
    """
    value = os.environ.get(SCRATCH_SYNCHRONOUS_ENV, "")
    return "OFF" if value.strip().upper() == "OFF" else None


@dataclass(frozen=True, slots=True)
class SQLiteConnectionProfile:
    """SQLite timeout and PRAGMA profile for one connection role."""

    role: Literal["read", "write"]
    timeout_seconds: float
    busy_timeout_ms: int
    cache_size_kib: int
    mmap_size_bytes: int
    foreign_keys: bool = False
    journal_mode: str | None = None
    synchronous: str | None = None
    temp_store: str = "MEMORY"
    wal_autocheckpoint_pages: int | None = None
    journal_size_limit_bytes: int | None = None
    query_only: bool = False
    locking_mode: str | None = None
    generation_identity: Literal["live", "sealed"] = "live"
    immutable: bool = False
    max_snapshot_age_s: float | None = None
    cancellation_supported: bool = False

    @property
    def pragma_statements(self) -> tuple[str, ...]:
        statements: list[str] = []
        synchronous = scratch_synchronous_override() or self.synchronous
        if self.foreign_keys:
            statements.append("PRAGMA foreign_keys = ON")
        if self.journal_mode is not None:
            statements.append(f"PRAGMA journal_mode={self.journal_mode}")
        statements.extend(
            (
                f"PRAGMA busy_timeout = {self.busy_timeout_ms}",
                f"PRAGMA cache_size = -{self.cache_size_kib}",
            )
        )
        if synchronous is not None:
            statements.append(f"PRAGMA synchronous = {synchronous}")
        statements.extend(
            (
                # Qualify the schema explicitly.  An unqualified mmap_size
                # pragma becomes the default for databases attached later,
                # charging every sibling tier against a budget that counts
                # this profile once.
                f"PRAGMA main.mmap_size = {self.mmap_size_bytes}",
                f"PRAGMA temp_store = {self.temp_store}",
            )
        )
        if self.wal_autocheckpoint_pages is not None:
            statements.append(f"PRAGMA wal_autocheckpoint = {self.wal_autocheckpoint_pages}")
        if self.journal_size_limit_bytes is not None:
            statements.append(f"PRAGMA journal_size_limit = {self.journal_size_limit_bytes}")
        if self.query_only:
            statements.append("PRAGMA query_only = ON")
        if self.locking_mode is not None:
            # Deliberately qualified to ``main``: an unqualified locking_mode
            # pragma also applies to every attached database (and becomes the
            # default for later ATTACHes), which would exclusively lock shared
            # durable tiers (user.db/ops.db) out from under concurrent readers.
            statements.append(f"PRAGMA main.locking_mode = {self.locking_mode}")
        return tuple(statements)


DB_TIMEOUT = 30
# Read busy_timeout. WAL readers normally don't block on a writer, but the
# brief window where a writer holds an exclusive lock (commit + TRUNCATE
# checkpoint) can exceed a second on a multi-GiB archive. A 1 s timeout turned
# that transient window into a hard "database is locked" error on interactive
# read surfaces (e.g. `polylogue find` during daemon ingest); 5 s lets the read
# wait out the checkpoint and succeed while staying far below the 30 s writer
# timeout, so reads remain responsive.
READ_DB_TIMEOUT = 5

# The four named lock-wait classes. ``interactive-read`` is READ_DB_TIMEOUT
# above: short enough that a stuck read surfaces rather than hangs. The other
# three wait out a full writer hold rather than fail a job a retry would only
# repeat, so they sit at the writer's own busy timeout.
TIMEOUT_CLASS_BACKGROUND_READ_S = 30.0
TIMEOUT_CLASS_PUBLICATION_S = 30.0
TIMEOUT_CLASS_OFFLINE_BULK_S = 30.0

MEMORY_BUDGET_ENV_VAR = "POLYLOGUE_MEMORY_BUDGET_BYTES"
DEFAULT_MEMORY_BUDGET_BYTES = 18 * 1024**3


def _read_declared_memory_budget_bytes() -> int:
    """Resolve the optional typed config/env budget, preserving current defaults."""
    from polylogue.config import load_polylogue_config

    configured = load_polylogue_config().memory_budget_bytes
    return configured if configured is not None else DEFAULT_MEMORY_BUDGET_BYTES


MEMORY_BUDGET_BYTES = _read_declared_memory_budget_bytes()


def _scale_profile_size(default_size: int) -> int:
    """Scale one mmap/cache limit proportionally to the effective budget."""
    return max(1, round(default_size * MEMORY_BUDGET_BYTES / DEFAULT_MEMORY_BUDGET_BYTES))


# The measured defaults remain unchanged when no budget is configured. The
# service unit can export MEMORY_BUDGET_ENV_VAR from the same declared budget
# used for its cgroup limits, moving every SQLite mmap/cache allowance together.
WRITE_CACHE_SIZE_KIB = _scale_profile_size(131072)  # 128 MiB
DAEMON_WRITE_CACHE_SIZE_KIB = _scale_profile_size(16384)  # 16 MiB
READ_CACHE_SIZE_KIB = _scale_profile_size(32768)  # 32 MiB
WRITE_MMAP_SIZE_BYTES = _scale_profile_size(1073741824)  # 1 GiB
DAEMON_WRITE_MMAP_SIZE_BYTES = _scale_profile_size(67108864)  # 64 MiB
READ_MMAP_SIZE_BYTES = _scale_profile_size(134217728)  # 128 MiB
# The bounded FTS repair connection is opened separately from the daemon's
# ordinary writer and must remain inside the same process budget.
BOUNDED_REPAIR_CACHE_SIZE_KIB = _scale_profile_size(32768)  # 32 MiB
BOUNDED_REPAIR_MMAP_SIZE_BYTES = _scale_profile_size(134217728)  # 128 MiB
# Schema inference keeps its own WAL journal connection alive while it scans
# provider artifacts. It has no mmap allowance, only this page-cache limit.
OBSERVATION_JOURNAL_CACHE_SIZE_KIB = _scale_profile_size(65536)  # 64 MiB
WAL_AUTOCHECKPOINT_PAGES = 10000
OWNED_WAL_AUTOCHECKPOINT_PAGES = 0
# #1614: soft cap on the WAL file. After any checkpoint that frees
# pages, SQLite truncates the WAL down to this size. Without this cap
# the WAL grows unbounded when a TRUNCATE checkpoint is blocked by a
# long-running reader — the dogfood probe reproducibly grew it from
# ~750 MB to ~1 GB in 60 s during catch-up. 160 MiB = 4x the
# autocheckpoint threshold (40 MiB), so a healthy autocheckpoint
# cycle does not trip the limit but a reader-blocked WAL eventually
# hits it and shrinks on the next successful checkpoint.
WAL_JOURNAL_SIZE_LIMIT_BYTES = 160 * 1024 * 1024

WRITE_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="write",
    timeout_seconds=DB_TIMEOUT,
    busy_timeout_ms=DB_TIMEOUT * 1000,
    cache_size_kib=WRITE_CACHE_SIZE_KIB,
    mmap_size_bytes=WRITE_MMAP_SIZE_BYTES,
    foreign_keys=True,
    journal_mode="WAL",
    synchronous="NORMAL",
    wal_autocheckpoint_pages=WAL_AUTOCHECKPOINT_PAGES,
    journal_size_limit_bytes=WAL_JOURNAL_SIZE_LIMIT_BYTES,
)

DAEMON_WRITE_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="write",
    timeout_seconds=DB_TIMEOUT,
    busy_timeout_ms=DB_TIMEOUT * 1000,
    cache_size_kib=DAEMON_WRITE_CACHE_SIZE_KIB,
    mmap_size_bytes=DAEMON_WRITE_MMAP_SIZE_BYTES,
    foreign_keys=True,
    journal_mode="WAL",
    synchronous="NORMAL",
    wal_autocheckpoint_pages=WAL_AUTOCHECKPOINT_PAGES,
    journal_size_limit_bytes=WAL_JOURNAL_SIZE_LIMIT_BYTES,
)

# An owned INACTIVE index generation is never read by anything until
# ``IndexGenerationStore.promote()`` swaps the ``index.db`` symlink, and is
# unconditionally discarded (``discard_if_inactive``) if the pass raises.
# That licenses a much more aggressive durability/speed tradeoff than the
# live writer profile above, which must survive a crash mid-write against the
# ONE active index a concurrent reader may be using right now:
#   - ``journal_mode=MEMORY`` (not WAL, not OFF): keeps the rollback journal
#     resident in RAM instead of round-tripping through the filesystem/WAL
#     checkpoint machinery, but still gives ``sqlite3.Connection.rollback()``
#     something to roll back to. ``revision_backfill.py``'s batched
#     census/replay loops call ``archive.rollback()`` on a recoverable batch
#     failure and re-processes that batch -- ``journal_mode=OFF`` disables
#     the rollback journal entirely, so that call would silently no-op and
#     the retry could double-apply against already-partially-written rows.
#     MEMORY is the fastest mode that keeps this real, already-exercised
#     recovery path correct.
#   - ``synchronous=OFF``: no fsync at all. A host crash mid-build can leave
#     ``index.db`` corrupt, but a corrupt INACTIVE generation is simply
#     discarded and rebuilt -- never promoted, never read.
#   - A much larger ``cache_size``/``mmap_size`` than even the live writer
#     profile: a bulk rebuild's working set (the whole generation being
#     built) is far larger than one incremental daemon write, and there is no
#     competing live-writer cgroup budget to share (this is a throwaway,
#     single-purpose process).
BULK_BUILD_CACHE_SIZE_KIB = _scale_profile_size(524288)  # 512 MiB
BULK_BUILD_MMAP_SIZE_BYTES = _scale_profile_size(4294967296)  # 4 GiB

BULK_BUILD_WRITE_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="write",
    timeout_seconds=DB_TIMEOUT,
    busy_timeout_ms=DB_TIMEOUT * 1000,
    cache_size_kib=BULK_BUILD_CACHE_SIZE_KIB,
    mmap_size_bytes=BULK_BUILD_MMAP_SIZE_BYTES,
    foreign_keys=True,
    journal_mode="MEMORY",
    synchronous="OFF",
    # An owned inactive generation has exactly one writer and zero readers
    # until promoted, so per-transaction lock acquisition/release syscall
    # churn is pure waste. EXCLUSIVE holds the file lock for the connection
    # lifetime. The promote path closes this connection before the pointer
    # swap, so the exclusive hold never outlives the build.
    locking_mode="EXCLUSIVE",
)


# polylogue-6xcqj: the cold-build shape for the ACTIVE index generation, held
# under the single-writer lease and proven empty before this profile is used.
#
# index.db is rebuildable, and an empty active generation has nothing a crash
# could lose that a restart would not simply re-derive from source.db, so the
# durability levers of the bulk-build profile apply by the same argument:
#   - ``synchronous=OFF``: no fsync per commit (measured ~15% of a cold build).
#   - a raised autocheckpoint threshold: a cold build commits constantly, and
#     an autocheckpoint inside a 256 MiB catch-up page charges its whole WAL
#     copy-back to whichever commit crossed the threshold.
#
# Foreign-key enforcement stays ON. Turning it off would need a verification
# pass at a boundary, and this shape has no boundary that may mutate the
# connection (see ``ArchiveStore.finish_active_cold_build``). Keeping it on
# means the cold shape relaxes durability only, and cannot change what a pass
# writes, defers or refuses -- which is the property that makes it safe to
# select automatically on the live route.
#
# What is deliberately NOT taken from ``BULK_BUILD_WRITE_CONNECTION_PROFILE``:
# ``journal_mode=MEMORY`` and ``locking_mode=EXCLUSIVE``. The active generation
# is read concurrently by the CLI, MCP and the daemon's own readers, and both
# of those would either lock them out or remove the WAL they read through.
# Those two -- and dropping reader indexes, which a read-only open reports as a
# schema manifest mismatch -- belong to an owned inactive generation.
COLD_BUILD_ACTIVE_WAL_AUTOCHECKPOINT_PAGES = 200_000

COLD_BUILD_ACTIVE_WRITE_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="write",
    timeout_seconds=DB_TIMEOUT,
    busy_timeout_ms=DB_TIMEOUT * 1000,
    # The live writer's cache and mmap budget, NOT the bulk build's. The bulk
    # profile's 512 MiB / 4 GiB window is sized for a throwaway single-purpose
    # process that owns the machine; this connection is the daemon's own live
    # writer, sharing a cgroup budget with its readers (see the mapped-bytes
    # note below). Measured 2026-09-16 on a 500-file synthetic cold build
    # through the dispatcher: the bulk sizes cost 433 MiB peak RSS against the
    # live profile's 261 MiB, for a shape whose window on this route is a
    # single intake page.
    cache_size_kib=WRITE_CACHE_SIZE_KIB,
    mmap_size_bytes=WRITE_MMAP_SIZE_BYTES,
    foreign_keys=True,
    journal_mode="WAL",
    synchronous="OFF",
    wal_autocheckpoint_pages=COLD_BUILD_ACTIVE_WAL_AUTOCHECKPOINT_PAGES,
    journal_size_limit_bytes=WAL_JOURNAL_SIZE_LIMIT_BYTES,
)

# What a live-generation reader pins is its *open read transaction*, not its
# connection. Two measurements on a synthetic WAL archive with
# ``wal_autocheckpoint=0`` (what the armed recurring owner leaves every daemon
# writer) and a 512-byte row payload:
#
#   40k rows written, then one PASSIVE. No reader and an idle ``mode=ro``
#   connection both checkpointed 6079 of 6079 frames; a reader holding one
#   lazily stepped cursor checkpointed 0 of 6079.
#
#   Eight bursts of 5k rows with one PASSIVE after each. Behind the idle
#   connection the WAL plateaued (2,962,312 -> 2,978,792 bytes); behind the
#   stepped cursor it grew monotonically every burst, 2,962,312 -> 23,776,552
#   bytes -- 8.0x, the full concurrent write volume, for the transaction's
#   whole lifetime.
#
# So the bound that matters is on how long a read transaction may stay open, and
# ``ReadFrame.stream`` below is the route that applies it per row. Every live
# read profile declares that maximum age; a sealed generation cannot change
# under a reader and declares none.
INTERACTIVE_READ_SNAPSHOT_AGE_S = 30.0
BACKGROUND_READ_SNAPSHOT_AGE_S = 300.0

READ_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="read",
    timeout_seconds=READ_DB_TIMEOUT,
    busy_timeout_ms=READ_DB_TIMEOUT * 1000,
    cache_size_kib=READ_CACHE_SIZE_KIB,
    mmap_size_bytes=READ_MMAP_SIZE_BYTES,
    # #1614: explicit read-only signal. ``open_readonly_connection``
    # opens with the ``mode=ro`` URI flag which is already enforced
    # by SQLite at the file level, but the pragma additionally
    # rejects accidental writes via the same connection at SQL parse
    # time instead of waiting for the write lock.
    query_only=True,
    generation_identity="live",
    max_snapshot_age_s=INTERACTIVE_READ_SNAPSHOT_AGE_S,
    cancellation_supported=True,
)

BACKGROUND_READ_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="read",
    timeout_seconds=TIMEOUT_CLASS_BACKGROUND_READ_S,
    busy_timeout_ms=int(TIMEOUT_CLASS_BACKGROUND_READ_S * 1000),
    cache_size_kib=READ_CACHE_SIZE_KIB,
    mmap_size_bytes=READ_MMAP_SIZE_BYTES,
    query_only=True,
    generation_identity="live",
    max_snapshot_age_s=BACKGROUND_READ_SNAPSHOT_AGE_S,
    cancellation_supported=True,
)

OFFLINE_BULK_READ_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="read",
    timeout_seconds=TIMEOUT_CLASS_OFFLINE_BULK_S,
    busy_timeout_ms=int(TIMEOUT_CLASS_OFFLINE_BULK_S * 1000),
    cache_size_kib=READ_CACHE_SIZE_KIB,
    mmap_size_bytes=READ_MMAP_SIZE_BYTES,
    query_only=True,
    generation_identity="live",
    max_snapshot_age_s=BACKGROUND_READ_SNAPSHOT_AGE_S,
    cancellation_supported=True,
)

# SQLite's ``immutable=1`` skips locking and WAL/journal detection outright, so
# it is a claim about the generation rather than about the caller's intent: it
# is correct only where nothing can still write the file. This is the one
# profile that carries it, and ``open_readonly_connection`` selects this profile
# whenever a caller asks for immutability, so the two cannot drift apart.
SEALED_READ_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="read",
    timeout_seconds=TIMEOUT_CLASS_OFFLINE_BULK_S,
    busy_timeout_ms=int(TIMEOUT_CLASS_OFFLINE_BULK_S * 1000),
    cache_size_kib=READ_CACHE_SIZE_KIB,
    mmap_size_bytes=READ_MMAP_SIZE_BYTES,
    query_only=True,
    generation_identity="sealed",
    immutable=True,
    max_snapshot_age_s=None,
    cancellation_supported=True,
)

# Historical continuity classification needs SQLite's connection-local TEMP
# relations for its bounded candidate stream.  ``query_only`` rejects TEMP
# writes as well as durable writes, so this deliberately private profile is
# not part of ``READ_PROFILES``: the factory below is the only route that can
# use it.  The URI's ``mode=ro&immutable=1`` still makes the authenticated
# main database immutable; only SQLite's private TEMP schema is writable.
SEALED_STAGING_CONNECTION_PROFILE = SQLiteConnectionProfile(
    role="read",
    timeout_seconds=TIMEOUT_CLASS_OFFLINE_BULK_S,
    busy_timeout_ms=int(TIMEOUT_CLASS_OFFLINE_BULK_S * 1000),
    cache_size_kib=READ_CACHE_SIZE_KIB,
    mmap_size_bytes=READ_MMAP_SIZE_BYTES,
    temp_store="MEMORY",
    query_only=False,
    generation_identity="sealed",
    immutable=True,
    max_snapshot_age_s=None,
    cancellation_supported=True,
)


# This is intentionally a small, positive allowlist.  The liveness and legacy
# hook matcher need these aggregate/scalar functions plus the registered UDF;
# all other function calls, including ``load_extension``, are refused.
_SEALED_STAGING_FUNCTIONS = frozenset(
    {
        "coalesce",
        "count",
        "min",
        "sum",
        "polylogue_deterministic_raw_session_id",
    }
)
_SEALED_STAGING_READ_PRAGMAS = frozenset(
    {"data_version", "query_only", "schema_version", "table_info", "table_list", "temp_store", "user_version"}
)


def _authorize_sealed_staging_operation(
    action: int,
    argument1: str | None,
    argument2: str | None,
    database: str | None,
    _trigger: str | None,
) -> int:
    """Allow only liveness reads and connection-local TEMP staging.

    SQLite invokes this callback while compiling each statement.  Returning
    ``SQLITE_DENY`` by default is important: adding a new operation to the
    classifier must explicitly earn an entry here rather than silently
    widening an authenticated immutable reader.
    """

    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_SAVEPOINT):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        # SQLite reports COUNT(*)'s synthetic empty-column read without a
        # database name; it still belongs to the statement's main/temp table.
        return sqlite3.SQLITE_OK if database in {None, "main", "temp"} else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        function_name = (argument2 or argument1 or "").lower()
        return sqlite3.SQLITE_OK if function_name in _SEALED_STAGING_FUNCTIONS else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_PRAGMA:
        pragma_name = (argument1 or "").lower()
        # A non-NULL second argument is a PRAGMA assignment.  Setup pragmas
        # run before this authorizer is installed; callers get reads only.
        # These PRAGMAs take a table/index lookup argument, never a setting.
        # Keep parameterized schema reads distinct from enforcement setters.
        if pragma_name in {"table_info", "table_xinfo", "foreign_key_list", "index_list", "index_info", "index_xinfo"}:
            return sqlite3.SQLITE_OK
        return (
            sqlite3.SQLITE_OK
            if argument2 is None and pragma_name in _SEALED_STAGING_READ_PRAGMAS
            else sqlite3.SQLITE_DENY
        )

    # TEMP DML and TEMP table/index creation, deletion, and reindexing are the
    # complete staging vocabulary.  SQLite reports its internal
    # sqlite_temp_master updates with database="temp", so those are included
    # by the same database check rather than by table-name exceptions.
    temp_dml = {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}
    temp_schema = {
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
    }
    if action in temp_dml:
        return sqlite3.SQLITE_OK if database == "temp" else sqlite3.SQLITE_DENY
    if action in temp_schema:
        return sqlite3.SQLITE_OK if database == "temp" else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_REINDEX:
        return sqlite3.SQLITE_OK if database == "temp" else sqlite3.SQLITE_DENY

    # This explicitly denies ATTACH/DETACH, all main-schema writes, virtual
    # tables, triggers/views, unsafe PRAGMAs and every future/unlisted action.
    return sqlite3.SQLITE_DENY


# Named timeout classes are the only supported policy vocabulary.  Callers
# select a role, not an arbitrary lock-wait duration.
TIMEOUT_CLASSES: Mapping[str, float] = {
    "interactive-read": float(READ_DB_TIMEOUT),
    "background-read": TIMEOUT_CLASS_BACKGROUND_READ_S,
    "publication": TIMEOUT_CLASS_PUBLICATION_S,
    "offline-bulk": TIMEOUT_CLASS_OFFLINE_BULK_S,
    # The active cold-build writer runs under the daemon's normal publication
    # hold budget and therefore uses the same declared lock-wait class.  Keep
    # it in the shared vocabulary so every named writer profile is validated
    # against an explicit timeout class.
    "active-cold-build": TIMEOUT_CLASS_PUBLICATION_S,
}
READ_PROFILES: Mapping[str, SQLiteConnectionProfile] = {
    "interactive-read": READ_CONNECTION_PROFILE,
    "background-read": BACKGROUND_READ_CONNECTION_PROFILE,
    "offline-bulk": OFFLINE_BULK_READ_CONNECTION_PROFILE,
}
# ``publication`` and ``offline-bulk`` name different profiles on the write side
# than on the read side, so the two vocabularies stay separate maps: merging
# them silently shadowed the offline-bulk *read* profile with the bulk-build
# writer.
WRITE_PROFILES: Mapping[str, SQLiteConnectionProfile] = {
    "publication": DAEMON_WRITE_CONNECTION_PROFILE,
    "offline-bulk": BULK_BUILD_WRITE_CONNECTION_PROFILE,
    "active-cold-build": COLD_BUILD_ACTIVE_WRITE_CONNECTION_PROFILE,
}

# One tier, no sibling attach. An excision apply and a backup snapshot both
# commit a single tier at a time so a mid-operation failure leaves at most one
# tier mutated; attaching siblings would draw them into the same transaction
# scope, which is the thing those routes exist to avoid. Journal mode and
# foreign-key enforcement are deliberately left as the file already has them:
# these routes adopt a tier, they do not reconfigure it. Autocheckpoint remains
# explicit because it is connection-local: an unowned process keeps the bounded
# fallback while the daemon's recurring owner disables it for this writer too.
ISOLATED_TIER_WRITE_PROFILE = SQLiteConnectionProfile(
    role="write",
    timeout_seconds=TIMEOUT_CLASS_PUBLICATION_S,
    busy_timeout_ms=int(TIMEOUT_CLASS_PUBLICATION_S * 1000),
    cache_size_kib=DAEMON_WRITE_CACHE_SIZE_KIB,
    mmap_size_bytes=DAEMON_WRITE_MMAP_SIZE_BYTES,
    wal_autocheckpoint_pages=WAL_AUTOCHECKPOINT_PAGES,
)

READ_CONNECTION_PRAGMA_STATEMENTS = READ_CONNECTION_PROFILE.pragma_statements


# ---------------------------------------------------------------------------
# Recurring checkpoint ownership and WAL escalation policy
# ---------------------------------------------------------------------------

#: What each escalation may attempt, in attempt order.
#:
#: PASSIVE never waits and never blocks a reader, so it is the only mode a
#: recurring owner may run against a live archive. RESTART additionally waits
#: for existing readers to drain before resetting the WAL, which is bounded
#: only at a declared quiescent boundary. TRUNCATE also takes the writer lock to
#: shrink the file, and belongs to seal, shutdown and offline generation
#: lifecycle after readers have drained -- never to fight a busy live reader.
CheckpointEscalation = Literal["recurring", "quiescent", "exclusive"]

CHECKPOINT_ESCALATION_MODES: Mapping[CheckpointEscalation, tuple[str, ...]] = {
    "recurring": ("PASSIVE",),
    "quiescent": ("PASSIVE", "RESTART"),
    "exclusive": ("PASSIVE", "RESTART", "TRUNCATE"),
}

#: WAL size at which a checkpoint is worth running at all, and the size past
#: which an escalation may reach its next mode.
WAL_WARN_BYTES = 256 * 1024 * 1024
WAL_ESCALATION_BYTES = 512 * 1024 * 1024

#: Checkpoint hold budget. Declared here rather than beside the publication
#: budgets in ``daemon/write_coordinator.py`` so checkpoint time is accounted
#: against its own ceiling instead of disappearing into whichever publication
#: hold happened to contain it.
CHECKPOINT_HOLD_BUDGET_S = 20.0

_RECURRING_CHECKPOINT_OWNER = threading.Event()


def recurring_checkpoint_owner_armed() -> bool:
    """Whether this process runs the recurring checkpoint coordinator."""
    return _RECURRING_CHECKPOINT_OWNER.is_set()


def _set_recurring_checkpoint_owner(armed: bool) -> None:
    if armed:
        _RECURRING_CHECKPOINT_OWNER.set()
    else:
        _RECURRING_CHECKPOINT_OWNER.clear()


@contextmanager
def arm_recurring_checkpoint_owner(*, armed: bool = True) -> Iterator[None]:
    """Claim recurring checkpoint ownership for this process.

    Process-global rather than thread-local like the write lease: the daemon
    opens writable connections from several threads and one coordinator owns
    checkpointing for all of them.
    """
    previous = _RECURRING_CHECKPOINT_OWNER.is_set()
    _set_recurring_checkpoint_owner(armed)
    try:
        yield
    finally:
        _set_recurring_checkpoint_owner(previous)


def write_connection_pragma_statements(profile: SQLiteConnectionProfile) -> tuple[str, ...]:
    """Write pragmas for ``profile`` under this process' checkpoint ownership.

    Under an armed recurring owner an implicit autocheckpoint runs inside
    whichever writer's commit crossed the page threshold, charging its wait and
    hold to that writer's publication budget where no checkpoint accounting can
    see it. Resolved per open rather than frozen at import so a process that
    arms ownership after loading this module still gets the owned profile.
    """
    if profile.role != "write" or profile.wal_autocheckpoint_pages is None:
        return profile.pragma_statements
    if not recurring_checkpoint_owner_armed():
        return profile.pragma_statements
    return replace(profile, wal_autocheckpoint_pages=OWNED_WAL_AUTOCHECKPOINT_PAGES).pragma_statements


def execute_pragma_statement(conn: sqlite3.Connection, statement: str) -> None:
    """Run one profile pragma, leaving an already-matching database mode alone.

    ``PRAGMA journal_mode=<mode>`` rewrites the database header even when the
    mode does not change, which moves the file's size and mtime under any
    reference seal prepared against it. Bootstrap establishes each tier's
    mode; an open only changes a mode that genuinely differs (a bulk build's
    MEMORY profile). Every other pragma is connection-local and runs as is.
    """
    if statement.startswith("PRAGMA journal_mode="):
        desired = statement.split("=", 1)[1].strip().lower()
        with closing(conn.execute("PRAGMA journal_mode")) as cursor:
            current = str(cursor.fetchone()[0]).lower()
        if current == desired:
            return
    conn.execute(statement)


def write_connection_local_pragma_statements(profile: SQLiteConnectionProfile) -> tuple[str, ...]:
    """Return a writer profile without database-mode initialization.

    ``journal_mode`` changes shared database state and needs an exclusive
    transition lock. A later connection to an already-initialized durable tier
    must therefore not replay it while another writer owns that tier's
    transaction; the remaining statements are connection-local policy.
    """
    return tuple(
        statement
        for statement in write_connection_pragma_statements(profile)
        if not statement.startswith("PRAGMA journal_mode")
    )


#: The journal mode a tier file is created in when it is not an ordinary
#: writer-profile (WAL) tier. A rollback-journal header is the SQLite default,
#: so creating the file in it is a no-op; it is named for the two contracts
#: that require it:
#: - the embeddings tier is published as sealed generation files that the
#:   lifecycle copies and validates immutably, refusing any -wal/-shm sidecar,
#:   and a WAL-mode file grows those on any read-only open;
#: - an inactive Index generation is built under the bulk-build profile
#:   (journal_mode=MEMORY, a per-connection mode that needs no header change
#:   from rollback, but an exclusive lock to leave WAL), and switches to WAL
#:   at promotion, its exclusive commit point.
ROLLBACK_TIER_JOURNAL_MODE = "DELETE"


def initialize_tier_database_mode(conn: sqlite3.Connection, *, rollback: bool = False) -> None:
    """Set a tier's declared journal mode while its bootstrap owns the file.

    This is deliberately separate from every writer open: a later open may
    run while a publication or GC transaction owns the tier's mode-transition
    lock, and a mode pragma rewrites the header under a prepared seal. A tier
    is created in the writer profile's mode (WAL/NORMAL; source.db's
    power-loss guarantee remains the durable publication/cursor boundary)
    unless its contract declares ``ROLLBACK_TIER_JOURNAL_MODE``. An ordinary
    open therefore never has a mode to change.
    """
    journal_mode = ROLLBACK_TIER_JOURNAL_MODE if rollback else WRITE_CONNECTION_PROFILE.journal_mode
    if journal_mode is None:
        raise RuntimeError("the tier writer profile must declare a journal mode")
    execute_pragma_statement(conn, f"PRAGMA journal_mode={journal_mode}")


def _connect_archive_writer(
    path: str | Path,
    *,
    profile: SQLiteConnectionProfile,
    archive_root: str | Path | None = None,
    timeout: float = DB_TIMEOUT,
    check_same_thread: bool = True,
    existing_only: bool = False,
    mutation_permit: KnownTierWriteAuthority | None = None,
) -> sqlite3.Connection:
    """Keep Source SQL authorization dynamic across admitted writer leases."""
    if profile.role != "write":
        raise ValueError("archive writer construction requires a write profile")
    selected = Path(path)
    root = configured_archive_root(path, archive_root)
    is_source = selected.resolve() == (root / "source.db").resolve()
    is_user = selected.resolve() == (root / "user.db").resolve()
    is_embeddings = selected.resolve() == (root / "embeddings.db").resolve()
    is_prepared_embeddings = (
        mutation_permit is not None
        and mutation_permit.tier == "embeddings"
        and selected.resolve() == (root / "embeddings.db").resolve()
    )
    selected_tier = "source" if is_source else "user" if is_user else "embeddings" if is_prepared_embeddings else None
    if mutation_permit is not None and mutation_permit.tier != selected_tier:
        raise UnleasedWriteError("known tier permit cannot authorize another physical tier")
    connection = connect_measured(
        f"{selected.resolve(strict=True).as_uri()}?mode=rw" if existing_only or mutation_permit is not None else path,
        # Sibling Source attachments use mode=ro URIs even when main is an
        # ordinary Path. URI handling belongs to the entire native connection.
        uri=True,
        timeout=timeout,
        check_same_thread=check_same_thread,
        **({"cached_statements": 0} if is_source or is_user or is_embeddings else {}),
    )
    owner = NativeSQLCustodyOwner(connection)
    try:
        if mutation_permit is None:
            for statement in write_connection_local_pragma_statements(profile):
                connection.execute(statement)
        if not (is_source or is_user or is_embeddings):
            return owner.handoff()
        creator_pid = os.getpid()
        creator_thread = threading.current_thread()
        creator_task = _native_owner_task()
        source_path = selected.resolve()
        metadata = source_path.stat()
        incarnation = metadata.st_dev, metadata.st_ino
        # Physical custody facts verified for one statement compile: the
        # file incarnation, the current custody, its root and namespace, and
        # (once a write is seen) the write lease. They cannot vary within one
        # compile, so they are verified at its first callback; every
        # action-dependent decision still runs per callback. Outside a
        # measured execute (no epoch) every callback verifies everything.
        verified_epoch: int | None = None
        verified_custody: ArchiveWriteCustody | None = None
        verified_lease = False

        def authorize(
            action: int, first: str | None, second: str | None, schema: str | None, trigger: str | None
        ) -> int:
            if os.getpid() != creator_pid or threading.current_thread() is not creator_thread:
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_TRANSACTION and first == "ROLLBACK":
                if _native_owner_task() is not creator_task:
                    return sqlite3.SQLITE_DENY
                # Cleanup stays available after failure, but cannot mint a
                # successful receipt for a transaction it has rolled back.
                custody = current_sql_custody()
                permit = None if custody is None else custody.known_tier_authority
                if permit is not None and not permit.authorize_tier_sql(
                    connection, action, first, second, schema, trigger
                ):
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            nonlocal verified_epoch, verified_custody, verified_lease
            epoch = getattr(connection, "_compile_epoch", None)
            fresh = epoch is None or epoch != verified_epoch
            try:
                if fresh:
                    verified_epoch = None
                    current_metadata = source_path.stat()
                    if (current_metadata.st_dev, current_metadata.st_ino) != incarnation:
                        return sqlite3.SQLITE_DENY
                # Context inheritance is not writer admission: a child task
                # can carry its parent's lease while owning no physical custody.
                writes = action not in {
                    sqlite3.SQLITE_READ,
                    sqlite3.SQLITE_SELECT,
                    sqlite3.SQLITE_FUNCTION,
                    sqlite3.SQLITE_RECURSIVE,
                } and not (action == sqlite3.SQLITE_PRAGMA and second is None)
                custody = current_sql_custody() if fresh else verified_custody
                known = None if custody is None else custody.known_tier_authority
                completion_write = (
                    first == "excision_embedding_completions"
                    and action
                    in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE, sqlite3.SQLITE_DROP_TABLE}
                ) or (action == sqlite3.SQLITE_ALTER_TABLE and second == "excision_embedding_completions")
                if (
                    is_embeddings
                    and completion_write
                    and (known is None or known is not mutation_permit or not is_prepared_embeddings)
                ):
                    return sqlite3.SQLITE_DENY
                lease_needed = writes and (is_source or known is not None)
                if lease_needed and (fresh or not verified_lease):
                    if require_write_lease("durable tier SQL execution", archive_root=root) is None:
                        return sqlite3.SQLITE_DENY
                    if not fresh:
                        verified_lease = True
                if custody is not None:
                    if fresh:
                        if custody.archive_root != root and custody.archive_root.resolve() != root.resolve():
                            return sqlite3.SQLITE_DENY
                        custody.assert_namespace()
                    permit = custody.known_tier_authority
                    if permit is not None and not permit.authorize_tier_sql(
                        connection, action, first, second, schema, trigger
                    ):
                        return sqlite3.SQLITE_DENY
            except (OSError, UnleasedWriteError):
                verified_epoch = None
                return sqlite3.SQLITE_DENY
            if fresh and epoch is not None:
                verified_epoch = epoch
                verified_custody = custody
                verified_lease = lease_needed
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        if mutation_permit is not None:
            custody = current_sql_custody()
            if custody is None or custody.known_tier_authority is not mutation_permit:
                raise UnleasedWriteError("Source connection does not own its current known mutation")
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    return owner.handoff()


def open_source_tier_write_connection(
    path: str | Path,
    *,
    archive_root: str | Path | None = None,
    mutation_permit: KnownTierWriteAuthority | None = None,
) -> sqlite3.Connection:
    """Open a source-tier writer with the normal local policy only.

    Bootstrap owns the one-time database-mode transition. This factory is
    shared by the persistent archive handle and publication reservations so a
    fresh source tier and a reservation transaction cannot drift on
    synchronous, busy-timeout, or foreign-key policy.
    """
    require_write_lease(f"open_source_tier_write_connection({path})", archive_root=archive_root)
    conn = _connect_archive_writer(
        path,
        profile=WRITE_CONNECTION_PROFILE,
        archive_root=archive_root,
        mutation_permit=mutation_permit,
        timeout=WRITE_CONNECTION_PROFILE.timeout_seconds,
    )
    owner = NativeSQLCustodyOwner(
        conn, terminal_parent=None if mutation_permit is None else mutation_permit.terminal_parent
    )
    try:
        statements = write_connection_local_pragma_statements(WRITE_CONNECTION_PROFILE)
        if mutation_permit is not None:
            # Profile setup precedes immutable row/FK/trigger enforcement on
            # the permit's one registered native connection.
            mutation_permit.configure_mutation_connection(conn, (*statements, "PRAGMA recursive_triggers = ON"))
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    return owner.handoff() if mutation_permit is None else owner.require_connection()


# ---------------------------------------------------------------------------
# Mapped-bytes budget vs. the cgroup memory limit (polylogue-e98k)
# ---------------------------------------------------------------------------
#
# 2026-07-31 incident: the mmap/cache profile sizes above (this file) and the
# systemd cgroup limits (sinnix repo, `modules/services/polylogue.nix`) were
# picked independently, in two different repos, with nothing tying them
# together. A 4 GiB `BULK_BUILD_MMAP_SIZE_BYTES` window over a 38 GB
# `index.db` fills completely under any scan-heavy work; one bulk-build
# connection alone therefore accounted for ~4.5 GiB against a 6 GiB
# `MemoryHigh` ceiling, leaving no headroom for the daemon's own writer and
# concurrent readers. `MemoryHigh` was the wrong instrument for reclaimable,
# file-backed mmap'd pages (it throttles anon growth; mapped pages just
# evict-and-refault under pressure -- the observed `slow_write` signature),
# not a leak, so pinning at the ceiling was structurally guaranteed rather
# than a bug in either repo. A runtime bump to `MemoryHigh=14G` stopped the
# throttling dead; that finding is now the committed default
# (`MemoryHigh=14G` / `MemoryMax=18G`) -- see the comment beside that
# override for the measurement. `mapped_bytes_budget` below is the
# mechanical anchor so nobody has to re-derive that arithmetic from scratch
# next time either side's constants move: the sinnix `MemoryMax`/`MemoryHigh`
# override for `polylogued.service` must stay comfortably above the value
# this returns, and `check_mapped_bytes_budget_against_cgroup_limit` makes
# that comparison observable at runtime instead of only discoverable hours
# into an incident.
#
# `mmap_size` is an upper bound SQLite MAY map into, never a guaranteed
# allocation, so `mapped_bytes_budget()` is a conservative ceiling estimate,
# not a live-RSS prediction -- it will typically overstate actual usage.


def mapped_bytes_budget(*, concurrent_read_connections: int = 4) -> int:
    """Plausible peak concurrent SQLite mmap+cache footprint for one polylogued process.

    Models the worst case that actually bit us: one bulk-build connection
    running concurrently with the daemon's own long-lived write connection
    (`DAEMON_WRITE_CONNECTION_PROFILE`) and a handful of concurrent
    short-lived read connections (CLI/MCP/API reads against the live
    archive while a rebuild is in flight), plus three ordinary writers
    (index, source, and a publication reservation), one bounded FTS repair
    connection, and one schema-observation journal
    connection. This is a conservative upper bound across the production
    profiles, including one-shot maintenance/CLI writers, so cgroup allowance
    does not depend on an assumed lifecycle ordering.

    `concurrent_read_connections` defaults to 4: a conservative but not
    extreme estimate of simultaneous interactive reads (CLI/MCP/API) during
    a bulk rebuild. Callers with better knowledge of their own concurrency
    (e.g. a fixed MCP worker pool size) may override it.
    """
    return (
        BULK_BUILD_MMAP_SIZE_BYTES
        + BULK_BUILD_CACHE_SIZE_KIB * 1024
        + 3 * (WRITE_MMAP_SIZE_BYTES + WRITE_CACHE_SIZE_KIB * 1024)
        + DAEMON_WRITE_MMAP_SIZE_BYTES
        + DAEMON_WRITE_CACHE_SIZE_KIB * 1024
        + concurrent_read_connections * (READ_MMAP_SIZE_BYTES + READ_CACHE_SIZE_KIB * 1024)
        + BOUNDED_REPAIR_MMAP_SIZE_BYTES
        + BOUNDED_REPAIR_CACHE_SIZE_KIB * 1024
        + OBSERVATION_JOURNAL_CACHE_SIZE_KIB * 1024
    )


@dataclass(frozen=True, slots=True)
class MappedBytesBudgetCheck:
    """Result of comparing :func:`mapped_bytes_budget` to the detected cgroup limits."""

    budget_bytes: int
    memory_max_bytes: int | None
    memory_high_bytes: int | None
    concurrent_read_connections: int
    memory_budget_bytes: int | None = None

    @property
    def budget_mb(self) -> float:
        return round(self.budget_bytes / (1024 * 1024), 1)

    @property
    def effective_memory_budget_bytes(self) -> int:
        return self.memory_budget_bytes if self.memory_budget_bytes is not None else MEMORY_BUDGET_BYTES

    @property
    def memory_budget_mb(self) -> float:
        return round(self.effective_memory_budget_bytes / (1024 * 1024), 1)

    @property
    def concurrent_read_budget_bytes(self) -> int:
        return self.concurrent_read_connections * (READ_MMAP_SIZE_BYTES + READ_CACHE_SIZE_KIB * 1024)

    @property
    def concurrent_profile_budget_bytes(self) -> int:
        return (
            BULK_BUILD_MMAP_SIZE_BYTES
            + BULK_BUILD_CACHE_SIZE_KIB * 1024
            + 3 * (WRITE_MMAP_SIZE_BYTES + WRITE_CACHE_SIZE_KIB * 1024)
            + DAEMON_WRITE_MMAP_SIZE_BYTES
            + DAEMON_WRITE_CACHE_SIZE_KIB * 1024
            + BOUNDED_REPAIR_MMAP_SIZE_BYTES
            + BOUNDED_REPAIR_CACHE_SIZE_KIB * 1024
            + OBSERVATION_JOURNAL_CACHE_SIZE_KIB * 1024
        )

    @property
    def concurrent_allowance_bytes(self) -> int:
        return self.concurrent_read_budget_bytes + self.concurrent_profile_budget_bytes

    @property
    def memory_max_mb(self) -> float | None:
        return round(self.memory_max_bytes / (1024 * 1024), 1) if self.memory_max_bytes is not None else None

    @property
    def memory_high_mb(self) -> float | None:
        return round(self.memory_high_bytes / (1024 * 1024), 1) if self.memory_high_bytes is not None else None

    @property
    def at_risk_limits(self) -> tuple[str, ...]:
        """Which cgroup limit file(s), if any, sit at or below the computed budget.

        Either limit landing at or below the budget reproduces the 2026-07-31
        incident shape: `memory.high` throttles mapped/reclaimable pages before
        `memory.max` would ever OOM-kill, so `memory.high` is actually the
        more precise reproduction of what happened -- but a `memory.max` this
        low is also worth flagging, since it means the hard ceiling itself
        cannot even hold one worst-case concurrent footprint.
        """
        at_risk: list[str] = []
        if self.memory_max_bytes is not None and self.memory_max_bytes <= self.budget_bytes:
            at_risk.append("memory.max")
        if self.memory_high_bytes is not None and self.memory_high_bytes <= self.budget_bytes:
            at_risk.append("memory.high")
        return tuple(at_risk)


def check_mapped_bytes_budget_against_cgroup_limit(*, concurrent_read_connections: int = 4) -> MappedBytesBudgetCheck:
    """Compare the computed mmap/cache budget to this process' cgroup v2 memory limits.

    Reads `memory.max`/`memory.high` under `/sys/fs/cgroup/<this process' unified
    cgroup path>` via `polylogue.core.metrics`. Both are `None` when cgroup v2
    is not mounted, the controller isn't delegated (e.g. outside a cgroup, or a
    container without the `memory` controller), or the limit is literally
    `max` (unlimited) -- callers must treat `None` as "no limit detected", not
    as an error.
    """
    from polylogue.core.metrics import read_cgroup_memory_high_bytes, read_cgroup_memory_max_bytes

    return MappedBytesBudgetCheck(
        budget_bytes=mapped_bytes_budget(concurrent_read_connections=concurrent_read_connections),
        memory_max_bytes=read_cgroup_memory_max_bytes(),
        memory_high_bytes=read_cgroup_memory_high_bytes(),
        concurrent_read_connections=concurrent_read_connections,
        memory_budget_bytes=MEMORY_BUDGET_BYTES,
    )


def log_mapped_bytes_budget_check(logger: BoundLoggerLike, check: MappedBytesBudgetCheck | None = None) -> None:
    """Log the mapped-bytes budget vs. detected cgroup memory limit at startup.

    Call once at daemon startup and once at the start of an offline bulk
    rebuild -- the two paths that can hold a `BULK_BUILD_WRITE_CONNECTION_PROFILE`
    connection. Degrades gracefully (a debug-level line, never a raised
    exception) when no cgroup limit is detected at all, since that is the
    ordinary case for a dev-machine or non-cgroup-confined run, not an error.
    """
    if check is None:
        check = check_mapped_bytes_budget_against_cgroup_limit()
    if check.memory_max_bytes is None and check.memory_high_bytes is None:
        logger.debug(
            "mmap_budget_no_cgroup_limit_detected",
            memory_budget_bytes=check.effective_memory_budget_bytes,
            memory_budget_mb=check.memory_budget_mb,
            budget_bytes=check.budget_bytes,
            budget_mb=check.budget_mb,
            concurrent_allowance_bytes=check.concurrent_allowance_bytes,
            concurrent_read_budget_bytes=check.concurrent_read_budget_bytes,
            concurrent_profile_budget_bytes=check.concurrent_profile_budget_bytes,
            concurrent_read_connections=check.concurrent_read_connections,
        )
        return
    at_risk = check.at_risk_limits
    if at_risk:
        logger.warning(
            "mmap_budget_at_or_above_cgroup_limit",
            memory_budget_bytes=check.effective_memory_budget_bytes,
            memory_budget_mb=check.memory_budget_mb,
            budget_bytes=check.budget_bytes,
            budget_mb=check.budget_mb,
            concurrent_allowance_bytes=check.concurrent_allowance_bytes,
            concurrent_read_budget_bytes=check.concurrent_read_budget_bytes,
            concurrent_profile_budget_bytes=check.concurrent_profile_budget_bytes,
            memory_max_mb=check.memory_max_mb,
            memory_high_mb=check.memory_high_mb,
            at_risk_limits=list(at_risk),
            concurrent_read_connections=check.concurrent_read_connections,
        )
    else:
        logger.info(
            "mmap_budget_within_cgroup_limit",
            memory_budget_bytes=check.effective_memory_budget_bytes,
            memory_budget_mb=check.memory_budget_mb,
            budget_bytes=check.budget_bytes,
            budget_mb=check.budget_mb,
            concurrent_allowance_bytes=check.concurrent_allowance_bytes,
            concurrent_read_budget_bytes=check.concurrent_read_budget_bytes,
            concurrent_profile_budget_bytes=check.concurrent_profile_budget_bytes,
            memory_max_mb=check.memory_max_mb,
            memory_high_mb=check.memory_high_mb,
            concurrent_read_connections=check.concurrent_read_connections,
        )


# ---------------------------------------------------------------------------
# Lightweight factory functions — open + apply pragmas, no caching / schema / vec
# ---------------------------------------------------------------------------


_SIBLING_TIER_ATTACHMENTS: tuple[tuple[str, str], ...] = (
    ("source_tier", "source.db"),
    ("user_tier", "user.db"),
    ("embeddings", "embeddings.db"),
    ("ops_tier", "ops.db"),
)


def configured_archive_root(path: str | Path, archive_root: str | Path | None = None) -> Path:
    """Carry configured archive authority independently of a resolved Index target."""
    custody = current_sql_custody()
    configured_path = Path(path)
    if archive_root is None and custody is None and ".index-generations" in configured_path.parts:
        raise RuntimeError("an offline Index generation requires its explicit configured archive root")
    root = (
        Path(archive_root)
        if archive_root is not None
        else (custody.archive_root if custody is not None else Path(path).parent)
    )
    if custody is not None:
        if root.resolve() != custody.archive_root.resolve():
            raise UnleasedWriteError("SQLite configuration belongs to another admitted archive")
        custody.assert_namespace()
    return root


def _attach_sibling_tiers(conn: sqlite3.Connection, *, archive_root: Path) -> None:
    """Attach sibling archive tiers to an ``index.db`` connection (idempotent).

    Lets one-shot sync connections resolve cross-tier tables (e.g. source.db's
    ``raw_sessions``/``blob_refs``) by unqualified name. SQLite resolves
    unqualified names to ``main`` first, so index-tier tables are unaffected;
    only sibling-only tables resolve to their attached tier.
    """
    main_path: str | None = None
    attached: set[str] = set()
    with closing(conn.execute("PRAGMA database_list")) as cursor:
        databases = cursor.fetchall()
    for row in databases:
        schema_name = str(row[1])
        if schema_name == "main":
            main_path = str(row[2]) if row[2] else None
        else:
            attached.add(schema_name)
    if not main_path:
        return
    main = Path(main_path)
    if main.name != "index.db":
        return
    root = configured_archive_root(main, archive_root)
    custody = current_sql_custody()
    for schema_name, filename in _SIBLING_TIER_ATTACHMENTS:
        if schema_name in attached:
            continue
        sibling = root / filename
        if sibling.exists():
            tier = _archive_tier_for_path(sibling)
            if tier is not None:
                sibling_conn = open_readonly_connection(
                    sibling, tier=tier, validate_schema=False, timeout_class="background-read"
                )
                sibling_owner = NativeSQLCustodyOwner(
                    sibling_conn, lifetime_dependencies=current_native_sql_lifetimes()
                )
                try:
                    # An attached sibling keeps the absent-tier read answer; a
                    # writer that mutates a tier opens that tier directly.
                    _assert_schema_supported(sibling_conn, sibling, tier, allow_uninitialized_read=True)
                except BaseException as primary:
                    _close_failed_native_construction(sibling_owner, primary)
                    raise
                else:
                    sibling_owner.close()
            attach_database(conn, sibling, alias=schema_name)
    if custody is not None:
        custody.assert_namespace()


def _archive_tier_for_path(path: str | Path) -> ArchiveTier | None:
    """Resolve a conventional archive filename without importing at module load."""
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    return next((tier for tier in ArchiveTier if Path(path).name == f"{tier.value}.db"), None)


def _schema_skew_remedy(tier: ArchiveTier) -> str:
    """Describe the safe recovery route for a mismatched archive tier."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import archive_tier_spec

    spec = archive_tier_spec(tier)
    if spec.durability in {"rebuildable", "expensive_rebuild", "disposable"}:
        return (
            f"{tier.value}.db is {spec.durability} derived state; rebuild or recreate this tier "
            "from durable evidence with the current runtime before retrying"
        )
    return (
        f"{tier.value}.db is durable state; do not rebuild it. This runtime declares no migration from its "
        "schema version; open it with the runtime that wrote it"
    )


def _tier_holds_no_schema(conn: sqlite3.Connection) -> bool:
    """Report whether a tier file carries any non-internal schema object."""
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 1").fetchone()
    return row is None


def _assert_schema_supported(
    conn: sqlite3.Connection,
    path: str | Path,
    tier: ArchiveTier | None,
    *,
    allow_uninitialized_read: bool = False,
) -> None:
    """Reject a known archive tier before any caller can issue SQL against it."""
    from polylogue.core.errors import SchemaSkew
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    resolved_tier = tier if tier is not None else _archive_tier_for_path(path)
    if resolved_tier is None:
        return
    try:
        expected = ARCHIVE_VERSION_BY_TIER[resolved_tier]
    except KeyError as exc:
        raise ValueError(f"unknown archive tier: {resolved_tier!r}") from exc
    found = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if allow_uninitialized_read and found == 0 and _tier_holds_no_schema(conn):
        # A tier file with neither a version stamp nor any schema object has
        # never been provisioned. Reading it is reading an absent tier: the
        # caller fails on the missing table it asked for, which is a truthful
        # not-provisioned answer, where skew would misreport corruption.
        return
    if resolved_tier is ArchiveTier.INDEX and found == 0 and _tier_holds_no_schema(conn):
        return
    if found != expected:
        raise SchemaSkew(
            tier=resolved_tier.value,
            expected=expected,
            found=found,
            remedy=_schema_skew_remedy(resolved_tier),
        )
    _assert_derived_identity_supported(conn, resolved_tier)


def _assert_derived_identity_supported(conn: sqlite3.Connection, tier: ArchiveTier | None) -> None:
    """Check a derived tier's identity, not merely its version cursor.

    ``user_version`` tracks the numbered schema; it says nothing about the
    lowering, materializer and routing fingerprints the derived identity also
    covers, and a bootstrap route restamps it before this check ever reads it.
    A tier stamped by a different runtime therefore passes the version gate
    while carrying read models this runtime cannot interpret.
    """
    from polylogue.storage.sqlite.archive_tiers.schema_identity import DerivedTier

    if tier is None:
        return
    try:
        derived_tier = DerivedTier(tier.value)
    except ValueError:
        return
    from polylogue.storage.sqlite.schema_bootstrap import assert_derived_schema_identity

    assert_derived_schema_identity(conn, derived_tier.value)


def assert_tier_schema_supported(
    conn: sqlite3.Connection,
    path: str | Path,
    tier: ArchiveTier | None = None,
) -> None:
    """Reject a tier this runtime cannot serve, by version and by identity.

    Public so a bootstrap route that opens a not-yet-materialised tier with
    ``validate_schema=False`` can apply the check once it has stamped the
    schema. That route rewrites ``user_version`` while materialising, so the
    version alone proves nothing about the tier it just wrote over; the
    derived identity is what the stamp is for and is checked here rather than
    on every ordinary open.
    """
    # Read-only inspection can admit an uninitialized file. Owned Index
    # writers call this after canonical initialization and before writer
    # pragmas or DDL; the identity check below admits their actual handle.
    _assert_schema_supported(conn, path, tier, allow_uninitialized_read=True)
    _assert_derived_identity_supported(conn, tier if tier is not None else _archive_tier_for_path(path))


def open_connection(
    path: str | Path,
    *,
    timeout: float = DB_TIMEOUT,
    tier: ArchiveTier | None = None,
    validate_schema: bool = True,
    profile: SQLiteConnectionProfile = WRITE_CONNECTION_PROFILE,
    archive_root: str | Path | None = None,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """Open a read-write SQLite connection with canonical write pragmas applied.

    This is a lightweight one-shot factory: it opens the file, applies the
    write-time PRAGMA profile, attaches sibling archive tiers (so cross-tier
    reads resolve), and returns the connection.  The caller owns the connection
    lifecycle (must close it).

    For the thread-local cached archive connection used by the async runtime,
    use ``connection_context`` from ``connection.py`` instead.

    ``check_same_thread=False`` is for a handle that outlives the thread that
    opened it -- the cold build's ``ops.db`` checkpoint holder is opened on one
    write-coordinator thread, re-asserted on the next pass' thread and closed
    on whichever thread settles the generation. It is not a concurrency
    licence: ``sqlite3`` here is serialized (``threadsafety == 3``), so the
    handle is thread-safe, but the *writes* it serves are still ordered by the
    single-writer lease. A caller that has no such ordering must leave this
    ``True`` and get the thread check.
    """
    if profile.role != "write":
        raise ValueError("open_connection requires a write profile")
    root = configured_archive_root(path, archive_root)
    require_write_lease(f"open_connection({path})", archive_root=root)
    conn = _connect_archive_writer(
        path, profile=profile, archive_root=root, timeout=timeout, check_same_thread=check_same_thread
    )
    owner = NativeSQLCustodyOwner(conn)
    try:
        if validate_schema:
            _assert_schema_supported(conn, path, tier)
        for stmt in write_connection_pragma_statements(profile):
            if stmt.startswith("PRAGMA journal_mode"):
                execute_pragma_statement(conn, stmt)
        _attach_sibling_tiers(conn, archive_root=root)
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    return owner.handoff()


def open_daemon_connection(
    path: str | Path,
    *,
    timeout: float = DB_TIMEOUT,
    busy_timeout_ms: int | None = None,
    tier: ArchiveTier | None = None,
    validate_schema: bool = True,
    archive_root: str | Path | None = None,
) -> sqlite3.Connection:
    """Open a read-write SQLite connection for daemon maintenance/ops writes.

    Long-running daemon loops write small status, cursor, telemetry, and
    maintenance rows. They should not inherit the full batch-ingest cache and
    mmap profile, because systemd charges their SQLite page cache to the
    service cgroup for the lifetime of the process.
    """
    root = configured_archive_root(path, archive_root)
    require_write_lease(f"open_daemon_connection({path})", archive_root=root)
    profile = (
        DAEMON_WRITE_CONNECTION_PROFILE
        if busy_timeout_ms is None
        else replace(DAEMON_WRITE_CONNECTION_PROFILE, busy_timeout_ms=busy_timeout_ms)
    )
    conn = _connect_archive_writer(path, profile=profile, archive_root=root, timeout=timeout)
    owner = NativeSQLCustodyOwner(conn)
    try:
        if validate_schema:
            _assert_schema_supported(conn, path, tier)
        for stmt in write_connection_pragma_statements(profile):
            if stmt.startswith("PRAGMA journal_mode"):
                execute_pragma_statement(conn, stmt)
        _attach_sibling_tiers(conn, archive_root=root)
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    return owner.handoff()


@contextmanager
def owned_daemon_connection(
    path: str | Path,
    *,
    archive_root: str | Path,
) -> Iterator[sqlite3.Connection]:
    """Keep one-shot daemon SQL in the admitted worker's actual custody."""
    owner = NativeSQLCustodyOwner(
        open_daemon_connection(path, archive_root=archive_root), lifetime_dependencies=current_native_sql_lifetimes()
    )
    try:
        connection = owner.connection
        assert connection is not None
        yield connection
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    else:
        owner.close()


def descriptor_alias_path(opened_fd: int) -> Path | None:
    """Return a validated portable pathname alias for an opened descriptor."""

    descriptor_metadata = os.fstat(opened_fd)
    for directory in ("/dev/fd", "/proc/self/fd"):
        candidate = Path(directory) / str(opened_fd)
        try:
            alias_metadata = os.stat(candidate)
        except OSError:
            continue
        if (alias_metadata.st_dev, alias_metadata.st_ino) == (
            descriptor_metadata.st_dev,
            descriptor_metadata.st_ino,
        ):
            return candidate
    return None


def _descriptor_database_uri(opened_main_fd: int, suffix: str) -> str | None:
    """Return a validated descriptor URI on platforms that expose one."""
    alias = descriptor_alias_path(opened_main_fd)
    return None if alias is None else f"file:{alias}{suffix}"


class LiveGenerationImmutableError(ValueError):
    """An ``immutable=1`` open was asked of a file that still carries live state.

    SQLite's immutable mode reads the main file alone. A non-empty ``-wal`` or
    rollback journal beside it holds committed state that such a reader would
    silently skip, and a writer may still be appending to it, so the file is
    not a sealed generation. Freezing a snapshot means checkpointing that state
    into the file first; this refusal is what makes skipping that step visible.
    """

    code = "immutable_over_live_state"


def _rollback_journal_header_is_zeroed(journal: Path) -> bool:
    """True when a rollback journal's magic header is zeroed (inactive, not hot)."""
    try:
        with journal.open("rb") as handle:
            header = handle.read(8)
    except FileNotFoundError:
        return True
    return header == bytes(len(header))


def _refuse_immutable_over_live_state(path: str | Path) -> None:
    # SQLite opens the symlink target, so its sidecars sit beside the target.
    database = Path(path).resolve()
    for suffix in ("-wal", "-journal"):
        sidecar = database.with_name(database.name + suffix)
        try:
            size = sidecar.stat().st_size
        except FileNotFoundError:
            continue
        if size > 0 and suffix == "-journal" and _rollback_journal_header_is_zeroed(sidecar):
            # journal_mode=PERSIST leaves a non-empty journal whose header is
            # zeroed after every commit; SQLite treats it as inactive, so the
            # main file already holds all committed state.
            continue
        if size > 0:
            raise LiveGenerationImmutableError(
                f"cannot open {database} immutable: {sidecar.name} holds {size} bytes of state "
                "the main file does not; checkpoint it into a sealed generation first"
            )


def open_readonly_connection(
    path: str | Path,
    *,
    timeout: float | None = None,
    immutable: bool = False,
    opened_main_fd: int | None = None,
    tier: ArchiveTier | None = None,
    validate_schema: bool = True,
    profile: SQLiteConnectionProfile = READ_CONNECTION_PROFILE,
    timeout_class: str = "interactive-read",
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """Open a read-only SQLite connection with canonical read pragmas applied.

    Uses ``file:...?mode=ro`` URI mode to guarantee no write locks are taken.
    Returns ``None`` / raises ``sqlite3.OperationalError`` if the database file
    does not exist.

    ``immutable`` additionally sets SQLite's ``immutable=1`` URI parameter,
    which tells SQLite the file is guaranteed not to change for the lifetime
    of the connection: it skips locking and WAL/journal presence checks, and
    will not create a ``-shm``/``-wal`` sidecar itself. This is only correct
    against a verified-stable snapshot (e.g. a stopped-daemon clone the caller
    has already confirmed has no WAL/SHM/journal sidecars) -- never against a
    database a live process (such as ``polylogued``) might still be writing.
    Callers passing ``immutable=True`` own that precondition check; this
    helper does not perform it, since the check is specific to how the caller
    obtained the snapshot.

    When ``opened_main_fd`` is supplied, the reader is bound to that opened
    inode through a validated ``/dev/fd`` or ``/proc/self/fd`` alias. A caller
    that needs descriptor binding fails closed when neither alias is available.

    ``validate_schema=False`` is reserved for diagnostic readers that need to
    inspect a tier before reporting its schema mismatch. It does not change the
    read-only connection profile or grant write access.

    ``check_same_thread=False`` is reserved for a cached handle whose caller
    already serializes access and may close it from a different thread.
    """
    return _open_readonly_owner(
        path,
        timeout=timeout,
        immutable=immutable,
        opened_main_fd=opened_main_fd,
        tier=tier,
        validate_schema=validate_schema,
        profile=profile,
        timeout_class=timeout_class,
        check_same_thread=check_same_thread,
    ).handoff()


def _open_readonly_owner(
    path: str | Path,
    *,
    timeout: float | None = None,
    immutable: bool = False,
    opened_main_fd: int | None = None,
    tier: ArchiveTier | None = None,
    validate_schema: bool = True,
    profile: SQLiteConnectionProfile = READ_CONNECTION_PROFILE,
    timeout_class: str = "interactive-read",
    check_same_thread: bool = True,
    lifetime_dependencies: tuple[object, ...] = (),
    terminal_parent: SQLCustodyOwner | None = None,
) -> NativeSQLCustodyOwner:
    """Register read construction and its explicit artifact lifetime before SQL."""
    from polylogue.storage.sqlite.population_admission import assert_population_admitted

    assert_population_admitted(path)
    if profile.role != "read" or not profile.query_only:
        raise ValueError("open_readonly_connection requires a query-only read profile")
    if timeout_class not in READ_PROFILES:
        raise ValueError(f"unknown SQLite timeout class: {timeout_class}")
    if profile is READ_CONNECTION_PROFILE and timeout_class != "interactive-read":
        profile = READ_PROFILES[timeout_class]
    if immutable and profile.generation_identity != "sealed":
        if not any(profile is named for named in (READ_CONNECTION_PROFILE, *READ_PROFILES.values())):
            raise ValueError(
                "immutable SQLite mode requires a sealed-generation read profile; a live-generation "
                "profile cannot promise the file will not change under the connection"
            )
        profile = SEALED_READ_CONNECTION_PROFILE
    immutable = immutable or profile.immutable
    if immutable and opened_main_fd is None:
        _refuse_immutable_over_live_state(path)
    # ``None`` selects the profile's lock wait. An explicit value is the
    # caller's bound and replaces the profile's busy_timeout as well: the
    # PRAGMA runs after connect and would otherwise silently win.
    explicit_timeout = timeout is not None
    timeout = profile.timeout_seconds if timeout is None else timeout
    suffix = "?mode=ro&immutable=1" if immutable else "?mode=ro"
    if opened_main_fd is not None and immutable:
        raise ValueError("an opened SQLite file descriptor cannot use immutable mode")
    opened_fd = opened_main_fd
    if opened_fd is None:
        # Percent-encode the path: an unescaped '?' or '#' in a filename would
        # otherwise be parsed as the URI's own query or fragment delimiter and
        # silently open a different file, or none.
        database_uri = f"file:{quote(str(path))}{suffix}"
    else:
        descriptor_uri = _descriptor_database_uri(opened_fd, suffix)
        if descriptor_uri is None:
            raise RuntimeError(f"cannot open selected SQLite database through a descriptor-bound path: {path}")
        database_uri = descriptor_uri
    conn = connect_measured(database_uri, uri=True, timeout=timeout, check_same_thread=check_same_thread)
    owner = NativeSQLCustodyOwner(conn, lifetime_dependencies=lifetime_dependencies, terminal_parent=terminal_parent)
    try:
        if validate_schema:
            _assert_schema_supported(conn, path, tier, allow_uninitialized_read=True)
        for stmt in profile.pragma_statements:
            if explicit_timeout and stmt.startswith("PRAGMA busy_timeout"):
                stmt = f"PRAGMA busy_timeout = {int(timeout * 1000)}"
            conn.execute(stmt)
        conn.set_authorizer(_authorize_read_operation)
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    return owner


def _authorize_read_operation(
    action: int,
    argument1: str | None,
    argument2: str | None,
    _database: str | None,
    _trigger: str | None,
) -> int:
    """Keep a profiled reader read-only after its connection setup."""
    if action in {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_SAVEPOINT,
        sqlite3.SQLITE_RECURSIVE,
        sqlite3.SQLITE_DETACH,
    }:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA:
        pragma = (argument1 or "").lower()
        read_pragmas = {
            "application_id",
            "busy_timeout",
            "cache_size",
            "compile_options",
            "database_list",
            "data_version",
            "encoding",
            "foreign_keys",
            "foreign_key_check",
            "foreign_key_list",
            "freelist_count",
            "index_info",
            "index_list",
            "index_xinfo",
            "integrity_check",
            "journal_mode",
            "journal_size_limit",
            "locking_mode",
            "mmap_size",
            "page_count",
            "page_size",
            "query_only",
            "recursive_triggers",
            "quick_check",
            "schema_version",
            "synchronous",
            "table_info",
            "table_list",
            "table_xinfo",
            "temp_store",
            "trusted_schema",
            "user_version",
            "wal_autocheckpoint",
        }
        parameterized_reads = {
            "foreign_key_check",
            "foreign_key_list",
            "index_info",
            "index_list",
            "index_xinfo",
            "table_info",
            "table_xinfo",
        }
        if pragma in read_pragmas and (argument2 is None or pragma in parameterized_reads):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_ATTACH and argument1 is not None:
        uri = urlsplit(argument1)
        if uri.scheme == "file" and parse_qs(uri.query).get("mode") == ["ro"]:
            return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def attach_readonly_database(
    conn: sqlite3.Connection,
    path: str | Path,
    *,
    alias: str,
    immutable: bool = False,
) -> None:
    """Attach a second read-only database to a profiled reader.

    SQLite's authorizer receives a NULL filename while preparing a
    parameterized ATTACH, so this tightly scoped helper temporarily removes
    it for the single ATTACH statement. ``query_only`` remains enabled, and
    the attached URI is always opened read-only.
    """
    from polylogue.storage.sqlite.population_admission import assert_population_admitted

    assert_population_admitted(path)
    if conn.execute("PRAGMA query_only").fetchone()[0] != 1:
        raise ValueError("read-only attachment requires a query-only connection")
    if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", alias) is None:
        raise ValueError(f"invalid SQLite attachment alias: {alias!r}")
    if immutable:
        _refuse_immutable_over_live_state(path)
    uri = f"file:{quote(str(path))}?mode=ro" + ("&immutable=1" if immutable else "")
    conn.set_authorizer(None)
    try:
        with closing(conn.execute(f"ATTACH DATABASE ? AS {alias}", (uri,))):
            pass
    finally:
        conn.set_authorizer(_authorize_read_operation)


def attach_database(conn: sqlite3.Connection, path: str | Path, *, alias: str) -> None:
    """Attach ``path`` as ``alias`` with the connection's own access mode.

    A query-only reader carries the read authorizer, which denies a plain
    parameterized ATTACH, so its attachment goes through
    :func:`attach_readonly_database` and is opened read-only. Any other
    connection attaches the file directly.
    """
    from polylogue.storage.sqlite.population_admission import assert_population_admitted

    assert_population_admitted(path)
    with closing(conn.execute("PRAGMA query_only")) as cursor:
        query_only = cursor.fetchone()[0]
    if query_only == 1:
        attach_readonly_database(conn, path, alias=alias)
        return
    if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", alias) is None:
        raise ValueError(f"invalid SQLite attachment alias: {alias!r}")
    # Source mutations use its admitted direct connection. Cross-tier handles
    # retain read access without acquiring another Source write surface.
    attachment = f"file:{quote(str(Path(path).resolve()))}?mode=ro" if Path(path).name == "source.db" else str(path)
    with closing(conn.execute(f"ATTACH DATABASE ? AS {alias}", (attachment,))):
        pass


def _authorize_read_temp_operation(
    action: int,
    argument1: str | None,
    argument2: str | None,
    database: str | None,
    trigger: str | None,
) -> int:
    if database == "temp" and action in {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
        sqlite3.SQLITE_REINDEX,
    }:
        return sqlite3.SQLITE_OK
    return _authorize_read_operation(action, argument1, argument2, database, trigger)


@contextmanager
def readonly_temp_staging(
    conn: sqlite3.Connection,
    *,
    temp_store: Literal["FILE", "MEMORY"] | None = None,
) -> Iterator[None]:
    """Permit TEMP projection rows while persistent attached tiers stay read-only.

    ``temp_store`` may be selected before the projection is created. SQLite
    drops existing TEMP objects when this setting changes, so this option is
    restricted to a connection whose TEMP schema is empty.
    """
    if conn.execute("PRAGMA query_only").fetchone()[0] != 1:
        raise ValueError("TEMP staging requires a query-only reader")
    if temp_store not in {None, "FILE", "MEMORY"}:
        raise ValueError("TEMP staging store must be FILE or MEMORY")
    if temp_store is not None and conn.execute("SELECT 1 FROM sqlite_temp_master LIMIT 1").fetchone() is not None:
        raise ValueError("TEMP staging store can only be selected before TEMP objects exist")
    conn.set_authorizer(None)
    try:
        conn.execute("PRAGMA query_only = OFF")
        if temp_store is not None:
            conn.execute(f"PRAGMA temp_store = {temp_store}")
        conn.set_authorizer(_authorize_read_temp_operation)
        yield
    finally:
        conn.set_authorizer(None)
        try:
            conn.execute("PRAGMA query_only = ON")
        finally:
            conn.set_authorizer(_authorize_read_operation)


@contextmanager
def one_shot_diagnostic_read(
    path: str | Path,
    *,
    tier: ArchiveTier | None = None,
) -> Iterator[sqlite3.Connection]:
    """Open one non-paginated diagnostic probe under the interactive read profile.

    This is deliberately narrower than :func:`open_readonly_connection`:
    it is for a small probe whose result is consumed before request or
    presentation work begins (for example, checking a schema version).  A
    caller that retains SQLite state while paging or rendering must use a
    :func:`read_frame` instead, so its live snapshot has a declared bounded
    lifetime and can rebind safely.

    Diagnostics may need to read an unsupported schema in order to explain
    it, hence schema validation is intentionally disabled here.  That does
    not relax SQLite's ``mode=ro`` or ``query_only`` enforcement.
    """
    conn = open_readonly_connection(
        path,
        tier=tier,
        validate_schema=False,
        timeout_class="interactive-read",
    )
    owner = NativeSQLCustodyOwner(conn, lifetime_dependencies=current_native_sql_lifetimes())
    try:
        yield conn
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    else:
        owner.close()


def open_sealed_staging_connection(
    path: str | Path,
    *,
    tier: ArchiveTier | None = None,
    validate_schema: bool = True,
) -> sqlite3.Connection:
    """Open an immutable source image with sealed, TEMP-only staging.

    This is a deliberately dedicated exception for historical liveness
    classification.  The main database is opened with SQLite's
    ``mode=ro&immutable=1`` URI and remains protected by a fail-closed
    authorizer; the only writes admitted after setup target the connection's
    in-memory TEMP schema.  It does not accept caller-selected profiles,
    descriptors, attachments, or write options.
    """

    profile = SEALED_STAGING_CONNECTION_PROFILE
    _refuse_immutable_over_live_state(path)
    database_uri = f"file:{quote(str(path))}?mode=ro&immutable=1"
    conn = connect_measured(database_uri, uri=True, timeout=profile.timeout_seconds)
    owner = NativeSQLCustodyOwner(conn)
    try:
        if validate_schema:
            _assert_schema_supported(conn, path, tier, allow_uninitialized_read=True)
        # Apply only this bounded profile's setup statements.  In particular,
        # do not copy READ_CONNECTION_PROFILE here: its query_only=ON is
        # exactly what prevents TEMP staging.
        for statement in profile.pragma_statements:
            conn.execute(statement)
        conn.set_authorizer(_authorize_sealed_staging_operation)
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    return owner.handoff()


def open_profiled_connection(
    path: str | Path,
    *,
    profile: SQLiteConnectionProfile,
    timeout: float | None = None,
    immutable: bool = False,
    opened_main_fd: int | None = None,
    tier: ArchiveTier | None = None,
) -> sqlite3.Connection:
    """Open a connection from an explicit named profile.

    Read profiles are always enforced at SQLite's boundary.  Write profiles
    use the existing writer factory so attachment and schema checks remain
    identical to ordinary archive writes.
    """
    if profile.role == "read":
        if not profile.query_only:
            raise ValueError("read profiles must enable query_only")
        return open_readonly_connection(
            path,
            timeout=profile.timeout_seconds if timeout is None else timeout,
            immutable=immutable,
            opened_main_fd=opened_main_fd,
            tier=tier,
            profile=profile,
        )
    if immutable or opened_main_fd is not None:
        raise ValueError("writer profiles cannot use immutable or descriptor-bound reads")
    return open_connection(
        path,
        timeout=profile.timeout_seconds if timeout is None else timeout,
        tier=tier,
        profile=profile,
    )


def open_isolated_write_connection(
    path: str | Path,
    *,
    purpose: str,
    profile: SQLiteConnectionProfile = ISOLATED_TIER_WRITE_PROFILE,
    timeout: float | None = None,
    archive_root: str | Path | None = None,
    mutation_permit: KnownTierWriteAuthority | None = None,
) -> sqlite3.Connection:
    """Open one writable tier without attaching sibling databases.

    Snapshot/checkpoint and other one-tier operations must still pass through
    the same lease boundary as ordinary archive writes.  Keeping this factory
    separate prevents those operations from accidentally widening their
    transaction to attached tiers.
    """
    if profile.role != "write":
        raise ValueError("open_isolated_write_connection requires a write profile")
    require_write_lease(purpose, archive_root=archive_root)
    conn = _connect_archive_writer(
        path,
        profile=profile,
        archive_root=archive_root,
        mutation_permit=mutation_permit,
        timeout=profile.timeout_seconds if timeout is None else timeout,
    )
    owner = NativeSQLCustodyOwner(
        conn, terminal_parent=None if mutation_permit is None else mutation_permit.terminal_parent
    )
    try:
        statements = write_connection_pragma_statements(profile)
        if mutation_permit is None:
            for statement in statements:
                if statement.startswith("PRAGMA journal_mode"):
                    execute_pragma_statement(conn, statement)
        else:
            mutation_permit.configure_mutation_connection(
                conn, (*statements, "PRAGMA foreign_keys = ON", "PRAGMA recursive_triggers = ON")
            )
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    return owner.handoff() if mutation_permit is None else owner.require_connection()


# ---------------------------------------------------------------------------
# Read frames: a bounded, rebindable read connection over one generation
# ---------------------------------------------------------------------------
#
# A ``mode=ro`` connection is not by itself a bounded read, and it is also not
# by itself a cost: an idle one holds no read mark and a recurring PASSIVE
# checkpoint recycles the log straight past it. What turns PASSIVE into a no-op
# and lets the WAL grow without limit is an *open read transaction* -- in
# practice a cursor that is still being stepped. ``sqlite3.Connection`` cannot
# report that (``in_transaction`` stays False for a SELECT in autocommit while
# the cursor pins frames), so the frame has to own the stepping to bound it.
#
# A read frame therefore gives a live-generation reader three things: an age
# past which it must rebind, a generation identity that says whether what it
# rebound to is still the thing it was reading, and ``stream`` -- the one route
# that steps a cursor while re-checking that age between rows and releases the
# cursor before raising, so an expiry ends the WAL pin instead of reporting it.


class ReadFrameExpiredError(RuntimeError):
    """A live read frame outlived the maximum snapshot age its profile declares."""

    code = "read_frame_expired"


class StaleContinuationError(RuntimeError):
    """A continuation cannot be resumed against an equivalent frame.

    Raised instead of resuming where the anchor no longer holds the position:
    continuing there would skip or duplicate rows, and only the caller can
    decide which of those it can tolerate.
    """

    code = "stale_continuation"


class ReadFrameCancelledError(RuntimeError):
    """The frame's in-flight statement was cancelled by its owner."""

    code = "read_frame_cancelled"


@dataclass(frozen=True, slots=True)
class GenerationToken:
    """Identity of the generation a frame is bound to.

    File identity, because that is what a generation swap moves and what stays
    comparable between two connections. Content freshness *within* one
    generation is a separate question that only the open connection can answer
    (``PRAGMA data_version`` is explicitly not meaningful across connections),
    so the frame tracks that separately.
    """

    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class ReadContinuation:
    """A resumable position in a paged read, and the anchor that proves it.

    ``anchor_sql`` must select the row the page stopped at and return its
    position as the first column, so a rebound frame can prove the position
    still means the same thing rather than assuming it.
    """

    position: object
    anchor_sql: str
    anchor_params: tuple[object, ...] = ()
    generation: GenerationToken | None = None
    #: Which frame incarnation produced this continuation. A rebind starts a
    #: new one, so a continuation can never be waved through on generation
    #: identity alone after the frame it was produced on was replaced.
    epoch: int = 0


@dataclass(frozen=True, slots=True)
class ReadFrameStatus:
    """One live read frame, as the recurring checkpoint owner sees it."""

    path: Path
    timeout_class: str
    age_s: float
    max_snapshot_age_s: float | None
    #: Whether a cursor opened through :meth:`ReadFrame.stream` is in flight.
    #: This is the only state that provably pins WAL frames, and the only
    #: reason a frame belongs in a blocked checkpoint's evidence.
    streaming: bool
    reason: str | None = None

    @property
    def overdue(self) -> bool:
        return self.max_snapshot_age_s is not None and self.age_s > self.max_snapshot_age_s

    def describe(self) -> str:
        declared = "none" if self.max_snapshot_age_s is None else f"{self.max_snapshot_age_s:.0f}s"
        suffix = f" ({self.reason})" if self.reason else ""
        return f"{self.path.name}:{self.timeout_class} age={self.age_s:.1f}s max={declared}{suffix}"


# The process-local read-snapshot registry. ``connection_profile`` owns both
# halves of the coupled decision, so the recurring checkpoint owner can name the
# frames that pinned it instead of only naming a PID from a ``/proc`` walk that
# cannot distinguish an idle handle from an open read transaction. Strong
# ownership also preserves failed cleanup until the actual handle closes;
# every frame caller must explicitly close its operation's frame.
_LIVE_READ_FRAMES: set[ReadFrame] = set()
_LIVE_READ_FRAMES_LOCK = threading.Lock()
_FORK_ABANDONED_READ_FRAMES: list[ReadFrame] = []


def _before_read_frame_fork() -> None:
    _LIVE_READ_FRAMES_LOCK.acquire()


def _after_read_frame_fork_parent() -> None:
    _LIVE_READ_FRAMES_LOCK.release()


def _abandon_read_frames_after_fork() -> None:
    global _LIVE_READ_FRAMES, _LIVE_READ_FRAMES_LOCK
    _FORK_ABANDONED_READ_FRAMES.extend(_LIVE_READ_FRAMES)
    _LIVE_READ_FRAMES = set()
    _LIVE_READ_FRAMES_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_before_read_frame_fork,
        after_in_parent=_after_read_frame_fork_parent,
        after_in_child=_abandon_read_frames_after_fork,
    )


def live_read_frames() -> tuple[ReadFrameStatus, ...]:
    """Every read frame still open in this process, oldest first."""
    with _LIVE_READ_FRAMES_LOCK:
        frames = tuple(_LIVE_READ_FRAMES)
    statuses = [frame.status() for frame in frames]
    return tuple(sorted(statuses, key=lambda status: status.age_s, reverse=True))


def pinning_read_frames(path: Path | str | None = None) -> tuple[ReadFrameStatus, ...]:
    """Live frames holding an open read transaction, optionally over one file.

    ``path`` is compared by resolved filesystem identity so a generation reached
    through a symlinked ``index.db`` matches the tier a checkpoint names.
    """
    target = Path(path).resolve(strict=False) if path is not None else None
    return tuple(
        status
        for status in live_read_frames()
        if status.streaming and (target is None or status.path.resolve(strict=False) == target)
    )


def _generation_token(path: Path) -> GenerationToken:
    stat = path.stat()
    return GenerationToken(device=stat.st_dev, inode=stat.st_ino)


def _data_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA main.data_version").fetchone()[0])


class ReadFrame:
    """A read connection bound to one generation for a declared maximum age.

    Not thread-safe: a frame belongs to the request or page loop that opened
    it. ``cancel`` is the one exception, and only for a profile that declares
    cancellation support -- it interrupts an in-flight statement from another
    thread, which is what SQLite's ``interrupt`` is for.
    """

    __slots__ = (
        "_cancelled",
        "_conn",
        "_cursors",
        "_data_version",
        "_epoch",
        "_generation",
        "_opened_at",
        "_path",
        "_profile",
        "_reason",
        "_sql_owner",
        "_tier",
        "_timeout_class",
    )

    def __init__(
        self,
        path: Path | str,
        *,
        profile: SQLiteConnectionProfile,
        tier: ArchiveTier | None = None,
        timeout_class: str = "unnamed",
        reason: str | None = None,
    ) -> None:
        if profile.role != "read" or not profile.query_only:
            raise ValueError("a read frame requires a query-only read profile")
        if profile.generation_identity == "live" and profile.max_snapshot_age_s is None:
            # A live generation changes under the reader, so a frame over one
            # with no declared maximum is precisely the unbounded WAL pin this
            # class exists to prevent. Refusing here is what stops a caller
            # from obtaining one by handing in a profile with the bound removed.
            raise ValueError(
                f"a live-generation read frame over {path} must declare max_snapshot_age_s; "
                "use read_frame(..., max_snapshot_age_s=..., reason=...) to extend the bound, "
                "or a sealed-generation profile if the file genuinely cannot change"
            )
        if profile.max_snapshot_age_s is not None and (
            not math.isfinite(profile.max_snapshot_age_s) or profile.max_snapshot_age_s <= 0
        ):
            raise ValueError("a read-frame snapshot bound must be finite and positive")
        self._path = Path(path)
        self._profile = profile
        self._tier = tier
        self._timeout_class = timeout_class
        self._reason = reason
        self._cancelled = False
        self._epoch = 0
        self._cursors: set[sqlite3.Cursor] = set()
        self._opened_at = time.monotonic()
        self._conn = self._open()
        self._initialize_opened_connection()

    def _initialize_opened_connection(self) -> None:
        try:
            self._opened_at = time.monotonic()
            self._data_version = _data_version(self._conn)
            self._install_progress_handler()
        except BaseException as primary:
            _close_failed_native_construction(self._sql_owner, primary)
            raise
        with _LIVE_READ_FRAMES_LOCK:
            _LIVE_READ_FRAMES.add(self)

    def _open(self) -> sqlite3.Connection:
        # Generation promotion swaps the configured pointer. Open its selected
        # physical leaf and capture that leaf's identity, so a later pointer
        # observation cannot label a predecessor handle with the successor inode.
        try:
            selected_path = self._path.resolve(strict=True)
            generation = _generation_token(selected_path)
        except FileNotFoundError as exc:
            raise sqlite3.OperationalError(f"unable to open database file: {self._path}") from exc
        try:
            conn = open_readonly_connection(
                selected_path,
                profile=self._profile,
                immutable=self._profile.immutable,
                tier=self._tier,
            )
        except NativeConnectionSettlementError as cleanup:
            self._sql_owner = cleanup.owner
            cleanup.owner.frame = self
            with _LIVE_READ_FRAMES_LOCK:
                _LIVE_READ_FRAMES.add(self)
            raise
        self._sql_owner = NativeSQLCustodyOwner(conn, frame=self)
        try:
            if _generation_token(selected_path) != generation:
                raise StaleContinuationError(f"selected read generation changed while opening {self._path}")
            self._generation = generation
            conn.row_factory = sqlite3.Row
        except BaseException as primary:
            _close_failed_native_construction(self._sql_owner, primary)
            raise
        return conn

    def _install_progress_handler(self) -> None:
        # A row can take arbitrarily long to compute. Checking only between
        # yielded rows would let one SQLite step pin a WAL frame past its age.
        self._conn.set_progress_handler(lambda: int(self._cancelled or self.expired), 1000)

    def _raise_if_interrupted(self, exc: sqlite3.OperationalError) -> None:
        if self._cancelled:
            raise ReadFrameCancelledError(f"read frame over {self._path} was cancelled") from exc
        if self.expired:
            self.check()

    # -- identity and lifetime ------------------------------------------------

    @property
    def connection(self) -> sqlite3.Connection:
        """The bound connection, refused once the frame has expired.

        Refusing here is what makes the declared maximum age load-bearing: a
        caller that holds a frame past it gets a typed error instead of a
        silently unbounded WAL pin.
        """
        self.check()
        return self._conn

    @property
    def generation(self) -> GenerationToken:
        return self._generation

    @property
    def epoch(self) -> int:
        """How many times this frame has been rebound."""
        return self._epoch

    @property
    def profile(self) -> SQLiteConnectionProfile:
        return self._profile

    @property
    def age_s(self) -> float:
        return time.monotonic() - self._opened_at

    @property
    def expired(self) -> bool:
        max_age = self._profile.max_snapshot_age_s
        return max_age is not None and self.age_s > max_age

    @property
    def streaming(self) -> bool:
        """Whether a :meth:`stream` cursor is in flight, i.e. pinning WAL frames."""
        return bool(self._cursors)

    def status(self) -> ReadFrameStatus:
        return ReadFrameStatus(
            path=self._path,
            timeout_class=self._timeout_class,
            age_s=self.age_s,
            max_snapshot_age_s=self._profile.max_snapshot_age_s,
            streaming=self.streaming,
            reason=self._reason,
        )

    def _require_read_owner(self) -> None:
        self._sql_owner._require_owner()
        if self._sql_owner.close_required:
            if self._sql_owner.connection is not None:
                raise NativeConnectionSettlementError(
                    self._sql_owner,
                    RuntimeError("read frame requires terminal cleanup"),
                )
            raise RuntimeError("read frame has already closed")

    def check(self) -> None:
        """Raise if this frame may no longer be read from."""
        self._require_read_owner()
        if self._cancelled:
            raise ReadFrameCancelledError(f"read frame over {self._path} was cancelled")
        if self.expired:
            declared = f"{self._profile.max_snapshot_age_s:.1f}s"
            extension = f" (extended for {self._reason})" if self._reason else ""
            raise ReadFrameExpiredError(
                f"read frame over {self._path} reached {self.age_s:.1f}s against a declared "
                f"{declared} maximum for timeout class {self._timeout_class}{extension}; "
                "rebind it or finish the read"
            )

    # -- bounded streaming ----------------------------------------------------

    def stream(
        self,
        sql: str,
        parameters: Sequence[object] = (),
    ) -> Generator[sqlite3.Row, None, None]:
        """Step one cursor under this frame's declared maximum snapshot age.

        This is the only supported way to hold a SQLite read transaction open
        across other work. A cursor that is stepped lazily pins every WAL frame
        written since it started, which is what makes the recurring PASSIVE
        checkpoint reclaim nothing; a caller that pulls rows through here gets
        the declared bound applied between rows rather than only at the moment
        it first asked for the connection.

        On expiry the cursor is closed before :class:`ReadFrameExpiredError`
        reaches the caller, so the refusal *ends* the pin rather than merely
        reporting it. That ordering is the point: a typed error that left the
        cursor open would name the problem and keep causing it.

        Deliberately a generator rather than a plain iterator: a caller that
        abandons the read part-way calls ``close()`` to end the pin at a point
        it chooses, instead of leaving it to garbage collection.
        """
        self.check()
        cursor: sqlite3.Cursor | None = None
        try:
            try:
                cursor = self._conn.execute(sql, tuple(parameters))
                self._cursors.add(cursor)
                while True:
                    self.check()
                    try:
                        row = next(cursor)
                    except StopIteration:
                        break
                    yield row
            except sqlite3.OperationalError as exc:
                self._raise_if_interrupted(exc)
                raise
        finally:
            # A suspended iterator can be resumed or closed by another task or
            # fork child. Refuse before touching SQLite, retaining the cursor
            # for terminal cleanup by the original frame owner.
            self._sql_owner._require_owner()
            if cursor is not None and cursor in self._cursors:
                close_connection_cursor(self._sql_owner.require_connection(), cursor)
                self._cursors.remove(cursor)

    def revalidate(self) -> bool:
        """Whether this frame still sees exactly what it was opened on.

        Both halves matter: the generation may have been swapped under the
        frame, or another connection may have committed into the same one.
        """
        self._require_read_owner()
        try:
            if _generation_token(self._path) != self._generation:
                return False
            return _data_version(self._conn) == self._data_version
        except (OSError, sqlite3.Error):
            return False

    def rebind(self) -> GenerationToken:
        """Reopen against the current generation, releasing the pinned frames.

        A sealed generation has nothing to rebind to: it cannot change, so a
        rebind request against one is a caller error rather than a no-op that
        hides a wrong profile choice.
        """
        if self._profile.generation_identity == "sealed":
            raise ValueError(f"a sealed-generation read frame over {self._path} has nothing to rebind to")
        if self.streaming:
            # Rebinding closes the connection the in-flight cursor is stepping.
            # Refusing is the loud half of the bound: a stream that outlived its
            # age must end with a typed expiry, not be silently re-opened
            # underneath and resumed against different rows.
            raise ReadFrameExpiredError(
                f"read frame over {self._path} cannot rebind while a stream is in flight; "
                "finish or abandon the stream first"
            )
        self._sql_owner.close()
        self._cancelled = False
        self._epoch += 1
        self._conn = self._open()
        self._initialize_opened_connection()
        return self._generation

    def cancel(self) -> None:
        """Interrupt an in-flight statement on this frame."""
        if self._sql_owner.pid != os.getpid():
            raise RuntimeError("read frame belongs to another process")
        if not self._profile.cancellation_supported:
            raise ValueError(f"read profile for {self._path} does not declare cancellation support")
        self._cancelled = True
        self._conn.interrupt()

    # -- continuations --------------------------------------------------------

    def bind(self, continuation: ReadContinuation) -> ReadContinuation:
        """Stamp a continuation with the frame incarnation that produced it."""
        return replace(continuation, generation=self._generation, epoch=self._epoch)

    def resume(self, continuation: ReadContinuation) -> ReadContinuation:
        """Return a continuation valid against a current frame, or refuse.

        Rebinds an expired or stale live snapshot first, then confirms the
        continuation is still equivalent -- same generation, or an anchor row that still
        holds the same position -- or raises :class:`StaleContinuationError`. It
        never advances or rewinds the position to make one fit.
        """
        self._require_read_owner()
        if self.expired or (
            self._profile.generation_identity == "live"
            and (self._conn.in_transaction or self.streaming or not self.revalidate())
        ):
            # A held read transaction (or an in-flight stream) pins the old
            # snapshot, and data_version inside it reports that snapshot, so
            # an anchor proven there says nothing about current rows. End it
            # first; a stream in flight makes rebind refuse with a typed error.
            self.rebind()
        unchanged = (
            continuation.generation == self._generation and continuation.epoch == self._epoch and self.revalidate()
        )
        if unchanged:
            return continuation
        row = self._conn.execute(continuation.anchor_sql, continuation.anchor_params).fetchone()
        if row is None or row[0] != continuation.position:
            raise StaleContinuationError(
                f"continuation at {continuation.position!r} cannot be resumed against the current "
                f"generation of {self._path}: its anchor row no longer holds that position"
            )
        return replace(continuation, generation=self._generation, epoch=self._epoch)

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        self._sql_owner.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is None:
            self.close()
        else:
            _close_failed_native_construction(self._sql_owner, exc)


def read_frame(
    path: Path | str,
    *,
    timeout_class: str = "interactive-read",
    tier: ArchiveTier | None = None,
    max_snapshot_age_s: float | None = None,
    reason: str | None = None,
) -> ReadFrame:
    """Open a read frame under one of the declared read timeout classes.

    A caller whose work legitimately outlives its class default extends the
    bound explicitly with ``max_snapshot_age_s`` and says why in ``reason``;
    the reason travels into the expiry error and into
    :func:`live_read_frames`, so a long snapshot is a declared decision rather
    than an anonymous one. There is deliberately no way to spell "no bound":
    that is what a sealed generation is for.
    """
    if timeout_class not in READ_PROFILES:
        raise ValueError(f"unknown SQLite timeout class: {timeout_class}")
    profile = READ_PROFILES[timeout_class]
    if max_snapshot_age_s is None:
        if reason is not None:
            raise ValueError("a read-frame reason describes an extended bound; pass max_snapshot_age_s with it")
    else:
        if not reason:
            raise ValueError(f"extending the {timeout_class} snapshot bound to {max_snapshot_age_s}s requires a reason")
        if not math.isfinite(max_snapshot_age_s) or max_snapshot_age_s <= 0:
            raise ValueError("an extended read-frame snapshot bound must be a finite positive number of seconds")
        profile = replace(profile, max_snapshot_age_s=float(max_snapshot_age_s))
    return ReadFrame(path, profile=profile, tier=tier, timeout_class=timeout_class, reason=reason)


def open_scratch_connection(
    path: Path,
    *,
    terminal_parent: SQLCustodyOwner | None = None,
    scratch_directory: tempfile.TemporaryDirectory[str] | None = None,
    lifetime_dependencies: tuple[object, ...] = (),
) -> NativeSQLCustodyOwner:
    """Register disposable SQL before its first pragma or schema statement."""
    connection = connect_measured(path)
    owner = NativeSQLCustodyOwner(
        connection,
        terminal_parent=terminal_parent,
        scratch_directory=scratch_directory,
        lifetime_dependencies=(*current_native_sql_lifetimes(), *lifetime_dependencies),
    )
    try:
        connection = owner.require_connection()
        connection.execute("PRAGMA journal_mode = MEMORY")
        connection.execute("PRAGMA synchronous = OFF")
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    return owner


@contextmanager
def scratch_connection_context(
    *, prefix: str, filename: str, directory: Path | None = None
) -> Iterator[sqlite3.Connection]:
    """Keep disposable artifacts until their actual creator closes SQL."""
    scratch = tempfile.TemporaryDirectory(prefix=prefix, dir=directory)
    try:
        connection = connect_measured(Path(scratch.name) / filename)
    except BaseException:
        scratch.cleanup()
        raise
    owner = NativeSQLCustodyOwner(
        connection, scratch_directory=scratch, lifetime_dependencies=current_native_sql_lifetimes()
    )
    try:
        connection = owner.require_connection()
        connection.execute("PRAGMA journal_mode = MEMORY")
        connection.execute("PRAGMA synchronous = OFF")
        yield connection
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    else:
        owner.close()


@contextmanager
def readonly_connection_context(
    path: str | Path,
    *,
    timeout: float = DB_TIMEOUT,
    validate_schema: bool = True,
    lifetime_dependencies: tuple[object, ...] = (),
) -> Iterator[sqlite3.Connection]:
    """Close a temporary reader on its creator, retaining a failed close."""
    owner = _open_readonly_owner(
        path, timeout=timeout, validate_schema=validate_schema, lifetime_dependencies=lifetime_dependencies
    )
    try:
        yield owner.require_connection()
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    else:
        owner.close()


@contextmanager
def connection_context(
    path: str | Path,
    *,
    timeout: float = DB_TIMEOUT,
    archive_root: str | Path | None = None,
) -> Iterator[sqlite3.Connection]:
    """Context manager for a single-use read-write connection.

    Opens a connection with write pragmas, yields it, and closes on exit.
    """
    conn = open_connection(path, timeout=timeout, archive_root=archive_root)
    owner = NativeSQLCustodyOwner(conn, lifetime_dependencies=current_native_sql_lifetimes())
    try:
        yield conn
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    else:
        owner.close()


__all__ = [
    "attach_database",
    "DB_TIMEOUT",
    "DEFAULT_MEMORY_BUDGET_BYTES",
    "BOUNDED_REPAIR_CACHE_SIZE_KIB",
    "BOUNDED_REPAIR_MMAP_SIZE_BYTES",
    "BULK_BUILD_CACHE_SIZE_KIB",
    "BULK_BUILD_MMAP_SIZE_BYTES",
    "BULK_BUILD_WRITE_CONNECTION_PROFILE",
    "COLD_BUILD_ACTIVE_WAL_AUTOCHECKPOINT_PAGES",
    "COLD_BUILD_ACTIVE_WRITE_CONNECTION_PROFILE",
    "DAEMON_WRITE_CACHE_SIZE_KIB",
    "DAEMON_WRITE_CONNECTION_PROFILE",
    "DAEMON_WRITE_MMAP_SIZE_BYTES",
    "MEMORY_BUDGET_BYTES",
    "MEMORY_BUDGET_ENV_VAR",
    "MappedBytesBudgetCheck",
    "OBSERVATION_JOURNAL_CACHE_SIZE_KIB",
    "READ_CACHE_SIZE_KIB",
    "READ_CONNECTION_PRAGMA_STATEMENTS",
    "READ_CONNECTION_PROFILE",
    "READ_DB_TIMEOUT",
    "READ_MMAP_SIZE_BYTES",
    "READ_PROFILES",
    "SEALED_READ_CONNECTION_PROFILE",
    "SEALED_STAGING_CONNECTION_PROFILE",
    "BACKGROUND_READ_CONNECTION_PROFILE",
    "OFFLINE_BULK_READ_CONNECTION_PROFILE",
    "BACKGROUND_READ_SNAPSHOT_AGE_S",
    "INTERACTIVE_READ_SNAPSHOT_AGE_S",
    "CHECKPOINT_ESCALATION_MODES",
    "CHECKPOINT_HOLD_BUDGET_S",
    "CheckpointEscalation",
    "OWNED_WAL_AUTOCHECKPOINT_PAGES",
    "WAL_ESCALATION_BYTES",
    "WAL_WARN_BYTES",
    "WRITE_PROFILES",
    "arm_recurring_checkpoint_owner",
    "recurring_checkpoint_owner_armed",
    "write_connection_pragma_statements",
    "write_connection_local_pragma_statements",
    "execute_pragma_statement",
    "initialize_tier_database_mode",
    "open_source_tier_write_connection",
    "SQLiteConnectionProfile",
    "TIMEOUT_CLASSES",
    "WAL_AUTOCHECKPOINT_PAGES",
    "WRITE_CACHE_SIZE_KIB",
    "WRITE_CONNECTION_PROFILE",
    "WRITE_MMAP_SIZE_BYTES",
    "check_mapped_bytes_budget_against_cgroup_limit",
    "GenerationToken",
    "LiveGenerationImmutableError",
    "ReadContinuation",
    "ReadFrame",
    "ReadFrameCancelledError",
    "ReadFrameExpiredError",
    "ReadFrameStatus",
    "StaleContinuationError",
    "live_read_frames",
    "pinning_read_frames",
    "read_frame",
    "connection_context",
    "descriptor_alias_path",
    "open_sealed_staging_connection",
    "one_shot_diagnostic_read",
    "log_mapped_bytes_budget_check",
    "mapped_bytes_budget",
    "assert_tier_schema_supported",
    "open_isolated_write_connection",
    "open_daemon_connection",
    "open_connection",
    "open_readonly_connection",
]

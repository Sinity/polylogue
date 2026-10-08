"""Descriptor-anchored access to the archive-owned ``audit.db`` leaf."""

from __future__ import annotations

import fcntl
import os
import sqlite3
import stat
from builtins import BaseExceptionGroup
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.connection_profile import (
    DB_TIMEOUT,
    NativeConnectionSettlementError,
    NativeSQLCustodyOwner,
    _close_failed_native_construction,
    open_readonly_connection,
)
from polylogue.storage.sqlite.write_lease import require_write_lease

_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


class AuditLeafError(RuntimeError):
    """The audit database cannot provide verified archive-owned access."""


@contextmanager
def _audit_sqlite_access(detail: str) -> Iterator[None]:
    """Keep SQLite failures inside the verified audit storage boundary."""

    try:
        yield
    except sqlite3.DatabaseError as exc:
        raise AuditLeafError(detail) from exc


@dataclass(frozen=True, slots=True)
class _AuditLeafIdentity:
    device: int
    inode: int


class VerifiedAuditLeaf:
    """Keep one archive directory descriptor and verify its ``audit.db`` leaf.

    A writer holds the verified main leaf while SQLite opens a child path that
    is proven to resolve back to that descriptor's directory. The main leaf
    and any SQLite sidecar are checked before and after opening, so a
    replacement or redirected sidecar is rejected before a caller receives a
    connection.
    """

    def __init__(
        self,
        archive_root: Path,
        *,
        filename: str = "audit.db",
        lock_writer: bool = False,
    ) -> None:
        self._archive_root = archive_root
        self._filename = filename
        self._lock_writer = lock_writer
        path_flag = getattr(os, "O_PATH", None)
        if path_flag is None:
            raise AuditLeafError("lock-preserving identity custody requires the pending portable custody capability")
        # Closing an ordinary descriptor for any SQLite inode releases every
        # POSIX lock this process holds on it, including other native readers.
        self._identity_open_flag = path_flag
        self._directory_fd: int | None = None
        self._leaf_fd: int | None = None
        self._directory_identity: _AuditLeafIdentity | None = None
        self._identity: _AuditLeafIdentity | None = None
        self._anchored_path: Path | None = None
        self._writer_lock_held = False
        self._sidecar_fds: dict[str, int] = {}
        self._sidecar_identities: dict[str, _AuditLeafIdentity] = {}
        self._first_transaction_guard_armed = False
        self._terminal_owner: NativeSQLCustodyOwner | None = None

    def __enter__(self) -> VerifiedAuditLeaf:
        from polylogue.storage.sqlite.population_admission import assert_population_admitted

        assert_population_admitted(self._archive_root)
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        try:
            self._directory_fd = os.open(self._archive_root, directory_flags | nofollow)
            directory_metadata = os.fstat(self._directory_fd)
            self._validate_directory(directory_metadata)
            self._directory_identity = _AuditLeafIdentity(directory_metadata.st_dev, directory_metadata.st_ino)
            expected = self._validate(self._lstat_leaf_metadata())
            self._leaf_fd = self._open_leaf()
            metadata = os.fstat(self._leaf_fd)
            self._identity = self._validate(metadata)
            if self._identity != expected:
                raise AuditLeafError(f"audit tier leaf changed while opening: {self._archive_root / self._filename}")
            if self._lock_writer:
                self._acquire_writer_lock()
            self._anchored_path = self._resolve_portable_child_path()
            self._assert_sidecar_namespace()
            self._assert_directory_namespace()
        except BaseException as exc:
            self._close_after_failed_enter(exc)
            if isinstance(exc, AuditLeafError):
                raise
            raise AuditLeafError(f"cannot safely open audit tier leaf: {self._archive_root / self._filename}") from exc
        return self

    def __exit__(self, _exc_type: object, exc: BaseException | None, _traceback: object) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if exc is not None:
                raise BaseExceptionGroup("Audit leaf operation and cleanup failed", [exc, cleanup]) from exc
            raise

    def sqlite_uri(self, *, readonly: bool = False) -> str:
        if readonly:
            # ``immutable=1`` is unsafe for the live authority database: a
            # committed head may still reside in WAL, and immutable readers
            # deliberately ignore locking and change detection. ``mode=ro``
            # preserves WAL visibility without granting write access.
            return f"{self.anchored_path.as_uri()}?mode=ro"
        return f"{self.anchored_path.as_uri()}?mode=rw"

    @property
    def anchored_path(self) -> Path:
        """Return the descriptor-anchored path SQLite and byte readers may open."""

        if self._anchored_path is None:
            raise RuntimeError("audit leaf descriptor is closed")
        return self._anchored_path

    def assert_unchanged(self) -> None:
        """Require the current directory entry to retain the inspected inode."""

        if self._identity is None or self._directory_identity is None:
            raise RuntimeError("audit leaf descriptor is closed")
        try:
            current = self._validate(self._open_leaf_metadata())
            self._assert_directory_namespace()
            anchored = self._stat_path(self.anchored_path)
            anchored_directory = self._stat_path(self.anchored_path.parent)
            self._assert_sidecar_namespace()
        except OSError as exc:
            raise AuditLeafError(f"cannot revalidate audit tier leaf: {self._archive_root / self._filename}") from exc
        if (
            current != self._identity
            or _AuditLeafIdentity(anchored.st_dev, anchored.st_ino) != self._identity
            or _AuditLeafIdentity(anchored_directory.st_dev, anchored_directory.st_ino) != self._directory_identity
        ):
            raise AuditLeafError(f"audit tier leaf changed during SQLite open: {self._archive_root / self._filename}")

    def _assert_directory_namespace(self) -> None:
        if self._directory_identity is None:
            raise RuntimeError("audit leaf descriptor is closed")
        metadata = os.stat(self._archive_root, follow_symlinks=False)
        self._validate_directory(metadata)
        if _AuditLeafIdentity(metadata.st_dev, metadata.st_ino) != self._directory_identity:
            raise AuditLeafError(f"audit tier directory changed during SQLite access: {self._archive_root}")

    def identity_metadata(self) -> os.stat_result:
        """Read metadata from the lifetime-pinned selected main inode."""
        if self._leaf_fd is None:
            raise RuntimeError("audit leaf descriptor is closed")
        self.assert_unchanged()
        return os.fstat(self._leaf_fd)

    def close(self) -> None:
        owner = self._terminal_owner
        if owner is not None:
            if owner.anchored_descriptors:
                owner.close()
                return
            # Native settlement calls back only after all transferred
            # descriptors retired. Never close their numeric slots again.
            self._terminal_owner = None
            self._directory_identity = None
            self._identity = None
            self._anchored_path = None
            self._writer_lock_held = False
            self._sidecar_identities = {}
            self._first_transaction_guard_armed = False
            return
        descriptors = (
            *self._sidecar_fds.values(),
            *((self._leaf_fd,) if self._leaf_fd is not None else ()),
            *((self._directory_fd,) if self._directory_fd is not None else ()),
        )
        # Move exact bindings into the existing native owner before closing.
        # Its descriptor ledger preserves ambiguity, original creator and
        # flock; a substituted closer cannot erase leaf cleanup obligations.
        self._sidecar_fds = {}
        self._leaf_fd = None
        self._directory_fd = None
        if not descriptors:
            self._directory_identity = None
            self._identity = None
            self._anchored_path = None
            self._writer_lock_held = False
            self._sidecar_identities = {}
            self._first_transaction_guard_armed = False
            return
        try:
            owner = NativeSQLCustodyOwner(None, leaf=self, anchored_descriptors=descriptors)
        except NativeConnectionSettlementError as failure:
            self._terminal_owner = failure.owner
            raise
        self._terminal_owner = owner
        owner.close()

    def _close_after_failed_enter(self, primary: BaseException) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            raise BaseExceptionGroup("Audit leaf construction and cleanup failed", [primary, cleanup]) from primary

    def _acquire_writer_lock(self) -> None:
        if self._directory_fd is None:
            raise RuntimeError("audit leaf descriptor is closed")
        try:
            # The existing anchored directory owns the writer flock. It is
            # never a SQLite inode, so closing it cannot drop native SQL locks.
            fcntl.flock(self._directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise AuditLeafError(
                f"audit tier already has an active writer: {self._archive_root / self._filename}"
            ) from exc
        self._writer_lock_held = True

    def _open_leaf(self) -> int:
        if self._directory_fd is None:
            raise RuntimeError("audit leaf descriptor is closed")
        flags = self._identity_open_flag | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        return os.open(self._filename, flags, dir_fd=self._directory_fd)

    def _open_leaf_metadata(self) -> os.stat_result:
        descriptor = self._open_leaf()
        owner = NativeSQLCustodyOwner(None, anchored_descriptors=(descriptor,))
        try:
            metadata = os.fstat(descriptor)
        except BaseException as primary:
            _close_failed_native_construction(owner, primary)
            raise
        owner.close()
        return metadata

    def _lstat_leaf_metadata(self) -> os.stat_result:
        if self._directory_fd is None:
            raise RuntimeError("audit leaf descriptor is closed")
        return os.stat(self._filename, dir_fd=self._directory_fd, follow_symlinks=False)

    def _resolve_portable_child_path(self) -> Path:
        if self._directory_fd is None or self._identity is None:
            raise RuntimeError("audit leaf descriptor is closed")
        directory = self._native_directory_path()
        if directory is not None:
            candidate = directory / self._filename
            if self._matches_identity(candidate, self._identity):
                return candidate
        descriptor_child = self._descriptor_child_path()
        if descriptor_child is not None:
            return descriptor_child
        raise AuditLeafError(f"cannot access audit tier through a verified descriptor: {self._archive_root}")

    def _descriptor_child_path(self) -> Path | None:
        """Return a descriptor-directory child only where the host proves it works."""

        if self._directory_fd is None or self._identity is None:
            raise RuntimeError("audit leaf descriptor is closed")
        for directory in (Path("/proc/self/fd"), Path("/dev/fd")):
            candidate = directory / str(self._directory_fd) / self._filename
            if self._matches_identity(candidate, self._identity):
                return candidate
        return None

    def _native_directory_path(self) -> Path | None:
        if self._directory_fd is None:
            raise RuntimeError("audit leaf descriptor is closed")
        request = getattr(fcntl, "F_GETPATH", None)
        if not isinstance(request, int):
            return None
        try:
            raw = fcntl.fcntl(self._directory_fd, request, b"\0" * 1024)
        except OSError:
            return None
        if not isinstance(raw, bytes):
            return None
        encoded = raw.split(b"\0", 1)[0]
        if not encoded:
            return None
        try:
            candidate = Path(os.fsdecode(encoded))
            directory = os.fstat(self._directory_fd)
            metadata = self._stat_path(candidate)
        except OSError:
            return None
        if (metadata.st_dev, metadata.st_ino) != (directory.st_dev, directory.st_ino):
            return None
        return candidate

    def _assert_sidecar_namespace(self) -> None:
        if self._directory_fd is None:
            raise RuntimeError("audit leaf descriptor is closed")
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            filename = f"{self._filename}{suffix}"
            try:
                expected = self._validate(
                    os.stat(filename, dir_fd=self._directory_fd, follow_symlinks=False),
                    description="audit tier sidecar",
                    filename=filename,
                )
            except FileNotFoundError:
                continue
            descriptor = os.open(
                filename,
                self._identity_open_flag | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self._directory_fd,
            )
            persistent = filename not in self._sidecar_fds
            if persistent:
                # Attach before validation so every constructor/namespace
                # failure leaves the actual descriptor with its existing leaf.
                self._sidecar_fds[filename] = descriptor
                actual = self._validate(os.fstat(descriptor), description="audit tier sidecar", filename=filename)
                if actual != expected:
                    raise AuditLeafError(f"audit tier sidecar changed while opening: {self._archive_root / filename}")
                self._sidecar_identities[filename] = actual
            else:
                owner = NativeSQLCustodyOwner(None, anchored_descriptors=(descriptor,))
                try:
                    actual = self._validate(os.fstat(descriptor), description="audit tier sidecar", filename=filename)
                    if actual != expected:
                        raise AuditLeafError(
                            f"audit tier sidecar changed while opening: {self._archive_root / filename}"
                        )
                except BaseException as primary:
                    _close_failed_native_construction(owner, primary)
                    raise
                owner.close()
        self._assert_pinned_sidecars()

    def prepare_writable_sqlite(self, connection: sqlite3.Connection) -> None:
        """Create and pin SQLite's WAL namespace before exposing a writer.

        Opening ``audit.db`` alone does not create WAL/SHM.  Force that setup
        while the verified directory lock is held, then retain descriptors for
        both files so a later pathname replacement is detectable before an
        application transaction is authorized.
        """

        if not self._lock_writer:
            raise RuntimeError("audit leaf is not a writer")
        with _audit_sqlite_access("cannot establish the audit SQLite WAL namespace"):
            journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()
            if journal_mode is None or str(journal_mode[0]).lower() != "wal":
                raise AuditLeafError("audit tier must use WAL before writable access")
            connection.execute("BEGIN IMMEDIATE")
            connection.commit()
            self._pin_writable_sidecars()
            self.assert_unchanged()
            self._first_transaction_guard_armed = True

    def install_transaction_guard(self, connection: sqlite3.Connection) -> None:
        """Reject a sidecar replacement before SQLite starts an application tx."""

        def authorize(
            action: int, argument1: str | None, _argument2: str | None, _database: str | None, _trigger: str | None
        ) -> int:
            if action == sqlite3.SQLITE_TRANSACTION and argument1 == "BEGIN" and self._first_transaction_guard_armed:
                self._assert_pinned_sidecars(allow_absent=False)
                self._first_transaction_guard_armed = False
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)

    def _pin_writable_sidecars(self) -> None:
        """Pin the WAL sidecars this leaf can prove exist.

        polylogue-x18ml: ``-wal`` is required and ``-shm`` is not, because the
        two files carry different guarantees.

        ``-wal`` holds committed frames -- it is durable authority, and a
        writable audit connection that has just forced WAL mode must have one.
        ``-shm`` is the wal-index: pure derived shared memory, carrying no
        committed data, which SQLite creates and destroys on its own schedule.
        It legitimately does not exist while a WAL database is opened in
        exclusive locking mode (the index lives in heap instead), it is removed
        on last close, and -- the case that surfaced this -- a restore that
        rebinds a new audit image leaves the connection without a ``-shm``
        pathname for the replaced inode even though the connection itself is
        open, in WAL mode, and has committed.

        Requiring it therefore turned a file SQLite promises nothing about into
        a hard precondition of the durable restore path, which is why an
        interrupted continuity commit could not be resumed. Requiring it also
        bought nothing: the pin exists to detect a pathname swap under a live
        connection, and ``_assert_pinned_sidecars`` only ever re-checks entries
        that were actually pinned, so a sidecar absent at pin time is simply
        outside that guarantee rather than silently weakening it. Every file
        that IS present is still pinned and still re-validated byte-identically.
        """
        if self._directory_fd is None:
            raise RuntimeError("audit leaf descriptor is closed")
        for suffix in ("-wal", "-shm"):
            filename = f"{self._filename}{suffix}"
            if filename in self._sidecar_fds:
                self._assert_pinned_sidecars()
                continue
            try:
                expected = self._validate(
                    os.stat(filename, dir_fd=self._directory_fd, follow_symlinks=False),
                    description="audit tier sidecar",
                    filename=filename,
                )
                descriptor = os.open(
                    filename,
                    self._identity_open_flag | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=self._directory_fd,
                )
            except FileNotFoundError as exc:
                if suffix == "-shm":
                    # Derived wal-index, not authority: absent is a legitimate
                    # SQLite state, so there is nothing to pin and nothing lost.
                    continue
                raise AuditLeafError(
                    f"audit tier did not create required WAL sidecar: {self._archive_root / filename}"
                ) from exc
            self._sidecar_fds[filename] = descriptor
            actual = self._validate(os.fstat(descriptor), description="audit tier sidecar", filename=filename)
            if actual != expected:
                raise AuditLeafError(f"audit tier sidecar changed while pinning: {self._archive_root / filename}")
            self._sidecar_identities[filename] = actual

    def _assert_pinned_sidecars(self, *, allow_absent: bool = True) -> None:
        if self._directory_fd is None:
            raise RuntimeError("audit leaf descriptor is closed")
        for filename, identity in self._sidecar_identities.items():
            try:
                current = self._validate(
                    os.stat(filename, dir_fd=self._directory_fd, follow_symlinks=False),
                    description="audit tier sidecar",
                    filename=filename,
                )
                pinned = self._validate(
                    os.fstat(self._sidecar_fds[filename]), description="audit tier sidecar", filename=filename
                )
            except FileNotFoundError as exc:
                if allow_absent:
                    continue
                raise AuditLeafError(
                    f"audit tier sidecar disappeared during SQLite access: {self._archive_root / filename}"
                ) from exc
            except OSError as exc:
                raise AuditLeafError(f"cannot inspect audit tier sidecar: {self._archive_root / filename}") from exc
            if current != identity or pinned != identity:
                raise AuditLeafError(
                    f"audit tier sidecar changed during SQLite access: {self._archive_root / filename}"
                )

    @staticmethod
    def _stat_path(path: Path) -> os.stat_result:
        return os.stat(path)

    def _matches_identity(self, path: Path, identity: _AuditLeafIdentity) -> bool:
        try:
            metadata = self._stat_path(path)
        except OSError:
            return False
        return (metadata.st_dev, metadata.st_ino) == (identity.device, identity.inode)

    def _validate(
        self,
        metadata: os.stat_result,
        *,
        description: str = "audit tier",
        filename: str | None = None,
    ) -> _AuditLeafIdentity:
        path = self._archive_root / (filename or self._filename)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise AuditLeafError(f"{description} must be an archive-owned regular file with one link: {path}")
        if metadata.st_uid != os.geteuid():
            raise AuditLeafError(f"{description} must be owned by the current effective user: {path}")
        if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise AuditLeafError(f"{description} must not be writable by group or other: {path}")
        return _AuditLeafIdentity(metadata.st_dev, metadata.st_ino)

    def _validate_directory(self, metadata: os.stat_result) -> None:
        if not stat.S_ISDIR(metadata.st_mode):
            raise AuditLeafError(f"audit tier directory must be a directory without a symlink: {self._archive_root}")
        if metadata.st_uid != os.geteuid():
            raise AuditLeafError(
                f"audit tier directory must be owned by the current effective user: {self._archive_root}"
            )
        if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise AuditLeafError(f"audit tier directory must not be writable by group or other: {self._archive_root}")


@contextmanager
def _owned_verified_leaf_connection(
    path: Path,
    *,
    readonly: bool = False,
    lock_writer: bool = False,
) -> Iterator[tuple[sqlite3.Connection, VerifiedAuditLeaf]]:
    """Keep verified descriptors with their actual SQL owner until close settles."""
    leaf = VerifiedAuditLeaf(path.parent, filename=path.name, lock_writer=lock_writer).__enter__()
    try:
        connection = (
            open_readonly_connection(leaf.anchored_path, validate_schema=False)
            if readonly
            else connect_measured(leaf.sqlite_uri(), uri=True, timeout=DB_TIMEOUT)
        )
        owner = NativeSQLCustodyOwner(connection, leaf=leaf)
    except NativeConnectionSettlementError as error:
        # A readonly profile can fail before this outer owner is constructed.
        # Transfer the still-pinned leaf to that already retained actual owner.
        error.owner.leaf = leaf
        raise
    except BaseException as primary:
        try:
            leaf.close()
        except BaseException as cleanup:
            primary.add_note(f"verified descriptor cleanup also failed: {type(cleanup).__name__}")
        raise
    try:
        yield connection, leaf
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    else:
        owner.close()


@contextmanager
def open_verified_audit_connection(path: Path) -> Iterator[sqlite3.Connection]:
    """Open one writable audit connection pinned to an owned leaf descriptor."""
    require_write_lease(f"open_verified_audit_connection({path})", archive_root=path.parent)
    with _owned_verified_leaf_connection(path, lock_writer=True) as (connection, leaf):
        leaf.prepare_writable_sqlite(connection)
        leaf.install_transaction_guard(connection)
        yield connection
        leaf.assert_unchanged()


@contextmanager
def open_verified_sqlite_read_connection(path: Path) -> Iterator[sqlite3.Connection]:
    """Open a read-only SQLite leaf through a no-follow directory descriptor."""
    with _owned_verified_leaf_connection(path, readonly=True) as (connection, leaf):
        leaf.assert_unchanged()
        yield connection
        leaf.assert_unchanged()


@contextmanager
def open_verified_sqlite_write_connection(path: Path) -> Iterator[sqlite3.Connection]:
    """Open an existing writable SQLite leaf through a no-follow descriptor."""
    require_write_lease(f"open_verified_sqlite_write_connection({path})", archive_root=path.parent)
    with _owned_verified_leaf_connection(path) as (connection, leaf):
        leaf.assert_unchanged()
        yield connection
        leaf.assert_unchanged()


@contextmanager
def open_verified_audit_read_connection(path: Path) -> Iterator[sqlite3.Connection]:
    """Open one live-WAL-aware, read-only audit connection."""

    with (
        _audit_sqlite_access("audit SQLite read is unavailable"),
        open_verified_sqlite_read_connection(path) as connection,
    ):
        yield connection


def assert_verified_audit_leaf(path: Path) -> None:
    """Check an existing audit leaf without exposing its descriptor to callers."""

    with VerifiedAuditLeaf(path.parent, filename=path.name):
        return


__all__ = [
    "AuditLeafError",
    "VerifiedAuditLeaf",
    "assert_verified_audit_leaf",
    "open_verified_audit_connection",
    "open_verified_audit_read_connection",
    "open_verified_sqlite_read_connection",
    "open_verified_sqlite_write_connection",
]

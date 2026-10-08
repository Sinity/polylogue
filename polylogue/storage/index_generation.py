"""Owned inactive index generations and atomic active-index promotion."""

from __future__ import annotations

import asyncio
import errno
import fcntl
import json
import os
import re
import shutil
import socket
import sqlite3
import stat
import sys
import threading
import time
import uuid
from builtins import BaseExceptionGroup
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from types import BuiltinFunctionType, TracebackType
from typing import Any, cast

from polylogue.logging import WARNING, emit
from polylogue.storage.archive_identity import (
    ACTIVE_POINTER_FILENAME,
    GENERATIONS_DIRNAME,
    LIFECYCLE_LOCK_FILENAME,
    ArchiveLocation,
    is_index_generation_member,
)
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import DEFAULT_ARCHIVE_PAGE_SIZE, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import descriptor_alias_path, open_readonly_connection
from polylogue.storage.sqlite.wal_checkpoint import checkpoint_connection

#: Durable and disposable archive members an index generation reaches by
#: read-through symlink rather than owning. ``audit.db`` joined the set with
#: the daemon's cold build (polylogue-b7dkb): revision governance reads the
#: excision policy out of the audit tier on every raw admission, so a
#: generation that cannot see it cannot host an ordinary ingest pass.
_GENERATION_READ_THROUGH_MEMBERS: tuple[str, ...] = (
    "source.db",
    "user.db",
    "embeddings.db",
    "audit.db",
    "ops.db",
    "blob",
)

_LOCK_PID_PATTERN = re.compile(r"pid=(\d+)")
_LOCK_HOST_PATTERN = re.compile(r"host=(\S+)")

#: Superseded generations kept after a promotion.  One is enough to roll back
#: to the previous index; each costs roughly the size of the index itself
#: (~35 GB on the reference archive), so keeping more is expensive storage,
#: not cheap insurance.
SUPERSEDED_GENERATION_RETENTION = 1
# Keep the current promotion receipt plus the immediately preceding one. This
# is intentionally larger than the rollback-generation boundary so automatic
# receipt pruning cannot erase the evidence for the active boundary.
RETENTION_RECEIPT_HISTORY = SUPERSEDED_GENERATION_RETENTION + 1
_RETENTION_RECEIPTS_DIRNAME = "retention-receipts"
_SAFE_LIFECYCLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LIFECYCLE_STATES = frozenset({"inactive", "promoting", "active", "retained", "eligible", "reclaimed"})


def _assert_no_symlink_ancestry(path: Path, *, label: str) -> Path:
    """Reject links in an untrusted lifecycle path before any use.

    ``Path.resolve`` is deliberately not a security check: it follows the very
    links this lifecycle must refuse.  Inspect every existing component with
    ``lstat`` and keep the lexical path for subsequent descriptor-relative
    operations.
    """
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(f"cannot inspect {label}: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{label} contains a symlink component: {current}")
    return absolute


def _ensure_lifecycle_directory(path: Path, *, label: str) -> Path:
    """Create a lifecycle directory without accepting a symlink replacement."""
    absolute = _assert_no_symlink_ancestry(path, label=label)
    try:
        absolute.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"cannot create {label}: {absolute}") from exc
    _assert_no_symlink_ancestry(absolute, label=label)
    if not absolute.is_dir():
        raise RuntimeError(f"{label} is not a directory: {absolute}")
    return absolute


def _remove_created_generation(
    parent: Path, name: str, parent_identity: tuple[int, int], created_identity: tuple[int, int]
) -> None:
    """Remove only the directory created by this call, through its pinned parent."""
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parent_stat = os.fstat(descriptor)
        if (parent_stat.st_dev, parent_stat.st_ino) != parent_identity:
            return
        try:
            target_stat = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(target_stat.st_mode) or (target_stat.st_dev, target_stat.st_ino) != created_identity:
            return
        shutil.rmtree(name, dir_fd=descriptor)
    finally:
        os.close(descriptor)


def _read_json_nofollow(path: Path, *, label: str) -> dict[str, object]:
    """Read one lifecycle record without following a replacement symlink."""
    _assert_no_symlink_ancestry(path.parent, label=f"{label} parent")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise RuntimeError(f"cannot securely read {label}: {path}") from exc
    try:
        with os.fdopen(fd, encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid {label}: {path}")
    return payload


def _atomic_json_write(path: Path, payload: dict[str, object], *, label: str) -> None:
    """Write a record via an exclusive, no-follow temporary file.

    A pre-existing ``*.tmp`` symlink is an integrity failure, not an invitation
    to follow it.  The fixed-name O_EXCL temporary plus the lifecycle lock make
    check-to-use replacement by cooperating writers impossible; the final
    lstat is a second fail-closed assertion about the installed inode.
    """
    parent = _assert_no_symlink_ancestry(path.parent, label=f"{label} parent")
    for candidate in parent.glob("*.tmp"):
        try:
            if candidate.is_symlink():
                raise RuntimeError(f"{label} temporary path is a symlink: {candidate}")
        except OSError as exc:
            raise RuntimeError(f"cannot inspect {label} temporary path: {candidate}") from exc
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise RuntimeError(f"cannot create atomic {label} temporary: {temporary}") from exc
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        installed = path.lstat()
        if stat.S_ISLNK(installed.st_mode):
            raise RuntimeError(f"atomic {label} installed a symlink: {path}")
    except BaseException:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        with suppress(OSError):
            if path.is_symlink():
                path.unlink()
        raise
    _fsync_directory(parent)


class GenerationRetentionState(StrEnum):
    """Durable lifecycle states for a promoted generation's retention record."""

    ACTIVE = "active"
    RETAINED = "retained"
    ELIGIBLE = "eligible"
    RECLAIMED = "reclaimed"


@dataclass(frozen=True, slots=True)
class GenerationRetentionRecord:
    generation_id: str
    generation_owner_id: str
    retention_owner_id: str
    state: GenerationRetentionState


@dataclass(frozen=True, slots=True)
class GenerationRetentionReceipt:
    """Evidence for automatic rollback retention and generation reclamation."""

    promoted_generation_id: str
    promoted_at_ns: int
    retention_boundary: int
    automatic: bool
    records: tuple[GenerationRetentionRecord, ...]
    eligible_generation_ids: tuple[str, ...] = ()
    reclaimed_marker_count: int = 0

    @property
    def states_by_generation_id(self) -> dict[str, str]:
        return {record.generation_id: record.state.value for record in self.records}

    @property
    def owner_by_generation_id(self) -> dict[str, str]:
        return {record.generation_id: record.retention_owner_id for record in self.records}


def _is_generation_member(path: Path) -> bool:
    """True when ``path`` lives inside a generation directory rather than beside one.

    The canonical index pointer must not name a path inside a generation,
    because ``generations_root`` is derived from the pointer's *parent*: a
    pointer at ``…/.index-generations/gen-X/index.db`` makes it nest as
    ``…/gen-X/.index-generations``, which is the shape the
    self-poisoning bug produced.

    The test is deliberately narrower than "the path mentions
    ``.index-generations`` somewhere". Two cases must stay distinguishable:

    * ``…/.index-generations/gen-X/index.db`` -- a generation *member*, refused.
      The pointer is at least one directory below the generations root, which is
      exactly how a generation's own files sit.
    * ``…/.index-generations/index.db`` -- a file sitting *directly* in a
      directory that happens to carry that name, allowed. An archive root may
      legitimately be named anything, including this, and there the derived
      roots stay self-consistent. Rejecting it on a name match alone would break
      a valid symlink-farm target for no invariant's benefit.

    Uses ``absolute()`` rather than ``resolve()`` on purpose: resolving would
    follow ``index.db``'s own promotion symlink into the generation it targets,
    so the canonical pointer would classify itself as poisoned.
    """
    return is_index_generation_member(path)


def canonical_active_index_path(location: ArchiveLocation) -> Path:
    """Resolve the active index path without repairing a missing or poisoned anchor."""
    anchored = location.active_pointer
    if anchored is not None and not _is_generation_member(anchored):
        return anchored
    configured_index = location.configured_tier("index").configured_path
    if configured_index.is_symlink():
        target = Path(os.readlink(configured_index))
        resolved = target if target.is_absolute() else configured_index.parent / target
        return configured_index if _is_generation_member(resolved) else resolved
    return configured_index


@dataclass(frozen=True, slots=True)
class IndexGeneration:
    generation_id: str
    owner_id: str
    archive_root: str
    index_path: str
    state: str
    created_at_ms: int
    source_snapshot: str = ""
    # Millisecond creation time remains for compatibility with existing
    # generation metadata. Lifecycle ordering uses these nanosecond values so
    # UUID text never decides which rollback target is newest.
    created_at_ns: int = 0
    promoted_at_ns: int = 0
    predecessor_generation_id: str | None = None
    retention_owner_id: str | None = None
    retention_state: str | None = None
    # polylogue-kc8eq: the SQLite page size this generation's index.db was
    # created with. Recorded because it is unrecoverable from anything else
    # once the file exists and cannot be changed without recreating it, so a
    # generation built before the choice existed has to read as 0 ("whatever
    # SQLite defaulted to") rather than as a claim about 8192.
    page_size: int = 0


@dataclass(slots=True)
class PreparedIndexPromotion:
    """Off-gate proof retained through one index-generation pointer swap."""

    generation_id: str
    candidate_path: Path
    prior_active_path: Path | None
    candidate_identity: tuple[int, int, int, int]
    reference_seal: Any
    missing_session_count: int = 0
    first_missing_session_id: str | None = None

    def close(self) -> None:
        self.reference_seal.close()

    def __enter__(self) -> PreparedIndexPromotion:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> None:
        try:
            self.close()
        except BaseException as close_error:
            if exc is None:
                raise
            raise BaseExceptionGroup("Index promotion and proof cleanup failed", [exc, close_error]) from exc


class RebuildLeaseUnavailableError(RuntimeError):
    """Another process owns the archive-wide rebuild lease."""


class RebuildLeaseSettlementError(RuntimeError):
    """Accepted SQL remains unsettled; exclusion stays with this rebuild owner."""

    def __init__(self, lease: RebuildLease) -> None:
        super().__init__("rebuild SQL remains unsettled; retry owner cleanup before closing exclusion")
        self.lease = lease


class UnpublishedPromotionRecoveryError(RuntimeError):
    """A cold candidate cannot be reclaimed without prior-pointer proof."""


def _lock_holder_pid(path: Path) -> int | None:
    """Best-effort recorded pid from an existing lock file; ``None`` if absent/unreadable."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _LOCK_PID_PATTERN.search(text)
    if match is None:
        return None
    return int(match.group(1))


def _lock_holder_host(path: Path) -> str | None:
    """Best-effort recorded hostname from an existing lock file; ``None`` if absent/unreadable."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _LOCK_HOST_PATTERN.search(text)
    return match.group(1) if match is not None else None


def _pid_is_alive(pid: int) -> bool:
    """Whether ``pid`` still names a live process, best-effort via ``kill(pid, 0)``."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Owned by another user but still running.
        return True
    return True


def _open_lock_fd(path: Path, lock_type: int, *, unavailable_message: str) -> int:
    """Open ``path`` and acquire ``lock_type`` (``LOCK_EX``/``LOCK_SH``), non-blocking.

    The kernel lock is authoritative.  Owner text is diagnostic only: a
    forked worker can legitimately outlive the pid recorded by its parent,
    and an active shared writer can hold an inode whose text was left by an
    earlier exclusive owner.  Replacing that still-locked inode would create
    a second lock domain and permit concurrent archive writers.
    """
    from polylogue.storage.sqlite.write_lease import _validate_custody_directory, _validate_custody_lock

    path.parent.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    fd = -1
    try:
        directory = os.fstat(directory_fd)
        _validate_custody_directory(directory, path.parent)
        try:
            before = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
            _validate_custody_lock(before, path)
            before_identity = (before.st_dev, before.st_ino)
        except FileNotFoundError:
            before_identity = None
        fd = os.open(path.name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory_fd)
        after = os.fstat(fd)
        _validate_custody_lock(after, path)
        if before_identity is not None and (after.st_dev, after.st_ino) != before_identity:
            raise RuntimeError(f"lock path was replaced: {path}")
        fcntl.flock(fd, lock_type | fcntl.LOCK_NB)
        path_metadata = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        _validate_custody_lock(path_metadata, path)
        parent_metadata = os.stat(path.parent, follow_symlinks=False)
        _validate_custody_directory(parent_metadata, path.parent)
        if (path_metadata.st_dev, path_metadata.st_ino) != (after.st_dev, after.st_ino) or (
            parent_metadata.st_dev,
            parent_metadata.st_ino,
        ) != (directory.st_dev, directory.st_ino):
            raise RuntimeError(f"lock namespace was replaced: {path}")
        return fd
    except BlockingIOError as exc:
        if fd >= 0:
            os.close(fd)
        holder_pid = _lock_holder_pid(path)
        suffix = f" (recorded pid={holder_pid})" if holder_pid is not None else ""
        raise RebuildLeaseUnavailableError(unavailable_message + suffix) from exc
    except BaseException:
        if fd >= 0:
            os.close(fd)
        raise
    finally:
        os.close(directory_fd)


class RebuildLease:
    """Process-held exclusive lease for an offline index rebuild."""

    def __init__(self, archive_root: Path) -> None:
        self.archive_root = archive_root.resolve()
        self.path = self.archive_root / ".index-rebuild.lock"
        self._fd: int | None = None
        self._owner_pid = os.getpid()
        self._segments = threading.Condition()
        self._accepting = False
        self._retired = False
        self._pending_segments = 0
        self._accepted_leases: list[Any] = []

    def __enter__(self) -> RebuildLease:
        if self._owner_pid != os.getpid():
            raise RuntimeError("cannot acquire rebuild exclusion inherited across fork")
        if self._retired or self._fd is not None:
            raise RuntimeError("a rebuild owner cannot be entered after admission or retirement")
        from polylogue.storage.sqlite.write_lease import archive_write_custody

        custody_context = archive_write_custody(self.archive_root)
        custody_context.__enter__()
        fd: int | None = None
        try:
            fd = _open_lock_fd(
                self.path,
                fcntl.LOCK_EX,
                unavailable_message=f"index rebuild lease is already held: {self.path}",
            )
            try:
                os.ftruncate(fd, 0)
                os.write(fd, f"pid={os.getpid()} host={socket.gethostname()}\n".encode())
                os.fsync(fd)
            except BaseException:
                os.close(fd)
                fd = None
                raise
            self._fd = fd
            self._accepting = True
            return self
        except BaseException:
            if fd is not None:
                with suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
                with suppress(OSError):
                    os.close(fd)
                self._fd = None
            raise
        finally:
            # EX exclusion persists, but physical custody belongs only to
            # acquisition and individual SQL segments, never proof preparation.
            custody_context.__exit__(None, None, None)

    @contextmanager
    def write_segment(self, actor: str = "offline.index_rebuild") -> Iterator[Any]:
        """Admit one physical writer after its off-gate preparation completed."""
        if self._owner_pid != os.getpid():
            raise RuntimeError("cannot admit rebuild work inherited across fork")
        from polylogue.storage.sqlite.write_lease import write_lease

        with self._segments:
            if self._owner_pid != os.getpid() or not self._accepting or self._fd is None:
                raise RuntimeError("the rebuild owner no longer accepts writer segments")
            self._pending_segments += 1
        try:
            with write_lease(actor, archive_root=self.archive_root) as lease:
                with self._segments:
                    if not self._accepting:
                        raise RuntimeError("the rebuild owner retired before writer admission")
                    self._accepted_leases.append(lease)
                yield lease
        finally:
            with self._segments:
                self._pending_segments -= 1
                self._accepted_leases[:] = [
                    accepted
                    for accepted in self._accepted_leases
                    if accepted.custody is not None and accepted.custody.held
                ]
                self._segments.notify_all()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._owner_pid != os.getpid():
            raise RuntimeError("cannot release rebuild exclusion inherited across fork")
        interruption: BaseException | None = None
        with self._segments:
            self._retired = True
            self._accepting = False
            while self._pending_segments:
                try:
                    self._segments.wait()
                except BaseException as error:
                    interruption = interruption or error
            accepted = tuple(self._accepted_leases)
        for lease in accepted:
            lease.retire()
            for delegation in tuple(lease.delegations):
                if delegation.retire():
                    try:
                        delegation.wait_until_settled()
                    except BaseException as error:
                        interruption = interruption or error
            for grant in tuple(lease.thread_grants):
                try:
                    grant.wait_until_settled()
                except BaseException as error:
                    interruption = interruption or error
            if lease.custody is not None and lease.custody.held:
                # A failed SQL close still owns physical custody. Keep EX
                # recoverable on this object; retry close after owner cleanup.
                raise RebuildLeaseSettlementError(self)
        if self._fd is not None:
            fd, self._fd = self._fd, None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        self._accepted_leases.clear()
        if interruption is not None:
            raise interruption


class ActiveWriterLeaseSettlementError(RuntimeError):
    """The exact exclusion descriptor remains owned after an uncertain close."""

    def __init__(self, lease: ActiveWriterLease, failure: BaseException) -> None:
        super().__init__("active-writer exclusion requires original-owner descriptor settlement")
        self.lease = lease
        self.failure = failure


class ActiveWriterLease:
    """Shared process-held lease refused while an offline rebuild owns the archive."""

    def __init__(self, archive_root: Path) -> None:
        self.path = archive_root / ".index-rebuild.lock"
        self._fd: int | None = None
        self._owner_pid = os.getpid()
        self._owner_thread = threading.current_thread()
        self._owner_task = self._task()
        self._identity: tuple[int, int] | None = None
        self._close_failure: BaseException | None = None

    @staticmethod
    def _task() -> object | None:
        try:
            return asyncio.current_task()
        except RuntimeError:
            return None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def require_owner(self, archive_root: Path) -> None:
        if (
            self._owner_pid != os.getpid()
            or self._owner_thread is not threading.current_thread()
            or self._owner_task is not self._task()
            or self.path.parent.resolve(strict=True) != archive_root.resolve(strict=True)
            or self._fd is None
            or self._close_failure is not None
        ):
            raise RuntimeError("publication exclusion requires its exact acquired creator and archive")
        opened = os.fstat(self._fd)
        linked = self.path.stat(follow_symlinks=False)
        if self._identity != (opened.st_dev, opened.st_ino) or self._identity != (linked.st_dev, linked.st_ino):
            raise RuntimeError("publication exclusion namespace changed after acquisition")

    def acquire(self) -> None:
        if self._owner_pid != os.getpid():
            raise RuntimeError("cannot acquire active-writer exclusion inherited across fork")
        if self._fd is not None:
            raise RuntimeError("active-writer exclusion is already acquired")
        self._fd = _open_lock_fd(
            self.path,
            fcntl.LOCK_SH,
            unavailable_message=f"offline index rebuild owns archive: {self.path}",
        )
        try:
            metadata = os.fstat(self._fd)
            self._identity = metadata.st_dev, metadata.st_ino
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Active-writer acquisition and cleanup failed", [primary, cleanup]
                ) from primary
            raise

    def _binding_retired(self, fd: int) -> bool:
        try:
            metadata = os.fstat(fd)
        except OSError as error:
            return error.errno == errno.EBADF
        return self._identity is not None and self._identity != (metadata.st_dev, metadata.st_ino)

    def close(self) -> None:
        if self._owner_pid != os.getpid():
            raise RuntimeError("cannot release active-writer exclusion inherited across fork")
        if self._fd is not None:
            fd = self._fd
            if self._close_failure is not None:
                if not self._binding_retired(fd):
                    raise ActiveWriterLeaseSettlementError(self, self._close_failure) from self._close_failure
                self._fd = None
                self._close_failure = None
                return
            closer = os.close
            native_linux_close = (
                sys.platform == "linux"
                and isinstance(closer, BuiltinFunctionType)
                and closer.__module__ == "posix"
                and closer.__name__ == "close"
            )
            try:
                # Closing the original open file description releases flock.
                # An earlier LOCK_UN would surrender exclusion even when a
                # controlled close fails before releasing that description.
                closer(fd)
            except BaseException as error:
                if (native_linux_close and isinstance(error, OSError)) or self._binding_retired(fd):
                    self._fd = None
                    raise
                self._close_failure = error
                raise ActiveWriterLeaseSettlementError(self, error) from error
            else:
                self._fd = None


@dataclass(frozen=True, slots=True)
class RebuildLeaseStatus:
    """Read-only snapshot of the archive-root rebuild lease, for status surfaces.

    polylogue-b5l.1 AC5: an operator/agent must be able to see who owns the
    lease, whether the recorded holder is actually still alive, and whether
    the lock looks reclaimable, without disturbing a real holder and without
    duplicating ``RebuildLease``/``ActiveWriterLease`` as the sole exclusion
    mechanism.
    """

    held: bool
    holder_pid: int | None
    holder_host: str | None
    #: ``None`` when ``held`` is False (nothing to check liveness against) or
    #: when no pid could be parsed from the lock file at all.
    holder_alive: bool | None
    #: True when the lease is held but its diagnostic owner pid is provably
    #: dead.  The kernel lock remains authoritative and is never bypassed;
    #: this flag tells an operator to locate the surviving fd holder.
    stale: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "held": self.held,
            "holder_pid": self.holder_pid,
            "holder_host": self.holder_host,
            "holder_alive": self.holder_alive,
            "stale": self.stale,
        }


def rebuild_lease_status(archive_root: Path) -> RebuildLeaseStatus:
    """Probe the rebuild lease without blocking or disturbing a genuine holder.

    Attempts a non-blocking exclusive ``flock``: if it succeeds, nothing
    currently holds the lease and the probe releases it immediately; if it
    fails with ``EAGAIN``/``EACCES`` (``BlockingIOError``), the lease is
    genuinely held and the lock file's recorded pid/host are reported
    best-effort for diagnosis (the file content may be stale or unreadable).
    """
    path = archive_root / ".index-rebuild.lock"
    if not path.exists():
        return RebuildLeaseStatus(held=False, holder_pid=None, holder_host=None, holder_alive=None, stale=False)
    holder_pid = _lock_holder_pid(path)
    holder_host = _lock_holder_host(path)
    fd = os.open(path, os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            alive = _pid_is_alive(holder_pid) if holder_pid is not None else None
            return RebuildLeaseStatus(
                held=True,
                holder_pid=holder_pid,
                holder_host=holder_host,
                holder_alive=alive,
                stale=holder_pid is not None and alive is False,
            )
        fcntl.flock(fd, fcntl.LOCK_UN)
        return RebuildLeaseStatus(
            held=False, holder_pid=holder_pid, holder_host=holder_host, holder_alive=None, stale=False
        )
    finally:
        os.close(fd)


def _configured_link_identity(path: Path) -> tuple[int, int, int, str | None, int, int, int, int]:
    metadata = path.lstat()
    target = os.readlink(path) if stat.S_ISLNK(metadata.st_mode) else None
    after = path.lstat()
    parent = path.parent.stat()
    target_parent = path.resolve(strict=True).parent.stat()
    identity = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        target,
        parent.st_dev,
        parent.st_ino,
        target_parent.st_dev,
        target_parent.st_ino,
    )
    if (after.st_dev, after.st_ino, after.st_mode) != identity[:3]:
        raise RuntimeError(f"configured tier link changed during capture: {path}")
    return identity


def _stable_link_target(
    source: Path, *, label: str
) -> tuple[Path, tuple[int, int], bool, tuple[int, int, int, str | None, int, int, int, int]]:
    """Capture the declared link incarnation and its exact durable leaf."""
    try:
        link_identity = _configured_link_identity(source)
        before = source.stat()
        resolved = source.resolve(strict=True)
        after = source.stat()
        if _configured_link_identity(source) != link_identity:
            raise RuntimeError(f"{label} namespace changed during identity capture: {source}")
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise RuntimeError(f"cannot inspect {label}: {source}") from exc
    before_identity = (before.st_dev, before.st_ino)
    after_identity = (after.st_dev, after.st_ino)
    if before_identity != after_identity:
        raise RuntimeError(f"{label} changed during identity capture: {source}")
    return resolved, after_identity, stat.S_ISDIR(after.st_mode), link_identity


def _require_path_identity(
    path: Path,
    identity: tuple[int, int],
    *,
    label: str,
    link_identity: tuple[int, int, int, str | None, int, int, int, int],
) -> None:
    """Require both the configured link and the selected leaf to survive."""
    try:
        if _configured_link_identity(path) != link_identity:
            raise RuntimeError(f"{label} configured link was replaced: {path}")
        metadata = path.stat()
    except OSError as exc:
        raise RuntimeError(f"cannot verify {label}: {path}") from exc
    current = (metadata.st_dev, metadata.st_ino)
    if current != identity:
        raise RuntimeError(f"{label} was replaced: {path}")


def _stable_directory(path: Path, *, label: str) -> tuple[int, int]:
    """Capture a lifecycle parent identity without following symlink ancestry."""
    _assert_no_symlink_ancestry(path, label=label)
    try:
        metadata = path.stat()
    except OSError as exc:
        raise RuntimeError(f"cannot inspect {label}: {path}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"{label} is not a directory: {path}")
    return (metadata.st_dev, metadata.st_ino)


def _atomic_text_write(path: Path, text: str, *, label: str) -> None:
    """Install a small text anchor with the same no-follow guarantees as JSON."""
    parent = _assert_no_symlink_ancestry(path.parent, label=f"{label} parent")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise RuntimeError(f"cannot create atomic {label} temporary: {temporary}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if stat.S_ISLNK(path.lstat().st_mode):
            raise RuntimeError(f"atomic {label} installed a symlink: {path}")
    except BaseException:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise
    _fsync_directory(parent)


class IndexGenerationStore:
    """Create, checkpoint, and atomically promote inactive generations.

    Constructed from an already-resolved :class:`~polylogue.storage.archive_identity.ArchiveLocation`
    rather than a bare ``archive_root: Path`` (polylogue-ovme.2.1): the two
    boundaries have deliberately different jobs. ``ArchiveLocation.resolve()``
    is a pure, side-effect-free *read* of whatever pointer state already
    exists; this store additionally performs first-touch pointer
    **bootstrap** -- writing ``.index-active-pointer`` the first time an
    archive is opened, before any generation has ever been promoted --
    which ``ArchiveLocation.resolve()`` intentionally never does (a read-only
    resolver must never mutate the archive it is describing). Passing an
    ``ArchiveLocation`` in still lets this constructor reuse its pointer-read
    outcome instead of re-deriving it, while keeping the bootstrap-write
    behavior (and the ``.index-generations``-anchor sanity check below, which
    ``ArchiveLocation.resolve()`` also does not perform) here where the write
    authority belongs.
    """

    def __init__(self, location: ArchiveLocation, *, repair_anchor: bool = True) -> None:
        self.archive_root = location.configured_root
        self.location = location
        anchor = location.configured_root / ACTIVE_POINTER_FILENAME
        anchored = location.active_pointer
        self.active_pointer = canonical_active_index_path(location)
        if (anchored is None or _is_generation_member(anchored)) and repair_anchor:
            # Recompute, and rewrite the anchor, when it is absent OR poisoned.
            #
            # The canonical pointer is the path ``promote()`` replaces with a
            # symlink -- i.e. the archive's own ``index.db`` -- never the
            # generation that symlink currently targets. Following the symlink
            # here wrote a ``.index-generations/gen-*/index.db`` path into the
            # anchor, which the next construction then rejected outright; the
            # store poisoned its own anchor on first use and refused every run
            # afterwards. It also made ``generations_root`` nest as
            # ``.index-generations/gen-*/.index-generations``.
            #
            # A symlink is still followed when it leaves the archive (an
            # archive root that is a symlink farm pointing at the real
            # location), because there the canonical pointer genuinely lives
            # elsewhere. Only a link that lands *inside a generation* is
            # refused -- see ``_is_generation_member``. Treating a poisoned
            # anchor as recoverable rather than fatal lets an archive already
            # carrying one heal on next open, instead of needing the file
            # repaired by hand.
            # Constructing the store must not require the archive root to have
            # been materialized first.  The anchor itself and its ancestry are
            # nevertheless untrusted until inspected with lstat.
            _ensure_lifecycle_directory(anchor.parent, label="active pointer parent")
            if anchor.is_symlink():
                raise RuntimeError(f"active pointer is a symlink: {anchor}")
            _atomic_text_write(anchor, str(self.active_pointer.absolute()), label="active pointer")
        self.generations_root = self.active_pointer.parent / GENERATIONS_DIRNAME
        _ensure_lifecycle_directory(self.generations_root, label="generation root")
        _ensure_lifecycle_directory(self.generations_root / _RETENTION_RECEIPTS_DIRNAME, label="retention receipt root")
        self._lifecycle_lock_path = self.active_pointer.parent / LIFECYCLE_LOCK_FILENAME
        # Only this process may resume a pre-swap rollback whose metadata
        # replace also failed. The prepared file is durable; this guard binds
        # it to the pointer inode and owner observed before promotion.
        self._unpublished_rollbacks: dict[
            str, tuple[IndexGeneration, Path, Path, tuple[int, int] | None, tuple[str, ...]]
        ] = {}
        self._active_parent_identity = _stable_directory(self.active_pointer.parent, label="active pointer parent")
        self._active_parent_link_identity = _configured_link_identity(self.active_pointer.parent)
        _require_path_identity(
            self.active_pointer.parent,
            self._active_parent_identity,
            label="active pointer parent",
            link_identity=self._active_parent_link_identity,
        )
        self._lifecycle_lock_fd: int | None = None
        if self._lifecycle_lock_path.is_symlink():
            raise RuntimeError(f"lifecycle lock is a symlink: {self._lifecycle_lock_path}")
        _assert_no_symlink_ancestry(self._lifecycle_lock_path.parent, label="lifecycle lock parent")

    @contextmanager
    def _lifecycle_lock(self) -> Iterator[None]:
        """Serialize lifecycle check/use sequences on one filesystem root."""
        if self._lifecycle_lock_fd is not None:
            yield
            return
        _require_path_identity(
            self.active_pointer.parent,
            self._active_parent_identity,
            label="active pointer parent",
            link_identity=self._active_parent_link_identity,
        )
        fd = os.open(self._lifecycle_lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            _require_path_identity(
                self.active_pointer.parent,
                self._active_parent_identity,
                label="active pointer parent",
                link_identity=self._active_parent_link_identity,
            )
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._lifecycle_lock_fd = fd
            yield
        finally:
            self._lifecycle_lock_fd = None
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _validate_generation(self, generation: IndexGeneration, requested_id: str) -> None:
        if requested_id != generation.generation_id or not _SAFE_LIFECYCLE_ID.fullmatch(requested_id):
            raise RuntimeError("generation metadata identity mismatch")
        expected_root = self.generations_root / requested_id
        _assert_no_symlink_ancestry(expected_root, label="generation metadata ancestry")
        expected_index = (expected_root / "index.db").absolute()
        if Path(generation.index_path).absolute() != expected_index:
            raise RuntimeError("generation metadata index path escapes generation root")
        try:
            index_metadata = expected_index.lstat()
        except FileNotFoundError:
            index_metadata = None
        if index_metadata is not None and (
            stat.S_ISLNK(index_metadata.st_mode) or not stat.S_ISREG(index_metadata.st_mode)
        ):
            raise RuntimeError("generation metadata index path is not a regular file")
        if Path(generation.archive_root).resolve(strict=False) != self.archive_root.resolve(strict=False):
            raise RuntimeError("generation metadata archive root mismatch")
        if generation.state not in _LIFECYCLE_STATES:
            raise RuntimeError(f"generation metadata has invalid state: {generation.state}")

    @classmethod
    def for_archive_root(
        cls,
        archive_root: Path,
        *,
        repair_anchor: bool = True,
    ) -> IndexGenerationStore:
        """Convenience constructor resolving ``archive_root`` into an :class:`ArchiveLocation` first."""
        return cls(ArchiveLocation.resolve(archive_root), repair_anchor=repair_anchor)

    def _require_write_lease(self, purpose: str) -> None:
        """Require archive-bound admission before mutating lifecycle state.

        Generation metadata, transaction receipts, and candidate teardown are
        part of the same archive publication as the candidate ``index.db``.
        Keeping this check at the store boundary prevents a recovery or
        checkpoint caller from bypassing the connection-level guard merely
        because its particular mutation is a filesystem write.
        """
        from polylogue.storage.sqlite.write_lease import require_write_lease

        require_write_lease(purpose, archive_root=self.archive_root)

    def observe_candidate_capacity(self, *, operation_id: str, generation_id: str) -> None:
        """Raise the recorded peak for this build so the next projection calibrates.

        Calibration is evidence about future builds, never a precondition of
        this one: a receipt that cannot be written is logged and the build
        continues.

        Public because the candidate builder is not always a rebuild
        transaction. The daemon's cold build (``sources/live/cold_build.py``)
        owns a generation directly and is the only production route that
        records a capacity prediction, so it has to be the route that closes
        the loop -- otherwise ``calibrated_index_ratio`` can never leave its
        default and every projection is made from an unmeasured constant.
        """
        from polylogue.maintenance.candidate_capacity import record_capacity_observation

        try:
            record_capacity_observation(
                self.archive_root,
                operation_id=operation_id,
                candidate_root=self.generations_root / generation_id,
            )
        except (OSError, RuntimeError) as exc:
            emit(
                "storage.index_generation.capacity_observation_failed",
                level=WARNING,
                outcome="unmeasured",
                reason="capacity_probe_failed",
                operation_id=operation_id,
                generation_id=generation_id,
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )

    def create(
        self,
        *,
        owner_id: str | None = None,
        source_snapshot: str,
        page_size: int = DEFAULT_ARCHIVE_PAGE_SIZE,
    ) -> IndexGeneration:
        """Materialise one inactive generation.

        ``page_size`` is fixed here and recorded on the generation: SQLite
        freezes it when the first page is allocated, so the only moment this
        archive can choose it is before the index DDL runs.
        """
        created_at_ns = self._next_lifecycle_timestamp_ns()
        generation_id = f"gen-{created_at_ns // 1_000_000}-{uuid.uuid4().hex[:8]}"
        owner = owner_id or str(uuid.uuid4())
        self._validate_lifecycle_id(generation_id, "generation")
        root = self.generations_root / generation_id
        _assert_no_symlink_ancestry(self.generations_root, label="generation root")
        parent_stat = self.generations_root.stat()
        parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
        try:
            root.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            raise RuntimeError(f"generation already exists: {generation_id}") from None
        created_stat = root.lstat()
        created_identity = (created_stat.st_dev, created_stat.st_ino)
        try:
            _assert_no_symlink_ancestry(root, label="generation directory")
            for filename in _GENERATION_READ_THROUGH_MEMBERS:
                source = self.archive_root / filename
                if source.exists() or source.is_symlink():
                    target, identity, is_directory, link_identity = _stable_link_target(
                        source, label=f"durable tier {filename}"
                    )
                    link = root / filename
                    link.symlink_to(target, target_is_directory=is_directory)
                    try:
                        linked = link.stat()
                    except OSError as exc:
                        raise RuntimeError(f"linked durable tier disappeared: {filename}") from exc
                    if (linked.st_dev, linked.st_ino) != identity:
                        raise RuntimeError(f"durable tier changed during linking: {filename}")
                    # The source pathname is still an authority boundary after the
                    # link is installed.  Do not proceed if it was replaced between
                    # capture and post-link verification.
                    _require_path_identity(
                        source, identity, label=f"durable tier {filename}", link_identity=link_identity
                    )
            index_path = root / "index.db"
            from polylogue.storage.sqlite.write_lease import require_write_lease

            require_write_lease(
                f"IndexGenerationStore.create(index={index_path})",
                archive_root=self.archive_root,
            )
            initialize_archive_database(
                index_path,
                ArchiveTier.INDEX,
                page_size=page_size,
                archive_root=self.archive_root,
                inactive_generation=True,
            )
            generation = IndexGeneration(
                generation_id=generation_id,
                owner_id=owner,
                archive_root=str(self.archive_root.resolve(strict=False)),
                index_path=str(index_path),
                state="inactive",
                created_at_ms=created_at_ns // 1_000_000,
                source_snapshot=source_snapshot,
                created_at_ns=created_at_ns,
                page_size=page_size,
            )
            self._write(generation)
            return generation
        except BaseException:
            # The metadata is the enumeration record.  Never leave a partially
            # materialized directory that cannot be loaded or reclaimed.
            with suppress(OSError):
                _remove_created_generation(self.generations_root, generation_id, parent_identity, created_identity)
            raise

    def load(self, generation_id: str) -> IndexGeneration:
        self._validate_lifecycle_id(generation_id, "generation")
        payload = _read_json_nofollow(self._metadata_path(generation_id), label="generation metadata")
        try:
            generation = IndexGeneration(**cast(dict[str, Any], payload))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("invalid generation metadata") from exc
        self._validate_generation(generation, generation_id)
        return generation

    def prepare_promotion(self, generation: IndexGeneration) -> PreparedIndexPromotion:
        """Prepare all archive-sized promotion proofs before writer admission."""
        from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

        current = self.load(generation.generation_id)
        if current.owner_id != generation.owner_id or current.state != "inactive":
            raise RuntimeError("only the owning inactive generation can be promoted")
        candidate_path = Path(current.index_path).resolve(strict=True)
        pointer = self.active_pointer
        prior_active_path = pointer.resolve(strict=True) if (pointer.exists() or pointer.is_symlink()) else None
        # On a first generation the candidate is also the only index against
        # which durable references can be interpreted. It cannot remove a
        # previously resolved target, and the retained candidate observer
        # still proves its schema/incarnation through pointer admission.
        proof_index = prior_active_path or candidate_path
        seal = PreparedIndexMutation(proof_index, archive_root=self.archive_root)
        try:
            candidate_identity = seal.prepare_candidate_reachability(candidate_path)
            return PreparedIndexPromotion(
                generation_id=current.generation_id,
                candidate_path=candidate_path,
                prior_active_path=prior_active_path,
                candidate_identity=candidate_identity,
                reference_seal=seal,
                missing_session_count=seal.candidate_missing_session_count,
                first_missing_session_id=seal.candidate_first_missing_session_id,
            )
        except BaseException as primary:
            try:
                seal.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Index promotion preparation and cleanup failed", [primary, cleanup]
                ) from primary
            raise

    def promote(
        self,
        generation: IndexGeneration,
        prepared: PreparedIndexPromotion | None = None,
    ) -> IndexGeneration:
        """Promote only with off-gate typed-reference and coverage proof."""
        from polylogue.storage.sqlite.write_lease import require_write_lease

        if prepared is None:
            # Direct/offline callers may use the convenience form only when
            # they have not already entered archive writer custody. Production
            # daemon publication passes a retained proof prepared on its
            # managed compute worker before bridge admission.
            from polylogue.core.write_lease import current_write_lease

            if current_write_lease() is not None:
                raise RuntimeError("promotion proof must be prepared before writer admission")
            require_write_lease(
                f"IndexGenerationStore.promote(index={generation.index_path})",
                archive_root=self.archive_root,
            )
            with self.prepare_promotion(generation) as off_gate:
                from polylogue.storage.sqlite.write_lease import write_lease

                with write_lease("storage.index_generation.promote", archive_root=self.archive_root):
                    return self.promote(generation, off_gate)

        require_write_lease(
            f"IndexGenerationStore.promote(index={generation.index_path})",
            archive_root=self.archive_root,
        )
        pointer = self.active_pointer
        if not isinstance(prepared, PreparedIndexPromotion):
            raise RuntimeError("promotion requires a prepared index proof")
        if prepared.generation_id != generation.generation_id:
            raise RuntimeError("promotion proof belongs to another generation")
        if prepared.reference_seal.archive_root != self.archive_root.resolve():
            raise RuntimeError("promotion proof belongs to another archive")
        if Path(generation.index_path).resolve(strict=True) != prepared.candidate_path:
            raise RuntimeError("promotion candidate path changed after proof preparation")
        with self._lifecycle_lock():
            if prepared.prior_active_path is None:
                if pointer.exists() or pointer.is_symlink():
                    raise RuntimeError("active index pointer appeared during promotion preparation")
            elif (
                not (pointer.exists() or pointer.is_symlink())
                or pointer.resolve(strict=True) != prepared.prior_active_path
            ):
                raise RuntimeError("active index pointer changed during promotion preparation")
            prepared.reference_seal.validate_observers_current()
            candidate_identity = prepared.reference_seal.validate_candidate_current(prepared.candidate_path)
            if candidate_identity != prepared.candidate_identity:
                raise RuntimeError("promotion candidate changed after durable-reference validation")
            return self._promote_unlocked(generation)

    def _promote_unlocked(self, generation: IndexGeneration) -> IndexGeneration:
        from polylogue.storage.sqlite.connection_profile import settle_cached_connections_on_current_thread
        from polylogue.storage.sqlite.write_lease import current_sql_custody

        custody = current_sql_custody()
        if custody is None:
            raise RuntimeError("Index promotion requires current physical archive custody")
        settle_cached_connections_on_current_thread(custody)
        current = self.load(generation.generation_id)
        if current.owner_id != generation.owner_id or current.state != "inactive":
            raise RuntimeError("only the owning inactive generation can be promoted")
        self._validate_retention_ownership()
        target_path = Path(current.index_path)
        _assert_no_symlink_ancestry(target_path.parent, label="generation index ancestry")
        target_metadata = target_path.lstat()
        if stat.S_ISLNK(target_metadata.st_mode) or not stat.S_ISREG(target_metadata.st_mode):
            raise RuntimeError("generation index is not a regular, non-symlink file")
        target = target_path.absolute()
        # The generation was built in rollback-journal mode; promotion is its
        # exclusive commit point and the one place it takes the live tiers'
        # WAL mode, before any reader can reach it through the pointer.
        _enter_wal_mode(target, label="new index", archive_root=self.archive_root)
        _checkpoint_truncate(target, label="new index", archive_root=self.archive_root)
        pointer = self.active_pointer
        predecessor_generation_id = self._generation_id_for_active_target(pointer)
        retired = self.generations_root / f"retired-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        sidecars: list[str] = []
        if pointer.exists() or pointer.is_symlink():
            _checkpoint_truncate(pointer, label="active index", archive_root=self.archive_root)
            for suffix in ("-wal", "-shm"):
                sidecar = pointer.with_name(pointer.name + suffix)
                if sidecar.exists():
                    if suffix == "-wal" and sidecar.stat().st_size != 0:
                        raise RuntimeError(f"non-empty active index sidecar blocks promotion: {sidecar}")
                    sidecars.append(suffix)
        promoting = IndexGeneration(
            **{
                **asdict(current),
                "state": "promoting",
                "predecessor_generation_id": predecessor_generation_id,
            }
        )
        temporary = pointer.parent / f".index.db.promote-{uuid.uuid4().hex}"
        # A storage failure before the swap must leave this same candidate
        # retryable. Keep the prior pointer identity as the rollback proof;
        # a failed replace is not permission to overwrite a pointer that moved.
        prior_pointer = pointer.lstat() if pointer.exists() or pointer.is_symlink() else None
        prior_identity = (prior_pointer.st_dev, prior_pointer.st_ino) if prior_pointer is not None else None
        rollback = self._metadata_path(current.generation_id).with_name("generation.rollback.json")
        pointer_proof = self._rollback_pointer_proof_path(current.generation_id)
        try:
            _atomic_json_write(
                pointer_proof,
                self._rollback_pointer_proof(current, prior_identity, tuple(sidecars), retired.name),
                label="prior pointer proof",
            )
            try:
                _atomic_json_write(rollback, asdict(current), label="generation rollback")
            except BaseException:
                # Setup is a pair: a lone pointer proof makes a later retry
                # fail O_EXCL before it can reproduce the rollback record.
                with suppress(OSError):
                    pointer_proof.unlink(missing_ok=True)
                    _fsync_directory(pointer_proof.parent)
                raise
        except RuntimeError as exc:
            if isinstance(exc.__cause__, OSError):
                raise exc.__cause__ from exc
            raise
        try:
            self._write(promoting)
            retired.mkdir(parents=True, exist_ok=False)
            _fsync_directory(self.generations_root)
            for suffix in sidecars:
                sidecar = pointer.with_name(pointer.name + suffix)
                os.replace(sidecar, retired / sidecar.name)
            if pointer.exists() or pointer.is_symlink():
                os.link(pointer, retired / "index.db", follow_symlinks=False)
            _fsync_directory(retired)
            _fsync_directory(pointer.parent)
            temporary.symlink_to(target)
            os.replace(temporary, pointer)
        except Exception as exc:
            storage_error = exc if isinstance(exc, OSError) else exc.__cause__
            if not isinstance(storage_error, OSError):
                raise
            with suppress(OSError):
                temporary.unlink()
            current_pointer = pointer.lstat() if pointer.exists() or pointer.is_symlink() else None
            current_identity = (current_pointer.st_dev, current_pointer.st_ino) if current_pointer is not None else None
            if current_identity == prior_identity:
                try:
                    self._restore_pre_swap_predecessor(retired, prior_identity, tuple(sidecars))
                    if self.load(current.generation_id).state == "promoting":
                        os.replace(rollback, self._metadata_path(current.generation_id))
                        _fsync_directory(rollback.parent)
                except (OSError, UnpublishedPromotionRecoveryError):
                    self._unpublished_rollbacks[current.generation_id] = (
                        current,
                        rollback,
                        retired,
                        prior_identity,
                        tuple(sidecars),
                    )
                else:
                    with suppress(OSError):
                        rollback.unlink(missing_ok=True)
                    with suppress(OSError):
                        pointer_proof.unlink(missing_ok=True)
            if storage_error is exc:
                raise
            raise storage_error from exc
        _fsync_directory(pointer.parent)
        promoted = IndexGeneration(
            **{
                **asdict(current),
                "state": "active",
                "promoted_at_ns": self._next_lifecycle_timestamp_ns(),
                "predecessor_generation_id": predecessor_generation_id,
                "retention_owner_id": current.generation_id,
                "retention_state": GenerationRetentionState.ACTIVE.value,
            }
        )
        self._write(promoted)
        with suppress(OSError):
            rollback.unlink(missing_ok=True)
        with suppress(OSError):
            pointer_proof.unlink(missing_ok=True)
        # Retention collection is part of promotion rather than a separate
        # cleanup surface. Its receipt is written before any eligible
        # generation is removed, so a completed pointer swap never reclaims
        # history without durable evidence of the retention boundary.
        try:
            self._collect_superseded_generations(promoted)
        except OSError as exc:
            emit(
                "storage.index_generation.retention_collection_failed",
                level=WARNING,
                outcome="degraded",
                reason="retention_collection_failed",
                phase="promotion",
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
        return promoted

    def unpublished_rollback_pending(self, generation_id: str) -> bool:
        return generation_id in self._unpublished_rollbacks

    def _restore_pre_swap_predecessor(
        self, retired: Path, prior_identity: tuple[int, int] | None, sidecars: tuple[str, ...]
    ) -> None:
        """Restore the prior pointer's sidecars before releasing an unpublished candidate."""
        pointer = self.active_pointer
        if retired.parent != self.generations_root or not re.fullmatch(r"retired-\d+-[0-9a-f]{8}", retired.name):
            raise UnpublishedPromotionRecoveryError("retired predecessor marker identity is invalid")
        try:
            marker_metadata = retired.lstat()
        except FileNotFoundError:
            marker_metadata = None
        if marker_metadata is not None and not stat.S_ISDIR(marker_metadata.st_mode):
            raise UnpublishedPromotionRecoveryError("retired predecessor marker is not a directory")
        if marker_metadata is not None:
            try:
                pointer_backup = (retired / "index.db").lstat()
            except FileNotFoundError:
                pointer_backup = None
            if (
                pointer_backup is not None
                and (
                    pointer_backup.st_dev,
                    pointer_backup.st_ino,
                )
                != prior_identity
            ):
                raise UnpublishedPromotionRecoveryError("retired predecessor pointer identity changed")
        for suffix in ("-wal", "-shm"):
            original = pointer.with_name(pointer.name + suffix)
            backup = retired / original.name
            try:
                backup_metadata = backup.lstat() if marker_metadata is not None else None
            except FileNotFoundError:
                backup_metadata = None
            if backup_metadata is not None and suffix not in sidecars:
                raise UnpublishedPromotionRecoveryError("retired predecessor has an unexpected sidecar")
            if backup_metadata is not None and not stat.S_ISREG(backup_metadata.st_mode):
                raise UnpublishedPromotionRecoveryError("retired predecessor sidecar is not a regular file")
            if suffix not in sidecars:
                continue
            try:
                original_metadata = original.lstat()
            except FileNotFoundError:
                original_metadata = None
            if backup_metadata is None and original_metadata is None:
                # Promotion records a sidecar only after checkpointing the WAL
                # to empty; -shm is rebuildable. Before the marker exists no
                # sidecar was moved, so a clean SQLite close may have removed
                # them at the pointer: nothing to restore. Once the marker
                # exists the move happened and its backup must still be there.
                if marker_metadata is None:
                    continue
                raise UnpublishedPromotionRecoveryError("predecessor sidecar is missing from both locations")
            if backup_metadata is not None:
                if original_metadata is None:
                    os.link(backup, original, follow_symlinks=False)
                elif (original_metadata.st_dev, original_metadata.st_ino) != (
                    backup_metadata.st_dev,
                    backup_metadata.st_ino,
                ):
                    raise UnpublishedPromotionRecoveryError("predecessor sidecar was replaced during rollback")
        _fsync_directory(pointer.parent)
        if marker_metadata is not None:
            for suffix in sidecars:
                (retired / f"{pointer.name}{suffix}").unlink(missing_ok=True)
            _fsync_directory(retired)
            if any(path.name != "index.db" for path in retired.iterdir()):
                raise UnpublishedPromotionRecoveryError("retired predecessor marker has unexpected contents")
            shutil.rmtree(retired)
            _fsync_directory(self.generations_root)

    def restore_unpublished_promotion(self, generation_id: str) -> None:
        """Finish a same-process pre-swap rollback after storage returns."""
        pending = self._unpublished_rollbacks.get(generation_id)
        if pending is None:
            return
        self._require_write_lease(f"IndexGenerationStore.restore_unpublished_promotion(generation={generation_id})")
        with self._lifecycle_lock():
            original, rollback, retired, prior_identity, sidecars = pending
            current = self.load(generation_id)
            if current.owner_id != original.owner_id or current.state not in {"promoting", "inactive"}:
                raise RuntimeError("unpublished promotion owner or state changed")
            pointer = self.active_pointer
            metadata = pointer.lstat() if pointer.exists() or pointer.is_symlink() else None
            identity = (metadata.st_dev, metadata.st_ino) if metadata is not None else None
            if identity != prior_identity:
                raise RuntimeError("active pointer changed during unpublished promotion rollback")
            self._restore_pre_swap_predecessor(retired, prior_identity, sidecars)
            if current.state == "promoting":
                if _read_json_nofollow(rollback, label="generation rollback") != asdict(original):
                    raise RuntimeError("generation rollback record changed")
                os.replace(rollback, self._metadata_path(generation_id))
            _fsync_directory(rollback.parent)
            self._unpublished_rollbacks.pop(generation_id, None)
            with suppress(OSError):
                self._rollback_pointer_proof_path(generation_id).unlink(missing_ok=True)

    def discard_unpublished_cold_promotion(self, generation: IndexGeneration) -> bool:
        """Reclaim an interrupted pre-swap cold candidate, never a published one."""
        self._require_write_lease(
            f"IndexGenerationStore.discard_unpublished_cold_promotion(generation={generation.generation_id})"
        )
        with self._lifecycle_lock():
            current = self.load(generation.generation_id)
            if current.owner_id != generation.owner_id or not current.owner_id.startswith("cold-build:"):
                raise UnpublishedPromotionRecoveryError("cold promotion ownership changed")
            if current.state != "promoting":
                return False
            pointer = self.active_pointer
            if pointer.resolve(strict=False) == Path(current.index_path).resolve(strict=False):
                raise UnpublishedPromotionRecoveryError("pointer-swapped cold promotion needs activation recovery")
            metadata = pointer.lstat() if pointer.exists() or pointer.is_symlink() else None
            identity = (metadata.st_dev, metadata.st_ino) if metadata is not None else None
            rollback = self._metadata_path(current.generation_id).with_name("generation.rollback.json")
            proof_path = self._rollback_pointer_proof_path(current.generation_id)
            try:
                original = IndexGeneration(
                    **cast(dict[str, Any], _read_json_nofollow(rollback, label="generation rollback"))
                )
                proof = _read_json_nofollow(proof_path, label="prior pointer proof")
                self._validate_generation(original, current.generation_id)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise UnpublishedPromotionRecoveryError("cold promotion rollback proof is unavailable") from exc
            expected = IndexGeneration(
                **{
                    **asdict(original),
                    "state": "promoting",
                    "predecessor_generation_id": current.predecessor_generation_id,
                }
            )
            raw_sidecars = proof.get("sidecars")
            retired_name = proof.get("retired_marker")
            if not isinstance(raw_sidecars, list) or not isinstance(retired_name, str):
                raise UnpublishedPromotionRecoveryError("cold promotion rollback proof has invalid custody")
            sidecars = tuple(cast(list[object], raw_sidecars))
            if sidecars != tuple(suffix for suffix in ("-wal", "-shm") if suffix in sidecars):
                raise UnpublishedPromotionRecoveryError("cold promotion rollback sidecars are invalid")
            if not re.fullmatch(r"retired-\d+-[0-9a-f]{8}", retired_name):
                raise UnpublishedPromotionRecoveryError("cold promotion rollback marker is invalid")
            if (
                original.state != "inactive"
                or original.owner_id != current.owner_id
                or asdict(expected) != asdict(current)
                or proof
                != self._rollback_pointer_proof(original, identity, cast(tuple[str, ...], sidecars), retired_name)
            ):
                raise UnpublishedPromotionRecoveryError("cold promotion rollback proof does not match active pointer")
            self._restore_pre_swap_predecessor(
                self.generations_root / retired_name,
                identity,
                cast(tuple[str, ...], sidecars),
            )
            shutil.rmtree(self._metadata_path(current.generation_id).parent)
            _fsync_directory(self.generations_root)
            return True

    def _validate_retention_ownership(self) -> None:
        """Require every prior promoted generation to name its build owner.

        This runs before the active pointer moves. An ownerless predecessor is
        ambiguous history, not a reclaimable candidate: promotion stops while
        the old generation is still live rather than creating a future GC path
        with no accountable owner.
        """
        for metadata_path in sorted(self.generations_root.glob("gen-*/generation.json")):
            generation = self.load(metadata_path.parent.name)
            if generation.state == "active" and not generation.owner_id.strip():
                raise RuntimeError(f"retention ownership is missing for generation {generation.generation_id}")

    def _collect_superseded_generations(self, promoted: IndexGeneration) -> GenerationRetentionReceipt:
        """Automatically retain one rollback target and reclaim older history.

        A promoted generation is large enough that a bounded retention window
        matters, but the immediately preceding generation remains rollback
        capable until the next promotion crosses the declared boundary. The
        receipt first records every eligible generation, then records its
        reclaimed state after filesystem removal.
        """
        generations_root_identity = _stable_directory(self.generations_root, label="generation root")
        generations_root_link_identity = _configured_link_identity(self.generations_root)
        _require_path_identity(
            self.generations_root,
            generations_root_identity,
            label="generation root",
            link_identity=generations_root_link_identity,
        )
        active_target = self.active_pointer.resolve(strict=True)
        candidates: list[tuple[int, int, str, Path, IndexGeneration]] = []
        for metadata_path in sorted(self.generations_root.glob("gen-*/generation.json")):
            generation = self.load(metadata_path.parent.name)
            if generation.state != "active":
                continue
            try:
                if Path(generation.index_path).resolve(strict=True) == active_target:
                    continue
            except OSError:
                continue  # an incomplete candidate remains retained
            candidates.append(
                (
                    _generation_lifecycle_recency_ns(generation),
                    _generation_creation_recency_ns(generation),
                    generation.generation_id,
                    metadata_path.parent,
                    generation,
                )
            )

        # Promotion order decides rollback capability. A normal promotion
        # records its actual predecessor before the pointer swap and pins it
        # first; recovered promotions have no pre-swap observation, so their
        # persisted promotion timestamp supplies the same chronology. UUIDs
        # are only a deterministic final tie-break, never the recency signal.
        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        predecessor = promoted.predecessor_generation_id
        if predecessor is not None:
            candidates.sort(key=lambda item: item[4].generation_id != predecessor)
        retained = candidates[:SUPERSEDED_GENERATION_RETENTION]
        eligible = candidates[SUPERSEDED_GENERATION_RETENTION:]
        records = [
            GenerationRetentionRecord(
                generation_id=promoted.generation_id,
                generation_owner_id=promoted.owner_id,
                retention_owner_id=promoted.generation_id,
                state=GenerationRetentionState.ACTIVE,
            )
        ]
        for _lifecycle_at_ns, _created_at_ns, _generation_id, _directory, generation in retained:
            retained_generation = IndexGeneration(
                **{
                    **asdict(generation),
                    "retention_owner_id": promoted.generation_id,
                    "retention_state": GenerationRetentionState.RETAINED.value,
                }
            )
            self._write(retained_generation)
            records.append(
                GenerationRetentionRecord(
                    generation_id=retained_generation.generation_id,
                    generation_owner_id=retained_generation.owner_id,
                    retention_owner_id=promoted.generation_id,
                    state=GenerationRetentionState.RETAINED,
                )
            )
        for _lifecycle_at_ns, _created_at_ns, _generation_id, _directory, generation in eligible:
            eligible_generation = IndexGeneration(
                **{
                    **asdict(generation),
                    "retention_owner_id": promoted.generation_id,
                    "retention_state": GenerationRetentionState.ELIGIBLE.value,
                }
            )
            self._write(eligible_generation)
        receipt = GenerationRetentionReceipt(
            promoted_generation_id=promoted.generation_id,
            promoted_at_ns=promoted.promoted_at_ns,
            retention_boundary=SUPERSEDED_GENERATION_RETENTION,
            automatic=True,
            records=tuple(records)
            + tuple(
                GenerationRetentionRecord(
                    generation_id=generation.generation_id,
                    generation_owner_id=generation.owner_id,
                    retention_owner_id=promoted.generation_id,
                    state=GenerationRetentionState.ELIGIBLE,
                )
                for _lifecycle_at_ns, _created_at_ns, _generation_id, _directory, generation in eligible
            ),
            eligible_generation_ids=tuple(
                generation.generation_id
                for _lifecycle_at_ns, _created_at_ns, _generation_id, _directory, generation in eligible
            ),
        )
        self._write_retention_receipt(receipt)

        reclaimed: list[str] = []
        try:
            generations_fd = os.open(self.generations_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError as exc:
            raise RuntimeError("cannot securely open generation root") from exc
        try:
            for _lifecycle_at_ns, _created_at_ns, generation_id, directory, _generation in eligible:
                _require_path_identity(
                    self.generations_root,
                    generations_root_identity,
                    label="generation root",
                    link_identity=generations_root_link_identity,
                )
                shutil.rmtree(directory.name, dir_fd=generations_fd)
                reclaimed.append(generation_id)
        finally:
            os.close(generations_fd)

        # The retired-* markers only point at superseded generations, so they
        # follow the same retention -- otherwise they accumulate as dangling
        # links to directories this method just removed.
        markers = sorted(
            (path for path in self.generations_root.glob("retired-*") if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        pruned_markers = 0
        try:
            markers_fd = os.open(self.generations_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError as exc:
            raise RuntimeError("cannot securely open generation root") from exc
        try:
            for marker in markers[SUPERSEDED_GENERATION_RETENTION:]:
                _require_path_identity(
                    self.generations_root,
                    generations_root_identity,
                    label="generation root",
                    link_identity=generations_root_link_identity,
                )
                shutil.rmtree(marker.name, dir_fd=markers_fd)
                pruned_markers += 1
        finally:
            os.close(markers_fd)

        if reclaimed or pruned_markers:
            _fsync_directory(self.generations_root)
        completed = GenerationRetentionReceipt(
            promoted_generation_id=receipt.promoted_generation_id,
            promoted_at_ns=receipt.promoted_at_ns,
            retention_boundary=receipt.retention_boundary,
            automatic=True,
            records=tuple(
                record
                if record.generation_id not in reclaimed
                else GenerationRetentionRecord(
                    generation_id=record.generation_id,
                    generation_owner_id=record.generation_owner_id,
                    retention_owner_id=record.retention_owner_id,
                    state=GenerationRetentionState.RECLAIMED,
                )
                for record in receipt.records
            ),
            eligible_generation_ids=receipt.eligible_generation_ids,
            reclaimed_marker_count=pruned_markers,
        )
        self._write_retention_receipt(completed)
        self._prune_retention_receipts(current_generation_id=completed.promoted_generation_id)
        if reclaimed:
            emit(
                "storage.index_generation.superseded_reclaimed",
                outcome="ok",
                generation_id=completed.promoted_generation_id,
                reclaimed=len(reclaimed),
            )
        return completed

    def load_retention_receipt(self, promoted_generation_id: str) -> GenerationRetentionReceipt:
        self._validate_lifecycle_id(promoted_generation_id, "promoted generation")
        payload = _read_json_nofollow(self._retention_receipt_path(promoted_generation_id), label="retention receipt")
        payload_any = cast(Any, payload)
        try:
            receipt = GenerationRetentionReceipt(
                promoted_generation_id=str(payload_any["promoted_generation_id"]),
                promoted_at_ns=int(payload_any.get("promoted_at_ns", 0)),
                retention_boundary=int(payload_any["retention_boundary"]),
                automatic=bool(payload_any["automatic"]),
                records=tuple(
                    GenerationRetentionRecord(
                        generation_id=str(record["generation_id"]),
                        generation_owner_id=str(record["generation_owner_id"]),
                        retention_owner_id=str(record["retention_owner_id"]),
                        state=GenerationRetentionState(str(record["state"])),
                    )
                    for record in payload_any["records"]
                ),
                eligible_generation_ids=tuple(
                    str(generation_id) for generation_id in payload_any["eligible_generation_ids"]
                ),
                reclaimed_marker_count=int(payload_any.get("reclaimed_marker_count", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("invalid retention receipt") from exc
        self._validate_receipt(receipt, promoted_generation_id)
        return receipt

    def _validate_receipt(self, receipt: GenerationRetentionReceipt, requested_id: str) -> None:
        if receipt.promoted_generation_id != requested_id:
            raise RuntimeError("retention receipt identity mismatch")
        if not receipt.automatic or receipt.retention_boundary != SUPERSEDED_GENERATION_RETENTION:
            raise RuntimeError("retention receipt provenance mismatch")
        for record in receipt.records:
            self._validate_lifecycle_id(record.generation_id, "receipt generation")
            self._validate_lifecycle_id(record.generation_owner_id, "receipt generation owner")
            self._validate_lifecycle_id(record.retention_owner_id, "receipt retention owner")
        for generation_id in receipt.eligible_generation_ids:
            self._validate_lifecycle_id(generation_id, "eligible generation")
        if receipt.promoted_generation_id not in {record.generation_id for record in receipt.records}:
            raise RuntimeError("retention receipt does not contain promoted generation")

    def _validate_lifecycle_id(self, value: str, label: str) -> None:
        if _SAFE_LIFECYCLE_ID.fullmatch(value) is None:
            raise RuntimeError(f"invalid {label} identifier")

    def recover_promotion(self, generation_id: str) -> IndexGeneration:
        """Reconcile an incomplete promotion without trusting the pointer alone.

        A pointer swap is only one durable step in promotion.  When the
        pointer already targets the promoting generation, leave the metadata
        in ``promoting`` until the activation caller has revalidated the
        candidate against the current archive and explicitly completes
        recovery.  A pointer mismatch means the swap never became visible, so
        the candidate can safely return to ``inactive``.
        """
        self._require_write_lease(f"IndexGenerationStore.recover_promotion(generation={generation_id})")
        generation = self.load(generation_id)
        if generation.state != "promoting":
            return generation
        pointer = self.active_pointer
        if pointer.exists() or pointer.is_symlink():
            try:
                pointer_target = pointer.resolve(strict=True)
                expected_target = Path(generation.index_path).resolve(strict=True)
            except OSError as exc:
                raise RuntimeError("cannot recover through an unreadable active pointer") from exc
            # A promoting record is authenticated only by its own generation
            # inode.  A pointer to an external file must never make us clear
            # or activate that record.
            if pointer_target == expected_target:
                return generation
            try:
                pointer_target.relative_to(self.generations_root.resolve(strict=True))
            except (OSError, ValueError) as exc:
                raise RuntimeError("active pointer targets outside generation root") from exc
        # A mismatch can mean a newer active generation won the race. Never
        # clear a promoting record while another active identity owns pointer.
        if pointer.exists() or pointer.is_symlink():
            try:
                pointer_target = pointer.resolve(strict=True)
            except OSError as exc:
                raise RuntimeError("cannot recover through an unreadable active pointer") from exc
            for metadata_path in sorted(self.generations_root.glob("gen-*/generation.json")):
                candidate = self.load(metadata_path.parent.name)
                if candidate.state == "active" and Path(candidate.index_path).resolve(strict=True) == pointer_target:
                    return generation
        recovered = IndexGeneration(**{**asdict(generation), "state": "inactive"})
        self._write(recovered)
        return recovered

    def complete_promotion_recovery(self, generation_id: str) -> IndexGeneration:
        """Record a pointer-swapped promotion as active after external validation."""
        self._require_write_lease(f"IndexGenerationStore.complete_promotion_recovery(generation={generation_id})")
        generation = self.load(generation_id)
        if generation.state != "promoting":
            return generation
        pointer = self.active_pointer
        if not (pointer.exists() or pointer.is_symlink()):
            raise RuntimeError("cannot complete promotion recovery without an active index pointer")
        try:
            pointer_target = pointer.resolve(strict=True)
            expected_target = Path(generation.index_path).resolve(strict=True)
            pointer_target.relative_to(self.generations_root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise RuntimeError("active pointer targets outside generation root") from exc
        if pointer_target != expected_target:
            raise RuntimeError("cannot complete promotion recovery for a non-active generation")
        self._validate_retention_ownership()
        # A prior attempt may have swapped the pointer but failed to fsync its
        # parent. Prove the pointer durable before active metadata or receipt
        # cleanup can complete.
        _fsync_directory(pointer.parent)
        recovered = IndexGeneration(
            **{
                **asdict(generation),
                "state": "active",
                "promoted_at_ns": self._next_lifecycle_timestamp_ns(),
                "retention_owner_id": generation.generation_id,
                "retention_state": GenerationRetentionState.ACTIVE.value,
            }
        )
        self._write(recovered)
        try:
            self._collect_superseded_generations(recovered)
        except OSError as exc:
            emit(
                "storage.index_generation.retention_collection_failed",
                level=WARNING,
                outcome="degraded",
                reason="retention_collection_failed",
                phase="recovered_promotion",
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
        return recovered

    def discard_if_inactive(self, generation: IndexGeneration) -> bool:
        """Remove a terminal failed candidate without risking an active target."""
        # Removing an inactive generation removes its writable ``index.db``
        # alongside lifecycle metadata.  It therefore belongs to the same
        # archive-bound authority as create, membership publication, and
        # promotion; the daemon must not race a candidate writer by treating
        # empty-build cleanup as ordinary background filesystem work.
        from polylogue.storage.sqlite.write_lease import require_write_lease

        require_write_lease(
            f"IndexGenerationStore.discard_if_inactive(index={generation.index_path})",
            archive_root=self.archive_root,
        )
        current = self.load(generation.generation_id)
        if current.owner_id != generation.owner_id or current.state != "inactive":
            return False
        shutil.rmtree(self._metadata_path(generation.generation_id).parent)
        _fsync_directory(self.generations_root)
        return True

    def _metadata_path(self, generation_id: str) -> Path:
        self._validate_lifecycle_id(generation_id, "generation")
        return self.generations_root / generation_id / "generation.json"

    def _rollback_pointer_proof_path(self, generation_id: str) -> Path:
        return self._metadata_path(generation_id).with_name("generation.rollback-pointer.json")

    def _rollback_pointer_proof(
        self,
        generation: IndexGeneration,
        pointer_identity: tuple[int, int] | None,
        sidecars: tuple[str, ...],
        retired_marker: str,
    ) -> dict[str, object]:
        return {
            "schema": "polylogue.generation-rollback-pointer.v2",
            "generation_id": generation.generation_id,
            "owner_id": generation.owner_id,
            "active_pointer": str(self.active_pointer.absolute()),
            "pointer_device": pointer_identity[0] if pointer_identity is not None else None,
            "pointer_inode": pointer_identity[1] if pointer_identity is not None else None,
            "sidecars": list(sidecars),
            "retired_marker": retired_marker,
        }

    def _retention_receipt_path(self, promoted_generation_id: str) -> Path:
        self._validate_lifecycle_id(promoted_generation_id, "promoted generation")
        return self.generations_root / _RETENTION_RECEIPTS_DIRNAME / f"{promoted_generation_id}.json"

    def _generation_id_for_active_target(self, pointer: Path) -> str | None:
        """Find the generation currently exposed by ``pointer``, if any."""
        if not (pointer.exists() or pointer.is_symlink()):
            return None
        try:
            active_target = pointer.resolve(strict=True)
        except OSError:
            return None
        for metadata_path in sorted(self.generations_root.glob("gen-*/generation.json")):
            try:
                generation = IndexGeneration(**json.loads(metadata_path.read_text(encoding="utf-8")))
                if generation.state == "active" and Path(generation.index_path).resolve(strict=True) == active_target:
                    return generation.generation_id
            except (OSError, ValueError, TypeError):
                continue
        return None

    def _next_lifecycle_timestamp_ns(self) -> int:
        """Return a persisted lifecycle timestamp that never moves backwards.

        Rebuild ownership serializes production promotion. Reading the small
        bounded generation set here also keeps a fresh store instance monotonic
        after a process restart or a coarse/frozen wall clock in a test.
        """
        latest = 0
        for metadata_path in self.generations_root.glob("gen-*/generation.json"):
            try:
                generation = IndexGeneration(**json.loads(metadata_path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                continue
            latest = max(latest, _generation_lifecycle_recency_ns(generation))
        return max(time.time_ns(), latest + 1)

    def _write(self, generation: IndexGeneration) -> None:
        self._validate_generation(generation, generation.generation_id)
        path = self._metadata_path(generation.generation_id)
        _ensure_lifecycle_directory(path.parent, label="generation directory")
        _atomic_json_write(path, asdict(generation), label="generation metadata")

    def _write_retention_receipt(self, receipt: GenerationRetentionReceipt) -> None:
        self._validate_lifecycle_id(receipt.promoted_generation_id, "promoted generation")
        path = self._retention_receipt_path(receipt.promoted_generation_id)
        _ensure_lifecycle_directory(path.parent, label="retention receipt root")
        _atomic_json_write(path, asdict(receipt), label="retention receipt")

    def _prune_retention_receipts(self, *, current_generation_id: str) -> None:
        """Bound receipt history without deleting current or unreadable proof."""
        receipts_root = self.generations_root / _RETENTION_RECEIPTS_DIRNAME
        candidates: list[tuple[int, str, Path]] = []
        for receipt_path in receipts_root.glob("*.json"):
            if receipt_path == self._retention_receipt_path(current_generation_id):
                continue
            try:
                payload = json.loads(receipt_path.read_text(encoding="utf-8"))
                promoted_at_ns = int(payload.get("promoted_at_ns", 0))
            except (OSError, ValueError, TypeError):
                continue  # malformed evidence remains visible for investigation
            candidates.append((promoted_at_ns, receipt_path.name, receipt_path))
        candidates.sort(reverse=True)
        for _promoted_at_ns, _name, receipt_path in candidates[RETENTION_RECEIPT_HISTORY - 1 :]:
            receipt_path.unlink()
        if len(candidates) >= RETENTION_RECEIPT_HISTORY:
            _fsync_directory(receipts_root)


@contextmanager
def _open_source_snapshot(archive_root: Path) -> Iterator[sqlite3.Connection]:
    """Open source.db through an already-open descriptor.

    The source snapshot is an authority boundary: opening by pathname and
    checking its inode afterwards still permits replacement in between those
    operations. Linux's proc fd view lets SQLite bind its main database to the
    descriptor we opened (and therefore to that inode), while retaining
    SQLite's normal read-only behavior. Keep the descriptor alive through
    connection close so SQLite cannot outlive the identity it was admitted
    against.
    """
    path = archive_root / "source.db"
    target, expected_identity, is_directory, link_identity = _stable_link_target(path, label="source snapshot")
    if is_directory:
        raise RuntimeError(f"source snapshot is not a regular file: {path}")
    from polylogue.storage.sqlite.connection_profile import (
        NativeConnectionSettlementError,
        NativeSQLCustodyOwner,
        _close_failed_native_construction,
    )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(target, flags)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != expected_identity:
            raise RuntimeError(f"source snapshot changed during descriptor admission: {path}")
        _require_path_identity(path, expected_identity, label="source snapshot", link_identity=link_identity)
        alias = descriptor_alias_path(fd)
        if alias is None:
            raise RuntimeError(f"no validated descriptor alias for source snapshot: {path}")
        try:
            conn = open_readonly_connection(
                path,
                opened_main_fd=fd,
                timeout_class="background-read",
                validate_schema=False,
            )
        except NativeConnectionSettlementError as cleanup:
            owned_fd, fd = fd, -1
            cleanup.owner.retain_anchored_descriptor(owned_fd)
            raise
        owned_fd, fd = fd, -1
        owner = NativeSQLCustodyOwner(conn, anchored_descriptors=(owned_fd,))
        try:
            yield conn
            _require_path_identity(path, expected_identity, label="source snapshot", link_identity=link_identity)
        except BaseException as primary:
            _close_failed_native_construction(owner, primary)
            raise
        else:
            owner.close()
    finally:
        if fd >= 0:
            settlement_primary = sys.exception()
            owned_fd, fd = fd, -1
            cleanup_owner = NativeSQLCustodyOwner(None, anchored_descriptors=(owned_fd,))
            if settlement_primary is not None:
                _close_failed_native_construction(cleanup_owner, settlement_primary)
            else:
                cleanup_owner.close()


def source_revision_snapshot(archive_root: Path) -> str:
    """Hash the full mutable raw-session state after a rebuild replay."""
    import hashlib

    digest = hashlib.sha256()
    with _open_source_snapshot(archive_root) as conn:
        for row in conn.execute("SELECT * FROM raw_sessions ORDER BY raw_id"):
            for value in row:
                encoded = value.hex() if isinstance(value, bytes) else str(value)
                digest.update(encoded.encode())
                digest.update(b"\0")
            digest.update(b"\n")
    return digest.hexdigest()


def rebuild_source_evidence_snapshot(archive_root: Path) -> str:
    """Hash the immutable source evidence a rebuild is allowed to replay.

    Parse, validation, and revision-governance state are rebuild outputs or
    post-acquisition interpretation. They can legitimately change while the
    rebuild runs, so they must never invalidate its before/after source proof.
    The selected columns capture every durable raw and authority field already
    consumed by revision backfill, together with the parser/lowering semantic
    fingerprints that give those fields meaning. Equivalent receipt snapshots
    use the same ordered evidence below, so a resumable pass cannot cross a
    changed revision authority boundary.
    """
    import hashlib

    from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin
    from polylogue.storage.blob_store import BlobStore

    digest = hashlib.sha256()
    with _open_source_snapshot(archive_root) as conn:
        rows = conn.execute(
            """
            SELECT raw_id, origin, capture_mode, native_id, source_path,
                   source_index, blob_hash, blob_size, acquired_at_ms,
                   file_mtime_ms, logical_source_key, revision_kind,
                   source_revision, predecessor_source_revision,
                   predecessor_raw_id, baseline_raw_id, append_start_offset,
                   append_end_offset, acquisition_generation,
                   revision_authority, revision_authority_evidence
            FROM raw_sessions
            ORDER BY raw_id
            """
        )
        origins: set[str] = set()
        for row in rows:
            origins.add(str(row[1]))
            for value in row:
                if value is None:
                    encoded = b"n"
                elif isinstance(value, bytes):
                    encoded = b"b" + value
                elif isinstance(value, str):
                    encoded = b"s" + value.encode()
                else:
                    encoded = b"i" + str(value).encode()
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        sorted_origins = sorted(origins)
        digest.update(b"parser-lowering-semantic-fingerprints\0")
        digest.update(b"lowering\0")
        lowering = lowering_fingerprint()
        digest.update(len(lowering).to_bytes(8, "big"))
        digest.update(lowering.encode())
        for origin in sorted_origins:
            parser = parser_fingerprint_for_origin(origin)
            for value in (origin, parser):
                encoded = value.encode()
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        raw_blob_hashes = {
            bytes(row[0]).hex() if isinstance(row[0], (bytes, bytearray, memoryview)) else str(row[0])
            for row in conn.execute("SELECT DISTINCT blob_hash FROM raw_sessions ORDER BY blob_hash")
        }
        blob_store = BlobStore(archive_root / "blob")
        digest.update(b"raw_payload_bytes_verified\0")
        for blob_hash in sorted(raw_blob_hashes):
            if not blob_store.verify(blob_hash):
                raise RuntimeError(f"raw payload blob bytes failed verification: {blob_hash}")
            # ``BlobStore.verify`` has re-hashed the bytes at this point. Keep
            # the verified content identity in the existing canonical digest,
            # rather than introducing a second blob hashing implementation.
            encoded = b"s" + blob_hash.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        digest.update(b"raw_payload_refs\0")
        blob_refs = conn.execute(
            """
            SELECT b.ref_type, b.ref_id, b.blob_hash, b.source_path,
                   b.size_bytes, b.acquired_at_ms
            FROM blob_refs AS b
            JOIN raw_sessions AS r ON r.raw_id = b.ref_id
            WHERE b.ref_type = 'raw_payload'
            ORDER BY b.ref_id, b.blob_hash
            """
        )
        for row in blob_refs:
            for value in row:
                if value is None:
                    encoded = b"n"
                elif isinstance(value, bytes):
                    encoded = b"b" + value
                elif isinstance(value, str):
                    encoded = b"s" + value.encode()
                else:
                    encoded = b"i" + str(value).encode()
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        digest.update(b"raw_capture_observations\0")
        observations = conn.execute(
            """
            SELECT raw_id, capture_mode, first_observed_at_ms
            FROM raw_capture_observations
            ORDER BY raw_id, capture_mode
            """
        )
        for row in observations:
            for value in row:
                encoded = b"s" + value.encode() if isinstance(value, str) else b"i" + str(value).encode()
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
    return digest.hexdigest()


def _enter_wal_mode(path: Path, *, label: str, archive_root: Path) -> None:
    """Switch a rollback-mode inactive generation to WAL at its promotion.

    Promotion is the generation's exclusive commit point: its build writer is
    closed and no reader can reach it before the pointer swap, so this is the
    one place its header may change. An already-WAL file is left as is.
    """
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner, execute_pragma_statement
    from polylogue.storage.sqlite.write_lease import require_write_lease

    require_write_lease(f"index generation {label} WAL entry({path})", archive_root=archive_root)
    open_path = path.resolve(strict=True)
    owner = NativeSQLCustodyOwner(connect_measured(f"{open_path.as_uri()}?mode=rw", uri=True))
    try:
        conn = owner.require_connection()
        execute_pragma_statement(conn, "PRAGMA journal_mode=WAL")
        mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        if mode != "wal":
            raise RuntimeError(f"{label} could not enter WAL mode: {mode!r}")
    finally:
        owner.close()


def _checkpoint_truncate(path: Path, *, label: str, archive_root: Path) -> None:
    """Checkpoint one inode without a path check-then-reopen race.

    An exclusive ``TRUNCATE`` checkpoint is a durable mutation of an archive
    tier: it rewrites the database file from the WAL and empties the WAL. It
    is also the one writable open in this module that the connection-level
    guard structurally cannot see. The open goes through
    ``/proc/self/fd/N`` -- deliberately, so the checked descriptor cannot be
    swapped between validation and ``sqlite3.connect`` -- and
    ``guarded_archive_tier_path`` decides tier membership from the *file
    name*, which for that alias is a descriptor number and never ``index.db``.

    So this site asserts ownership itself, before any descriptor is opened,
    naming the real path rather than the alias. ``archive_root`` is required
    rather than defaulted: the assertion is archive-bound or it is not an
    assertion, and every caller already knows which archive it is promoting
    into (polylogue-8qm4k AC1).
    """
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner, _close_failed_native_construction
    from polylogue.storage.sqlite.write_lease import require_write_lease

    require_write_lease(f"index generation {label} WAL checkpoint({path})", archive_root=archive_root)
    fd = reopened_fd = -1
    try:
        try:
            open_path = path.resolve(strict=True)
            fd = os.open(open_path, os.O_RDWR | os.O_NOFOLLOW)
            before = os.fstat(fd)
            reopened_fd = os.open(open_path, os.O_RDWR | os.O_NOFOLLOW)
            after = os.fstat(reopened_fd)
        except OSError as exc:
            raise RuntimeError(f"cannot securely open {label}: {path}") from exc
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise RuntimeError(f"{label} changed during descriptor validation: {path}")
        consumed_fd, reopened_fd = reopened_fd, -1
        NativeSQLCustodyOwner(None, anchored_descriptors=(consumed_fd,)).close()
        alias = descriptor_alias_path(fd)
        if alias is None:
            raise RuntimeError(f"no validated descriptor alias for {label}: {path}")
        conn = connect_measured(str(alias))
        owned_fd, fd = fd, -1
        owner = NativeSQLCustodyOwner(conn, anchored_descriptors=(owned_fd,))
        try:
            checkpoint = checkpoint_connection(conn, "TRUNCATE", boundary="exclusive")
        except BaseException as primary:
            _close_failed_native_construction(owner, primary)
            raise
        else:
            owner.close()
    finally:
        settlement_primary = sys.exception()
        descriptors = tuple(descriptor for descriptor in (reopened_fd, fd) if descriptor >= 0)
        reopened_fd = fd = -1
        if descriptors:
            cleanup_owner = NativeSQLCustodyOwner(None, anchored_descriptors=descriptors)
            if settlement_primary is not None:
                _close_failed_native_construction(cleanup_owner, settlement_primary)
            else:
                cleanup_owner.close()
    if int(checkpoint[0]) != 0:
        raise RuntimeError(f"{label} WAL checkpoint failed: {checkpoint!r}")


def _generation_creation_recency_ns(generation: IndexGeneration) -> int:
    return generation.created_at_ns or generation.created_at_ms * 1_000_000


def _generation_lifecycle_recency_ns(generation: IndexGeneration) -> int:
    return generation.promoted_at_ns or _generation_creation_recency_ns(generation)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


__all__ = [
    "ActiveWriterLease",
    "IndexGeneration",
    "IndexGenerationStore",
    "UnpublishedPromotionRecoveryError",
    "RebuildLeaseStatus",
    "RebuildLease",
    "RebuildLeaseUnavailableError",
    "rebuild_lease_status",
    "rebuild_source_evidence_snapshot",
    "source_revision_snapshot",
]

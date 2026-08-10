"""Typed identity for one split-tier archive file set."""

from __future__ import annotations

import fcntl
import logging
import os
import socket
import stat
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal

from polylogue.version import VERSION_INFO

logger = logging.getLogger(__name__)

_LOCAL_ARCHIVE_OWNERS_LOCK = threading.RLock()
_LOCAL_ARCHIVE_OWNERS: dict[tuple[int, int], tuple[int, int, int, str]] = {}

ArchiveTierName = Literal["source", "index", "embeddings", "user", "ops", "audit"]
TIER_FILENAMES: tuple[tuple[ArchiveTierName, str], ...] = (
    ("source", "source.db"),
    ("index", "index.db"),
    ("embeddings", "embeddings.db"),
    ("user", "user.db"),
    ("ops", "ops.db"),
    ("audit", "audit.db"),
)


class ArchiveLocationError(RuntimeError):
    """A configured archive does not name one coherent active file set."""


def resolve_active_index_path(archive_root: Path) -> Path:
    """Resolve the effective ``index.db`` path for ``archive_root``.

    Pure function of ``archive_root`` alone (no ambient environment/cwd
    reads): follows ``.index-active-pointer`` when present so ordinary
    config resolution can never silently diverge from a promoted generation
    (polylogue-k8kj finding 1). When the plain ``archive_root/index.db``
    path exists as a stale regular file that diverges from the active
    pointer target -- the signature of an interrupted rebuild that updated
    the pointer without ever swapping the conventional path over to a
    symlink -- that divergence is reported loudly via a warning log line
    rather than served silently; the correct (pointer) data is still what
    gets returned.
    """
    location = ArchiveLocation.resolve(archive_root)
    if location.shadow_index is not None:
        logger.warning(
            "stale conventional index path %s diverges from the active generation %s "
            "(likely an interrupted rebuild promotion that never replaced the conventional "
            "path with a symlink); resolving to the active pointer target instead of the stale file",
            location.shadow_index.resolved_path,
            location.active_index.resolved_path,
        )
    return location.active_index_path


def archive_file_set_root(*, archive_root: Path, db_path: Path) -> Path:
    """Return the file-set root housing the durable tiers alongside ``db_path``.

    ``db_path`` (typically ``Config.db_path``) always names a concrete
    ``index.db`` in the ordinary case -- either an explicit
    ``Config(db_path=...)`` split-root override (polylogue-yla8.1) or the
    resolved active generation (``resolve_active_index_path``, already
    ``.index-active-pointer``-aware) -- so its parent is the correct root.

    Falls back to ``archive_root`` when ``db_path`` was overridden to a path
    that is not named ``index.db`` (e.g. a "never initialized" test
    sentinel that should resolve as if no index existed there at all).

    Callers pass plain ``archive_root``/``db_path`` values (not a
    :class:`~polylogue.config.Config` instance) so this also serves fakes
    that only duck-type those two attributes.
    """
    # A promoted generation may live outside the durable-tier root.  The
    # pointer is the explicit marker that its source/user/blob siblings still
    # belong to the configured archive root, not to the generation directory.
    if (archive_root / ".index-active-pointer").exists():
        return archive_root
    return db_path.parent if db_path.name == "index.db" else archive_root


@dataclass(frozen=True)
class ArchiveLocation:
    """Resolved archive topology with the active index kept distinct from its root.

    A split-tier installation intentionally keeps durable tiers under the
    configured root while an atomic pointer selects an index generation
    elsewhere.  Treating the parent directory of that index as the archive
    root silently invents sibling tier paths; treating ``root/index.db`` as
    authoritative silently bypasses a promoted generation.  This value is the
    one place where those two facts are joined.
    """

    configured_root: Path
    configured_tiers: tuple[TierFileIdentity, ...]
    active_index: TierFileIdentity
    active_pointer: Path | None
    shadow_index: TierFileIdentity | None

    @classmethod
    def resolve(cls, root: Path) -> ArchiveLocation:
        configured_root = root.absolute()
        configured = tuple(
            TierFileIdentity.resolve(name, configured_root / filename) for name, filename in TIER_FILENAMES
        )
        configured_index = next(tier for tier in configured if tier.name == "index")
        pointer_file = configured_root / ".index-active-pointer"
        pointer: Path | None = None
        try:
            pointer_metadata = pointer_file.lstat()
        except FileNotFoundError:
            pointer_metadata = None
        except OSError as exc:
            raise ArchiveLocationError(f"cannot inspect active index pointer: {pointer_file}") from exc
        if pointer_metadata is not None:
            try:
                raw = (
                    os.readlink(pointer_file)
                    if stat.S_ISLNK(pointer_metadata.st_mode)
                    else pointer_file.read_text(encoding="utf-8")
                ).strip()
            except (OSError, ValueError) as exc:
                raise ArchiveLocationError(f"cannot read active index pointer: {pointer_file}") from exc
            candidate = Path(raw)
            if not candidate.is_absolute() or candidate.name != "index.db":
                raise ArchiveLocationError(f"invalid active index pointer: {candidate}")
            pointer = candidate
        active_index = TierFileIdentity.resolve("index", pointer or configured_index.configured_path)
        shadow_index: TierFileIdentity | None = None
        if pointer is not None and configured_index.exists and not configured_index.same_file(active_index):
            shadow_index = configured_index
        return cls(
            configured_root=configured_root,
            configured_tiers=configured,
            active_index=active_index,
            active_pointer=pointer,
            shadow_index=shadow_index,
        )

    def configured_tier(self, name: ArchiveTierName) -> TierFileIdentity:
        return next(tier for tier in self.configured_tiers if tier.name == name)

    def active_tier(self, name: ArchiveTierName) -> TierFileIdentity:
        return self.active_index if name == "index" else self.configured_tier(name)

    @property
    def active_index_path(self) -> Path:
        return self.active_index.configured_path

    @property
    def active_generation(self) -> str:
        return self.active_index.stable_id

    def as_dict(self) -> dict[str, object]:
        return {
            "configured_root": str(self.configured_root),
            "active_pointer": str(self.active_pointer) if self.active_pointer is not None else None,
            "active_index": self.active_index.as_dict(),
            "configured_tiers": [tier.as_dict() for tier in self.configured_tiers],
            "shadow_index": self.shadow_index.as_dict() if self.shadow_index is not None else None,
        }


@dataclass(frozen=True)
class TierFileIdentity:
    name: ArchiveTierName
    configured_path: Path
    resolved_path: Path
    device: int | None
    inode: int | None

    @classmethod
    def resolve(cls, name: ArchiveTierName, path: Path) -> TierFileIdentity:
        resolved = path.resolve(strict=False)
        try:
            stat = path.stat()
        except OSError:
            device = inode = None
        else:
            device, inode = stat.st_dev, stat.st_ino
        return cls(name, path, resolved, device, inode)

    @property
    def exists(self) -> bool:
        return self.device is not None

    @property
    def stable_id(self) -> str:
        if self.device is not None and self.inode is not None:
            return f"dev:{self.device}:ino:{self.inode}"
        return f"path:{self.resolved_path}"

    def same_file(self, other: TierFileIdentity) -> bool:
        if self.exists and other.exists:
            return (self.device, self.inode) == (other.device, other.inode)
        return self.resolved_path == other.resolved_path

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "configured_path": str(self.configured_path),
            "resolved_path": str(self.resolved_path),
            "device": self.device,
            "inode": self.inode,
            "stable_id": self.stable_id,
            "exists": self.exists,
        }


@dataclass(frozen=True)
class ArchiveIdentity:
    configured_root: Path
    tiers: tuple[TierFileIdentity, ...]
    active_generation: str
    generation_owner: str | None = None
    generation_state: Literal["active", "inactive"] = "active"

    @classmethod
    def resolve(
        cls,
        root: Path,
        *,
        generation_owner: str | None = None,
        generation_state: Literal["active", "inactive"] = "active",
    ) -> ArchiveIdentity:
        tiers = tuple(TierFileIdentity.resolve(name, root / filename) for name, filename in TIER_FILENAMES)
        index = next(tier for tier in tiers if tier.name == "index")
        return cls(root, tiers, index.stable_id, generation_owner, generation_state)

    @classmethod
    def resolve_location(
        cls,
        location: ArchiveLocation,
        *,
        generation_owner: str | None = None,
        generation_state: Literal["active", "inactive"] = "active",
    ) -> ArchiveIdentity:
        tiers = tuple(location.active_tier(name) for name, _filename in TIER_FILENAMES)
        index = next(tier for tier in tiers if tier.name == "index")
        return cls(location.configured_root, tiers, index.stable_id, generation_owner, generation_state)

    def tier(self, name: ArchiveTierName) -> TierFileIdentity:
        return next(tier for tier in self.tiers if tier.name == name)

    @property
    def durable_id(self) -> str:
        return "|".join((self.tier("source").stable_id, self.tier("user").stable_id))

    @property
    def authority_identity_digest(self) -> str:
        """Digest the live archive identity without recursively including audit.db."""

        import hashlib
        import json

        payload = {
            "configured_root": str(self.configured_root.absolute()),
            "active_generation": self.active_generation,
            "durable_id": self.durable_id,
            "tiers": {tier.name: tier.stable_id for tier in self.tiers if tier.name != "audit"},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def conflicts_with(self, other: ArchiveIdentity) -> bool:
        shared_durable = self.tier("source").same_file(other.tier("source")) or self.tier("user").same_file(
            other.tier("user")
        )
        distinct_indexes = not self.tier("index").same_file(other.tier("index"))
        # A missing index is not an exemption: writable bootstrap would create
        # it immediately after preflight and produce the second generation we
        # are trying to prevent.
        return shared_durable and distinct_indexes

    def as_dict(self, *, unit: str | None = None) -> dict[str, object]:
        return {
            "configured_root": str(self.configured_root),
            "durable_id": self.durable_id,
            "active_generation": self.active_generation,
            "generation_owner": self.generation_owner,
            "generation_state": self.generation_state,
            "process_id": os.getpid(),
            "executable": os.path.realpath(sys.executable),
            "unit": unit,
            "invocation_id": os.environ.get("INVOCATION_ID"),
            "version": VERSION_INFO.version,
            "build_commit": VERSION_INFO.commit,
            "build_dirty": VERSION_INFO.dirty,
            "tiers": [tier.as_dict() for tier in self.tiers],
        }


class ArchiveIdentityConflictError(RuntimeError):
    """A durable archive is paired with two distinct writable indexes."""


def archive_identity_conflicts(*, configured_root: Path, active_root: Path) -> tuple[ArchiveIdentity, ...]:
    configured_location = ArchiveLocation.resolve(configured_root)
    active_location = ArchiveLocation.resolve(active_root)
    configured = ArchiveIdentity.resolve_location(configured_location)
    active = ArchiveIdentity.resolve_location(active_location)
    if configured.active_generation == active.active_generation:
        return ()
    return (configured,) if configured.conflicts_with(active) else ()


def assert_writable_archive_identity(*, configured_root: Path, active_root: Path) -> ArchiveIdentity:
    """Fail before bootstrap when one durable archive has two active indexes."""
    active = ArchiveIdentity.resolve_location(ArchiveLocation.resolve(active_root))
    conflicts = archive_identity_conflicts(configured_root=configured_root, active_root=active_root)
    if conflicts:
        configured = conflicts[0]
        raise ArchiveIdentityConflictError(
            "archive identity conflict: durable tiers are shared but writable index generations differ; "
            f"configured_index={configured.tier('index').resolved_path} "
            f"active_index={active.tier('index').resolved_path}"
        )
    return active


class ArchiveOwnershipError(RuntimeError):
    """A maintenance/campaign writer could not prove exclusive ownership of an archive location."""


class OwnedArchiveLocation:
    """Exclusive maintenance/campaign ownership proof over one :class:`ArchiveLocation`.

    ``ArchiveLocation``/``ArchiveIdentity`` answer *what* a configured root,
    tier, or resolved generation names.  They deliberately say nothing about
    *who* may mutate it.  A maintenance or campaign writer (an offline index
    rebuild, a devtools benchmark/scale campaign, a b5l generation
    transition) must call :meth:`acquire` and receive a distinct
    ``OwnedArchiveLocation`` token before opening SQLite against the
    location's tiers -- an ``ArchiveLocation`` alone is never sufficient
    authorization to write.

    Acquisition is a pure filesystem preflight: it takes an exclusive
    ``flock`` on a lock file under the configured root and never opens a
    sqlite3 connection.  A location already owned by another live process
    therefore fails closed -- ``ArchiveOwnershipError`` -- before any tier
    file is created or touched, rather than surfacing later as a confusing
    runtime error against an already-open writer connection (or, worse, two
    writers silently racing the same generation).  This mirrors
    ``assert_writable_archive_identity``'s preflight shape one level up: that
    function proves *one coherent generation*; this one proves *one writer*
    for that generation.
    """

    def __init__(self, location: ArchiveLocation, *, root_fd: int, root_identity: tuple[int, int]) -> None:
        self.location = location
        self.lock_path = location.configured_root / ".archive-ownership.lock"
        self.owner_id: str | None = None
        self._fd: int | None = None
        self._root_fd = root_fd
        self.root_identity = root_identity

    @property
    def directory_fd(self) -> int:
        """Return the descriptor that pins the owned archive-root directory."""
        if self._root_fd < 0:
            raise ArchiveOwnershipError("archive ownership root descriptor is closed")
        return self._root_fd

    @classmethod
    def acquire(
        cls,
        location: ArchiveLocation,
        *,
        owner_id: str | None = None,
        allow_reentrant: bool = False,
    ) -> OwnedArchiveLocation:
        """Claim exclusive ownership of ``location`` for a maintenance/campaign writer.

        Raises :class:`ArchiveOwnershipError` immediately -- before any
        SQLite tier file is opened -- when another live process already
        holds the lock.  A lock file left behind by a process that is no
        longer running is reclaimed rather than treated as a permanent
        blocker, matching ``index_generation``'s stale-lease handling.
        """
        owner = owner_id or f"pid={os.getpid()} host={socket.gethostname()} token={uuid.uuid4().hex}"
        root_fd = _open_archive_root_fd(location.configured_root)
        root_metadata = os.fstat(root_fd)
        instance = cls(
            location,
            root_fd=root_fd,
            root_identity=(root_metadata.st_dev, root_metadata.st_ino),
        )
        root_key = instance.root_identity
        with _LOCAL_ARCHIVE_OWNERS_LOCK:
            try:
                existing = _LOCAL_ARCHIVE_OWNERS.get(root_key)
                if existing is None or not allow_reentrant:
                    fd = _acquire_ownership_lock_fd(instance.lock_path, owner=owner, dir_fd=root_fd)
                    _LOCAL_ARCHIVE_OWNERS[root_key] = (fd, root_fd, 1, owner)
                    instance.owner_id = owner
                else:
                    fd, existing_root_fd, references, existing_owner = existing
                    os.close(root_fd)
                    instance._root_fd = existing_root_fd
                    _LOCAL_ARCHIVE_OWNERS[root_key] = (fd, existing_root_fd, references + 1, existing_owner)
                    instance.owner_id = existing_owner
                instance._fd = fd
                _assert_archive_root_identity(
                    instance.location.configured_root,
                    instance.directory_fd,
                    instance.root_identity,
                )
            except BaseException:
                if instance._fd is not None:
                    instance.release()
                elif instance._root_fd >= 0:
                    os.close(instance._root_fd)
                    instance._root_fd = -1
                raise
        return instance

    def release(self) -> None:
        if self._fd is not None:
            with _LOCAL_ARCHIVE_OWNERS_LOCK:
                existing = _LOCAL_ARCHIVE_OWNERS.get(self.root_identity)
                if existing is not None and existing[0] == self._fd:
                    fd, root_fd, references, owner = existing
                    if references <= 1:
                        _LOCAL_ARCHIVE_OWNERS.pop(self.root_identity, None)
                        fcntl.flock(fd, fcntl.LOCK_UN)
                        os.close(fd)
                        os.close(root_fd)
                    else:
                        _LOCAL_ARCHIVE_OWNERS[self.root_identity] = (fd, root_fd, references - 1, owner)
            self._fd = None
            self._root_fd = -1

    def __enter__(self) -> OwnedArchiveLocation:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.release()


def assert_owns_archive_location(owned: OwnedArchiveLocation, location: ArchiveLocation) -> None:
    """Fail before SQLite opens when an ownership proof does not cover ``location``.

    An ``OwnedArchiveLocation`` is a token for one configured root at one
    resolved active generation.  Reusing it against a different root (wrong
    archive entirely) or the same root after its active generation moved on
    (stale proof -- the pointer rotated after acquisition, e.g. a concurrent
    promotion) must fail here rather than let the caller silently write
    against tiers it never actually claimed.
    """
    if owned.location.configured_root != location.configured_root:
        raise ArchiveOwnershipError(
            "archive ownership proof does not cover this location: "
            f"owned={owned.location.configured_root} target={location.configured_root}"
        )
    _assert_archive_root_identity(owned.location.configured_root, owned.directory_fd, owned.root_identity)
    if not owned.location.active_index.same_file(location.active_index):
        raise ArchiveOwnershipError(
            "archive ownership proof is stale for the current active generation: "
            f"owned={owned.location.active_index.stable_id} "
            f"current={location.active_index.stable_id}"
        )


def _lock_holder_pid(path: Path, *, fd: int | None = None) -> int | None:
    """Best-effort recorded pid from an existing lock file; ``None`` if absent/unreadable."""
    try:
        text = path.read_text(encoding="utf-8") if fd is None else os.pread(fd, 4096, 0).decode("utf-8")
    except (OSError, ValueError):
        return None
    for token in text.split():
        if token.startswith("pid="):
            try:
                return int(token[len("pid=") :])
            except ValueError:
                return None
    return None


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


_STALE_RECLAIM_ATTEMPTS = 20
_STALE_RECLAIM_RETRY_SECONDS = 0.01


def _open_archive_root_fd(path: Path) -> int:
    """Open one archive-root directory and reject a path swap around the open."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise ArchiveOwnershipError(f"cannot open archive root directory: {path}") from exc
    try:
        _assert_archive_root_identity(path, fd, None)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _assert_archive_root_identity(path: Path, fd: int, expected: tuple[int, int] | None) -> None:
    """Prove the configured root still names the directory held by ``fd``."""
    metadata = os.fstat(fd)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ArchiveOwnershipError(f"archive root is not a directory: {path}")
    identity = (metadata.st_dev, metadata.st_ino)
    if expected is not None and identity != expected:
        raise ArchiveOwnershipError(f"archive root descriptor changed: {path}")
    try:
        path_metadata = path.stat()
    except OSError as exc:
        raise ArchiveOwnershipError(f"archive root disappeared during ownership validation: {path}") from exc
    if (path_metadata.st_dev, path_metadata.st_ino) != identity:
        raise ArchiveOwnershipError(f"archive root changed during ownership validation: {path}")


def _acquire_ownership_lock_fd(path: Path, *, owner: str, dir_fd: int | None = None) -> int:
    """Open ``path`` and take an exclusive, non-blocking ``flock`` proving ownership.

    Never opens a sqlite3 connection -- callers rely on this to fail before
    any SQLite tier file is touched.

    A lock file whose recorded holder pid is no longer running is reclaimed
    by retrying the *same* ``flock`` call against the *same* inode, never by
    creating a second one. An earlier version of this function raced two
    reclaimers by opening a fresh inode and ``os.replace``-ing it over
    ``path``: if the true holder had just died, one reclaimer could win the
    direct ``flock`` on the original (now-unlocked) inode while a second
    reclaimer, having already observed ``BlockingIOError``, swapped in a
    brand-new inode via rename -- leaving two processes each holding a lock
    on a *different* inode and both believing they own the location. A dead
    holder's ``flock`` is released by the kernel the moment its last fd
    closes (process exit tears down the fd table), which happens at or
    before the point another process can read pid= from the file and
    confirm it's dead -- so retrying the identical ``flock`` call converges
    quickly without ever creating a second inode to race against.
    """
    lock_flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    if dir_fd is None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, lock_flags, 0o600)
        except OSError as exc:
            raise ArchiveOwnershipError(f"cannot open archive ownership lock: {path}") from exc
    else:
        try:
            fd = os.open(
                path.name,
                lock_flags,
                0o600,
                dir_fd=dir_fd,
            )
        except OSError as exc:
            raise ArchiveOwnershipError(f"cannot open archive ownership lock: {path}") from exc
    try:
        _validate_ownership_lock_fd(path, fd, dir_fd=dir_fd)
    except BaseException:
        os.close(fd)
        raise
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        holder_pid = _lock_holder_pid(path, fd=fd)
        if holder_pid is None or _pid_is_alive(holder_pid):
            os.close(fd)
            suffix = f" (pid={holder_pid})" if holder_pid is not None else ""
            raise ArchiveOwnershipError(f"archive location already owned: {path}{suffix}") from exc
        logger.warning(
            "reclaiming stale archive ownership lock %s: recorded holder pid=%d is no longer running",
            path,
            holder_pid,
        )
        for _ in range(_STALE_RECLAIM_ATTEMPTS):
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(_STALE_RECLAIM_RETRY_SECONDS)
        else:
            os.close(fd)
            raise ArchiveOwnershipError(f"archive location already owned: {path}") from exc
    try:
        _validate_ownership_lock_fd(path, fd, dir_fd=dir_fd)
    except BaseException:
        os.close(fd)
        raise
    try:
        os.ftruncate(fd, 0)
        os.write(fd, owner.encode("utf-8"))
        os.fsync(fd)
    except OSError as exc:
        os.close(fd)
        raise ArchiveOwnershipError(f"cannot record archive ownership lock owner: {path}") from exc
    return fd


def _validate_ownership_lock_fd(path: Path, fd: int, *, dir_fd: int | None) -> None:
    """Require the canonical lock entry to name the locked private inode."""
    try:
        metadata = os.fstat(fd)
    except OSError as exc:
        raise ArchiveOwnershipError(f"cannot inspect archive ownership lock: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ArchiveOwnershipError(f"archive ownership lock is not a regular file: {path}")
    if metadata.st_nlink != 1:
        raise ArchiveOwnershipError(f"archive ownership lock has unexpected link count: {path}")
    try:
        if dir_fd is None:
            path_metadata = os.stat(path, follow_symlinks=False)
        else:
            path_metadata = os.stat(path.name, dir_fd=dir_fd, follow_symlinks=False)
    except OSError as exc:
        raise ArchiveOwnershipError(f"cannot inspect archive ownership lock pathname: {path}") from exc
    if (path_metadata.st_dev, path_metadata.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise ArchiveOwnershipError(f"archive ownership lock pathname changed during validation: {path}")

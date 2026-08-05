"""Offline, dry-run-first quarantine for invalid blob namespace entries.

This actuator repairs only the *namespace* boundary owned by
``BlobStore.iter_namespace``.  It never decides whether a canonical blob is
referenced, obsolete, or safe for garbage collection.  Canonical paths stay
in place, and every non-canonical entry is moved atomically into a sibling
quarantine tree only after a complete hash census proves the canonical
inventory is sound.

The before receipt is deliberately immutable and written before the first
move.  If the process stops after that point, recovery is classification-only:
it reports rolled-back, committed, or indeterminate state without attempting
another move.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import time
import uuid
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from polylogue.config import Config
from polylogue.maintenance.offline_guard import running_daemon_pid
from polylogue.paths import render_root
from polylogue.storage.blob_store import BlobNamespaceEntry, BlobNamespaceIssue, BlobStore
from polylogue.storage.index_generation import RebuildLease
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import validate_migration_backup_manifest

TOOL_VERSION = "blob-namespace-quarantine-v1"
_CHUNK_SIZE = 1024 * 1024
_QUARANTINE_DIRNAME = "blob-namespace-quarantine"


class BlobNamespaceMoveCapability(StrEnum):
    """Filesystem movement capability granted by a namespace cleanup plan."""

    NONE = "none"


class BlobNamespaceQuarantineError(RuntimeError):
    """Raised when the namespace quarantine cannot prove a safe apply."""


@dataclass(frozen=True, slots=True)
class BlobNamespaceQuarantineEntry:
    """A no-follow snapshot of one invalid namespace entry."""

    relative_path: str
    kind: str
    issue: str
    file_type: str
    device: int
    inode: int
    size_bytes: int
    content_sha256: str | None
    tree_sha256: str
    destination: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BlobNamespaceCanonicalEntry:
    """A fully re-hashed canonical blob observed during a complete census."""

    relative_path: str
    hash_hex: str
    size_bytes: int
    device: int
    inode: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BlobNamespaceCensus:
    """Complete no-follow namespace census used for preflight and postflight."""

    canonical: tuple[BlobNamespaceCanonicalEntry, ...]
    candidates: tuple[BlobNamespaceQuarantineEntry, ...]
    blockers: tuple[str, ...]

    @property
    def canonical_inventory_digest(self) -> str:
        return _json_digest([entry.to_dict() for entry in self.canonical])

    @property
    def candidate_inventory_digest(self) -> str:
        return _json_digest([entry.to_dict() for entry in self.candidates])

    @property
    def safe_to_apply(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_inventory_count": len(self.canonical),
            "canonical_inventory_digest": self.canonical_inventory_digest,
            "invalid_namespace_entries": len(self.candidates),
            "candidate_inventory_digest": self.candidate_inventory_digest,
            "blockers": list(self.blockers),
            "truncated": False,
            "canonical": [entry.to_dict() for entry in self.canonical],
            "candidates": [entry.to_dict() for entry in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class BlobNamespaceCleanupPlan:
    """Backup-attested, read-only plan for invalid namespace entries.

    The plan is deliberately separate from the quarantine report. Producing
    it never creates a receipt, moves a filesystem entry, checkpoints SQLite,
    or changes an archive row. The later quarantine actuator may consume the
    same census only after its own offline and lock-held revalidation.
    """

    archive_root: Path
    blob_root: Path
    backup_manifest: Path
    backup_verification_receipt: Path
    census: BlobNamespaceCensus
    move_capability: BlobNamespaceMoveCapability = BlobNamespaceMoveCapability.NONE
    deletes_files: bool = False
    moves_files: bool = False
    mutates_sqlite: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "blob_root": str(self.blob_root),
            "backup_manifest": str(self.backup_manifest),
            "backup_verification_receipt": str(self.backup_verification_receipt),
            "move_capability": self.move_capability.value,
            "deletes_files": self.deletes_files,
            "moves_files": self.moves_files,
            "mutates_sqlite": self.mutates_sqlite,
            **self.census.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BlobNamespaceQuarantineReport:
    """Dry-run or applied namespace-quarantine outcome."""

    archive_root: Path
    blob_root: Path
    dry_run: bool
    census: BlobNamespaceCensus
    applied: bool
    moved_count: int = 0
    receipt_dir: Path | None = None
    quarantine_root: Path | None = None
    backup_manifest: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "blob_root": str(self.blob_root),
            "dry_run": self.dry_run,
            "applied": self.applied,
            "moved_count": self.moved_count,
            "receipt_dir": str(self.receipt_dir) if self.receipt_dir is not None else None,
            "quarantine_root": str(self.quarantine_root) if self.quarantine_root is not None else None,
            "backup_manifest": str(self.backup_manifest) if self.backup_manifest is not None else None,
            **self.census.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BlobNamespaceRecoveryReport:
    """Read-only classification of an interrupted quarantine operation."""

    receipt_dir: Path
    outcome: str
    source_present: int
    destination_present: int
    matching_destinations: int
    conflicts: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "receipt_dir": str(self.receipt_dir),
            "outcome": self.outcome,
            "source_present": self.source_present,
            "destination_present": self.destination_present,
            "matching_destinations": self.matching_destinations,
            "conflicts": list(self.conflicts),
        }


def _json_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _absolute(path: Path) -> Path:
    return path.absolute()


def _relative_path(root: Path, path: Path) -> str:
    try:
        relative = _absolute(path).relative_to(_absolute(root))
    except ValueError as exc:
        raise BlobNamespaceQuarantineError(f"path escapes blob root: {path}") from exc
    if relative == Path(".") or relative.is_absolute() or ".." in relative.parts:
        raise BlobNamespaceQuarantineError(f"path is not a contained namespace entry: {path}")
    return relative.as_posix()


def _file_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "regular"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "character-device"
    if stat.S_ISBLK(mode):
        return "block-device"
    return "unknown"


def _hash_regular_file(path: Path, expected: os.stat_result | None = None) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise BlobNamespaceQuarantineError(f"not a regular file: {path}")
        if expected is not None and (observed.st_dev, observed.st_ino, observed.st_size) != (
            expected.st_dev,
            expected.st_ino,
            expected.st_size,
        ):
            raise BlobNamespaceQuarantineError(f"file changed while hashing: {path}")
        hasher = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, _CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
        return hasher.hexdigest(), observed.st_size
    finally:
        os.close(descriptor)


def _tree_digest(path: Path) -> tuple[str | None, str]:
    """Hash an opaque candidate tree without ever resolving a symlink."""

    def _node(node: Path) -> tuple[dict[str, object], str | None]:
        details = os.lstat(node)
        kind = _file_type(details.st_mode)
        payload: dict[str, object] = {
            "name": node.name,
            "file_type": kind,
            "mode": stat.S_IMODE(details.st_mode),
            "size_bytes": details.st_size,
        }
        content_hash: str | None = None
        if stat.S_ISREG(details.st_mode):
            content_hash, _ = _hash_regular_file(node, details)
            payload["content_sha256"] = content_hash
        elif stat.S_ISLNK(details.st_mode):
            link_value = os.readlink(node)
            content_hash = hashlib.sha256(os.fsencode(link_value)).hexdigest()
            payload["link_sha256"] = content_hash
        elif stat.S_ISDIR(details.st_mode):
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(node, flags)
            try:
                children = []
                for name in sorted(os.listdir(descriptor)):
                    child_payload, _ = _node(node / name)
                    children.append(child_payload)
            finally:
                os.close(descriptor)
            payload["children"] = children
        return payload, content_hash

    payload, content_hash = _node(path)
    return content_hash, _json_digest(payload)


def _fingerprint_regular_file(path: Path) -> dict[str, object]:
    details = os.lstat(path)
    if not stat.S_ISREG(details.st_mode):
        raise BlobNamespaceQuarantineError(f"required regular file is not regular: {path}")
    sha256, size = _hash_regular_file(path, details)
    return {
        "path": str(_absolute(path)),
        "device": details.st_dev,
        "inode": details.st_ino,
        "size_bytes": size,
        "sha256": sha256,
    }


def _require_real_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        details = os.lstat(path)
    except OSError as exc:
        raise BlobNamespaceQuarantineError(f"cannot stat {label}: {path}: {exc}") from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise BlobNamespaceQuarantineError(f"{label} must be a non-symlink directory: {path}")
    return details


def _ensure_new_directory(path: Path, *, parent: Path, label: str) -> None:
    _require_real_directory(parent, label=f"{label} parent")
    if _path_exists(path):
        raise BlobNamespaceQuarantineError(f"{label} already exists: {path}")
    try:
        os.mkdir(path, 0o700)
    except OSError as exc:
        raise BlobNamespaceQuarantineError(f"could not create {label}: {path}: {exc}") from exc
    _require_real_directory(path, label=label)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_immutable_json(path: Path, payload: dict[str, object]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise BlobNamespaceQuarantineError(f"immutable receipt already exists: {path}") from exc
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written == 0:
                raise BlobNamespaceQuarantineError(f"could not write immutable receipt: {path}")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_source_db(conn: sqlite3.Connection) -> None:
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except sqlite3.Error as exc:
        raise BlobNamespaceQuarantineError("could not checkpoint source.db before quarantine") from exc
    if row is None or len(row) != 3:
        raise BlobNamespaceQuarantineError("source.db WAL checkpoint returned no complete status")
    busy, log_frames, checkpointed_frames = (int(value) for value in row)
    if busy != 0 or log_frames != checkpointed_frames:
        raise BlobNamespaceQuarantineError(
            "source.db WAL checkpoint was not clean; stop every writer and retry when no frames remain busy"
        )


def _offline_config(archive_root: Path) -> Config:
    return Config(archive_root=archive_root, render_root=render_root(), sources=[])


def _require_offline(archive_root: Path) -> None:
    from polylogue.daemon.write_coordinator import daemon_write_lease_active

    if daemon_write_lease_active():
        raise BlobNamespaceQuarantineError("refusing blob namespace quarantine while a daemon writer lease is active")
    pid = running_daemon_pid(_offline_config(archive_root))
    if pid is not None:
        raise BlobNamespaceQuarantineError(
            f"refusing blob namespace quarantine while polylogued PID {pid} is running; stop the daemon first"
        )


def _destination_for(quarantine_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise BlobNamespaceQuarantineError(f"invalid candidate relative path: {relative_path!r}")
    destination = quarantine_root / relative
    _relative_path(quarantine_root, destination)
    return destination


def _candidate_from_entry(
    entry: BlobNamespaceEntry, *, blob_root: Path, quarantine_root: Path
) -> BlobNamespaceQuarantineEntry:
    relative_path = _relative_path(blob_root, entry.path)
    if relative_path != entry.relative_path:
        raise BlobNamespaceQuarantineError(
            f"namespace entry relative path disagrees with containment check: {entry.relative_path!r} != {relative_path!r}"
        )
    details = os.lstat(entry.path)
    content_sha256, tree_sha256 = _tree_digest(entry.path)
    destination = _destination_for(quarantine_root, relative_path)
    return BlobNamespaceQuarantineEntry(
        relative_path=relative_path,
        kind=entry.kind.value,
        issue=entry.issue.value if entry.issue is not None else "unknown",
        file_type=_file_type(details.st_mode),
        device=details.st_dev,
        inode=details.st_ino,
        size_bytes=details.st_size,
        content_sha256=content_sha256,
        tree_sha256=tree_sha256,
        destination=str(_absolute(destination)),
    )


def _census(blob_root: Path, *, quarantine_root: Path) -> BlobNamespaceCensus:
    _require_real_directory(blob_root, label="blob root")
    canonical: list[BlobNamespaceCanonicalEntry] = []
    candidates: list[BlobNamespaceQuarantineEntry] = []
    blockers: list[str] = []
    try:
        entries = tuple(BlobStore(blob_root).iter_namespace())
    except OSError as exc:
        return BlobNamespaceCensus(
            canonical=(), candidates=(), blockers=(f"could not enumerate blob namespace: {exc}",)
        )

    for entry in entries:
        if entry.kind.value == "blob":
            try:
                relative_path = _relative_path(blob_root, entry.path)
                details = os.lstat(entry.path)
                if not stat.S_ISREG(details.st_mode):
                    raise BlobNamespaceQuarantineError("canonical path is no longer a regular file")
                observed_hash, size_bytes = _hash_regular_file(entry.path, details)
                expected_hash = entry.hash_hex or ""
                if observed_hash != expected_hash:
                    blockers.append(
                        f"canonical-shaped hash mismatch at {relative_path}: expected {expected_hash}, got {observed_hash}"
                    )
                    continue
                canonical.append(
                    BlobNamespaceCanonicalEntry(
                        relative_path=relative_path,
                        hash_hex=expected_hash,
                        size_bytes=size_bytes,
                        device=details.st_dev,
                        inode=details.st_ino,
                    )
                )
            except (BlobNamespaceQuarantineError, OSError) as exc:
                blockers.append(f"canonical blob stat/read failure at {entry.relative_path}: {exc}")
            continue

        if entry.issue is BlobNamespaceIssue.STAT_FAILED:
            blockers.append(f"namespace stat failure at {entry.relative_path}")
            continue
        try:
            candidates.append(_candidate_from_entry(entry, blob_root=blob_root, quarantine_root=quarantine_root))
        except (BlobNamespaceQuarantineError, OSError) as exc:
            blockers.append(f"invalid namespace stat/read failure at {entry.relative_path}: {exc}")

    canonical.sort(key=lambda item: item.relative_path)
    candidates.sort(key=lambda item: item.relative_path)
    return BlobNamespaceCensus(canonical=tuple(canonical), candidates=tuple(candidates), blockers=tuple(blockers))


def plan_blob_namespace_cleanup(
    archive_root: Path,
    *,
    backup_manifest: Path | None,
) -> BlobNamespaceCleanupPlan:
    """Build a backup-gated, non-mutating plan for invalid blob entries.

    This command is intentionally usable before any cleanup decision. The
    backup manifest is authenticated against an immutable source-tier read,
    and the namespace census records existing sidecars, ``.blob.*`` temps,
    malformed shards, and other invalid entries without deleting or moving
    them.
    """
    if backup_manifest is None:
        raise BlobNamespaceQuarantineError(
            "planning blob namespace cleanup requires a verified source backup manifest (--backup-manifest)"
        )

    archive_root = _absolute(archive_root)
    backup_manifest = _absolute(backup_manifest)
    _require_offline(archive_root)
    source_db = archive_root / "source.db"
    try:
        source_uri = source_db.resolve().as_uri() + "?mode=ro&immutable=1"
        with sqlite3.connect(source_uri, uri=True) as source_conn:
            verification_receipt = validate_migration_backup_manifest(
                backup_manifest,
                ArchiveTier.SOURCE,
                connection=source_conn,
            )
    except Exception as exc:
        raise BlobNamespaceQuarantineError(f"backup manifest validation failed: {exc}") from exc

    census = _census(
        archive_root / "blob",
        quarantine_root=archive_root / _QUARANTINE_DIRNAME / "plan",
    )
    return BlobNamespaceCleanupPlan(
        archive_root=archive_root,
        blob_root=archive_root / "blob",
        backup_manifest=backup_manifest,
        backup_verification_receipt=verification_receipt,
        census=census,
    )


def _same_census(left: BlobNamespaceCensus, right: BlobNamespaceCensus) -> bool:
    def _candidate_identity(entry: BlobNamespaceQuarantineEntry) -> dict[str, object]:
        payload = entry.to_dict()
        # The operation root is minted only after the lock-held preflight, so
        # destination paths intentionally differ between these two otherwise
        # identical source censuses.
        payload.pop("destination")
        return payload

    return (
        [entry.to_dict() for entry in left.canonical] == [entry.to_dict() for entry in right.canonical]
        and [_candidate_identity(entry) for entry in left.candidates]
        == [_candidate_identity(entry) for entry in right.candidates]
        and left.blockers == right.blockers
    )


def _prepare_quarantine_root(archive_root: Path, *, operation_id: str, blob_device: int) -> Path:
    archive_details = _require_real_directory(archive_root, label="archive root")
    base = archive_root / _QUARANTINE_DIRNAME
    if not _path_exists(base):
        _ensure_new_directory(base, parent=archive_root, label="quarantine base directory")
        _fsync_directory(archive_root)
    _require_real_directory(base, label="quarantine base directory")
    root = base / operation_id
    _ensure_new_directory(root, parent=base, label="quarantine operation directory")
    details = _require_real_directory(root, label="quarantine operation directory")
    if details.st_dev != blob_device or archive_details.st_dev != blob_device:
        raise BlobNamespaceQuarantineError("quarantine destination is not on the blob root filesystem")
    _fsync_directory(base)
    return root


def _ensure_destination_parent(quarantine_root: Path, destination: Path) -> None:
    relative_parent = destination.parent.relative_to(quarantine_root)
    current = quarantine_root
    for component in relative_parent.parts:
        current = current / component
        if not _path_exists(current):
            _ensure_new_directory(current, parent=current.parent, label="quarantine destination directory")
            _fsync_directory(current.parent)
        _require_real_directory(current, label="quarantine destination directory")


def _candidate_still_matches(candidate: BlobNamespaceQuarantineEntry, source: Path) -> None:
    details = os.lstat(source)
    if (details.st_dev, details.st_ino, details.st_size, _file_type(details.st_mode)) != (
        candidate.device,
        candidate.inode,
        candidate.size_bytes,
        candidate.file_type,
    ):
        raise BlobNamespaceQuarantineError(f"candidate changed after preflight: {candidate.relative_path}")
    content_sha256, tree_sha256 = _tree_digest(source)
    if content_sha256 != candidate.content_sha256 or tree_sha256 != candidate.tree_sha256:
        raise BlobNamespaceQuarantineError(f"candidate content changed after preflight: {candidate.relative_path}")


def _before_receipt(
    *,
    archive_root: Path,
    blob_root: Path,
    source_fingerprint: dict[str, object],
    backup_manifest: Path,
    verification_receipt: Path,
    census: BlobNamespaceCensus,
    quarantine_root: Path,
    operation_id: str,
) -> dict[str, object]:
    blob_details = os.lstat(blob_root)
    return {
        "kind": "blob_namespace_quarantine",
        "phase": "before",
        "tool_version": TOOL_VERSION,
        "operation_id": operation_id,
        "created_at_ms": int(time.time() * 1000),
        "archive_root": str(_absolute(archive_root)),
        "blob_root": str(_absolute(blob_root)),
        "blob_root_device": blob_details.st_dev,
        "blob_root_inode": blob_details.st_ino,
        "quarantine_root": str(_absolute(quarantine_root)),
        "source_db": source_fingerprint,
        "backup_manifest": _fingerprint_regular_file(backup_manifest),
        "backup_verification_receipt": _fingerprint_regular_file(verification_receipt),
        **census.to_dict(),
    }


def _after_receipt(
    *,
    before_digest: str,
    source_before: dict[str, object],
    source_after: dict[str, object],
    preflight: BlobNamespaceCensus,
    postflight: BlobNamespaceCensus,
    moved: tuple[BlobNamespaceQuarantineEntry, ...],
    destination_inventory_digest: str,
    full_verify: dict[str, object],
) -> dict[str, object]:
    return {
        "kind": "blob_namespace_quarantine",
        "phase": "after",
        "tool_version": TOOL_VERSION,
        "created_at_ms": int(time.time() * 1000),
        "before_receipt_sha256": before_digest,
        "source_db_before": source_before,
        "source_db_after": source_after,
        "moved": [entry.to_dict() for entry in moved],
        "moved_count": len(moved),
        "quarantined_candidate_inventory_digest": destination_inventory_digest,
        "skipped_count": 0,
        "conflict_count": 0,
        "preflight_canonical_inventory_digest": preflight.canonical_inventory_digest,
        "postflight": postflight.to_dict(),
        "full_blob_verification": full_verify,
    }


def _verify_quarantine_destinations(
    moved: tuple[BlobNamespaceQuarantineEntry, ...],
) -> str:
    """Prove every moved opaque entry still matches its before receipt."""
    for candidate in moved:
        destination = Path(candidate.destination)
        if not _path_exists(destination):
            raise BlobNamespaceQuarantineError(f"quarantine destination disappeared: {destination}")
        _candidate_still_matches(candidate, destination)
    return _json_digest([entry.to_dict() for entry in moved])


def _verify_postflight(
    blob_root: Path, preflight: BlobNamespaceCensus
) -> tuple[BlobNamespaceCensus, dict[str, object]]:
    postflight = _census(blob_root, quarantine_root=blob_root.parent / _QUARANTINE_DIRNAME / "postflight")
    if postflight.blockers or postflight.candidates:
        raise BlobNamespaceQuarantineError("postflight namespace is not pristine")
    if postflight.canonical_inventory_digest != preflight.canonical_inventory_digest:
        raise BlobNamespaceQuarantineError("canonical inventory changed during namespace quarantine")
    verification = BlobStore(blob_root).verify_all(max_failures=1)
    verification_payload: dict[str, object] = {
        "canonical_checked": verification.checked,
        "checked_bytes": verification.checked_bytes,
        "hash_failures": len(verification.failures),
        "invalid_namespace_entries": 0,
        "truncated": verification.truncated,
    }
    if verification.failures or verification.truncated or verification.checked != len(preflight.canonical):
        raise BlobNamespaceQuarantineError("full canonical hash verification failed after namespace quarantine")
    return postflight, verification_payload


def quarantine_blob_namespace(
    archive_root: Path,
    *,
    backup_manifest: Path | None = None,
    receipt_dir: Path | None = None,
    dry_run: bool = True,
) -> BlobNamespaceQuarantineReport:
    """Census invalid blob namespace entries, or move the exact safe set.

    Apply is intentionally narrow: it requires an offline archive, the
    archive-wide exclusive lease, a verified attested source backup, a clean
    WAL checkpoint, a new explicit receipt directory, and two identical full
    censuses before any ``os.replace`` call.
    """

    archive_root = _absolute(archive_root)
    blob_root = archive_root / "blob"
    placeholder_quarantine = archive_root / _QUARANTINE_DIRNAME / "dry-run"
    if dry_run:
        census = _census(blob_root, quarantine_root=placeholder_quarantine)
        return BlobNamespaceQuarantineReport(
            archive_root=archive_root,
            blob_root=blob_root,
            dry_run=True,
            census=census,
            applied=False,
        )

    if backup_manifest is None:
        raise BlobNamespaceQuarantineError(
            "applying blob namespace quarantine requires a verified source backup manifest (--backup-manifest)"
        )
    if receipt_dir is None:
        raise BlobNamespaceQuarantineError(
            "applying blob namespace quarantine requires a new explicit receipt directory (--receipt-dir)"
        )
    receipt_dir = _absolute(receipt_dir)
    if _path_exists(receipt_dir):
        raise BlobNamespaceQuarantineError(f"receipt directory already exists: {receipt_dir}")
    _require_real_directory(receipt_dir.parent, label="receipt directory parent")
    if _absolute(receipt_dir).is_relative_to(_absolute(blob_root)):
        raise BlobNamespaceQuarantineError("receipt directory must be outside the live blob namespace")
    _require_offline(archive_root)

    with RebuildLease(archive_root):
        # The second check is authoritative.  The lease excludes archive
        # writers; this catches a daemon that started between the first check
        # and acquisition, before we checkpoint or touch the filesystem.
        _require_offline(archive_root)
        blob_details = _require_real_directory(blob_root, label="blob root")
        source_db = archive_root / "source.db"
        source_before: dict[str, object]
        verification_receipt: Path
        with sqlite3.connect(f"file:{source_db}?mode=rw", uri=True) as source_conn:
            _checkpoint_source_db(source_conn)
            verification_receipt = validate_migration_backup_manifest(
                backup_manifest, ArchiveTier.SOURCE, connection=source_conn
            )
            source_before = _fingerprint_regular_file(source_db)
            preflight = _census(
                blob_root,
                quarantine_root=archive_root / _QUARANTINE_DIRNAME / "pending",
            )
        if not preflight.safe_to_apply:
            raise BlobNamespaceQuarantineError("refusing namespace quarantine: " + "; ".join(preflight.blockers))

        _ensure_new_directory(receipt_dir, parent=receipt_dir.parent, label="receipt directory")
        _fsync_directory(receipt_dir.parent)
        operation_id = uuid.uuid4().hex
        quarantine_root = _prepare_quarantine_root(
            archive_root, operation_id=operation_id, blob_device=blob_details.st_dev
        )
        # Destinations are part of the immutable plan.  Re-census with the
        # real operation root, then require every source observation to match
        # the lock-held preflight before the receipt is written.
        apply_census = _census(blob_root, quarantine_root=quarantine_root)
        if not _same_census(preflight, apply_census):
            raise BlobNamespaceQuarantineError("blob namespace changed between preflight and apply rescan")
        before = _before_receipt(
            archive_root=archive_root,
            blob_root=blob_root,
            source_fingerprint=source_before,
            backup_manifest=backup_manifest,
            verification_receipt=verification_receipt,
            census=apply_census,
            quarantine_root=quarantine_root,
            operation_id=operation_id,
        )
        before_digest = _write_immutable_json(receipt_dir / "before.json", before)

        moved: list[BlobNamespaceQuarantineEntry] = []
        for candidate in apply_census.candidates:
            source = blob_root / candidate.relative_path
            destination = Path(candidate.destination)
            _relative_path(blob_root, source)
            _relative_path(quarantine_root, destination)
            _candidate_still_matches(candidate, source)
            _ensure_destination_parent(quarantine_root, destination)
            if _path_exists(destination):
                raise BlobNamespaceQuarantineError(f"quarantine destination already exists: {destination}")
            try:
                os.replace(source, destination)
            except OSError as exc:
                raise BlobNamespaceQuarantineError(
                    f"could not atomically move {candidate.relative_path} into quarantine: {exc}"
                ) from exc
            _fsync_directory(source.parent)
            _fsync_directory(destination.parent)
            moved.append(candidate)

        postflight, full_verify = _verify_postflight(blob_root, apply_census)
        destination_inventory_digest = _verify_quarantine_destinations(tuple(moved))
        if destination_inventory_digest != apply_census.candidate_inventory_digest:
            raise BlobNamespaceQuarantineError("quarantined entry inventory differs from the before receipt")
        source_after = _fingerprint_regular_file(source_db)
        if source_after != source_before:
            raise BlobNamespaceQuarantineError("source.db changed during filesystem-only namespace quarantine")
        with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as source_conn:
            quick_check = source_conn.execute("PRAGMA quick_check").fetchone()
        if quick_check is None or str(quick_check[0]).lower() != "ok":
            raise BlobNamespaceQuarantineError(f"source.db quick_check failed after quarantine: {quick_check!r}")
        _write_immutable_json(
            receipt_dir / "after.json",
            _after_receipt(
                before_digest=before_digest,
                source_before=source_before,
                source_after=source_after,
                preflight=apply_census,
                postflight=postflight,
                moved=tuple(moved),
                destination_inventory_digest=destination_inventory_digest,
                full_verify=full_verify,
            ),
        )
        return BlobNamespaceQuarantineReport(
            archive_root=archive_root,
            blob_root=blob_root,
            dry_run=False,
            census=apply_census,
            applied=True,
            moved_count=len(moved),
            receipt_dir=receipt_dir,
            quarantine_root=quarantine_root,
            backup_manifest=backup_manifest,
        )


def classify_blob_namespace_quarantine_recovery(receipt_dir: Path) -> BlobNamespaceRecoveryReport:
    """Classify crash state from immutable receipts without mutating anything."""

    receipt_dir = _absolute(receipt_dir)
    before_path = receipt_dir / "before.json"
    try:
        before = json.loads(before_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BlobNamespaceQuarantineError(
            f"could not read namespace quarantine before receipt: {before_path}"
        ) from exc
    if before.get("kind") != "blob_namespace_quarantine" or before.get("phase") != "before":
        raise BlobNamespaceQuarantineError(f"not a blob namespace quarantine before receipt: {before_path}")
    blob_root = Path(str(before.get("blob_root", "")))
    raw_candidates = before.get("candidates")
    if not isinstance(raw_candidates, list):
        raise BlobNamespaceQuarantineError(f"invalid candidate list in before receipt: {before_path}")

    source_present = 0
    destination_present = 0
    matching_destinations = 0
    conflicts: list[str] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            conflicts.append("malformed candidate receipt row")
            continue
        try:
            candidate = BlobNamespaceQuarantineEntry(**raw)
        except TypeError:
            conflicts.append("malformed candidate receipt fields")
            continue
        source = blob_root / candidate.relative_path
        destination = Path(candidate.destination)
        source_exists = _path_exists(source)
        destination_exists = _path_exists(destination)
        if source_exists:
            source_present += 1
        if destination_exists:
            destination_present += 1
        if source_exists and destination_exists:
            conflicts.append(f"both source and destination exist: {candidate.relative_path}")
            continue
        if destination_exists:
            try:
                _candidate_still_matches(candidate, destination)
            except (BlobNamespaceQuarantineError, OSError) as exc:
                conflicts.append(f"destination mismatch: {candidate.relative_path}: {exc}")
            else:
                matching_destinations += 1
        elif source_exists:
            try:
                _candidate_still_matches(candidate, source)
            except (BlobNamespaceQuarantineError, OSError) as exc:
                conflicts.append(f"source mismatch: {candidate.relative_path}: {exc}")
        else:
            conflicts.append(f"source and destination are both absent: {candidate.relative_path}")

    total = len(raw_candidates)
    if conflicts:
        outcome = "indeterminate"
    elif source_present == total and destination_present == 0:
        outcome = "rolled_back"
    elif source_present == 0 and destination_present == total and matching_destinations == total:
        outcome = "committed"
    else:
        outcome = "indeterminate"
    return BlobNamespaceRecoveryReport(
        receipt_dir=receipt_dir,
        outcome=outcome,
        source_present=source_present,
        destination_present=destination_present,
        matching_destinations=matching_destinations,
        conflicts=tuple(conflicts),
    )


__all__ = [
    "TOOL_VERSION",
    "BlobNamespaceCanonicalEntry",
    "BlobNamespaceCensus",
    "BlobNamespaceCleanupPlan",
    "BlobNamespaceMoveCapability",
    "BlobNamespaceQuarantineEntry",
    "BlobNamespaceQuarantineError",
    "BlobNamespaceQuarantineReport",
    "BlobNamespaceRecoveryReport",
    "classify_blob_namespace_quarantine_recovery",
    "plan_blob_namespace_cleanup",
    "quarantine_blob_namespace",
]

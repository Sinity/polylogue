"""Lossless cut boundaries for mutable source inputs.

The cut is a filesystem operation, not a campaign state machine.  Preflight
binds the configured roots and source-owned strategy names.  Execution copies
the candidate cohort into a private immutable tree, inventories the live roots
again, and publishes two digests: candidate bytes and the material that stayed
in the ordinary source roots after the cut.

The source roots are never moved, renamed, or acknowledged by this module.
Consequently, a later daemon catch-up sees carry-forward files through its
normal acquisition route.  A candidate can only be read from the published
private tree, and every read rechecks its content digest.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from polylogue.maintenance.source_manifest_continuity import SourceDeclaration, SourceRole
from polylogue.sources.sqlite_snapshot import snapshot_sqlite_database, sqlite_source_revision

_FICLONE = 0x40049409
_MANIFEST_VERSION = 1
_COMPLETE_MARKER = ".source-cut-complete"


class SourceSnapshotError(RuntimeError):
    """A source cannot be cut or a published candidate cannot be trusted."""


class SourceMutationError(SourceSnapshotError):
    """A source changed while its candidate bytes were being copied."""


class CandidateCohortError(SourceSnapshotError):
    """A requested candidate item is outside the published cohort."""


class SnapshotMode(StrEnum):
    IMMUTABLE_EXPORT = "immutable-export"
    ARCHIVE_MEMBER = "archive-member"
    COMPLETE_COPY = "complete-copy"
    SPOOL_HANDOFF = "spool-generation-handoff"
    SQLITE_BACKUP = "sqlite-online-backup"
    DIRECTORY_COPY = "directory-copy"


@dataclass(frozen=True, slots=True)
class SourceCutPolicy:
    """Execution policy bound at preflight, before mutable bytes are read."""

    mode: SnapshotMode
    adapter_version: str = "v1"
    prefer_reflink: bool = True
    allow_full_copy_fallback: bool = True
    capacity_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.adapter_version.strip():
            raise ValueError("adapter_version must be non-empty")
        if self.capacity_bytes is not None and self.capacity_bytes < 0:
            raise ValueError("capacity_bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class SourceRootIdentity:
    """The root identity bound by preflight; content is intentionally absent."""

    device: int
    inode: int
    kind: str
    ctime_ns: int = 0


@dataclass(frozen=True, slots=True)
class SourceCutBinding:
    source: SourceDeclaration
    root_identity: SourceRootIdentity
    policy: SourceCutPolicy


@dataclass(frozen=True, slots=True)
class SourceCutPreflight:
    bindings: tuple[SourceCutBinding, ...]
    request_id: str
    binding_digest: str

    def verify_roots(self) -> None:
        for binding in self.bindings:
            observed = _root_identity(binding.source.root)
            if observed != binding.root_identity:
                raise SourceMutationError(f"source root identity changed: {binding.source.source_id}")


@dataclass(frozen=True, slots=True)
class CutItem:
    source_id: str
    coordinate: str
    identity: str
    content_sha256: str
    size_bytes: int
    snapshot_path: str | None = None
    readmission: bool = False

    @property
    def key(self) -> tuple[str, str, str, str]:
        # Content is part of ownership: inode/path reuse with identical bytes
        # is still a new physical observation and must remain visible on the
        # carry-forward side of the cut.
        return self.source_id, self.coordinate, self.identity, self.content_sha256


@dataclass(frozen=True, slots=True)
class CutManifest:
    kind: str
    items: tuple[CutItem, ...]
    item_count: int
    byte_count: int
    digest: str

    def __post_init__(self) -> None:
        if self.item_count != len(self.items):
            raise ValueError("manifest item denominator does not match items")
        if self.byte_count != sum(item.size_bytes for item in self.items):
            raise ValueError("manifest byte denominator does not match items")
        if len({item.key for item in self.items}) != len(self.items):
            raise ValueError("manifest contains duplicate-owned items")

    def as_dict(self) -> dict[str, object]:
        return {
            "version": _MANIFEST_VERSION,
            "kind": self.kind,
            "items": [
                {
                    "source_id": item.source_id,
                    "coordinate": item.coordinate,
                    "identity": item.identity,
                    "content_sha256": item.content_sha256,
                    "size_bytes": item.size_bytes,
                    "snapshot_path": item.snapshot_path,
                    "readmission": item.readmission,
                }
                for item in self.items
            ],
            "item_count": self.item_count,
            "byte_count": self.byte_count,
            "digest": self.digest,
        }

    def verify_integrity(self) -> None:
        expected = _manifest_digest(self.kind, self.items)
        if expected != self.digest:
            raise SourceSnapshotError(f"{self.kind} manifest integrity check failed")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CutManifest:
        raw_items = payload.get("items")
        if payload.get("version") != _MANIFEST_VERSION or not isinstance(raw_items, list):
            raise SourceSnapshotError("invalid source cut manifest")
        items = tuple(
            CutItem(
                str(item["source_id"]),
                str(item["coordinate"]),
                str(item["identity"]),
                str(item["content_sha256"]),
                int(item["size_bytes"]),
                None if item.get("snapshot_path") is None else str(item["snapshot_path"]),
                bool(item.get("readmission", False)),
            )
            for item in raw_items
            if isinstance(item, dict)
        )
        result = cls(
            str(payload.get("kind")),
            items,
            _required_int(payload, "item_count"),
            _required_int(payload, "byte_count"),
            str(payload.get("digest")),
        )
        result.verify_integrity()
        return result


@dataclass(frozen=True, slots=True)
class SourceSeal:
    cut_identity: str
    candidate_manifest_digest: str
    carry_forward_manifest_digest: str
    binding_digest: str

    @property
    def digest(self) -> str:
        return _sha256(
            {
                "cut_identity": self.cut_identity,
                "candidate_manifest_digest": self.candidate_manifest_digest,
                "carry_forward_manifest_digest": self.carry_forward_manifest_digest,
                "binding_digest": self.binding_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class SourceCutCounts:
    observed_items: int
    observed_bytes: int
    candidate_items: int
    candidate_bytes: int
    carry_forward_items: int
    carry_forward_bytes: int
    missing_items: int = 0
    duplicate_owned_items: int = 0
    unknown_items: int = 0

    @property
    def conserved(self) -> bool:
        return (
            self.observed_items == self.candidate_items + self.carry_forward_items
            and self.observed_bytes == self.candidate_bytes + self.carry_forward_bytes
            and self.missing_items == self.duplicate_owned_items == self.unknown_items == 0
        )


@dataclass(frozen=True, slots=True)
class SourceCutResult:
    cut_identity: str
    candidate_root: Path
    candidate_manifest: CutManifest
    carry_forward_manifest: CutManifest
    seal: SourceSeal
    counts: SourceCutCounts
    observed_manifest: CutManifest
    ownership_modes: tuple[tuple[str, SnapshotMode], ...]

    def verify(self) -> None:
        calculated = _counts_for_partition(
            self.observed_manifest.items,
            self.candidate_manifest.items,
            self.carry_forward_manifest.items,
            dict(self.ownership_modes),
        )
        if calculated != self.counts or not calculated.conserved:
            raise SourceSnapshotError("source cut conservation failed")
        self.candidate_manifest.verify_integrity()
        self.carry_forward_manifest.verify_integrity()
        if self.seal.candidate_manifest_digest != self.candidate_manifest.digest:
            raise SourceSnapshotError("source seal candidate digest mismatch")
        if self.seal.carry_forward_manifest_digest != self.carry_forward_manifest.digest:
            raise SourceSnapshotError("source seal carry-forward digest mismatch")
        self.observed_manifest.verify_integrity()
        if self.seal.cut_identity != self.cut_identity:
            raise SourceSnapshotError("source cut identity mismatch")


def _load_published_source_cut(destination: Path, preflight: SourceCutPreflight | None = None) -> SourceCutResult:
    marker = destination / _COMPLETE_MARKER
    if not marker.is_file():
        raise FileNotFoundError(destination)
    try:
        payload = json.loads((destination / "candidate-manifest.json").read_text(encoding="utf-8"))
        candidate = CutManifest.from_dict(payload["candidate"])
        carry = CutManifest.from_dict(payload["carry_forward"])
        observed = CutManifest.from_dict(payload["observed"])
        raw_seal = payload["seal"]
        seal = SourceSeal(
            str(raw_seal["cut_identity"]),
            str(raw_seal["candidate_manifest_digest"]),
            str(raw_seal["carry_forward_manifest_digest"]),
            str(raw_seal["binding_digest"]),
        )
        cut_identity = str(payload["cut_identity"])
        modes = tuple((str(source_id), SnapshotMode(mode)) for source_id, mode in payload["ownership_modes"])
        raw_counts = payload["counts"]
        counts = SourceCutCounts(
            **{name: _required_int(raw_counts, name) for name in SourceCutCounts.__dataclass_fields__}
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SourceSnapshotError(f"published source cut is unreadable: {destination}") from exc
    if cut_identity != seal.cut_identity:
        raise SourceSnapshotError("published source cut identity mismatch")
    try:
        marker_identity = marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise FileNotFoundError(destination) from exc
    if marker_identity != cut_identity:
        raise FileNotFoundError(destination)
    result = SourceCutResult(cut_identity, destination / "candidate", candidate, carry, seal, counts, observed, modes)
    result.verify()
    if preflight is not None:
        if seal.binding_digest != preflight.binding_digest:
            raise SourceSnapshotError("published source cut binding does not match preflight")
        expected_identity = _cut_identity(preflight, candidate, carry)
        if cut_identity != expected_identity:
            raise SourceSnapshotError("published source cut identity does not match preflight")
    return result


@dataclass(frozen=True, slots=True)
class CandidateInput:
    source_id: str
    coordinate: str
    path: Path
    content_sha256: str
    size_bytes: int


class SourceSnapshotStrategy(Protocol):
    mode: SnapshotMode

    def snapshot(
        self,
        binding: SourceCutBinding,
        destination: Path,
        baseline: tuple[CutItem, ...],
    ) -> tuple[CutItem, ...]: ...


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SourceSnapshotError(f"source member is unreadable: {path}") from exc
    return digest.hexdigest()


def _root_identity(root: Path) -> SourceRootIdentity:
    try:
        info = root.lstat()
    except OSError as exc:
        raise SourceSnapshotError(f"source root is unreadable: {root}") from exc
    if stat.S_ISLNK(info.st_mode) or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise SourceSnapshotError(f"source root is not a regular file or directory: {root}")
    return SourceRootIdentity(
        info.st_dev,
        info.st_ino,
        "file" if stat.S_ISREG(info.st_mode) else "directory",
        0,
    )


def _identity(info: os.stat_result) -> str:
    return f"dev:{info.st_dev}:ino:{info.st_ino}:ctime:{info.st_ctime_ns}"


def _walk_files(root: Path) -> tuple[tuple[str, Path, os.stat_result], ...]:
    if root.is_file():
        try:
            info = root.stat()
        except OSError as exc:
            raise SourceSnapshotError(f"source member disappeared: {root}") from exc
        return ((root.name, root, info),)
    result: list[tuple[str, Path, os.stat_result]] = []
    try:
        paths = sorted(root.rglob("*"))
    except OSError as exc:
        raise SourceSnapshotError(f"source root is unreadable: {root}") from exc
    for path in paths:
        try:
            info = path.lstat()
        except OSError as exc:
            raise SourceSnapshotError(f"source member disappeared: {path}") from exc
        if stat.S_ISDIR(info.st_mode):
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise SourceSnapshotError(f"source member is not a regular file: {path}")
        result.append((path.relative_to(root).as_posix(), path, info))
    return tuple(result)


def _observe(binding: SourceCutBinding) -> tuple[CutItem, ...]:
    root = Path(binding.source.root)
    mode = binding.policy.mode
    if mode is SnapshotMode.ARCHIVE_MEMBER:
        if not root.is_file():
            raise SourceSnapshotError("archive-member sources must name an archive file")
        archive_identity = _root_identity(root)
        try:
            with zipfile.ZipFile(root) as archive:
                return tuple(
                    CutItem(
                        binding.source.source_id,
                        f"{root.name}!{info.filename}",
                        f"{archive_identity.device}:{archive_identity.inode}:{archive_identity.ctime_ns}:{info.header_offset}",
                        hashlib.sha256(archive.read(info)).hexdigest(),
                        info.file_size,
                    )
                    for info in sorted(archive.infolist(), key=lambda item: item.filename)
                    if not info.is_dir()
                )
        except (OSError, zipfile.BadZipFile, KeyError) as exc:
            raise SourceSnapshotError(f"archive member inventory failed: {root}") from exc
    result: list[CutItem] = []
    for coordinate, path, info in _walk_files(root):
        if mode is SnapshotMode.SQLITE_BACKUP:
            identity = sqlite_source_revision(path)
            # The revision is the SQLite strategy's logical identity. Hashing a
            # live database here adds a race and is discarded by the strategy.
            content_sha256 = identity
        else:
            identity = _identity(info)
            content_sha256 = _sha256_path(path)
        result.append(CutItem(binding.source.source_id, coordinate, identity, content_sha256, info.st_size))
    return tuple(result)


def _try_reflink(source: Path, destination: Path) -> bool:
    try:
        with source.open("rb") as source_stream:
            destination_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                fcntl.ioctl(destination_fd, _FICLONE, source_stream.fileno())
                os.fsync(destination_fd)
            finally:
                os.close(destination_fd)
        return True
    except OSError as exc:
        destination.unlink(missing_ok=True)
        if exc.errno not in {errno.EOPNOTSUPP, errno.ENOTTY, errno.EINVAL, errno.EXDEV, errno.ENOSPC, errno.EIO}:
            raise SourceSnapshotError(f"reflink failed for {source}") from exc
        return False


def _copy_file(source: Path, destination: Path, policy: SourceCutPolicy) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if policy.prefer_reflink and _try_reflink(source, destination):
        return
    if not policy.allow_full_copy_fallback:
        raise SourceSnapshotError(f"reflink unavailable and full copy is disabled: {source}")
    if policy.capacity_bytes is not None and source.stat().st_size > policy.capacity_bytes:
        raise SourceSnapshotError(f"capacity preflight rejects full copy: {source}")
    shutil.copyfile(source, destination)
    with destination.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(root: Path) -> None:
    for directory, _children, _files in os.walk(root, topdown=False):
        _fsync_directory(Path(directory))


def _write_durable(path: Path, payload: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_candidates(
    binding: SourceCutBinding,
    baseline: tuple[CutItem, ...],
    destination: Path,
) -> tuple[CutItem, ...]:
    root = Path(binding.source.root)
    if binding.policy.mode is SnapshotMode.ARCHIVE_MEMBER:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_file(root, destination, binding.policy)
        try:
            with zipfile.ZipFile(destination) as archive:
                members = []
                for item in baseline:
                    member_name = item.coordinate.split("!", 1)[1]
                    payload = archive.read(member_name)
                    if hashlib.sha256(payload).hexdigest() != item.content_sha256 or len(payload) != item.size_bytes:
                        raise SourceMutationError(f"archive member changed during cut: {item.coordinate}")
                    members.append(item)
                return tuple(
                    CutItem(
                        item.source_id,
                        item.coordinate,
                        item.identity,
                        item.content_sha256,
                        item.size_bytes,
                        str(destination),
                    )
                    for item in members
                )
        except (OSError, zipfile.BadZipFile, KeyError) as exc:
            raise SourceMutationError(f"archive changed during cut: {root}") from exc
    result: list[CutItem] = []
    for item in baseline:
        source = root / item.coordinate if root.is_dir() else root
        target = destination / item.coordinate if root.is_dir() else destination
        _copy_file(source, target, binding.policy)
        # A changed source is deliberately carried forward by the post-cut
        # inventory.  Only a torn candidate copy is unsafe; a stable copy of
        # the pre-cut bytes remains a valid candidate even when the writer
        # appends or rewrites immediately after it was copied.
        if target.stat().st_size != item.size_bytes or _sha256_path(target) != item.content_sha256:
            raise SourceMutationError(f"source mutated during cut: {source}")
        result.append(
            CutItem(item.source_id, item.coordinate, item.identity, item.content_sha256, item.size_bytes, str(target))
        )
    return tuple(result)


class _FilesystemStrategy:
    def __init__(self, mode: SnapshotMode) -> None:
        self.mode = mode

    def snapshot(
        self, binding: SourceCutBinding, destination: Path, baseline: tuple[CutItem, ...]
    ) -> tuple[CutItem, ...]:
        return _copy_candidates(binding, baseline, destination)


class _SQLiteStrategy(_FilesystemStrategy):
    def snapshot(
        self, binding: SourceCutBinding, destination: Path, baseline: tuple[CutItem, ...]
    ) -> tuple[CutItem, ...]:
        root = Path(binding.source.root)
        if root.is_dir():
            raise SourceSnapshotError("mutable-sqlite declarations must name one database")
        destination.parent.mkdir(parents=True, exist_ok=True)
        snapshot_sqlite_database(root, destination)
        if not destination.exists():
            raise SourceSnapshotError(f"SQLite backup was not published: {root}")
        if sqlite_source_revision(root) != baseline[0].identity:
            raise SourceMutationError(f"SQLite source changed during backup: {root}")
        digest = _sha256_path(destination)
        size = destination.stat().st_size
        return tuple(
            CutItem(item.source_id, item.coordinate, item.identity, digest, size, str(destination)) for item in baseline
        )


class _SpoolHandoffStrategy(_FilesystemStrategy):
    """Atomically give writers a new spool generation before copying the old one."""

    def snapshot(
        self, binding: SourceCutBinding, destination: Path, baseline: tuple[CutItem, ...]
    ) -> tuple[CutItem, ...]:
        root = Path(binding.source.root)
        if not root.is_dir():
            raise SourceSnapshotError("spool handoff requires a directory root")
        retired = root.with_name(f".{root.name}.{binding.source.source_id}.cut")
        if retired.exists():
            raise SourceSnapshotError(f"stale spool handoff generation exists: {retired}")
        os.replace(root, retired)
        root.mkdir(mode=0o700)
        _fsync_directory(root.parent)
        retired_binding = SourceCutBinding(
            SourceDeclaration(binding.source.source_id, binding.source.role, retired, binding.source.mutable),
            _root_identity(retired),
            binding.policy,
        )
        try:
            copied = _copy_candidates(retired_binding, baseline, destination)
        except BaseException:
            # The old generation is still the only copy of pre-cut spool
            # material. Keep it for recovery if candidate copying fails.
            raise
        else:
            shutil.rmtree(retired, ignore_errors=True)
            return copied


def _default_policy(role: SourceRole) -> SourceCutPolicy:
    if role is SourceRole.IMMUTABLE_EXPORT:
        return SourceCutPolicy(SnapshotMode.IMMUTABLE_EXPORT)
    if role is SourceRole.ARCHIVE_MEMBER:
        return SourceCutPolicy(SnapshotMode.ARCHIVE_MEMBER)
    if role is SourceRole.MUTABLE_SQLITE:
        return SourceCutPolicy(SnapshotMode.SQLITE_BACKUP)
    if role in {SourceRole.SPOOL, SourceRole.QUEUE}:
        return SourceCutPolicy(SnapshotMode.SPOOL_HANDOFF)
    return SourceCutPolicy(
        SnapshotMode.COMPLETE_COPY
        if role in {SourceRole.APPEND_JSONL, SourceRole.REWRITE_JSONL}
        else SnapshotMode.DIRECTORY_COPY
    )


_ALLOWED_MODES: dict[SourceRole, frozenset[SnapshotMode]] = {
    SourceRole.IMMUTABLE_EXPORT: frozenset({SnapshotMode.IMMUTABLE_EXPORT}),
    SourceRole.ARCHIVE_MEMBER: frozenset({SnapshotMode.ARCHIVE_MEMBER}),
    SourceRole.APPEND_JSONL: frozenset({SnapshotMode.COMPLETE_COPY}),
    SourceRole.REWRITE_JSONL: frozenset({SnapshotMode.COMPLETE_COPY}),
    SourceRole.MUTABLE_SQLITE: frozenset({SnapshotMode.SQLITE_BACKUP}),
    SourceRole.SPOOL: frozenset({SnapshotMode.SPOOL_HANDOFF, SnapshotMode.DIRECTORY_COPY}),
    SourceRole.QUEUE: frozenset({SnapshotMode.SPOOL_HANDOFF, SnapshotMode.DIRECTORY_COPY}),
    SourceRole.ATTACHMENT: frozenset({SnapshotMode.DIRECTORY_COPY, SnapshotMode.COMPLETE_COPY}),
    SourceRole.SIDECAR: frozenset({SnapshotMode.DIRECTORY_COPY, SnapshotMode.COMPLETE_COPY}),
    SourceRole.PROVIDER_CACHE: frozenset({SnapshotMode.DIRECTORY_COPY, SnapshotMode.COMPLETE_COPY}),
    SourceRole.DIRECTORY: frozenset({SnapshotMode.DIRECTORY_COPY, SnapshotMode.COMPLETE_COPY}),
}


def _strategy(policy: SourceCutPolicy) -> SourceSnapshotStrategy:
    if policy.mode is SnapshotMode.SQLITE_BACKUP:
        return _SQLiteStrategy(policy.mode)
    if policy.mode is SnapshotMode.SPOOL_HANDOFF:
        return _SpoolHandoffStrategy(policy.mode)
    return _FilesystemStrategy(policy.mode)


def _sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _required_int(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourceSnapshotError(f"source cut manifest field {field!r} must be an integer")
    return value


def _manifest_digest(kind: str, items: Iterable[CutItem]) -> str:
    return _sha256(
        {
            "version": _MANIFEST_VERSION,
            "kind": kind,
            "items": [
                (
                    item.source_id,
                    item.coordinate,
                    item.identity,
                    item.content_sha256,
                    item.size_bytes,
                    item.snapshot_path,
                    item.readmission,
                )
                for item in sorted(items, key=lambda value: value.key)
            ],
        }
    )


def _manifest(kind: str, items: Iterable[CutItem]) -> CutManifest:
    ordered = tuple(sorted(items, key=lambda value: value.key))
    return CutManifest(
        kind, ordered, len(ordered), sum(item.size_bytes for item in ordered), _manifest_digest(kind, ordered)
    )


def _ownership_key(item: CutItem, *, mode: SnapshotMode) -> tuple[str, str, str, str] | tuple[str, str, str]:
    """Return the logical version key for a candidate/post-cut comparison."""
    if mode is SnapshotMode.SQLITE_BACKUP:
        return item.source_id, item.coordinate, item.identity
    return item.key


def _counts_for_partition(
    observed: Iterable[CutItem],
    candidate: Iterable[CutItem],
    carry_forward: Iterable[CutItem],
    modes: Mapping[str, SnapshotMode],
) -> SourceCutCounts:
    """Measure ownership against the independently observed cut population."""
    observed_items = tuple(observed)
    candidate_items = tuple(item for item in candidate if not item.readmission)
    carry_items = tuple(item for item in carry_forward if not item.readmission)

    def key(item: CutItem) -> tuple[str, str, str, str] | tuple[str, str, str]:
        try:
            return _ownership_key(item, mode=modes[item.source_id])
        except KeyError as exc:
            raise SourceSnapshotError(f"source cut item has no declared strategy: {item.source_id}") from exc

    observed_by_key = {key(item): item for item in observed_items}
    candidate_by_key = {key(item): item for item in candidate_items}
    carry_by_key = {key(item): item for item in carry_items}
    candidate_keys = set(candidate_by_key)
    carry_keys = set(carry_by_key)
    owned = candidate_keys | carry_keys
    missing = set(observed_by_key) - owned
    duplicate = candidate_keys & carry_keys
    unknown = owned - set(observed_by_key)

    def logical_size(item: CutItem) -> int:
        return observed_by_key.get(key(item), item).size_bytes

    return SourceCutCounts(
        observed_items=len(observed_by_key),
        observed_bytes=sum(item.size_bytes for item in observed_by_key.values()),
        candidate_items=len(candidate_by_key),
        candidate_bytes=sum(logical_size(item) for item in candidate_by_key.values()),
        carry_forward_items=len(carry_by_key),
        carry_forward_bytes=sum(logical_size(item) for item in carry_by_key.values()),
        missing_items=len(missing),
        duplicate_owned_items=len(duplicate),
        unknown_items=len(unknown),
    )


def _cut_identity(preflight: SourceCutPreflight, candidate: CutManifest, carry: CutManifest) -> str:
    return _sha256(
        {
            "request_id": preflight.request_id,
            "binding_digest": preflight.binding_digest,
            "candidate": candidate.digest,
            "carry": carry.digest,
        }
    )


def preflight_source_cut(
    declarations: Iterable[SourceDeclaration],
    *,
    request_id: str = "source-cut",
    policies: Mapping[str, SourceCutPolicy] | None = None,
) -> SourceCutPreflight:
    """Bind roots and strategies without asserting that bytes stay stable."""
    rows = tuple(declarations)
    if not rows:
        raise SourceSnapshotError("source cut requires at least one declaration")
    for row in rows:
        _validate_path_component(row.source_id, label="source_id")
    _validate_path_component(request_id, label="request_id")
    policy_map = policies or {}
    bindings = tuple(
        SourceCutBinding(row, _root_identity(row.root), policy_map.get(row.source_id, _default_policy(row.role)))
        for row in rows
    )
    invalid = next(
        (
            binding
            for binding in bindings
            if binding.policy.mode not in _ALLOWED_MODES.get(binding.source.role, frozenset())
        ),
        None,
    )
    if invalid is not None:
        raise SourceSnapshotError(
            f"snapshot strategy {invalid.policy.mode.value!r} is not valid for {invalid.source.role.value!r}"
        )
    if len({binding.source.source_id for binding in bindings}) != len(bindings):
        raise SourceSnapshotError("source cut declarations contain duplicate source IDs")
    digest = _sha256(
        [
            (
                binding.source.source_id,
                str(binding.source.root),
                binding.root_identity.device,
                binding.root_identity.inode,
                binding.root_identity.kind,
                binding.root_identity.ctime_ns,
                binding.policy.mode.value,
                binding.policy.adapter_version,
            )
            for binding in bindings
        ]
    )
    return SourceCutPreflight(bindings, request_id, digest)


def _validate_path_component(value: str, *, label: str) -> None:
    path = Path(value)
    if not value or path.is_absolute() or path.name != value or value in {".", ".."} or "\\" in value:
        raise SourceSnapshotError(f"{label} must be one relative path component")


def execute_source_cut(preflight: SourceCutPreflight, destination: Path) -> SourceCutResult:
    """Create and atomically publish one immutable candidate cohort."""
    destination = destination.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reclaim_orphaned_staging(destination.parent, preflight.request_id)
    if destination.exists():
        preflight.verify_roots()
        try:
            return _load_published_source_cut(destination, preflight)
        except FileNotFoundError:
            # A destination without the final marker was never published. It
            # contains only this operation's private output and is safe to
            # reclaim before repeating the immutable request.
            shutil.rmtree(destination)
    preflight.verify_roots()
    staging = Path(tempfile.mkdtemp(prefix=f".{preflight.request_id}.", dir=destination.parent))
    try:
        baselines = {binding.source.source_id: _observe(binding) for binding in preflight.bindings}
        candidate_items: list[CutItem] = []
        for binding in preflight.bindings:
            source_destination = staging / "candidate" / binding.source.source_id
            if binding.policy.mode is SnapshotMode.ARCHIVE_MEMBER or not Path(binding.source.root).is_dir():
                source_destination = source_destination.with_name(
                    source_destination.name + Path(binding.source.root).suffix
                )
            candidate_items.extend(
                _strategy(binding.policy).snapshot(binding, source_destination, baselines[binding.source.source_id])
            )
        candidate_items = [
            CutItem(
                item.source_id,
                item.coordinate,
                item.identity,
                item.content_sha256,
                item.size_bytes,
                str(Path(item.snapshot_path or "").relative_to(staging / "candidate")),
                item.readmission,
            )
            for item in candidate_items
        ]
        # A source-root rename or replacement invalidates the bound source
        # identity.  Per-file arrivals/replacements are handled by the
        # carry-forward manifest; replacing the declared root is unknown.
        for binding in preflight.bindings:
            if (
                binding.policy.mode is not SnapshotMode.SPOOL_HANDOFF
                and _root_identity(binding.source.root) != binding.root_identity
            ):
                raise SourceMutationError(f"source root identity changed: {binding.source.source_id}")
        post_items = [item for binding in preflight.bindings for item in _observe(binding)]
        modes = {binding.source.source_id: binding.policy.mode for binding in preflight.bindings}
        candidate_keys = {_ownership_key(item, mode=modes[item.source_id]) for item in candidate_items}
        baseline_coordinates = {(item.source_id, item.coordinate) for items in baselines.values() for item in items}
        carry_items: list[CutItem] = []
        for item in post_items:
            mode = modes[item.source_id]
            if mode is SnapshotMode.COMPLETE_COPY and (item.source_id, item.coordinate) in baseline_coordinates:
                # Complete copies define the cut boundary for live JSONL. The
                # active path remains a normal, idempotent future read rather
                # than a second logically-owned byte population.
                carry_items.append(replace(item, readmission=True))
            elif _ownership_key(item, mode=mode) not in candidate_keys:
                carry_items.append(item)
        candidate = _manifest("candidate", candidate_items)
        carry = _manifest("carry-forward", carry_items)
        observed_items = [item for items in baselines.values() for item in items]
        observed_items.extend(item for item in carry_items if not item.readmission)
        observed = _manifest("observed", observed_items)
        cut_identity = _cut_identity(preflight, candidate, carry)
        seal = SourceSeal(cut_identity, candidate.digest, carry.digest, preflight.binding_digest)
        ownership_modes = tuple(sorted(modes.items()))
        counts = _counts_for_partition(observed.items, candidate.items, carry.items, modes)
        result = SourceCutResult(
            cut_identity, destination / "candidate", candidate, carry, seal, counts, observed, ownership_modes
        )
        result.verify()
        manifest_payload = {
            "candidate": candidate.as_dict(),
            "carry_forward": carry.as_dict(),
            "observed": observed.as_dict(),
            "ownership_modes": [(source_id, mode.value) for source_id, mode in ownership_modes],
            "counts": {name: getattr(counts, name) for name in SourceCutCounts.__dataclass_fields__},
            "seal": {
                "cut_identity": seal.cut_identity,
                "candidate_manifest_digest": seal.candidate_manifest_digest,
                "carry_forward_manifest_digest": seal.carry_forward_manifest_digest,
                "binding_digest": seal.binding_digest,
                "digest": seal.digest,
            },
            "cut_identity": cut_identity,
        }
        _write_durable(staging / "candidate-manifest.json", json.dumps(manifest_payload, sort_keys=True, indent=2))
        _fsync_tree(staging)
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
        _write_durable(destination / _COMPLETE_MARKER, f"{cut_identity}\n")
        _fsync_directory(destination.parent)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _reclaim_orphaned_staging(parent: Path, request_id: str) -> None:
    prefix = f".{request_id}."
    for path in parent.iterdir():
        if path.name.startswith(prefix) and path.is_dir():
            shutil.rmtree(path)


def reacquire_candidate(
    result: SourceCutResult, *, source_id: str | None = None, coordinates: Iterable[str] | None = None
) -> tuple[CandidateInput, ...]:
    """Return only immutable candidate paths, rejecting outside coordinates."""
    result.verify()
    selected = set(coordinates) if coordinates is not None else None
    if source_id is not None and source_id not in {item.source_id for item in result.candidate_manifest.items}:
        raise CandidateCohortError(f"candidate request names outside cohort: {source_id}")
    inputs: list[CandidateInput] = []
    for item in result.candidate_manifest.items:
        if source_id is not None and item.source_id != source_id:
            continue
        if selected is not None and item.coordinate not in selected:
            continue
        if item.snapshot_path is None:
            raise CandidateCohortError(f"candidate item has no immutable snapshot: {item.coordinate}")
        path = result.candidate_root / item.snapshot_path
        if not path.is_relative_to(result.candidate_root):
            raise CandidateCohortError(f"candidate path escapes published snapshot: {item.coordinate}")
        if not path.is_file() or path.stat().st_size != item.size_bytes or _sha256_path(path) != item.content_sha256:
            raise SourceMutationError(f"candidate snapshot mutated: {item.coordinate}")
        inputs.append(CandidateInput(item.source_id, item.coordinate, path, item.content_sha256, item.size_bytes))
    if selected is not None:
        actual = {item.coordinate for item in inputs}
        missing = selected - actual
        if missing:
            raise CandidateCohortError(f"candidate request names outside cohort: {sorted(missing)}")
    return tuple(inputs)


__all__ = [
    "CandidateCohortError",
    "CandidateInput",
    "CutItem",
    "CutManifest",
    "SnapshotMode",
    "SourceCutBinding",
    "SourceCutCounts",
    "SourceCutPolicy",
    "SourceCutPreflight",
    "SourceCutResult",
    "SourceMutationError",
    "SourceRootIdentity",
    "SourceSeal",
    "SourceSnapshotError",
    "SourceSnapshotStrategy",
    "execute_source_cut",
    "load_source_cut",
    "preflight_source_cut",
    "reacquire_candidate",
]


load_source_cut = _load_published_source_cut

"""Lossless cut boundaries for mutable source inputs.

The cut is a filesystem operation, not a campaign state machine.  Preflight
binds the configured roots and source-owned strategy names.  Execution copies
the candidate cohort into a private immutable tree, inventories the live roots
again, and publishes two digests: candidate bytes and the material that stayed
in the ordinary source roots after the cut.

The declared spool strategy hands its original generation aside and creates
a fresh active root; other strategies copy their roots. A later daemon
catch-up sees carry-forward files through its normal acquisition route. A
candidate is read from the published private tree, with its digest rechecked.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import zipfile
from bisect import bisect_left
from collections.abc import Iterable, Iterator, Mapping
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from polylogue.sources.source_layout import SourceLayout

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.maintenance.source_manifest_continuity import SourceDeclaration, SourceRole
from polylogue.sources.file_alias import contained_file_alias_coordinate
from polylogue.sources.origin_specs import pre_acquisition_path_exclusion
from polylogue.sources.source_staging import bind_source_input
from polylogue.sources.sqlite_export import BinaryWriteSink, _logical_export_digest_bound, _write_logical_export_bound
from polylogue.sources.sqlite_snapshot import member_export_scope

_FICLONE = 0x40049409
_MANIFEST_VERSION = 2
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
    SQLITE_LOGICAL_EXPORT = "sqlite-logical-export"
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

    def verify_roots(self, *, allow_handed_off_spools: bool = False) -> None:
        for binding in self.bindings:
            if allow_handed_off_spools and binding.policy.mode is SnapshotMode.SPOOL_HANDOFF:
                continue
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
    post_cut_arrival: bool = False

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
    version: int = _MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.item_count != len(self.items):
            raise ValueError("manifest item denominator does not match items")
        if self.byte_count != sum(item.size_bytes for item in self.items):
            raise ValueError("manifest byte denominator does not match items")
        if len({item.key for item in self.items}) != len(self.items):
            raise ValueError("manifest contains duplicate-owned items")

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
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
                    "post_cut_arrival": item.post_cut_arrival,
                }
                for item in self.items
            ],
            "item_count": self.item_count,
            "byte_count": self.byte_count,
            "digest": self.digest,
        }

    def verify_integrity(self) -> None:
        expected = _manifest_digest(self.kind, self.items, version=self.version)
        if expected != self.digest:
            raise SourceSnapshotError(f"{self.kind} manifest integrity check failed")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CutManifest:
        raw_items = payload.get("items")
        version = payload.get("version")
        if isinstance(version, bool) or version not in {1, _MANIFEST_VERSION} or not isinstance(raw_items, list):
            raise SourceSnapshotError("invalid source cut manifest")
        assert isinstance(version, int)
        items = tuple(
            CutItem(
                str(item["source_id"]),
                str(item["coordinate"]),
                str(item["identity"]),
                str(item["content_sha256"]),
                int(item["size_bytes"]),
                None if item.get("snapshot_path") is None else str(item["snapshot_path"]),
                bool(item.get("readmission", False)),
                bool(item.get("post_cut_arrival", False)),
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
            version,
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


@dataclass(frozen=True, slots=True)
class SourceSnapshotResult:
    candidate_items: tuple[CutItem, ...]
    observation_binding: SourceCutBinding


class SourceSnapshotStrategy(Protocol):
    mode: SnapshotMode

    def snapshot(
        self,
        binding: SourceCutBinding,
        destination: Path,
        baseline: tuple[CutItem, ...],
    ) -> SourceSnapshotResult: ...


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                check_compute_cancelled()
                digest.update(chunk)
    except OSError as exc:
        raise SourceSnapshotError(f"source member is unreadable: {path}") from exc
    return digest.hexdigest()


@contextmanager
def _open_source_root(binding: SourceCutBinding) -> Iterator[tuple[int, os.stat_result, Path]]:
    """Resolve accepted parent aliases once and bind the actual declared root."""
    descriptor: int | None = None
    root = Path(binding.source.root)
    try:
        if _root_identity(root) != binding.root_identity:
            raise SourceMutationError(f"source root identity changed: {root}")
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        if binding.root_identity.kind == "directory":
            flags |= os.O_DIRECTORY
        physical_root = root.resolve(strict=True)
        descriptor = os.open(physical_root, flags)
        info = os.fstat(descriptor)
        kind = "directory" if stat.S_ISDIR(info.st_mode) else "file" if stat.S_ISREG(info.st_mode) else "other"
        if (info.st_dev, info.st_ino, kind) != (
            binding.root_identity.device,
            binding.root_identity.inode,
            binding.root_identity.kind,
        ):
            raise SourceMutationError(f"source root identity changed: {root}")
        yield descriptor, info, physical_root
    except OSError as exc:
        raise SourceSnapshotError(f"source root is unreadable: {root}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
def _open_source_file(
    anchor: int,
    coordinate: str,
    path: Path,
    expected: tuple[int, int] | None,
    *,
    directory: bool = False,
) -> Iterator[tuple[int, os.stat_result]]:
    """Open relative to the bound root, refusing substituted internal symlinks."""
    parent: int | None = None
    descriptor: int | None = None
    try:
        parent = os.dup(anchor)
        if coordinate:
            parts = Path(coordinate).parts
            for component in parts[:-1]:
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                os.close(parent)
                parent = child
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if directory:
                flags |= os.O_DIRECTORY
            descriptor = os.open(parts[-1], flags, dir_fd=parent)
        else:
            descriptor = os.dup(anchor)
        info = os.fstat(descriptor)
        valid_kind = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
        if not valid_kind or (expected is not None and (info.st_dev, info.st_ino) != expected):
            raise SourceMutationError(f"source member identity changed: {path}")
        yield descriptor, info
    except OSError as exc:
        raise SourceSnapshotError(f"source member is unreadable: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent is not None:
            os.close(parent)


def _snapshot_regular_file(
    path: Path, expected: os.stat_result, *, anchor: int, coordinate: str
) -> tuple[str, int, str]:
    """Hash one descriptor's captured prefix and return its matching identity.

    Enumeration binds the inode; opening refuses symlink substitution. An
    append after fstat keeps the original prefix's size, hash and identity.
    A changed prefix or truncation refuses the whole observation.
    """
    with _open_source_file(anchor, coordinate, path, (expected.st_dev, expected.st_ino)) as (descriptor, info):
        if not coordinate:
            info = expected
        first = _hash_prefix(descriptor, info.st_size, path)
        after = os.fstat(descriptor)
        truncated = after.st_size < info.st_size
        changed = after.st_ctime_ns != info.st_ctime_ns
        if truncated or (changed and _hash_prefix(descriptor, info.st_size, path) != first):
            raise SourceSnapshotError(f"source member was rewritten while reading: {path}")
    return first, info.st_size, _identity(info)


def _hash_prefix(descriptor: int, size: int, path: Path) -> str:
    """SHA-256 of the first ``size`` bytes of an open descriptor."""
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    remaining = size
    while remaining:
        check_compute_cancelled()
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            raise SourceSnapshotError(f"source member was truncated while reading: {path}")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _root_identity(root: Path) -> SourceRootIdentity:
    """Identify the actual root an explicit declaration names.

    A declared file root may be an alias (the snapshot route accepts one for
    a SQLite database); its identity is the resolved actual file, so a
    retargeted alias reads as a changed root. A directory root is never an
    alias: its members would be enumerated through the link.
    """
    try:
        info = root.lstat()
        if stat.S_ISLNK(info.st_mode):
            info = root.resolve(strict=True).lstat()
            if not stat.S_ISREG(info.st_mode):
                raise SourceSnapshotError(f"source root alias does not name a regular file: {root}")
    except OSError as exc:
        raise SourceSnapshotError(f"source root is unreadable: {root}") from exc
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise SourceSnapshotError(f"source root is not a regular file or directory: {root}")
    return SourceRootIdentity(
        info.st_dev,
        info.st_ino,
        "file" if stat.S_ISREG(info.st_mode) else "directory",
        0,
    )


def _identity(info: os.stat_result) -> str:
    return f"dev:{info.st_dev}:ino:{info.st_ino}:ctime:{info.st_ctime_ns}"


def _walk_files(
    root: Path,
    anchor: int,
    root_info: os.stat_result,
    physical_root: Path,
    *,
    layout_name: str | None = None,
    exclude_coordinates: tuple[str, ...] = (),
) -> Iterator[tuple[str, Path, os.stat_result, int | None, Path]]:
    """Enumerate every member, propagating scan and stat faults to the root owner."""
    try:
        if stat.S_ISREG(root_info.st_mode):
            yield root.name, root, root_info, None, physical_root.parent
            return
        if not stat.S_ISDIR(root_info.st_mode):
            raise SourceSnapshotError(f"source root is not a directory: {root}")
        layout = None
        excluded_database_names: frozenset[str] = frozenset()
        if layout_name is not None:
            from polylogue.sources.source_layout import source_layout_for

            layout = source_layout_for(layout_name)
            from polylogue.sources.origin_specs import database_capability_for_provider

            capability = database_capability_for_provider(layout.provider) if layout.provider is not None else None
            if capability is not None:
                excluded_database_names = frozenset(
                    member.filename for member in capability.members if member.disposition == "out-of-scope"
                )
        with (
            tempfile.TemporaryDirectory(prefix="polylogue-alias-coverage-") as directory,
            closing(sqlite3.connect(Path(directory) / "coverage.db")) as coverage,
        ):
            coverage.execute("PRAGMA journal_mode=OFF")
            coverage.execute("PRAGMA synchronous=OFF")
            coverage.execute("PRAGMA cache_size=-1024")
            coverage.execute(
                "CREATE TABLE targets(coordinate TEXT PRIMARY KEY, device INTEGER, inode INTEGER) WITHOUT ROWID"
            )
            coverage.execute(
                "CREATE TABLE aliases(coordinate TEXT PRIMARY KEY, device INTEGER, inode INTEGER, ctime INTEGER, "
                "link_text TEXT, target TEXT, target_device INTEGER, target_inode INTEGER) WITHOUT ROWID"
            )
            yield from _walk_bound_directory_files(
                root, anchor, root_info, physical_root, layout, exclude_coordinates, excluded_database_names, coverage
            )
    except (OSError, sqlite3.Error) as exc:
        raise SourceSnapshotError(f"source root inventory failed: {root}") from exc


def _walk_bound_directory_files(
    root: Path,
    anchor: int,
    root_info: os.stat_result,
    physical_root: Path,
    layout: SourceLayout | None,
    exclude_coordinates: tuple[str, ...],
    excluded_database_names: frozenset[str],
    coverage: sqlite3.Connection,
) -> Iterator[tuple[str, Path, os.stat_result, int | None, Path]]:
    directories: list[tuple[Path, os.stat_result, tuple[str, ...]]] = [(root, root_info, ())]
    while directories:
        check_compute_cancelled()
        directory, expected, relative = directories.pop()
        with (
            _open_source_file(
                anchor,
                directory.relative_to(root).as_posix() if directory != root else "",
                directory,
                (expected.st_dev, expected.st_ino),
                directory=True,
            ) as (descriptor, _info),
            os.scandir(descriptor) as entries,
        ):
            children = sorted(entries, key=lambda entry: entry.name)
            for entry in children:
                check_compute_cancelled()
                path = directory / entry.name
                info = entry.stat(follow_symlinks=False)
                child_parts = (*relative, entry.name)
                if stat.S_ISDIR(info.st_mode):
                    if layout is None or layout.admits_directory(child_parts):
                        directories.append((path, info, child_parts))
                elif stat.S_ISREG(info.st_mode):
                    if "/".join(child_parts) in exclude_coordinates:
                        continue
                    if layout is not None and layout.artifact_kind(child_parts) is None:
                        continue
                    if entry.name in excluded_database_names:
                        # The acquisition registry owns this exclusion. An
                        # arriving projection must not become a byte obligation
                        # or independently cover a selected alias target.
                        continue
                    if (
                        layout is not None
                        and layout.provider is not None
                        and pre_acquisition_path_exclusion(layout.provider, path) is not None
                    ):
                        continue
                    coordinate = path.relative_to(root).as_posix()
                    coverage.execute("INSERT INTO targets VALUES (?,?,?)", (coordinate, info.st_dev, info.st_ino))
                    yield (
                        coordinate,
                        path,
                        info,
                        descriptor,
                        (physical_root / path.relative_to(root)).parent,
                    )
                elif stat.S_ISLNK(info.st_mode) and (layout is None or layout.artifact_kind(child_parts) is not None):
                    coordinate = "/".join(child_parts)
                    link_text = os.readlink(entry.name, dir_fd=descriptor)
                    after = os.stat(entry.name, dir_fd=descriptor, follow_symlinks=False)
                    if (after.st_dev, after.st_ino, after.st_ctime_ns) != (info.st_dev, info.st_ino, info.st_ctime_ns):
                        raise SourceMutationError(f"source alias changed during inventory: {path}")
                    named = Path(link_text)
                    target = Path(
                        os.path.normpath(
                            str(named if named.is_absolute() else physical_root / Path(coordinate).parent / named)
                        )
                    )
                    # An absolute target may use the declaration's stable parent alias.
                    if target.is_relative_to(root):
                        target = physical_root / target.relative_to(root)
                    target_coordinate = contained_file_alias_coordinate(physical_root, target, stat.S_IFREG)
                    if target_coordinate is None or (
                        layout is not None and layout.artifact_kind(Path(target_coordinate).parts) is None
                    ):
                        raise SourceSnapshotError(f"source alias target is outside the declared layout: {path}")
                    with _open_source_file(anchor, target_coordinate, target, None) as (_target, target_info):
                        if contained_file_alias_coordinate(physical_root, target, target_info.st_mode) is None:
                            raise SourceSnapshotError(f"source alias target is not a regular file: {path}")
                    coverage.execute(
                        "INSERT INTO aliases VALUES (?,?,?,?,?,?,?,?)",
                        (
                            coordinate,
                            info.st_dev,
                            info.st_ino,
                            info.st_ctime_ns,
                            link_text,
                            target_coordinate,
                            target_info.st_dev,
                            target_info.st_ino,
                        ),
                    )
                elif (
                    layout is None
                    or layout.admits_directory(child_parts)
                    or layout.artifact_kind(child_parts) is not None
                ):
                    raise SourceSnapshotError(f"source member is not a regular file: {path}")
    for coordinate, device, inode, ctime, link_text, target, target_device, target_inode in coverage.execute(
        "SELECT coordinate,device,inode,ctime,link_text,target,target_device,target_inode FROM aliases"
    ):
        check_compute_cancelled()
        observed = coverage.execute("SELECT device,inode FROM targets WHERE coordinate=?", (target,)).fetchone()
        if observed != (target_device, target_inode):
            raise SourceSnapshotError(f"source alias target was not independently observed: {root / coordinate}")
        parent_coordinate = str(Path(coordinate).parent)
        parent_coordinate = "" if parent_coordinate == "." else parent_coordinate
        with (
            _open_source_file(
                anchor,
                parent_coordinate,
                root / parent_coordinate,
                None,
                directory=True,
            ) as (parent, _info),
            _open_source_file(anchor, target, root / target, (target_device, target_inode)),
        ):
            current = os.stat(Path(coordinate).name, dir_fd=parent, follow_symlinks=False)
            current_link = os.readlink(Path(coordinate).name, dir_fd=parent)
            if (current.st_dev, current.st_ino, current.st_ctime_ns, current_link) != (
                device,
                inode,
                ctime,
                link_text,
            ):
                raise SourceMutationError(f"source alias changed during inventory: {root / coordinate}")


def _observe(binding: SourceCutBinding) -> tuple[CutItem, ...]:
    return tuple(_iter_observe(binding))


def _iter_observe(binding: SourceCutBinding) -> Iterator[CutItem]:
    root = Path(binding.source.root)
    if binding.policy.mode is SnapshotMode.SQLITE_LOGICAL_EXPORT:
        if _root_identity(root) != binding.root_identity:
            raise SourceMutationError(f"source root identity changed: {root}")
        if binding.root_identity.kind == "directory":
            with _open_source_root(binding) as (anchor, root_info, physical_root):
                yield from _observe_sqlite_members(
                    binding,
                    _walk_files(
                        root,
                        anchor,
                        root_info,
                        physical_root,
                        layout_name=binding.source.layout_name,
                        exclude_coordinates=binding.source.exclude_coordinates,
                    ),
                )
        else:
            yield from _observe_sqlite_members(binding, ((root.name, root, root.stat(), None, root.parent),))
        if _root_identity(root) != binding.root_identity:
            raise SourceMutationError(f"source root identity changed: {root}")
        return
    with _open_source_root(binding) as (anchor, root_info, physical_root):
        yield from _observe_root(binding, anchor, root_info, physical_root)
        if _root_identity(root) != binding.root_identity:
            raise SourceMutationError(f"source root identity changed: {root}")
        return


def _observe_sqlite_members(
    binding: SourceCutBinding, members: Iterable[tuple[str, Path, os.stat_result, int | None, Path]]
) -> Iterator[CutItem]:
    for coordinate, path, before, parent_anchor, semantic_parent in members:
        check_compute_cancelled()
        expected = before.st_dev, before.st_ino
        for info in (before, path.stat() if parent_anchor is None else path.lstat()):
            if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != expected:
                raise SourceMutationError(f"source database identity changed: {path}")
        # The isolated export owner binds SQLite's actual descriptors to
        # this enumerated identity without closing any guard database fd.
        try:
            with bind_source_input(
                path, parent_anchor=parent_anchor, semantic_parent=semantic_parent
            ) as source_binding:
                identity = _logical_export_digest_bound(
                    path,
                    scope=member_export_scope(source_binding.source_path),
                    expected_identity=expected,
                    parent_anchor=source_binding.parent_anchor,
                    source_binding=source_binding,
                )
        except sqlite3.Error as exc:
            raise SourceSnapshotError(f"SQLite source observation failed: {path}") from exc
        after = path.stat() if parent_anchor is None else path.lstat()
        if not stat.S_ISREG(after.st_mode) or (after.st_dev, after.st_ino) != expected:
            raise SourceMutationError(f"source database identity changed: {path}")
        yield CutItem(binding.source.source_id, coordinate, identity, identity, before.st_size)


def _observe_root(
    binding: SourceCutBinding, anchor: int, root_info: os.stat_result, physical_root: Path
) -> Iterator[CutItem]:
    root = Path(binding.source.root)
    mode = binding.policy.mode
    if mode is SnapshotMode.ARCHIVE_MEMBER:
        if not root.is_file():
            raise SourceSnapshotError("archive-member sources must name an archive file")
        archive_info = root_info
        try:
            with (
                os.fdopen(os.dup(anchor), "rb") as stream,
                zipfile.ZipFile(stream) as archive,
            ):
                for info in sorted(archive.infolist(), key=lambda item: item.filename):
                    check_compute_cancelled()
                    if info.is_dir():
                        continue
                    digest = hashlib.sha256()
                    size = 0
                    with archive.open(info) as member:
                        while chunk := member.read(1024 * 1024):
                            check_compute_cancelled()
                            digest.update(chunk)
                            size += len(chunk)
                    yield CutItem(
                        binding.source.source_id,
                        f"{root.name}!{info.filename}",
                        f"{archive_info.st_dev}:{archive_info.st_ino}:{archive_info.st_ctime_ns}:{info.header_offset}",
                        digest.hexdigest(),
                        size,
                    )
                after = os.fstat(anchor)
                if (after.st_size, after.st_ctime_ns) != (archive_info.st_size, archive_info.st_ctime_ns):
                    raise SourceMutationError(f"archive changed during inventory: {root}")
                return
        except DaemonOperationCancelled:
            raise
        except (OSError, zipfile.BadZipFile, KeyError, RuntimeError) as exc:
            raise SourceSnapshotError(f"archive member inventory failed: {root}") from exc
    logical_members: set[str] = set()
    if binding.source.layout_name is not None:
        from polylogue.sources.origin_specs import database_capability_for_provider
        from polylogue.sources.source_layout import source_layout_for

        provider = source_layout_for(binding.source.layout_name).provider
        capability = database_capability_for_provider(provider) if provider is not None else None
        if capability is not None:
            logical_members = {member.filename for member in capability.members if member.disposition != "out-of-scope"}
    for coordinate, path, member_info, _parent_anchor, _semantic_parent in _walk_files(
        root,
        anchor,
        root_info,
        physical_root,
        layout_name=binding.source.layout_name,
        exclude_coordinates=binding.source.exclude_coordinates,
    ):
        if path.name in logical_members:
            # A database created after declaration discovery must never
            # enter the ordinary byte inventory as a physical page image.
            raise SourceMutationError(f"source database requires a logical declaration: {path}")
        content_sha256, captured_size, identity = _snapshot_regular_file(
            path,
            member_info,
            anchor=anchor,
            coordinate=coordinate if binding.root_identity.kind == "directory" else "",
        )
        yield CutItem(binding.source.source_id, coordinate, identity, content_sha256, captured_size)


def bind_source_observation(declaration: SourceDeclaration) -> SourceCutBinding:
    """Capture the root identity shared by observation and member addressing."""
    return SourceCutBinding(declaration, _root_identity(declaration.root), _default_policy(declaration.role))


def iter_observe_source_members(binding: SourceCutBinding) -> Iterator[CutItem]:
    """Stream a captured source inventory without retaining its members.

    Callers derive member paths from this binding's root kind. Exhaust the
    iterator to receive the final root identity check.
    """
    try:
        yield from _iter_observe(binding)
    except sqlite3.DatabaseError as exc:
        raise SourceSnapshotError(f"source database unreadable: {exc}") from exc


def observe_source_members(declaration: SourceDeclaration) -> tuple[CutItem, ...]:
    """Enumerate one declared source without copying or mutating it.

    This is the read-only observation law shared by source-cut and source
    conservation.  In particular, archive members and mutable SQLite roots
    are observed at their declared logical granularity rather than being
    reduced to one root row or a filesystem byte count.
    """
    return tuple(iter_observe_source_members(bind_source_observation(declaration)))


def _try_reflink(descriptor: int, destination: Path) -> bool:
    try:
        destination_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            fcntl.ioctl(destination_fd, _FICLONE, descriptor)
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
        return True
    except OSError as exc:
        destination.unlink(missing_ok=True)
        if exc.errno not in {errno.EOPNOTSUPP, errno.ENOTTY, errno.EINVAL, errno.EXDEV, errno.ENOSPC, errno.EIO}:
            raise SourceSnapshotError("reflink failed for bound source descriptor") from exc
        return False


def _copy_file(
    source: Path,
    destination: Path,
    policy: SourceCutPolicy,
    *,
    expected: tuple[int, int],
    captured_size: int | None,
    anchor: int,
    coordinate: str,
) -> None:
    """Copy the enumerated inode, preserving the captured append-log prefix."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _open_source_file(anchor, coordinate, source, expected) as (descriptor, info):
        size = info.st_size if captured_size is None else captured_size
        if size > info.st_size:
            raise SourceMutationError(f"source truncated before copying: {source}")
        if policy.prefer_reflink and size == info.st_size and _try_reflink(descriptor, destination):
            return
        if not policy.allow_full_copy_fallback:
            raise SourceSnapshotError(f"reflink unavailable and full copy is disabled: {source}")
        if policy.capacity_bytes is not None and size > policy.capacity_bytes:
            raise SourceSnapshotError(f"capacity preflight rejects full copy: {source}")
        remaining = size
        with destination.open("xb") as output:
            while remaining:
                check_compute_cancelled()
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise SourceMutationError(f"source truncated while copying: {source}")
                output.write(chunk)
                remaining -= len(chunk)
            output.flush()
            os.fsync(output.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(root: Path) -> None:
    def refuse_unreadable_directory(error: OSError) -> None:
        raise SourceSnapshotError(f"candidate directory sync inventory failed: {root}") from error

    for directory, _children, _files in os.walk(root, topdown=False, onerror=refuse_unreadable_directory):
        check_compute_cancelled()
        _fsync_directory(Path(directory))


def _write_durable(path: Path, payload: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _archive_member_info(archive: zipfile.ZipFile, item: CutItem) -> zipfile.ZipInfo:
    """Select the already captured member offset, including duplicate names."""
    try:
        offset = int(item.identity.rsplit(":", 1)[1])
    except (ValueError, IndexError) as exc:
        raise SourceMutationError(f"archive member identity changed: {item.coordinate}") from exc
    entries = archive.infolist()
    position = bisect_left(entries, offset, key=lambda entry: entry.header_offset)
    if position == len(entries):
        raise SourceMutationError(f"archive member disappeared: {item.coordinate}")
    entry = entries[position]
    if entry.header_offset != offset or entry.is_dir() or not item.coordinate.endswith("!" + entry.filename):
        raise SourceMutationError(f"archive member coordinate changed: {item.coordinate}")
    return entry


def _copy_candidates(
    binding: SourceCutBinding,
    baseline: tuple[CutItem, ...],
    destination: Path,
) -> tuple[CutItem, ...]:
    with _open_source_root(binding) as (anchor, _root_info, _physical_root):
        return _copy_bound_candidates(binding, baseline, destination, anchor)


def _copy_bound_candidates(
    binding: SourceCutBinding, baseline: tuple[CutItem, ...], destination: Path, anchor: int
) -> tuple[CutItem, ...]:
    root = Path(binding.source.root)
    if binding.policy.mode is SnapshotMode.ARCHIVE_MEMBER:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_file(
            root,
            destination,
            binding.policy,
            expected=(binding.root_identity.device, binding.root_identity.inode),
            captured_size=None,
            anchor=anchor,
            coordinate="",
        )
        try:
            with zipfile.ZipFile(destination) as archive:
                archive.infolist().sort(key=lambda entry: entry.header_offset)
                members = []
                for item in baseline:
                    check_compute_cancelled()
                    info = _archive_member_info(archive, item)
                    if item.coordinate != f"{root.name}!{info.filename}":
                        raise SourceMutationError(f"archive member coordinate changed: {item.coordinate}")
                    digest = hashlib.sha256()
                    size = 0
                    with archive.open(info) as stream:
                        while chunk := stream.read(1024 * 1024):
                            check_compute_cancelled()
                            digest.update(chunk)
                            size += len(chunk)
                    if digest.hexdigest() != item.content_sha256 or size != item.size_bytes:
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
        except DaemonOperationCancelled:
            raise
        except (OSError, zipfile.BadZipFile, KeyError, RuntimeError) as exc:
            raise SourceMutationError(f"archive changed during cut: {root}") from exc
    result: list[CutItem] = []
    for item in baseline:
        check_compute_cancelled()
        source = root / item.coordinate if binding.root_identity.kind == "directory" else root
        target = destination / item.coordinate if binding.root_identity.kind == "directory" else destination
        _copy_file(
            source,
            target,
            binding.policy,
            expected=(int(item.identity.split(":")[1]), int(item.identity.split(":")[3])),
            captured_size=item.size_bytes,
            anchor=anchor,
            coordinate=item.coordinate if binding.root_identity.kind == "directory" else "",
        )
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
    ) -> SourceSnapshotResult:
        return SourceSnapshotResult(_copy_candidates(binding, baseline, destination), binding)


class _BoundedSnapshotWriter:
    """Write one candidate member without exceeding its declared capacity."""

    def __init__(self, handle: BinaryWriteSink, *, capacity_bytes: int | None) -> None:
        self._handle = handle
        self._capacity_bytes = capacity_bytes
        self._written = 0

    def write(self, payload: bytes) -> int:
        next_size = self._written + len(payload)
        if self._capacity_bytes is not None and next_size > self._capacity_bytes:
            raise SourceSnapshotError(
                f"capacity preflight rejects logical SQLite export: requires more than {self._capacity_bytes} bytes"
            )
        written = self._handle.write(payload)
        self._written += written
        return written


class _SQLiteLogicalExportStrategy(_FilesystemStrategy):
    def snapshot(
        self, binding: SourceCutBinding, destination: Path, baseline: tuple[CutItem, ...]
    ) -> SourceSnapshotResult:
        root = Path(binding.source.root)
        if root.is_dir():
            raise SourceSnapshotError("mutable-sqlite declarations must name one database")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with bind_source_input(root) as source_binding, destination.open("xb") as raw_handle:
            handle = _BoundedSnapshotWriter(raw_handle, capacity_bytes=binding.policy.capacity_bytes)
            _write_logical_export_bound(
                root,
                handle,
                scope=member_export_scope(source_binding.source_path),
                parent_anchor=source_binding.parent_anchor,
                source_binding=source_binding,
                expected_identity=(binding.root_identity.device, binding.root_identity.inode),
            )
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        if not destination.exists():
            raise SourceSnapshotError(f"SQLite logical export was not published: {root}")
        digest = _sha256_path(destination)
        if digest != baseline[0].identity:
            raise SourceMutationError(f"SQLite logical export does not match source logical revision: {root}")
        with bind_source_input(root) as source_binding:
            if (
                _logical_export_digest_bound(
                    root,
                    scope=member_export_scope(source_binding.source_path),
                    parent_anchor=source_binding.parent_anchor,
                    source_binding=source_binding,
                    expected_identity=(binding.root_identity.device, binding.root_identity.inode),
                )
                != baseline[0].identity
            ):
                raise SourceMutationError(f"SQLite source changed during logical export: {root}")
        size = destination.stat().st_size
        return SourceSnapshotResult(
            tuple(
                CutItem(item.source_id, item.coordinate, item.identity, digest, size, str(destination))
                for item in baseline
            ),
            binding,
        )


class _SpoolHandoffStrategy(_FilesystemStrategy):
    """Atomically give writers a new spool generation before copying the old one."""

    def snapshot(
        self, binding: SourceCutBinding, destination: Path, baseline: tuple[CutItem, ...]
    ) -> SourceSnapshotResult:
        root = Path(binding.source.root)
        if not root.is_dir():
            raise SourceSnapshotError("spool handoff requires a directory root")
        retired = _retired_spool_root(binding)
        if retired.exists():
            raise SourceSnapshotError(f"stale spool handoff generation exists: {retired}")
        os.replace(root, retired)
        root.mkdir(mode=0o700)
        active_binding = SourceCutBinding(binding.source, _root_identity(root), binding.policy)
        _fsync_directory(root.parent)
        retired_binding = SourceCutBinding(
            SourceDeclaration(
                binding.source.source_id,
                binding.source.role,
                retired,
                binding.source.mutable,
                binding.source.layout_name,
                binding.source.exclude_coordinates,
            ),
            _root_identity(retired),
            binding.policy,
        )
        try:
            copied = _copy_candidates(retired_binding, baseline, destination)
        except BaseException:
            # The old generation is still the only copy of pre-cut spool
            # material. Keep it for recovery if candidate copying fails.
            raise
        return SourceSnapshotResult(copied, active_binding)


def _default_policy(role: SourceRole) -> SourceCutPolicy:
    if role is SourceRole.IMMUTABLE_EXPORT:
        return SourceCutPolicy(SnapshotMode.IMMUTABLE_EXPORT)
    if role is SourceRole.ARCHIVE_MEMBER:
        return SourceCutPolicy(SnapshotMode.ARCHIVE_MEMBER)
    if role is SourceRole.MUTABLE_SQLITE:
        return SourceCutPolicy(SnapshotMode.SQLITE_LOGICAL_EXPORT)
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
    SourceRole.MUTABLE_SQLITE: frozenset({SnapshotMode.SQLITE_LOGICAL_EXPORT}),
    SourceRole.SPOOL: frozenset({SnapshotMode.SPOOL_HANDOFF, SnapshotMode.DIRECTORY_COPY}),
    SourceRole.QUEUE: frozenset({SnapshotMode.SPOOL_HANDOFF, SnapshotMode.DIRECTORY_COPY}),
    SourceRole.ATTACHMENT: frozenset({SnapshotMode.DIRECTORY_COPY, SnapshotMode.COMPLETE_COPY}),
    SourceRole.SIDECAR: frozenset({SnapshotMode.DIRECTORY_COPY, SnapshotMode.COMPLETE_COPY}),
    SourceRole.PROVIDER_CACHE: frozenset({SnapshotMode.DIRECTORY_COPY, SnapshotMode.COMPLETE_COPY}),
    SourceRole.DIRECTORY: frozenset({SnapshotMode.DIRECTORY_COPY, SnapshotMode.COMPLETE_COPY}),
}


def _strategy(policy: SourceCutPolicy) -> SourceSnapshotStrategy:
    if policy.mode is SnapshotMode.SQLITE_LOGICAL_EXPORT:
        return _SQLiteLogicalExportStrategy(policy.mode)
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


def _manifest_digest(kind: str, items: Iterable[CutItem], *, version: int = _MANIFEST_VERSION) -> str:
    return _sha256(
        {
            "version": version,
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
                    *(() if version == 1 else (item.post_cut_arrival,)),
                )
                for item in sorted(items, key=lambda value: value.key)
            ],
        }
    )


def _manifest(kind: str, items: Iterable[CutItem], *, version: int = _MANIFEST_VERSION) -> CutManifest:
    ordered = tuple(sorted(items, key=lambda value: value.key))
    return CutManifest(
        kind,
        ordered,
        len(ordered),
        sum(item.size_bytes for item in ordered),
        _manifest_digest(kind, ordered, version=version),
        version,
    )


def _ownership_key(item: CutItem, *, mode: SnapshotMode) -> tuple[str, str, str, str] | tuple[str, str, str]:
    """Return the logical version key for a candidate/post-cut comparison."""
    if mode is SnapshotMode.SQLITE_LOGICAL_EXPORT:
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


def _retired_spool_root(binding: SourceCutBinding) -> Path:
    root = Path(binding.source.root)
    return root.with_name(f".{root.name}.{binding.source.source_id}.cut")


def _recover_markerless_spool_handoffs(preflight: SourceCutPreflight) -> None:
    """Restore pre-cut spool roots before reclaiming incomplete output."""
    for binding in preflight.bindings:
        if binding.policy.mode is not SnapshotMode.SPOOL_HANDOFF:
            continue
        retired = _retired_spool_root(binding)
        if not retired.exists():
            continue
        root = Path(binding.source.root)
        arrivals = root.with_name(f".{root.name}.{binding.source.source_id}.arrivals")
        if arrivals.exists():
            raise SourceSnapshotError(f"spool recovery arrivals path already exists: {arrivals}")
        if root.exists():
            os.replace(root, arrivals)
        os.replace(retired, root)
        if arrivals.exists():
            os.replace(arrivals, root / arrivals.name)
            _fsync_directory(root)
        _fsync_directory(root.parent)


def _retire_published_spool_handoffs(preflight: SourceCutPreflight) -> None:
    """Release retired spool roots only after the candidate is durable."""
    for binding in preflight.bindings:
        if binding.policy.mode is not SnapshotMode.SPOOL_HANDOFF:
            continue
        retired = _retired_spool_root(binding)
        if retired.exists():
            shutil.rmtree(retired)
            _fsync_directory(retired.parent)


def _preflight_copy_capacity(
    preflight: SourceCutPreflight,
    baselines: Mapping[str, tuple[CutItem, ...]],
    staging_parent: Path,
) -> None:
    """Reject daemon-bounded fallback copies before any candidate bytes publish."""
    capacity_limits = [
        binding.policy.capacity_bytes for binding in preflight.bindings if binding.policy.capacity_bytes is not None
    ]
    if not capacity_limits:
        return
    required_bytes = sum(item.size_bytes for items in baselines.values() for item in items)
    capacity = min(capacity_limits)
    if required_bytes > capacity:
        raise SourceSnapshotError(
            f"capacity preflight rejects source cut: requires {required_bytes} bytes, policy permits {capacity}"
        )
    available_bytes = shutil.disk_usage(staging_parent).free
    if required_bytes > available_bytes:
        raise SourceSnapshotError(
            f"capacity preflight rejects source cut: requires {required_bytes} bytes, only {available_bytes} available"
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
    physical_destination = destination.resolve()
    physical_parent = destination.parent.resolve()
    for binding in preflight.bindings:
        source_root = Path(binding.source.root).resolve()
        if physical_destination == source_root or (
            binding.root_identity.kind == "directory"
            and (physical_parent.is_relative_to(source_root) or source_root.is_relative_to(physical_destination))
        ):
            raise SourceSnapshotError(f"candidate destination overlaps source root: {binding.source.source_id}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reclaim_orphaned_staging(destination.parent, preflight.request_id)
    if not destination.exists():
        # A crash can land after the active spool was renamed but before the
        # candidate directory was published. Restore the retired generation
        # before root verification so the same preflight remains retryable.
        _recover_markerless_spool_handoffs(preflight)
    if destination.exists():
        try:
            preflight.verify_roots(allow_handed_off_spools=True)
            return _load_published_source_cut(destination, preflight)
        except FileNotFoundError:
            # A moved spool's retired root remains the only authoritative
            # pre-cut population until this destination's marker is durable.
            _recover_markerless_spool_handoffs(preflight)
            preflight.verify_roots()
            shutil.rmtree(destination)
    preflight.verify_roots()
    staging = Path(tempfile.mkdtemp(prefix=f".{preflight.request_id}.", dir=destination.parent))
    try:
        _write_durable(staging / _STAGING_OWNER_MARKER, f"{preflight.request_id}\n")
        baselines = {binding.source.source_id: _observe(binding) for binding in preflight.bindings}
        _preflight_copy_capacity(preflight, baselines, staging.parent)
        candidate_items: list[CutItem] = []
        observation_bindings: list[SourceCutBinding] = []
        for binding in preflight.bindings:
            check_compute_cancelled()
            source_destination = staging / "candidate" / binding.source.source_id
            if binding.policy.mode is SnapshotMode.SQLITE_LOGICAL_EXPORT:
                source_destination = source_destination.with_name(source_destination.name + ".jsonl")
            elif binding.policy.mode is SnapshotMode.ARCHIVE_MEMBER or not Path(binding.source.root).is_dir():
                source_destination = source_destination.with_name(
                    source_destination.name + Path(binding.source.root).suffix
                )
            snapshot = _strategy(binding.policy).snapshot(
                binding, source_destination, baselines[binding.source.source_id]
            )
            candidate_items.extend(snapshot.candidate_items)
            observation_bindings.append(snapshot.observation_binding)
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
        post_items = [item for binding in observation_bindings for item in _observe(binding)]
        modes = {binding.source.source_id: binding.policy.mode for binding in preflight.bindings}
        candidate_keys = {_ownership_key(item, mode=modes[item.source_id]) for item in candidate_items}
        baseline_coordinates = {(item.source_id, item.coordinate) for items in baselines.values() for item in items}
        carry_items: list[CutItem] = []
        for item in post_items:
            check_compute_cancelled()
            mode = modes[item.source_id]
            if (
                mode in {SnapshotMode.COMPLETE_COPY, SnapshotMode.DIRECTORY_COPY}
                and (
                    item.source_id,
                    item.coordinate,
                )
                in baseline_coordinates
            ):
                # Complete copies define the cut boundary for live JSONL. The
                # active path remains a normal, idempotent future read rather
                # than a second logically-owned byte population.
                carry_items.append(replace(item, readmission=True))
            elif _ownership_key(item, mode=mode) not in candidate_keys:
                carry_items.append(
                    replace(item, post_cut_arrival=(item.source_id, item.coordinate) not in baseline_coordinates)
                )
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
        # The owner marker only licenses reclaiming a crashed staging
        # directory; the published cut does not carry it.
        (staging / _STAGING_OWNER_MARKER).unlink()
        _fsync_tree(staging)
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
        _write_durable(destination / _COMPLETE_MARKER, f"{cut_identity}\n")
        _fsync_directory(destination)
        _retire_published_spool_handoffs(preflight)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


#: Written first into every staging directory this module creates. A name that
#: merely shares the request prefix proves no ownership, so reclamation
#: requires this marker naming the same request.
_STAGING_OWNER_MARKER = ".source-cut-staging-owner"


def _reclaim_orphaned_staging(parent: Path, request_id: str) -> None:
    prefix = f".{request_id}."
    for path in parent.iterdir():
        if not path.name.startswith(prefix) or path.is_symlink() or not path.is_dir():
            continue
        try:
            owner = (path / _STAGING_OWNER_MARKER).read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        if owner == f"{request_id}\n":
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
    modes = dict(result.ownership_modes)
    # Keep each central directory open once, rather than reparsing a ZIP for
    # every member. The manifest names member bytes, not container bytes, and
    # each member is selected by its captured physical header offset.
    archives: dict[Path, zipfile.ZipFile] = {}
    with ExitStack() as stack:
        for item in result.candidate_manifest.items:
            check_compute_cancelled()
            if source_id is not None and item.source_id != source_id:
                continue
            if selected is not None and item.coordinate not in selected:
                continue
            if item.snapshot_path is None:
                raise CandidateCohortError(f"candidate item has no immutable snapshot: {item.coordinate}")
            path = result.candidate_root / item.snapshot_path
            if not path.is_relative_to(result.candidate_root):
                raise CandidateCohortError(f"candidate path escapes published snapshot: {item.coordinate}")
            if modes[item.source_id] is SnapshotMode.ARCHIVE_MEMBER:
                if "!" not in item.coordinate:
                    raise CandidateCohortError(f"candidate member has no archive coordinate: {item.coordinate}")
                try:
                    if path not in archives:
                        archives[path] = stack.enter_context(zipfile.ZipFile(path))
                        archives[path].infolist().sort(key=lambda entry: entry.header_offset)
                    info = _archive_member_info(archives[path], item)
                    digest = hashlib.sha256()
                    size = 0
                    with archives[path].open(info) as stream:
                        while chunk := stream.read(1024 * 1024):
                            check_compute_cancelled()
                            digest.update(chunk)
                            size += len(chunk)
                    unchanged = size == item.size_bytes and digest.hexdigest() == item.content_sha256
                except DaemonOperationCancelled:
                    raise
                except (OSError, zipfile.BadZipFile, KeyError, RuntimeError) as exc:
                    raise SourceMutationError(f"candidate archive mutated: {item.coordinate}") from exc
            else:
                unchanged = (
                    path.is_file()
                    and path.stat().st_size == item.size_bytes
                    and _sha256_path(path) == item.content_sha256
                )
            if not unchanged:
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
    "SourceSnapshotResult",
    "execute_source_cut",
    "bind_source_observation",
    "iter_observe_source_members",
    "load_source_cut",
    "preflight_source_cut",
    "observe_source_members",
    "reacquire_candidate",
]


load_source_cut = _load_published_source_cut

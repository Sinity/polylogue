"""Typed identity for one split-tier archive file set."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from polylogue.version import VERSION_INFO

ArchiveTierName = Literal["source", "index", "embeddings", "user", "ops"]
TIER_FILENAMES: tuple[tuple[ArchiveTierName, str], ...] = (
    ("source", "source.db"),
    ("index", "index.db"),
    ("embeddings", "embeddings.db"),
    ("user", "user.db"),
    ("ops", "ops.db"),
)


class ArchiveLocationError(RuntimeError):
    """A configured archive does not name one coherent active file set."""


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
        if pointer_file.exists():
            raw = pointer_file.read_text(encoding="utf-8").strip()
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

"""Typed identity for one split-tier archive file set."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ArchiveTierName = Literal["source", "index", "embeddings", "user", "ops"]
TIER_FILENAMES: tuple[tuple[ArchiveTierName, str], ...] = (
    ("source", "source.db"),
    ("index", "index.db"),
    ("embeddings", "embeddings.db"),
    ("user", "user.db"),
    ("ops", "ops.db"),
)


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

    def tier(self, name: ArchiveTierName) -> TierFileIdentity:
        return next(tier for tier in self.tiers if tier.name == name)

    @property
    def durable_id(self) -> str:
        return "|".join((self.tier("source").stable_id, self.tier("user").stable_id))

    def conflicts_with(self, other: ArchiveIdentity) -> bool:
        shared_durable = self.tier("source").same_file(other.tier("source")) and self.tier("user").same_file(
            other.tier("user")
        )
        distinct_indexes = not self.tier("index").same_file(other.tier("index"))
        return shared_durable and distinct_indexes and self.tier("index").exists and other.tier("index").exists

    def as_dict(self) -> dict[str, object]:
        return {
            "configured_root": str(self.configured_root),
            "durable_id": self.durable_id,
            "active_generation": self.active_generation,
            "generation_owner": self.generation_owner,
            "generation_state": self.generation_state,
            "process_id": os.getpid(),
            "executable": os.path.realpath(sys.executable),
            "unit": os.environ.get("SYSTEMD_UNIT") or os.environ.get("INVOCATION_ID"),
            "tiers": [tier.as_dict() for tier in self.tiers],
        }


class ArchiveIdentityConflictError(RuntimeError):
    """A durable archive is paired with two distinct writable indexes."""


def archive_identity_conflicts(*, configured_root: Path, active_root: Path) -> tuple[ArchiveIdentity, ...]:
    configured = ArchiveIdentity.resolve(configured_root)
    active = ArchiveIdentity.resolve(active_root)
    if configured_root.resolve(strict=False) == active_root.resolve(strict=False):
        return ()
    return (configured,) if configured.conflicts_with(active) else ()


def assert_writable_archive_identity(*, configured_root: Path, active_root: Path) -> ArchiveIdentity:
    """Fail before bootstrap when one durable archive has two active indexes."""
    active = ArchiveIdentity.resolve(active_root)
    conflicts = archive_identity_conflicts(configured_root=configured_root, active_root=active_root)
    if conflicts:
        configured = conflicts[0]
        raise ArchiveIdentityConflictError(
            "archive identity conflict: durable tiers are shared but writable index generations differ; "
            f"configured_index={configured.tier('index').resolved_path} "
            f"active_index={active.tier('index').resolved_path}"
        )
    return active

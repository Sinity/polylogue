"""Typed, inactive whole-archive tuple allocation.

``ArchiveLocation`` answers which archive is active.  This module answers a
different, narrower question: where a complete replacement tuple may be
written before a caller is allowed to promote it.  The two questions must not
be answered by deriving siblings from an arbitrary database path.

The allocator intentionally reserves paths only.  It never opens SQLite,
changes an active pointer, or links durable authority into a candidate.  A
writer must present the returned :class:`InactiveTierDestination` back to the
validation seam immediately before opening its database.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

ARCHIVE_TUPLE_MANIFEST_VERSION = 1
ARCHIVE_TUPLES_DIRNAME = ".archive-tuples"
ARCHIVE_TUPLE_MANIFEST_FILENAME = "tuple.json"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TUPLE_ID = re.compile(r"^tuple-[0-9]+-[0-9a-f]{16}$")
_CANDIDATE_TIERS: tuple[ArchiveTier, ...] = (
    ArchiveTier.SOURCE,
    ArchiveTier.INDEX,
    ArchiveTier.EMBEDDINGS,
)
_TIER_FILENAMES = {tier: f"{tier.value}.db" for tier in ArchiveTier}


class ArchiveTupleError(RuntimeError):
    """Base class for malformed, foreign, stale, or unsafe tuple state."""


class ArchiveTupleCollisionError(ArchiveTupleError):
    """An allocation name was already claimed."""


class ArchiveTuplePathError(ArchiveTupleError):
    """A destination is not an archive-owned candidate path."""


class ArchiveTupleActiveError(ArchiveTupleError):
    """A writer attempted to use an active tier as an inactive destination."""


class ArchiveTupleForeignError(ArchiveTupleError):
    """A destination belongs to another archive or tuple."""


class ArchiveTupleStaleError(ArchiveTupleError):
    """The active tuple changed after a candidate was allocated."""


class ArchiveTupleState(StrEnum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    RETIRED = "retired"


class DisposableOpsPolicy(StrEnum):
    """The only supported policy for ops in an inactive tuple."""

    FRESH = "fresh"


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _absolute(path: Path) -> Path:
    return path.absolute()


def _assert_no_symlink_ancestry(path: Path, *, label: str) -> Path:
    """Return a lexical absolute path after rejecting existing symlink parts."""

    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ArchiveTuplePathError(f"cannot inspect {label}: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ArchiveTuplePathError(f"{label} contains a symlink component: {current}")
    return absolute


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _require_safe_id(value: str, label: str) -> str:
    if _SAFE_ID.fullmatch(value) is None:
        raise ArchiveTupleError(f"invalid {label} identity")
    return value


def _schema_fingerprints() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                tier.value,
                _digest({"version": ARCHIVE_VERSION_BY_TIER[tier], "ddl": ARCHIVE_DDL_BY_TIER[tier]}),
            )
            for tier in ArchiveTier
        )
    )


@dataclass(frozen=True, slots=True)
class ArchiveTupleManifest:
    """Versioned authority record for one whole-archive replacement tuple."""

    manifest_version: int
    tuple_id: str
    owner_id: str
    archive_root: str
    archive_identity_digest: str
    source_generation: str
    index_generation: str
    embeddings_generation: str
    user_identity: str
    audit_identity: str
    ops_policy: str
    schema_fingerprints: tuple[tuple[str, str], ...]
    semantic_fingerprints: tuple[tuple[str, str], ...]
    expected_seals: tuple[tuple[str, str], ...]
    state: str = ArchiveTupleState.INACTIVE.value
    created_at_ns: int = 0

    @classmethod
    def for_location(
        cls,
        location: ArchiveLocation,
        *,
        tuple_id: str,
        owner_id: str,
        source_generation: str,
        index_generation: str,
        embeddings_generation: str,
        semantic_fingerprints: dict[str, str] | None = None,
        expected_seals: dict[str, str] | None = None,
        ops_policy: str = DisposableOpsPolicy.FRESH.value,
        created_at_ns: int | None = None,
    ) -> ArchiveTupleManifest:
        _require_safe_id(tuple_id, "tuple")
        _require_safe_id(owner_id, "owner")
        for label, value in (
            ("source generation", source_generation),
            ("index generation", index_generation),
            ("embeddings generation", embeddings_generation),
        ):
            _require_safe_id(value, label)
        if ops_policy != DisposableOpsPolicy.FRESH.value:
            raise ArchiveTupleError(f"unsupported inactive ops policy: {ops_policy}")
        identity = ArchiveIdentity.resolve_location(location)
        stable_seals = {
            "user": identity.tier("user").stable_id,
            "audit": identity.tier("audit").stable_id,
        }
        stable_seals.update(expected_seals or {})
        semantic = semantic_fingerprints or {
            "archive-tuple": _digest(
                {
                    "manifest_version": ARCHIVE_TUPLE_MANIFEST_VERSION,
                    "candidate_tiers": [tier.value for tier in _CANDIDATE_TIERS],
                    "active_identity": identity.authority_identity_digest,
                }
            )
        }
        return cls(
            manifest_version=ARCHIVE_TUPLE_MANIFEST_VERSION,
            tuple_id=tuple_id,
            owner_id=owner_id,
            archive_root=str(location.configured_root),
            archive_identity_digest=identity.authority_identity_digest,
            source_generation=source_generation,
            index_generation=index_generation,
            embeddings_generation=embeddings_generation,
            user_identity=identity.tier("user").stable_id,
            audit_identity=identity.tier("audit").stable_id,
            ops_policy=ops_policy,
            schema_fingerprints=_schema_fingerprints(),
            semantic_fingerprints=tuple(sorted((str(k), str(v)) for k, v in semantic.items())),
            expected_seals=tuple(sorted((str(k), str(v)) for k, v in stable_seals.items())),
            created_at_ns=created_at_ns if created_at_ns is not None else time.time_ns(),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "manifest_version": self.manifest_version,
            "tuple_id": self.tuple_id,
            "owner_id": self.owner_id,
            "archive_root": self.archive_root,
            "archive_identity_digest": self.archive_identity_digest,
            "generations": {
                "source": self.source_generation,
                "index": self.index_generation,
                "embeddings": self.embeddings_generation,
            },
            "stable_identities": {"user": self.user_identity, "audit": self.audit_identity},
            "ops_policy": self.ops_policy,
            "schema_fingerprints": dict(self.schema_fingerprints),
            "semantic_fingerprints": dict(self.semantic_fingerprints),
            "expected_seals": dict(self.expected_seals),
            "state": self.state,
            "created_at_ns": self.created_at_ns,
        }

    @property
    def manifest_digest(self) -> str:
        return _digest(self._payload())

    @property
    def seal(self) -> str:
        """The content seal written beside the manifest digest."""

        return self.manifest_digest

    def as_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["manifest_digest"] = self.manifest_digest
        payload["manifest_seal"] = self.seal
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ArchiveTupleManifest:
        payload_any = cast(dict[str, Any], payload)
        try:
            generations = payload_any["generations"]
            stable = payload_any["stable_identities"]
            if not isinstance(generations, dict) or not isinstance(stable, dict):
                raise TypeError("manifest nested fields must be objects")
            result = cls(
                manifest_version=int(payload_any["manifest_version"]),
                tuple_id=str(payload_any["tuple_id"]),
                owner_id=str(payload_any["owner_id"]),
                archive_root=str(payload_any["archive_root"]),
                archive_identity_digest=str(payload_any["archive_identity_digest"]),
                source_generation=str(generations["source"]),
                index_generation=str(generations["index"]),
                embeddings_generation=str(generations["embeddings"]),
                user_identity=str(stable["user"]),
                audit_identity=str(stable["audit"]),
                ops_policy=str(payload_any["ops_policy"]),
                schema_fingerprints=tuple(
                    sorted((str(k), str(v)) for k, v in dict(payload_any["schema_fingerprints"]).items())
                ),
                semantic_fingerprints=tuple(
                    sorted((str(k), str(v)) for k, v in dict(payload_any["semantic_fingerprints"]).items())
                ),
                expected_seals=tuple(sorted((str(k), str(v)) for k, v in dict(payload_any["expected_seals"]).items())),
                state=str(payload_any.get("state", ArchiveTupleState.INACTIVE.value)),
                created_at_ns=int(payload_any["created_at_ns"]),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ArchiveTupleError("invalid archive tuple manifest") from exc
        _validate_manifest_shape(result)
        if (
            str(payload.get("manifest_digest")) != result.manifest_digest
            or str(payload.get("manifest_seal")) != result.seal
        ):
            raise ArchiveTupleError("archive tuple manifest seal mismatch")
        return result


def _validate_manifest_shape(manifest: ArchiveTupleManifest) -> None:
    if manifest.manifest_version != ARCHIVE_TUPLE_MANIFEST_VERSION:
        raise ArchiveTupleError(f"unsupported archive tuple manifest version: {manifest.manifest_version}")
    _require_safe_id(manifest.tuple_id, "tuple")
    _require_safe_id(manifest.owner_id, "owner")
    for label, value in (
        ("source generation", manifest.source_generation),
        ("index generation", manifest.index_generation),
        ("embeddings generation", manifest.embeddings_generation),
    ):
        _require_safe_id(value, label)
    if manifest.state not in {state.value for state in ArchiveTupleState}:
        raise ArchiveTupleError(f"invalid archive tuple state: {manifest.state}")
    if manifest.ops_policy != DisposableOpsPolicy.FRESH.value:
        raise ArchiveTupleError(f"invalid archive tuple ops policy: {manifest.ops_policy}")
    if manifest.created_at_ns <= 0:
        raise ArchiveTupleError("archive tuple manifest has no creation timestamp")
    if not manifest.archive_root or not manifest.archive_identity_digest:
        raise ArchiveTupleError("archive tuple manifest has no archive identity")
    if set(dict(manifest.schema_fingerprints)) != {tier.value for tier in ArchiveTier}:
        raise ArchiveTupleError("archive tuple manifest schema inventory is incomplete")
    if (
        dict(manifest.expected_seals).get("user") != manifest.user_identity
        or dict(manifest.expected_seals).get("audit") != manifest.audit_identity
    ):
        raise ArchiveTupleError("archive tuple stable-tier seals are inconsistent")


@dataclass(frozen=True, slots=True)
class InactiveTierDestination:
    """A capability-like, manifest-bound path for one replaceable tier."""

    tuple_id: str
    owner_id: str
    tier: ArchiveTier
    generation_id: str
    path: Path
    candidate_root: Path
    manifest_digest: str

    @property
    def archive_root(self) -> Path:
        return self.candidate_root.parent.parent


@dataclass(frozen=True, slots=True)
class ArchiveTupleLocation:
    """One reserved inactive tuple and its three writer destinations."""

    manifest: ArchiveTupleManifest
    candidate_root: Path
    source: InactiveTierDestination
    index: InactiveTierDestination
    embeddings: InactiveTierDestination
    manifest_path: Path

    @property
    def tuple_id(self) -> str:
        return self.manifest.tuple_id

    @property
    def owner_id(self) -> str:
        return self.manifest.owner_id

    @property
    def destinations(self) -> tuple[InactiveTierDestination, ...]:
        return (self.source, self.index, self.embeddings)

    def destination(self, tier: ArchiveTier) -> InactiveTierDestination:
        if tier is ArchiveTier.SOURCE:
            return self.source
        if tier is ArchiveTier.INDEX:
            return self.index
        if tier is ArchiveTier.EMBEDDINGS:
            return self.embeddings
        raise ArchiveTupleError(f"tier {tier.value} has no inactive tuple destination")

    def validate(self, location: ArchiveLocation, *, expected_tier: ArchiveTier | None = None) -> None:
        validate_inactive_tuple(self, location, expected_tier=expected_tier)


def _candidate_root_for(location: ArchiveLocation) -> Path:
    return _assert_no_symlink_ancestry(
        location.configured_root / ARCHIVE_TUPLES_DIRNAME,
        label="archive tuple root",
    )


def _destination_for(
    manifest: ArchiveTupleManifest, candidate_root: Path, tier: ArchiveTier
) -> InactiveTierDestination:
    generation = {
        ArchiveTier.SOURCE: manifest.source_generation,
        ArchiveTier.INDEX: manifest.index_generation,
        ArchiveTier.EMBEDDINGS: manifest.embeddings_generation,
    }[tier]
    return InactiveTierDestination(
        tuple_id=manifest.tuple_id,
        owner_id=manifest.owner_id,
        tier=tier,
        generation_id=generation,
        path=candidate_root / _TIER_FILENAMES[tier],
        candidate_root=candidate_root,
        manifest_digest=manifest.manifest_digest,
    )


def _make_location(manifest: ArchiveTupleManifest, candidate_root: Path) -> ArchiveTupleLocation:
    return ArchiveTupleLocation(
        manifest=manifest,
        candidate_root=candidate_root,
        source=_destination_for(manifest, candidate_root, ArchiveTier.SOURCE),
        index=_destination_for(manifest, candidate_root, ArchiveTier.INDEX),
        embeddings=_destination_for(manifest, candidate_root, ArchiveTier.EMBEDDINGS),
        manifest_path=candidate_root / ARCHIVE_TUPLE_MANIFEST_FILENAME,
    )


def _atomic_manifest_write(path: Path, manifest: ArchiveTupleManifest) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(temporary, flags, 0o600)
    except OSError as exc:
        raise ArchiveTuplePathError(f"cannot reserve tuple manifest: {path}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(manifest.as_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if path.is_symlink():
            raise ArchiveTuplePathError(f"tuple manifest became a symlink: {path}")
        _fsync_directory(path.parent)
    except BaseException:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise


def validate_inactive_destination(
    destination: InactiveTierDestination,
    location: ArchiveLocation,
    *,
    path: Path | None = None,
    expected_tier: ArchiveTier | None = None,
    expected_generation: str | None = None,
) -> None:
    """Validate a typed candidate before any SQLite connection is opened."""

    candidate = _absolute(destination.candidate_root)
    archive_root = _absolute(location.configured_root)
    if candidate.parent != archive_root / ARCHIVE_TUPLES_DIRNAME:
        raise ArchiveTupleForeignError("inactive destination belongs to a foreign archive tuple root")
    if destination.tuple_id != candidate.name or _TUPLE_ID.fullmatch(destination.tuple_id) is None:
        raise ArchiveTuplePathError("inactive destination has an invalid tuple identity")
    _assert_no_symlink_ancestry(archive_root, label="archive root")
    _assert_no_symlink_ancestry(candidate, label="inactive tuple")
    if candidate.is_symlink() or not candidate.is_dir():
        raise ArchiveTuplePathError("inactive tuple root is not an owned directory")
    expected_path = candidate / _TIER_FILENAMES.get(destination.tier, "")
    if destination.tier not in _CANDIDATE_TIERS or _absolute(destination.path) != expected_path:
        raise ArchiveTuplePathError("inactive destination is not the canonical replaceable tier path")
    if expected_tier is not None and destination.tier is not expected_tier:
        raise ArchiveTupleForeignError("inactive destination tier does not match the writer")
    if expected_generation is not None and destination.generation_id != expected_generation:
        raise ArchiveTupleStaleError("inactive destination generation does not match the writer")
    active = location.active_tier(destination.tier.value)
    if destination.path == active.configured_path or destination.path.resolve(strict=False) == active.resolved_path:
        raise ArchiveTupleActiveError("inactive destination is the active tier")
    if path is not None and _absolute(path) != expected_path:
        raise ArchiveTuplePathError("writer path does not match its typed inactive destination")
    manifest_path = candidate / ARCHIVE_TUPLE_MANIFEST_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ArchiveTuplePathError("inactive tuple manifest is missing or unsafe")
    try:
        manifest = ArchiveTupleManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ArchiveTupleError("inactive tuple manifest cannot be read") from exc
    if manifest.tuple_id != destination.tuple_id or manifest.owner_id != destination.owner_id:
        raise ArchiveTupleForeignError("inactive destination owner or tuple identity mismatch")
    if manifest.manifest_digest != destination.manifest_digest:
        raise ArchiveTupleForeignError("inactive destination manifest digest mismatch")
    current_identity = ArchiveIdentity.resolve_location(location)
    if manifest.archive_identity_digest != current_identity.authority_identity_digest:
        raise ArchiveTupleStaleError("active archive generation changed after tuple allocation")
    if manifest.state != ArchiveTupleState.INACTIVE.value:
        raise ArchiveTupleActiveError("inactive destination is no longer inactive")


def validate_inactive_tuple(
    tuple_location: ArchiveTupleLocation,
    location: ArchiveLocation,
    *,
    expected_tier: ArchiveTier | None = None,
) -> None:
    """Validate the complete tuple and every candidate path before opening SQLite."""

    manifest = tuple_location.manifest
    _validate_manifest_shape(manifest)
    if _absolute(Path(manifest.archive_root)) != _absolute(location.configured_root):
        raise ArchiveTupleForeignError("tuple manifest archive root mismatch")
    if tuple_location.candidate_root != _absolute(tuple_location.candidate_root):
        raise ArchiveTuplePathError("tuple candidate root must be absolute")
    if tuple_location.manifest_path != tuple_location.candidate_root / ARCHIVE_TUPLE_MANIFEST_FILENAME:
        raise ArchiveTuplePathError("tuple manifest path is not canonical")
    for destination in tuple_location.destinations:
        validate_inactive_destination(destination, location, expected_tier=expected_tier)


class ArchiveTupleAllocator:
    """Reserve collision-safe inactive tuples from an existing ArchiveLocation."""

    def __init__(self, location: ArchiveLocation) -> None:
        self.location = location
        self.archive_root = _absolute(location.configured_root)
        self.tuples_root = _candidate_root_for(location)
        if self.tuples_root.exists() and (self.tuples_root.is_symlink() or not self.tuples_root.is_dir()):
            raise ArchiveTuplePathError("archive tuple root is not an owned directory")
        self.tuples_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        _assert_no_symlink_ancestry(self.tuples_root, label="archive tuple root")

    @classmethod
    def for_archive_root(cls, archive_root: Path) -> ArchiveTupleAllocator:
        return cls(ArchiveLocation.resolve(archive_root))

    def allocate(
        self,
        *,
        owner_id: str,
        source_generation: str | None = None,
        index_generation: str | None = None,
        embeddings_generation: str | None = None,
        semantic_fingerprints: dict[str, str] | None = None,
        expected_seals: dict[str, str] | None = None,
        allocation_id_factory: Any | None = None,
        max_attempts: int = 32,
    ) -> ArchiveTupleLocation:
        """Reserve one complete inactive tuple without touching active state."""

        _require_safe_id(owner_id, "owner")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        factory = allocation_id_factory or (lambda: f"tuple-{time.time_ns()}-{uuid.uuid4().hex[:16]}")
        for _attempt in range(max_attempts):
            tuple_id = str(factory())
            if _TUPLE_ID.fullmatch(tuple_id) is None:
                raise ArchiveTupleError("allocation factory returned an invalid tuple identity")
            candidate_root = self.tuples_root / tuple_id
            try:
                candidate_root.mkdir(mode=0o700, exist_ok=False)
            except FileExistsError:
                continue
            try:
                manifest = ArchiveTupleManifest.for_location(
                    self.location,
                    tuple_id=tuple_id,
                    owner_id=owner_id,
                    source_generation=source_generation or f"source-{uuid.uuid4().hex}",
                    index_generation=index_generation or f"index-{uuid.uuid4().hex}",
                    embeddings_generation=embeddings_generation or f"embeddings-{uuid.uuid4().hex}",
                    semantic_fingerprints=semantic_fingerprints,
                    expected_seals=expected_seals,
                )
                location = _make_location(manifest, candidate_root.absolute())
                _atomic_manifest_write(location.manifest_path, manifest)
                validate_inactive_tuple(location, self.location)
                return location
            except BaseException:
                shutil.rmtree(candidate_root, ignore_errors=False)
                raise
        raise ArchiveTupleCollisionError("could not reserve a collision-free inactive tuple")

    def load(self, tuple_id: str) -> ArchiveTupleLocation:
        _require_safe_id(tuple_id, "tuple")
        if _TUPLE_ID.fullmatch(tuple_id) is None:
            raise ArchiveTupleError("invalid tuple identity")
        candidate_root = self.tuples_root / tuple_id
        _assert_no_symlink_ancestry(candidate_root, label="inactive tuple")
        manifest_path = candidate_root / ARCHIVE_TUPLE_MANIFEST_FILENAME
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ArchiveTupleError(f"cannot load archive tuple: {tuple_id}") from exc
        manifest = ArchiveTupleManifest.from_dict(payload)
        if manifest.tuple_id != tuple_id:
            raise ArchiveTupleForeignError("tuple manifest identity does not match its directory")
        location = _make_location(manifest, candidate_root.absolute())
        validate_inactive_tuple(location, self.location)
        return location


def allocate_inactive_archive_tuple(
    location: ArchiveLocation,
    *,
    owner_id: str,
    **kwargs: object,
) -> ArchiveTupleLocation:
    """Convenience entry point that does not perform a second resolution."""

    return ArchiveTupleAllocator(location).allocate(owner_id=owner_id, **cast(Any, kwargs))


def is_archive_tuple_candidate_path(path: Path) -> bool:
    """Whether a path is lexically below an archive tuple reservation root."""

    absolute = _absolute(path)
    return ARCHIVE_TUPLES_DIRNAME in absolute.parts


__all__ = [
    "ARCHIVE_TUPLE_MANIFEST_FILENAME",
    "ARCHIVE_TUPLE_MANIFEST_VERSION",
    "ARCHIVE_TUPLES_DIRNAME",
    "ArchiveTupleActiveError",
    "ArchiveTupleAllocator",
    "ArchiveTupleCollisionError",
    "ArchiveTupleError",
    "ArchiveTupleForeignError",
    "ArchiveTupleLocation",
    "ArchiveTupleManifest",
    "ArchiveTuplePathError",
    "ArchiveTupleState",
    "ArchiveTupleStaleError",
    "DisposableOpsPolicy",
    "InactiveTierDestination",
    "allocate_inactive_archive_tuple",
    "is_archive_tuple_candidate_path",
    "validate_inactive_destination",
    "validate_inactive_tuple",
]

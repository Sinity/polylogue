"""Member-level source continuity checks used by source-authority coordinators."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class SourceContinuityError(ValueError):
    """A source declaration, manifest, or continuity receipt is unsafe."""


class SourceRole(StrEnum):
    IMMUTABLE_EXPORT = "immutable-export"
    ARCHIVE_MEMBER = "archive-member"
    APPEND_JSONL = "append-jsonl"
    REWRITE_JSONL = "rewrite-leading-jsonl"
    MUTABLE_SQLITE = "mutable-sqlite"
    SPOOL = "spool"
    QUEUE = "queue"
    ATTACHMENT = "attachment"
    SIDECAR = "sidecar"
    PROVIDER_CACHE = "provider-cache"
    DIRECTORY = "directory"


class FrontierState(StrEnum):
    """Observation state for one configured source root."""

    PRESENT = "present"
    VALID_EMPTY = "valid-empty"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class SourceDeclaration:
    source_id: str
    role: SourceRole
    root: Path
    mutable: bool = False
    layout_name: str | None = None
    exclude_coordinates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise SourceContinuityError("source_id must be non-empty")
        mutable_roles = {
            SourceRole.APPEND_JSONL,
            SourceRole.REWRITE_JSONL,
            SourceRole.MUTABLE_SQLITE,
            SourceRole.SPOOL,
            SourceRole.QUEUE,
            SourceRole.SIDECAR,
            SourceRole.PROVIDER_CACHE,
        }
        if self.role in mutable_roles and not self.mutable:
            raise SourceContinuityError(f"mutable source {self.source_id} must be marked mutable")


@dataclass(frozen=True, slots=True)
class FrontierMember:
    """One independently enumerated source member at a stable cut."""

    source_id: str
    coordinate: str
    identity: str
    content_sha256: str
    size: int
    logical_sha256: str | None = None

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.source_id, self.coordinate, self.identity, self.content_sha256


@dataclass(frozen=True, slots=True)
class SourceFrontier:
    """Complete configured-source denominator used by conservation checks."""

    declarations: tuple[SourceDeclaration, ...]
    members: tuple[FrontierMember, ...]
    root_states: Mapping[str, FrontierState]
    blockers: tuple[str, ...]
    frontier_sha256: str

    @property
    def item_count(self) -> int:
        return len(self.members)

    @property
    def byte_count(self) -> int:
        return sum(member.size for member in self.members)

    @property
    def complete(self) -> bool:
        return not self.blockers and all(state is not FrontierState.UNAVAILABLE for state in self.root_states.values())

    def as_dict(self) -> dict[str, object]:
        return {
            "declarations": [
                {
                    "source_id": declaration.source_id,
                    "role": declaration.role.value,
                    "root": str(declaration.root),
                    "mutable": declaration.mutable,
                    "layout_name": declaration.layout_name,
                    "exclude_coordinates": list(declaration.exclude_coordinates),
                }
                for declaration in self.declarations
            ],
            "frontier_sha256": self.frontier_sha256,
            "item_count": self.item_count,
            "byte_count": self.byte_count,
            "complete": self.complete,
            "blockers": list(self.blockers),
            "root_states": {key: value.value for key, value in sorted(self.root_states.items())},
            "members": [
                {
                    "source_id": member.source_id,
                    "coordinate": member.coordinate,
                    "identity": member.identity,
                    "content_sha256": member.content_sha256,
                    "size": member.size,
                    "logical_sha256": member.logical_sha256,
                }
                for member in self.members
            ],
        }

    def verify_integrity(self) -> None:
        declaration_ids = {declaration.source_id for declaration in self.declarations}
        if len(declaration_ids) != len(self.declarations):
            raise SourceContinuityError("source frontier contains duplicate source IDs")
        if set(self.root_states) != declaration_ids:
            raise SourceContinuityError("source frontier root states do not cover declarations")
        member_keys = [member.key for member in self.members]
        if len(set(member_keys)) != len(member_keys):
            raise SourceContinuityError("source frontier contains duplicate members")
        if any(member.source_id not in declaration_ids for member in self.members):
            raise SourceContinuityError("source frontier member is outside its declarations")
        if any(member.size < 0 for member in self.members):
            raise SourceContinuityError("source frontier member size is negative")
        payload = {
            "declarations": [
                (d.source_id, d.role.value, str(d.root), d.mutable, d.layout_name, d.exclude_coordinates)
                for d in self.declarations
            ],
            "members": [
                (m.source_id, m.coordinate, m.identity, m.content_sha256, m.size, m.logical_sha256)
                for m in self.members
            ],
            "root_states": sorted((key, value.value) for key, value in self.root_states.items()),
            "blockers": list(self.blockers),
        }
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if expected != self.frontier_sha256:
            raise SourceContinuityError("source frontier integrity check failed")


_CAPTURED_INODE = re.compile(r"dev:(\d+):ino:(\d+)")


def build_source_frontier(declarations: Iterable[SourceDeclaration]) -> SourceFrontier:
    """Enumerate every configured root, retaining unavailable roots as blockers."""
    rows = tuple(declarations)
    if not rows:
        raise SourceContinuityError("source frontier declaration is empty")
    if len({row.source_id for row in rows}) != len(rows):
        raise SourceContinuityError("source frontier contains duplicate source IDs")
    if len({Path(row.root).resolve(strict=False) for row in rows}) != len(rows):
        raise SourceContinuityError("source frontier contains duplicate roots")
    from polylogue.sources.source_snapshot import SourceSnapshotError, observe_source_members

    members: list[FrontierMember] = []
    physical_identities: set[tuple[int, ...]] = set()
    states: dict[str, FrontierState] = {}
    blockers: list[str] = []
    for declaration in rows:
        try:
            observed = observe_source_members(declaration)
        except (OSError, SourceSnapshotError, ValueError) as exc:
            states[declaration.source_id] = FrontierState.UNAVAILABLE
            blockers.append(f"unavailable:{declaration.source_id}:{declaration.root}:{exc}")
            continue
        states[declaration.source_id] = FrontierState.PRESENT if observed else FrontierState.VALID_EMPTY
        disappeared: Path | None = None
        for item in observed:
            identity: tuple[int, ...]
            if declaration.role is SourceRole.ARCHIVE_MEMBER:
                # Archive member identity includes archive device/inode and
                # the ZIP header offset; equal payloads remain distinct.
                device, inode, _ctime, offset = item.identity.split(":")
                identity = (int(device), int(inode), int(offset))
            else:
                # Reuse the identity observe_source_members captured with the
                # bytes; re-stating the live path could see a replaced inode
                # or raise after a concurrent deletion.
                captured = _CAPTURED_INODE.search(item.identity)
                if captured is not None:
                    identity = (int(captured.group(1)), int(captured.group(2)))
                else:
                    # A SQLite logical-export member's identity is its export
                    # digest, not an inode; stat the member for duplicate
                    # detection and type a concurrent disappearance.
                    member_path = (
                        declaration.root if not declaration.root.is_dir() else declaration.root / item.coordinate
                    )
                    try:
                        info = member_path.stat()
                    except OSError:
                        disappeared = member_path
                        break
                    identity = (info.st_dev, info.st_ino)
            if identity in physical_identities:
                raise SourceContinuityError("source frontier contains duplicate physical member identity")
            physical_identities.add(identity)
        if disappeared is not None:
            # A member that vanished after observation makes this source's
            # frontier incoherent: a typed blocker, not an unhandled error.
            states[declaration.source_id] = FrontierState.UNAVAILABLE
            blockers.append(f"unavailable:{declaration.source_id}:{declaration.root}:member disappeared:{disappeared}")
            continue
        members.extend(
            FrontierMember(
                item.source_id,
                item.coordinate,
                item.identity,
                item.content_sha256,
                item.size_bytes,
                item.content_sha256 if declaration.role is SourceRole.MUTABLE_SQLITE else None,
            )
            for item in observed
        )
    payload = {
        "declarations": [
            (d.source_id, d.role.value, str(d.root), d.mutable, d.layout_name, d.exclude_coordinates) for d in rows
        ],
        "members": [
            (m.source_id, m.coordinate, m.identity, m.content_sha256, m.size, m.logical_sha256) for m in members
        ],
        "root_states": sorted((key, value.value) for key, value in states.items()),
        "blockers": blockers,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return SourceFrontier(rows, tuple(members), states, tuple(blockers), digest)


def configured_source_frontier(archive_root: Path) -> SourceFrontier:
    """Observe the configured provider layouts and Polylogue capture spools.

    The denominator is assembled from the same resolved source roots used by
    daemon intake, before consulting any acquisition ledger. Hook carriers,
    hook pending envelopes, and browser captures are explicit source roots.
    """
    from polylogue.config import resolve_runtime_config
    from polylogue.paths import archive_root as configured_archive_root
    from polylogue.sources.hooks import hook_spool_sources
    from polylogue.sources.origin_specs import database_capability_for_provider
    from polylogue.sources.source_layout import source_layout_for
    from polylogue.sources.source_walk import layout_source_candidates

    runtime = resolve_runtime_config()
    expected_archive = Path(archive_root).expanduser().resolve(strict=False)
    if Path(configured_archive_root()).expanduser().resolve(strict=False) != expected_archive:
        raise SourceContinuityError("archive root differs from the resolved source-spool configuration")

    rows: list[SourceDeclaration] = []
    for source in runtime.sources:
        if source.path is None:
            continue
        path = Path(source.path)
        if source.name == "hooks":
            # The complete hook root is declared below so pending envelopes
            # and provider carriers share the same spool owner.
            continue
        # The resolver omits optional local sources that are disabled or
        # absent in configuration. Once a source is present in runtime.sources,
        # its path is part of the denominator even if it disappears before this
        # observation; build_source_frontier turns that race into a blocker.
        name = source.name
        layout_name = name if path.is_dir() else None
        role = SourceRole.APPEND_JSONL if name == "claude-code-history" else SourceRole.DIRECTORY
        sqlite_paths: list[Path] = []
        if layout_name is not None:
            layout = source_layout_for(layout_name)
            provider = layout.provider
            capability = database_capability_for_provider(provider) if provider is not None else None
            accepted_members = (
                {item.filename for item in capability.members if item.disposition != "out-of-scope"}
                if capability is not None
                else set()
            )
            if accepted_members:
                sqlite_paths = [
                    member
                    for member in layout_source_candidates(layout_name, path)
                    if member.name in accepted_members and member.is_file()
                ]
        excluded = tuple(sorted(member.relative_to(path).as_posix() for member in sqlite_paths))
        rows.append(
            SourceDeclaration(
                f"configured:{name}",
                role,
                path,
                True,
                layout_name,
                excluded,
            )
        )
        rows.extend(
            SourceDeclaration(
                f"configured:{name}:sqlite:{member.relative_to(path).as_posix()}",
                SourceRole.MUTABLE_SQLITE,
                member,
                True,
            )
            for member in sqlite_paths
        )
    for spec in hook_spool_sources():
        spool_root = Path(spec.root)
        for carrier_provider in ("claude-code", "codex", "hermes"):
            carrier_root = spool_root / "carriers" / carrier_provider
            rows.append(
                SourceDeclaration(
                    f"{spec.source_id}:carrier:{carrier_provider}",
                    SourceRole.SPOOL,
                    carrier_root,
                    True,
                    f"{carrier_provider}-hooks",
                )
            )
        pending = spool_root / "pending"
        rows.append(SourceDeclaration(f"{spec.source_id}:pending", SourceRole.SPOOL, pending, True))
    unique: dict[str, SourceDeclaration] = {row.source_id: row for row in rows}
    return build_source_frontier(unique.values())


def canonical_source_declarations(
    *,
    configured: Iterable[SourceDeclaration] = (),
    hook_primary: Path | None = None,
    hook_legacy: Iterable[Path] = (),
    restored_spools: Iterable[Path] = (),
    browser_queue: Path | None = None,
    attachments: Path | None = None,
    sidecars: Iterable[Path] = (),
    provider_caches: Iterable[Path] = (),
    exports: Iterable[Path] = (),
    live_sources: Iterable[Path] = (),
) -> tuple[SourceDeclaration, ...]:
    """Assemble the single typed declaration for all source families."""
    rows = list(configured)
    if hook_primary is not None:
        rows.append(SourceDeclaration("hooks-primary", SourceRole.SPOOL, hook_primary, True))
    rows.extend(
        SourceDeclaration(f"hooks-legacy-{n}", SourceRole.SPOOL, path, True) for n, path in enumerate(hook_legacy)
    )
    rows.extend(
        SourceDeclaration(f"restored-spool-{n}", SourceRole.SPOOL, path, True) for n, path in enumerate(restored_spools)
    )
    if browser_queue is not None:
        rows.append(SourceDeclaration("browser-queue", SourceRole.QUEUE, browser_queue, True))
    if attachments is not None:
        rows.append(SourceDeclaration("attachments", SourceRole.ATTACHMENT, attachments))
    rows.extend(SourceDeclaration(f"sidecar-{n}", SourceRole.SIDECAR, path, True) for n, path in enumerate(sidecars))
    rows.extend(
        SourceDeclaration(f"provider-cache-{n}", SourceRole.PROVIDER_CACHE, path, True)
        for n, path in enumerate(provider_caches)
    )
    rows.extend(SourceDeclaration(f"export-{n}", SourceRole.IMMUTABLE_EXPORT, path) for n, path in enumerate(exports))
    rows.extend(
        SourceDeclaration(f"live-source-{n}", SourceRole.DIRECTORY, path, True) for n, path in enumerate(live_sources)
    )
    ids = [row.source_id for row in rows]
    roots = [Path(row.root).resolve(strict=False) for row in rows]
    if len(ids) != len(set(ids)):
        raise SourceContinuityError("duplicate source_id in canonical source declaration")
    if len(roots) != len(set(roots)):
        raise SourceContinuityError("duplicate root in canonical source declaration")
    # Resolved roots are for duplicate detection only; the declaration keeps the
    # configured absolute path so a symlinked root still reaches the source
    # snapshot's fail-closed root refusal instead of being replaced by its target.
    return tuple(
        SourceDeclaration(
            row.source_id,
            row.role,
            Path(row.root).absolute(),
            row.mutable,
            row.layout_name,
            row.exclude_coordinates,
        )
        for row in rows
    )

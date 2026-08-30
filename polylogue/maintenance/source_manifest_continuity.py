"""Member-level source continuity checks used by source-authority coordinators."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Iterable, Mapping
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


class MemberState(StrEnum):
    UNCHANGED = "unchanged"
    DECLARED_APPEND = "declared-append"
    DECLARED_REWRITE = "declared-rewrite"
    AUTHENTICATED_ROTATION = "authenticated-rotation"
    CONSUMED_AND_ACQUIRED = "consumed-and-durably-acquired"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class SourceDeclaration:
    source_id: str
    role: SourceRole
    root: Path
    mutable: bool = False

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
class MemberEvidence:
    source_id: str
    relative_path: str
    identity: str
    content_sha256: str
    size: int
    logical_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ConsumptionReceipt:
    content_sha256: str
    sealed_generation: str
    authenticated: bool = True


@dataclass(frozen=True, slots=True)
class SourceManifest:
    declarations: tuple[SourceDeclaration, ...]
    members: tuple[MemberEvidence, ...]
    manifest_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "declarations": [
                {"source_id": d.source_id, "role": d.role.value, "root": str(d.root), "mutable": d.mutable}
                for d in self.declarations
            ],
            "members": [
                {
                    "source_id": m.source_id,
                    "relative_path": m.relative_path,
                    "identity": m.identity,
                    "content_sha256": m.content_sha256,
                    "size": m.size,
                    "logical_sha256": m.logical_sha256,
                }
                for m in self.members
            ],
            "manifest_sha256": self.manifest_sha256,
        }

    def verify_integrity(self) -> None:
        """Reject a changed baseline instead of accepting its replacement hash."""
        payload = json.dumps(
            {
                "declarations": [(d.source_id, d.role.value, str(d.root), d.mutable) for d in self.declarations],
                "members": [
                    (m.source_id, m.relative_path, m.identity, m.content_sha256, m.size, m.logical_sha256)
                    for m in self.members
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if hashlib.sha256(payload).hexdigest() != self.manifest_sha256:
            raise SourceContinuityError("source manifest integrity check failed")


@dataclass(frozen=True, slots=True)
class ContinuityResult:
    states: Mapping[str, MemberState]
    blocked: tuple[str, ...] = ()

    @property
    def safe(self) -> bool:
        return not self.blocked and all(state is not MemberState.BLOCKED for state in self.states.values())


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
    roots = [Path(row.root).absolute() for row in rows]
    if len(ids) != len(set(ids)):
        raise SourceContinuityError("duplicate source_id in canonical source declaration")
    if len(roots) != len(set(roots)):
        raise SourceContinuityError("duplicate root in canonical source declaration")
    return tuple(
        SourceDeclaration(row.source_id, row.role, root, row.mutable) for row, root in zip(rows, roots, strict=True)
    )


def _real_root(root: Path) -> Path:
    try:
        info = root.lstat()
    except OSError as exc:
        raise SourceContinuityError(f"source root is unreadable: {root}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode))
        or not os.access(root, os.R_OK)
    ):
        raise SourceContinuityError(f"source root is unreadable or not a real directory: {root}")
    return root


def _sha256(path: Path, *, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = limit
    try:
        with path.open("rb") as handle:
            while remaining is None or remaining:
                chunk = handle.read(1024 * 1024 if remaining is None else min(1024 * 1024, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                if remaining is not None:
                    remaining -= len(chunk)
    except OSError as exc:
        raise SourceContinuityError(f"source member is unreadable: {path}") from exc
    if remaining:
        raise SourceContinuityError(f"source member was truncated while reading: {path}")
    return digest.hexdigest()


def _members(declaration: SourceDeclaration, logical: Mapping[str, str] | None = None) -> list[MemberEvidence]:
    root = _real_root(Path(declaration.root))
    if root.is_file():
        paths = [root]
        relative_paths = {root: root.name}
    else:
        try:
            paths = sorted(root.rglob("*"))
        except OSError as exc:
            raise SourceContinuityError(f"source root is unreadable: {root}") from exc
        relative_paths = {path: path.relative_to(root).as_posix() for path in paths}
    result: list[MemberEvidence] = []
    identities: set[str] = set()
    for path in paths:
        try:
            info = path.lstat()
        except OSError as exc:
            raise SourceContinuityError(f"source member disappeared: {path}") from exc
        if stat.S_ISDIR(info.st_mode):
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise SourceContinuityError(f"source member is not a regular file: {path}")
        relative = relative_paths[path]
        identity = f"dev:{info.st_dev}:ino:{info.st_ino}"
        if identity in identities:
            raise SourceContinuityError(f"source member identity collision: {declaration.source_id}:{relative}")
        identities.add(identity)
        result.append(
            MemberEvidence(
                declaration.source_id,
                relative,
                identity,
                _sha256(path),
                info.st_size,
                None if logical is None else logical.get(relative),
            )
        )
    if logical is not None and set(logical) != {member.relative_path for member in result}:
        raise SourceContinuityError(f"logical snapshot does not match source members: {declaration.source_id}")
    return result


def build_source_manifest(
    declarations: Iterable[SourceDeclaration],
    *,
    logical_snapshot: Callable[[SourceDeclaration], Mapping[str, str]] | None = None,
) -> SourceManifest:
    declarations = tuple(declarations)
    if not declarations:
        raise SourceContinuityError("source declaration is empty")
    if len({d.source_id for d in declarations}) != len(declarations) or len(
        {Path(d.root).absolute() for d in declarations}
    ) != len(declarations):
        raise SourceContinuityError("source declaration contains duplicate IDs or roots")
    members = tuple(
        member
        for declaration in declarations
        for member in _members(
            declaration,
            logical_snapshot(declaration)
            if declaration.role is SourceRole.MUTABLE_SQLITE and logical_snapshot
            else None,
        )
    )
    payload = json.dumps(
        {
            "declarations": [(d.source_id, d.role.value, str(d.root), d.mutable) for d in declarations],
            "members": [
                (m.source_id, m.relative_path, m.identity, m.content_sha256, m.size, m.logical_sha256) for m in members
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return SourceManifest(declarations, members, hashlib.sha256(payload).hexdigest())


def recheck_source_manifest(
    baseline: SourceManifest,
    *,
    consumed: Mapping[str, str] | None = None,
    consumption_receipts: Mapping[str, ConsumptionReceipt] | None = None,
    rotation: Mapping[str, str] | None = None,
    logical_snapshot: Callable[[SourceDeclaration], Mapping[str, str]] | None = None,
) -> ContinuityResult:
    """Classify every baseline member. Unrecognised loss blocks the result."""
    baseline.verify_integrity()
    consumed = consumed or {}
    consumption_receipts = consumption_receipts or {}
    rotation = rotation or {}
    declarations = {d.source_id: d for d in baseline.declarations}
    current: dict[str, MemberEvidence] = {}
    for declaration in baseline.declarations:
        logical = (
            logical_snapshot(declaration)
            if declaration.role is SourceRole.MUTABLE_SQLITE and logical_snapshot
            else None
        )
        for member in _members(declaration, logical):
            key = f"{member.source_id}:{member.relative_path}"
            if key in current:
                raise SourceContinuityError(f"duplicate current source member: {key}")
            current[key] = member
    states: dict[str, MemberState] = {}
    blocked: list[str] = []
    for member in baseline.members:
        key = f"{member.source_id}:{member.relative_path}"
        declaration = declarations[member.source_id]
        observed = current.get(key)
        if observed is None:
            receipt = consumption_receipts.get(key)
            if (
                receipt
                and receipt.authenticated
                and receipt.sealed_generation
                and receipt.content_sha256 == member.content_sha256
            ):
                states[key] = MemberState.CONSUMED_AND_ACQUIRED
            elif consumed.get(key) == member.content_sha256:
                states[key] = MemberState.BLOCKED
                blocked.append(f"unauthenticated-consumption:{key}")
            elif rotation.get(key, rotation.get(member.source_id)) == member.identity:
                states[key] = MemberState.AUTHENTICATED_ROTATION
            else:
                states[key] = MemberState.BLOCKED
                blocked.append(f"missing:{key}")
        elif declaration.role is SourceRole.MUTABLE_SQLITE:
            if member.logical_sha256 and observed.logical_sha256 == member.logical_sha256:
                states[key] = MemberState.UNCHANGED
            else:
                states[key] = MemberState.BLOCKED
                blocked.append(f"sqlite-logical-change:{key}")
        elif observed.identity == member.identity and observed.content_sha256 == member.content_sha256:
            states[key] = MemberState.UNCHANGED
        elif (
            declaration.role is SourceRole.APPEND_JSONL
            and observed.size >= member.size
            and _sha256(Path(declaration.root) / member.relative_path, limit=member.size) == member.content_sha256
        ):
            states[key] = MemberState.DECLARED_APPEND
        elif declaration.role is SourceRole.REWRITE_JSONL:
            states[key] = MemberState.DECLARED_REWRITE
        else:
            states[key] = MemberState.BLOCKED
            blocked.append(f"replacement:{key}")
    return ContinuityResult(states, tuple(blocked))


def validate_backup_evidence(reference: Mapping[str, object], *, now_ms: int, max_age_ms: int) -> None:
    """Require a fresh external authentication reference; do not create one."""
    if (
        reference.get("authenticated") is not True
        or not isinstance(reference.get("reference"), str)
        or not reference["reference"]
    ):
        raise SourceContinuityError("authenticated external backup evidence is required")
    observed = reference.get("observed_at_ms")
    if not isinstance(observed, int) or observed < 0 or now_ms < observed or now_ms - observed > max_age_ms:
        raise SourceContinuityError("external backup evidence is stale")

"""Member-level source continuity laws.

This module deliberately contains no campaign state and no repair action.  It
is a small value-oriented validator used by a coordinator to bind a source
population, then classify the next observation without treating aggregate
statistics as evidence of survival.
"""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class SourceContinuityError(ValueError):
    """A source declaration, manifest, or recheck is unsafe."""


class SourceRole(StrEnum):
    IMMUTABLE_EXPORT = "immutable-export"
    APPEND_JSONL = "append-jsonl"
    REWRITE_JSONL = "rewrite-leading-jsonl"
    MUTABLE_SQLITE = "mutable-sqlite"
    SPOOL = "spool"
    QUEUE = "queue"
    ATTACHMENT = "attachment"
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
        if (
            self.role
            in {
                SourceRole.APPEND_JSONL,
                SourceRole.REWRITE_JSONL,
                SourceRole.MUTABLE_SQLITE,
                SourceRole.SPOOL,
                SourceRole.QUEUE,
            }
            and not self.mutable
        ):
            raise SourceContinuityError(f"mutable source {self.source_id} must be marked mutable")


@dataclass(frozen=True, slots=True)
class MemberEvidence:
    source_id: str
    relative_path: str
    identity: str
    content_sha256: str
    size: int
    logical_sha256: str | None = None
    prefix_sha256: str | None = None
    append_offset: int | None = None


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
                    "prefix_sha256": m.prefix_sha256,
                    "append_offset": m.append_offset,
                }
                for m in self.members
            ],
            "manifest_sha256": self.manifest_sha256,
        }


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
    exports: Iterable[Path] = (),
    live_sources: Iterable[Path] = (),
) -> tuple[SourceDeclaration, ...]:
    """Build the sole typed source list; duplicate roots/IDs are rejected."""
    rows = list(configured)
    if hook_primary is not None:
        rows.append(SourceDeclaration("hooks-primary", SourceRole.SPOOL, hook_primary, True))
    rows.extend(SourceDeclaration(f"hooks-legacy-{n}", SourceRole.SPOOL, p, True) for n, p in enumerate(hook_legacy))
    rows.extend(
        SourceDeclaration(f"restored-spool-{n}", SourceRole.SPOOL, p, True) for n, p in enumerate(restored_spools)
    )
    if browser_queue is not None:
        rows.append(SourceDeclaration("browser-queue", SourceRole.QUEUE, browser_queue, True))
    if attachments is not None:
        rows.append(SourceDeclaration("attachments", SourceRole.ATTACHMENT, attachments))
    rows.extend(SourceDeclaration(f"export-{n}", SourceRole.IMMUTABLE_EXPORT, p) for n, p in enumerate(exports))
    rows.extend(
        SourceDeclaration(f"live-source-{n}", SourceRole.DIRECTORY, p, True) for n, p in enumerate(live_sources)
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
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise SourceContinuityError(f"source root is not a real directory: {root}")
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SourceContinuityError(f"source member is unreadable: {path}") from exc
    return digest.hexdigest()


def _members(declaration: SourceDeclaration) -> list[MemberEvidence]:
    root = _real_root(Path(declaration.root))
    result: list[MemberEvidence] = []
    try:
        paths = sorted(path for path in root.rglob("*") if path.is_file())
    except OSError as exc:
        raise SourceContinuityError(f"source root is unreadable: {root}") from exc
    for path in paths:
        try:
            info = path.lstat()
        except OSError as exc:
            raise SourceContinuityError(f"source member disappeared: {path}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise SourceContinuityError(f"source member is a symlink: {path}")
        relative = path.relative_to(root).as_posix()
        result.append(
            MemberEvidence(
                declaration.source_id, relative, f"dev:{info.st_dev}:ino:{info.st_ino}", _sha256(path), info.st_size
            )
        )
    return result


def build_source_manifest(declarations: Iterable[SourceDeclaration]) -> SourceManifest:
    declarations = tuple(declarations)
    if not declarations:
        raise SourceContinuityError("source declaration is empty")
    if len({d.source_id for d in declarations}) != len(declarations) or len(
        {Path(d.root).absolute() for d in declarations}
    ) != len(declarations):
        raise SourceContinuityError("source declaration contains duplicate IDs or roots")
    members = tuple(member for declaration in declarations for member in _members(declaration))
    member_rows = [
        (
            m.source_id,
            m.relative_path,
            m.identity,
            m.content_sha256,
            m.size,
            m.logical_sha256,
            m.prefix_sha256,
            m.append_offset,
        )
        for m in members
    ]
    payload = json.dumps(
        {
            "declarations": [(d.source_id, d.role.value, str(d.root), d.mutable) for d in declarations],
            "members": member_rows,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return SourceManifest(declarations, members, hashlib.sha256(payload).hexdigest())


def recheck_source_manifest(
    baseline: SourceManifest,
    *,
    consumed: Mapping[str, str] = {},
    rotation: Mapping[str, str] = {},
    logical_snapshot: Callable[[SourceDeclaration], Mapping[str, str]] | None = None,
) -> ContinuityResult:
    """Classify every baseline member; any unrecognised loss blocks."""
    current = {m.source_id + ":" + m.relative_path: m for d in baseline.declarations for m in _members(d)}
    states: dict[str, MemberState] = {}
    blocked: list[str] = []
    for member in baseline.members:
        key = member.source_id + ":" + member.relative_path
        observed = current.get(key)
        if observed is None:
            if key in consumed and consumed[key] == member.content_sha256:
                states[key] = MemberState.CONSUMED_AND_ACQUIRED
            elif member.source_id in rotation and rotation[member.source_id] == member.identity:
                states[key] = MemberState.AUTHENTICATED_ROTATION
            else:
                states[key] = MemberState.BLOCKED
                blocked.append(f"missing:{key}")
            continue
        if observed.content_sha256 == member.content_sha256:
            states[key] = MemberState.UNCHANGED
        elif (
            observed.size >= member.size
            and baseline.declarations[
                next(i for i, d in enumerate(baseline.declarations) if d.source_id == member.source_id)
            ].role
            is SourceRole.APPEND_JSONL
        ):
            states[key] = MemberState.DECLARED_APPEND
        elif (
            baseline.declarations[
                next(i for i, d in enumerate(baseline.declarations) if d.source_id == member.source_id)
            ].role
            is SourceRole.REWRITE_JSONL
        ):
            states[key] = MemberState.DECLARED_REWRITE
        else:
            states[key] = MemberState.BLOCKED
            blocked.append(f"replacement:{key}")
    if logical_snapshot is not None:
        for declaration in baseline.declarations:
            if declaration.role is SourceRole.MUTABLE_SQLITE:
                try:
                    logical_snapshot(declaration)
                except Exception as exc:
                    blocked.append(f"sqlite:{declaration.source_id}:{exc}")
    return ContinuityResult(states, tuple(blocked))


def validate_backup_evidence(reference: Mapping[str, object], *, now_ms: int, max_age_ms: int) -> None:
    """Require an authenticated, fresh external backup/runtime reference."""
    if reference.get("authenticated") is not True or not isinstance(reference.get("reference"), str):
        raise SourceContinuityError("authenticated external backup evidence is required")
    observed = reference.get("observed_at_ms")
    if not isinstance(observed, int) or observed < 0 or now_ms - observed > max_age_ms:
        raise SourceContinuityError("external backup evidence is stale")

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
from typing import Final

from polylogue.maintenance.receipt_fs import (
    atomic_replace_receipt,
    existing_maintenance_receipt_directory,
    maintenance_receipt_directory,
    read_optional_receipt,
)
from polylogue.storage.archive_identity import MAINTENANCE_STATE_DIRNAME


class SourceContinuityError(ValueError):
    """A source declaration, manifest, or continuity receipt is unsafe."""


class WantedSourceReceiptError(SourceContinuityError):
    """A wanted-source receipt cannot authorize a rebuild."""


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


class FrontierState(StrEnum):
    """Observation state for one configured source root."""

    PRESENT = "present"
    VALID_EMPTY = "valid-empty"
    UNAVAILABLE = "unavailable"


WANTED_SOURCE_RECEIPT_SCHEMA: Final = "polylogue.wanted-source.v1"
WANTED_SOURCE_RECEIPT_DIRNAME: Final = "wanted-sources"
WANTED_SOURCE_RECEIPT_FILENAME: Final = "selected.json"
_DEFAULT_EXCLUDED_SOURCE_KINDS: Final = (
    "discovery-only",
    "experimental",
    "optional",
    "native-sinex",
)


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
            "declarations": [(d.source_id, d.role.value, str(d.root), d.mutable) for d in self.declarations],
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


@dataclass(frozen=True, slots=True)
class WantedSourcePolicy:
    """The explicit source population policy bound to a rebuild receipt.

    The default policy is intentionally conservative: every declaration that
    reaches this function is wanted, including raw evidence that a parser may
    later reject. Discovery-only, experimental/optional, and native Sinex
    populations are not declarations in the configured-source frontier. A
    caller that has those classifications can pass them in ``source_kinds``
    to :func:`build_wanted_source_receipt`; they are recorded as excluded
    rather than silently becoming wanted members.
    """

    name: str = "campaign-default"
    revision: str = "1"
    included_roles: tuple[SourceRole, ...] = tuple(SourceRole)
    excluded_kinds: tuple[str, ...] = _DEFAULT_EXCLUDED_SOURCE_KINDS

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.revision.strip():
            raise WantedSourceReceiptError("wanted-source policy name and revision must be non-empty")
        if len(set(self.included_roles)) != len(self.included_roles):
            raise WantedSourceReceiptError("wanted-source policy contains duplicate roles")
        if len(set(self.excluded_kinds)) != len(self.excluded_kinds):
            raise WantedSourceReceiptError("wanted-source policy contains duplicate exclusions")

    @property
    def identity(self) -> str:
        payload = {
            "name": self.name,
            "revision": self.revision,
            "included_roles": [role.value for role in self.included_roles],
            "excluded_kinds": list(self.excluded_kinds),
        }
        return hashlib.sha256(_canonical_json(payload)).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "revision": self.revision,
            "included_roles": [role.value for role in self.included_roles],
            "excluded_kinds": list(self.excluded_kinds),
            "identity": self.identity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> WantedSourcePolicy:
        try:
            roles_raw = payload["included_roles"]
            excluded_raw = payload["excluded_kinds"]
            if not isinstance(roles_raw, list) or not isinstance(excluded_raw, list):
                raise TypeError
            policy = cls(
                name=str(payload["name"]),
                revision=str(payload["revision"]),
                included_roles=tuple(SourceRole(str(role)) for role in roles_raw),
                excluded_kinds=tuple(str(kind) for kind in excluded_raw),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WantedSourceReceiptError("invalid wanted-source policy") from exc
        if payload.get("identity") != policy.identity:
            raise WantedSourceReceiptError("wanted-source policy identity mismatch")
        return policy

    def select(
        self,
        declarations: Iterable[SourceDeclaration],
        *,
        source_kinds: Mapping[str, str] | None = None,
    ) -> tuple[tuple[SourceDeclaration, ...], tuple[str, ...]]:
        kinds = source_kinds or {}
        selected: list[SourceDeclaration] = []
        excluded: list[str] = []
        for declaration in declarations:
            kind = kinds.get(declaration.source_id)
            if kind in self.excluded_kinds or declaration.role not in self.included_roles:
                excluded.append(declaration.source_id)
            else:
                selected.append(declaration)
        return tuple(selected), tuple(excluded)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _declaration_digest(declarations: Iterable[SourceDeclaration]) -> str:
    return hashlib.sha256(
        _canonical_json([(row.source_id, row.role.value, str(row.root), row.mutable) for row in declarations])
    ).hexdigest()


def _frontier_digest(
    declarations: Iterable[SourceDeclaration],
    members: Iterable[FrontierMember],
    root_states: Mapping[str, FrontierState],
    blockers: Iterable[str],
) -> str:
    payload = {
        "declarations": [(d.source_id, d.role.value, str(d.root), d.mutable) for d in declarations],
        "members": [
            (m.source_id, m.coordinate, m.identity, m.content_sha256, m.size, m.logical_sha256) for m in members
        ],
        "root_states": sorted((key, value.value) for key, value in root_states.items()),
        "blockers": list(blockers),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class WantedSourceReceipt:
    """Private, policy-bound denominator for a final rebuild."""

    policy: WantedSourcePolicy
    declarations: tuple[SourceDeclaration, ...]
    members: tuple[FrontierMember, ...]
    root_states: Mapping[str, FrontierState]
    blockers: tuple[str, ...]
    frontier_sha256: str
    excluded_source_ids: tuple[str, ...] = ()
    receipt_sha256: str = ""

    @property
    def policy_identity(self) -> str:
        return self.policy.identity

    @property
    def declaration_sha256(self) -> str:
        return _declaration_digest(self.declarations)

    @property
    def complete(self) -> bool:
        return not self.blockers and all(state is not FrontierState.UNAVAILABLE for state in self.root_states.values())

    @property
    def item_count(self) -> int:
        return len(self.members)

    @property
    def byte_count(self) -> int:
        return sum(member.size for member in self.members)

    def _payload(self, *, include_receipt_digest: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": WANTED_SOURCE_RECEIPT_SCHEMA,
            "policy": self.policy.as_dict(),
            "declaration_sha256": self.declaration_sha256,
            "declarations": [
                {
                    "source_id": declaration.source_id,
                    "role": declaration.role.value,
                    "root": str(declaration.root),
                    "mutable": declaration.mutable,
                }
                for declaration in self.declarations
            ],
            "excluded_source_ids": list(self.excluded_source_ids),
            "root_states": {key: value.value for key, value in sorted(self.root_states.items())},
            "blockers": list(self.blockers),
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
            "frontier_sha256": self.frontier_sha256,
            "item_count": self.item_count,
            "byte_count": self.byte_count,
            "complete": self.complete,
        }
        if include_receipt_digest:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload

    def as_dict(self) -> dict[str, object]:
        return self._payload(include_receipt_digest=True)

    def verify_integrity(self) -> None:
        self.policy.__post_init__()
        declaration_ids = {declaration.source_id for declaration in self.declarations}
        if len(declaration_ids) != len(self.declarations):
            raise WantedSourceReceiptError("wanted-source receipt contains duplicate declarations")
        if set(self.root_states) != declaration_ids:
            raise WantedSourceReceiptError("wanted-source receipt root states do not cover declarations")
        if tuple(sorted(self.excluded_source_ids)) != self.excluded_source_ids:
            raise WantedSourceReceiptError("wanted-source receipt exclusions are not canonical")
        if any(member.source_id not in declaration_ids for member in self.members):
            raise WantedSourceReceiptError("wanted-source receipt member is outside its declarations")
        keys = [member.key for member in self.members]
        if len(keys) != len(set(keys)):
            raise WantedSourceReceiptError("wanted-source receipt contains duplicate members")
        coordinates = [(member.source_id, member.coordinate) for member in self.members]
        if len(coordinates) != len(set(coordinates)):
            raise WantedSourceReceiptError("wanted-source receipt contains duplicate member coordinates")
        if any(member.size < 0 for member in self.members):
            raise WantedSourceReceiptError("wanted-source receipt member size is negative")
        expected_frontier = _frontier_digest(self.declarations, self.members, self.root_states, self.blockers)
        if expected_frontier != self.frontier_sha256:
            raise WantedSourceReceiptError("wanted-source frontier integrity check failed")
        payload = self._payload(include_receipt_digest=False)
        expected_receipt = hashlib.sha256(_canonical_json(payload)).hexdigest()
        if expected_receipt != self.receipt_sha256:
            raise WantedSourceReceiptError("wanted-source receipt integrity check failed")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> WantedSourceReceipt:
        try:
            if payload.get("schema") != WANTED_SOURCE_RECEIPT_SCHEMA:
                raise WantedSourceReceiptError("unsupported wanted-source receipt schema")
            policy_payload = payload["policy"]
            declarations_payload = payload["declarations"]
            members_payload = payload["members"]
            states_payload = payload["root_states"]
            if not isinstance(policy_payload, Mapping) or not isinstance(declarations_payload, list):
                raise TypeError
            if not isinstance(members_payload, list) or not isinstance(states_payload, Mapping):
                raise TypeError
            declarations = tuple(
                SourceDeclaration(
                    str(row["source_id"]),
                    SourceRole(str(row["role"])),
                    Path(str(row["root"])),
                    bool(row["mutable"]),
                )
                for row in declarations_payload
                if isinstance(row, Mapping)
            )
            members = tuple(
                FrontierMember(
                    str(row["source_id"]),
                    str(row["coordinate"]),
                    str(row["identity"]),
                    str(row["content_sha256"]),
                    int(row["size"]),
                    None if row.get("logical_sha256") is None else str(row["logical_sha256"]),
                )
                for row in members_payload
                if isinstance(row, Mapping)
            )
            root_states = {str(key): FrontierState(str(value)) for key, value in states_payload.items()}
            excluded = payload.get("excluded_source_ids", [])
            blockers = payload.get("blockers", [])
            if not isinstance(excluded, list) or not isinstance(blockers, list):
                raise TypeError
            result = cls(
                policy=WantedSourcePolicy.from_dict(policy_payload),
                declarations=declarations,
                members=members,
                root_states=root_states,
                blockers=tuple(str(item) for item in blockers),
                frontier_sha256=str(payload["frontier_sha256"]),
                excluded_source_ids=tuple(str(item) for item in excluded),
                receipt_sha256=str(payload["receipt_sha256"]),
            )
        except WantedSourceReceiptError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise WantedSourceReceiptError("invalid wanted-source receipt") from exc
        result.verify_integrity()
        if payload.get("declaration_sha256") != result.declaration_sha256:
            raise WantedSourceReceiptError("wanted-source declaration digest mismatch")
        if payload.get("item_count") != result.item_count or payload.get("byte_count") != result.byte_count:
            raise WantedSourceReceiptError("wanted-source receipt denominators mismatch")
        if payload.get("complete") is not result.complete:
            raise WantedSourceReceiptError("wanted-source receipt completeness mismatch")
        return result


@dataclass(frozen=True, slots=True)
class RebuildPreflightReceipt:
    """Operator-safe receipt emitted after validating a wanted-source file."""

    receipt_sha256: str
    policy_identity: str
    declaration_sha256: str
    frontier_sha256: str
    item_count: int
    byte_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "outcome": "ok",
            "receipt_sha256": self.receipt_sha256,
            "policy_identity": self.policy_identity,
            "declaration_sha256": self.declaration_sha256,
            "frontier_sha256": self.frontier_sha256,
            "item_count": self.item_count,
            "byte_count": self.byte_count,
        }


def campaign_default_wanted_source_policy() -> WantedSourcePolicy:
    """Return a fresh default policy for the campaign's final rebuild."""
    return WantedSourcePolicy()


def build_wanted_source_receipt(
    declarations: Iterable[SourceDeclaration],
    *,
    policy: WantedSourcePolicy | None = None,
    source_kinds: Mapping[str, str] | None = None,
) -> WantedSourceReceipt:
    """Enumerate the selected declarations once and bind their denominator."""
    selected_policy = policy or campaign_default_wanted_source_policy()
    rows = tuple(declarations)
    if len({row.source_id for row in rows}) != len(rows):
        raise WantedSourceReceiptError("wanted-source declarations contain duplicate source IDs")
    selected, excluded = selected_policy.select(rows, source_kinds=source_kinds)
    if not selected:
        members: tuple[FrontierMember, ...] = ()
        root_states: Mapping[str, FrontierState] = {}
        blockers: tuple[str, ...] = ()
    else:
        frontier = build_source_frontier(selected)
        members = frontier.members
        root_states = frontier.root_states
        blockers = frontier.blockers
    frontier_digest = _frontier_digest(selected, members, root_states, blockers)
    provisional = WantedSourceReceipt(
        policy=selected_policy,
        declarations=selected,
        members=members,
        root_states=root_states,
        blockers=blockers,
        frontier_sha256=frontier_digest,
        excluded_source_ids=tuple(sorted(excluded)),
    )
    digest = hashlib.sha256(_canonical_json(provisional._payload(include_receipt_digest=False))).hexdigest()
    # Construct the sealed value explicitly; the receipt is immutable and has
    # no mutable refresh path.
    result = WantedSourceReceipt(
        selected_policy,
        selected,
        members,
        root_states,
        blockers,
        frontier_digest,
        tuple(sorted(excluded)),
        digest,
    )
    result.verify_integrity()
    return result


def _ensure_wanted_source_state(archive_root: Path) -> None:
    root = Path(archive_root)
    metadata = root.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise WantedSourceReceiptError("archive root is not a real directory")
    state = root / MAINTENANCE_STATE_DIRNAME
    try:
        state_metadata = state.lstat()
    except FileNotFoundError:
        state.mkdir(mode=0o700)
        return
    if stat.S_ISLNK(state_metadata.st_mode) or not stat.S_ISDIR(state_metadata.st_mode):
        raise WantedSourceReceiptError("maintenance state is not a real directory")


def write_wanted_source_receipt(
    archive_root: Path,
    declarations: Iterable[SourceDeclaration],
    *,
    policy: WantedSourcePolicy | None = None,
    source_kinds: Mapping[str, str] | None = None,
) -> WantedSourceReceipt:
    """Atomically publish one private wanted-source receipt."""
    receipt = build_wanted_source_receipt(declarations, policy=policy, source_kinds=source_kinds)
    _ensure_wanted_source_state(Path(archive_root))
    with maintenance_receipt_directory(Path(archive_root), WANTED_SOURCE_RECEIPT_DIRNAME) as directory_fd:
        atomic_replace_receipt(
            directory_fd,
            WANTED_SOURCE_RECEIPT_FILENAME,
            json.dumps(receipt.as_dict(), indent=2, sort_keys=True).encode("utf-8"),
        )
    return receipt


def _read_wanted_source_receipt(archive_root: Path) -> WantedSourceReceipt:
    with existing_maintenance_receipt_directory(Path(archive_root), WANTED_SOURCE_RECEIPT_DIRNAME) as directory_fd:
        if directory_fd is None:
            raise WantedSourceReceiptError("wanted-source receipt is missing")
        raw = read_optional_receipt(directory_fd, WANTED_SOURCE_RECEIPT_FILENAME)
    if raw is None:
        raise WantedSourceReceiptError("wanted-source receipt is missing")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WantedSourceReceiptError("wanted-source receipt is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise WantedSourceReceiptError("wanted-source receipt must be a JSON object")
    return WantedSourceReceipt.from_dict(payload)


def load_wanted_source_receipt(
    archive_root: Path,
    *,
    policy: WantedSourcePolicy | None = None,
    declarations: Iterable[SourceDeclaration] | None = None,
) -> WantedSourceReceipt:
    """Load and validate a private receipt before any rebuild publication."""
    receipt = _read_wanted_source_receipt(Path(archive_root))
    expected_policy = policy or campaign_default_wanted_source_policy()
    if receipt.policy.identity != expected_policy.identity:
        raise WantedSourceReceiptError("wanted-source receipt policy mismatch")
    if declarations is not None:
        expected, _excluded = expected_policy.select(tuple(declarations))
        if _declaration_digest(expected) != receipt.declaration_sha256:
            raise WantedSourceReceiptError("wanted-source receipt declaration mismatch")
    for declaration in receipt.declarations:
        try:
            _real_root(Path(declaration.root))
        except SourceContinuityError as exc:
            raise WantedSourceReceiptError(
                f"wanted-source receipt root is missing or unavailable: {declaration.source_id}"
            ) from exc
    if not receipt.complete:
        raise WantedSourceReceiptError("wanted-source receipt is incomplete")
    return receipt


def preflight_rebuild(
    archive_root: Path,
    *,
    policy: WantedSourcePolicy | None = None,
    declarations: Iterable[SourceDeclaration] | None = None,
) -> RebuildPreflightReceipt:
    """Authorize a rebuild from the frozen receipt, never a fresh source walk."""
    receipt = load_wanted_source_receipt(Path(archive_root), policy=policy, declarations=declarations)
    return RebuildPreflightReceipt(
        receipt_sha256=receipt.receipt_sha256,
        policy_identity=receipt.policy_identity,
        declaration_sha256=receipt.declaration_sha256,
        frontier_sha256=receipt.frontier_sha256,
        item_count=receipt.item_count,
        byte_count=receipt.byte_count,
    )


# Explicit aliases make the operation seam discoverable to callers that name
# the receipt as the authorization rather than as a generic preflight.
require_rebuild_preflight = preflight_rebuild


def build_source_frontier(declarations: Iterable[SourceDeclaration]) -> SourceFrontier:
    """Enumerate every configured root, retaining unavailable roots as blockers."""
    rows = tuple(declarations)
    if not rows:
        raise SourceContinuityError("source frontier declaration is empty")
    if len({row.source_id for row in rows}) != len(rows):
        raise SourceContinuityError("source frontier contains duplicate source IDs")
    from polylogue.sources.source_snapshot import SourceSnapshotError, observe_source_members

    members: list[FrontierMember] = []
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
        "declarations": [(d.source_id, d.role.value, str(d.root), d.mutable) for d in rows],
        "members": [
            (m.source_id, m.coordinate, m.identity, m.content_sha256, m.size, m.logical_sha256) for m in members
        ],
        "root_states": sorted((key, value.value) for key, value in states.items()),
        "blockers": blockers,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return SourceFrontier(rows, tuple(members), states, tuple(blockers), digest)


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

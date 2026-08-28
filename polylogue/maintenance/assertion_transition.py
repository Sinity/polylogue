"""Exact, candidate-bound reconciliation of durable assertion references.

This module deliberately knows nothing about similarity or historical index
rows.  A reference is preserved when its exact identity is in the candidate,
or is classified from a sealed source claim.  Identity changes are accepted
only as a complete, producer-supplied map.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from polylogue.core.refs import ObjectRef


class ObjectRefDisposition(StrEnum):
    PRESERVED = "preserved"
    EXPECTED_RESTORED = "expected-restored"
    EXPLICITLY_MIGRATED = "explicitly-migrated"
    ORPHANED = "orphaned"
    BLOCKING_MISSING = "blocking-missing"
    AMBIGUOUS = "ambiguous"


class ObjectRefReconciliationError(ValueError):
    """Raised when exact identity evidence is incomplete or contradictory."""


@dataclass(frozen=True, slots=True)
class SourceIdentityClaims:
    """The sealed source identity set used by one reconciliation."""

    refs: frozenset[str]

    @classmethod
    def from_refs(cls, refs: Iterable[str | ObjectRef]) -> SourceIdentityClaims:
        return cls(frozenset(_ref_text(ref) for ref in refs))


@dataclass(frozen=True, slots=True)
class IdentityMigrationMap:
    """A producer-issued total old-to-new identity map.

    A sequence is used instead of a mapping so duplicate and conflicting
    endpoints cannot be silently collapsed by the caller.
    """

    producer: str
    entries: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.producer.strip():
            raise ObjectRefReconciliationError("identity migration producer is required")
        old_to_new: dict[str, str] = {}
        new_to_old: dict[str, str] = {}
        for old, new in self.entries:
            old_text, new_text = _ref_text(old), _ref_text(new)
            prior = old_to_new.get(old_text)
            if prior is not None and prior != new_text:
                raise ObjectRefReconciliationError("identity migration map has conflicting old endpoints")
            prior_old = new_to_old.get(new_text)
            if prior_old is not None and prior_old != old_text:
                raise ObjectRefReconciliationError("identity migration map has conflicting new endpoints")
            old_to_new[old_text] = new_text
            new_to_old[new_text] = old_text

    def target_for(self, ref: str) -> str | None:
        return dict(self.entries).get(ref)

    @property
    def inverse(self) -> tuple[tuple[str, str], ...]:
        return tuple((new, old) for old, new in self.entries)


@dataclass(frozen=True, slots=True)
class ReconciledObjectRef:
    source: str
    disposition: ObjectRefDisposition
    target: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {"source": self.source, "disposition": self.disposition.value, "target": self.target}


@dataclass(frozen=True, slots=True)
class TransitionBinding:
    """Immutable identities which make a plan applicable to one candidate."""

    predecessor_digest: str
    candidate_digest: str
    source_seal: str
    package_version: str
    schema_versions: tuple[tuple[str, int], ...]
    migration_map_digest: str | None = None

    def digest(self) -> str:
        return _digest(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "predecessor_digest": self.predecessor_digest,
            "candidate_digest": self.candidate_digest,
            "source_seal": self.source_seal,
            "package_version": self.package_version,
            "schema_versions": list(self.schema_versions),
            "migration_map_digest": self.migration_map_digest,
        }


@dataclass(frozen=True, slots=True)
class AssertionTransitionPlan:
    binding: TransitionBinding
    rows: tuple[ReconciledObjectRef, ...]
    forward: tuple[tuple[str, str], ...]
    inverse: tuple[tuple[str, str], ...]

    @property
    def is_empty(self) -> bool:
        return not self.forward

    def digest(self) -> str:
        return _digest(
            {
                "binding": self.binding.as_dict(),
                "rows": [row.as_dict() for row in self.rows],
                "forward": self.forward,
                "inverse": self.inverse,
            }
        )


def reconcile_object_refs(
    refs: Iterable[str | ObjectRef],
    *,
    candidate_refs: Iterable[str | ObjectRef],
    source_claims: SourceIdentityClaims,
    predecessor_refs: Iterable[str | ObjectRef] = (),
    migration_map: IdentityMigrationMap | None = None,
    binding: TransitionBinding,
) -> AssertionTransitionPlan:
    """Classify every durable reference against exact candidate evidence."""

    requested = tuple(dict.fromkeys(_ref_text(ref) for ref in refs))
    candidate = frozenset(_ref_text(ref) for ref in candidate_refs)
    predecessor = frozenset(_ref_text(ref) for ref in predecessor_refs)
    if migration_map is not None:
        mapped_old = {old for old, _ in migration_map.entries}
        missing = mapped_old - set(requested)
        if missing:
            raise ObjectRefReconciliationError("identity migration map is not total for durable references")

    rows: list[ReconciledObjectRef] = []
    forward: list[tuple[str, str]] = []
    for source in requested:
        target = migration_map.target_for(source) if migration_map else None
        if target is not None:
            if target not in candidate:
                disposition = ObjectRefDisposition.BLOCKING_MISSING
                target = None
            else:
                disposition = ObjectRefDisposition.EXPLICITLY_MIGRATED
                forward.append((source, target))
        elif source in candidate:
            disposition = ObjectRefDisposition.PRESERVED
        elif source in source_claims.refs:
            disposition = ObjectRefDisposition.BLOCKING_MISSING
        else:
            disposition = ObjectRefDisposition.ORPHANED
        if disposition is ObjectRefDisposition.PRESERVED and source not in predecessor and source in source_claims.refs:
            disposition = ObjectRefDisposition.EXPECTED_RESTORED
        rows.append(ReconciledObjectRef(source, disposition, target))

    inverse = tuple((new, old) for old, new in forward)
    return AssertionTransitionPlan(binding, tuple(rows), tuple(forward), inverse)


def enumerate_assertion_object_refs(conn: sqlite3.Connection) -> tuple[str, ...]:
    """Enumerate every ObjectRef stored by the durable assertions table."""

    refs: list[str] = []
    rows = conn.execute(
        "SELECT scope_ref, target_ref, evidence_refs_json, supersedes_json FROM assertions ORDER BY assertion_id"
    )
    for row in rows:
        values: list[object] = [row[0], row[1]]
        for index in (2, 3):
            try:
                decoded = json.loads(str(row[index]))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ObjectRefReconciliationError("assertion reference JSON is invalid") from exc
            if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
                raise ObjectRefReconciliationError("assertion reference JSON must be a string list")
            values.extend(decoded)
        for value in values:
            if value is not None:
                refs.append(_ref_text(str(value)))
    return tuple(dict.fromkeys(refs))


def apply_assertion_transition(
    conn: sqlite3.Connection,
    plan: AssertionTransitionPlan,
    *,
    binding: TransitionBinding,
    verified_backup: bool,
    append_audit: Callable[[AssertionTransitionPlan], None] | None = None,
) -> None:
    """Apply a declared map atomically to assertion references.

    The caller must supply the same binding and a verified durable backup.
    Audit is appended inside the same transaction when supplied.
    """

    if binding != plan.binding:
        raise ObjectRefReconciliationError("transition binding is stale")
    if not verified_backup:
        raise ObjectRefReconciliationError("explicit identity transition requires a verified backup")
    if any(row.disposition is ObjectRefDisposition.BLOCKING_MISSING for row in plan.rows):
        raise ObjectRefReconciliationError("cannot apply a plan with blocking missing references")
    if any(row.disposition is ObjectRefDisposition.AMBIGUOUS for row in plan.rows):
        raise ObjectRefReconciliationError("cannot apply an ambiguous plan")
    if not plan.forward:
        return
    savepoint = "assertion_transition"
    conn.execute(f"SAVEPOINT {savepoint}")
    try:
        for old, new in plan.forward:
            conn.execute("UPDATE assertions SET target_ref = ? WHERE target_ref = ?", (new, old))
            conn.execute("UPDATE assertions SET scope_ref = ? WHERE scope_ref = ?", (new, old))
            for column in ("evidence_refs_json", "supersedes_json"):
                rows = conn.execute(f"SELECT assertion_id, {column} FROM assertions").fetchall()
                for assertion_id, raw in rows:
                    values = json.loads(str(raw))
                    changed = [new if value == old else value for value in values]
                    if changed != values:
                        conn.execute(
                            f"UPDATE assertions SET {column} = ? WHERE assertion_id = ?",
                            (json.dumps(changed), assertion_id),
                        )
        if append_audit is not None:
            append_audit(plan)
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
    except Exception:
        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise


def _ref_text(value: str | ObjectRef) -> str:
    text = value.format() if isinstance(value, ObjectRef) else value
    try:
        return ObjectRef.parse(text).format()
    except (TypeError, ValueError) as exc:
        raise ObjectRefReconciliationError(f"invalid ObjectRef: {text!r}") from exc


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AssertionTransitionPlan",
    "IdentityMigrationMap",
    "ObjectRefDisposition",
    "ObjectRefReconciliationError",
    "ReconciledObjectRef",
    "SourceIdentityClaims",
    "TransitionBinding",
    "apply_assertion_transition",
    "enumerate_assertion_object_refs",
    "reconcile_object_refs",
]

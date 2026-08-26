"""Compile the finite exact-source/candidate check plan.

This is deliberately a projection, not a catalogue.  Domain owners remain
responsible for the predicate, oracle, production route, and result.  The
compiler only freezes the membership and bindings needed by a bounded run.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from polylogue.core.json import JSONDocument, json_document
from polylogue.operations.specs import RUNTIME_OPERATION_SPECS

PlanPhase = Literal["source", "candidate"]
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_KNOWN_OPERATIONS = frozenset(spec.name for spec in RUNTIME_OPERATION_SPECS)


class DomainCheckPlanError(ValueError):
    """Raised when declarations cannot safely authorize a bounded run."""


@dataclass(frozen=True, slots=True)
class DomainCheckDeclaration:
    """The plan-facing facts naturally owned by one domain law."""

    identity: str
    version: int
    owner_operation: str
    phase: PlanPhase
    denominator: str
    target_bindings: tuple[str, ...]
    production_route: str
    oracle_reference: str
    candidate_applicability: Literal["required", "not-applicable"] = "required"

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "identity": self.identity,
                "version": self.version,
                "owner_operation": self.owner_operation,
                "phase": self.phase,
                "denominator": self.denominator,
                "target_bindings": list(self.target_bindings),
                "production_route": self.production_route,
                "oracle_reference": self.oracle_reference,
                "candidate_applicability": self.candidate_applicability,
            }
        )


@dataclass(frozen=True, slots=True)
class DomainCheckPlanRow:
    """A minimal immutable plan row; semantic results live with the owner."""

    identity: str
    version: int
    owner_operation: str
    phase: PlanPhase
    denominator: str
    target_bindings: tuple[str, ...]

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "identity": self.identity,
                "version": self.version,
                "owner_operation": self.owner_operation,
                "phase": self.phase,
                "denominator": self.denominator,
                "target_bindings": list(self.target_bindings),
            }
        )


@dataclass(frozen=True, slots=True)
class DomainCheckPlan:
    """Deterministically ordered, disposable source/candidate run evidence."""

    phase: PlanPhase
    rows: tuple[DomainCheckPlanRow, ...]

    @property
    def member_identities(self) -> tuple[str, ...]:
        return tuple(f"{row.identity}@{row.version}" for row in self.rows)

    def to_dict(self) -> JSONDocument:
        return json_document({"phase": self.phase, "rows": [row.to_dict() for row in self.rows]})

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> DomainCheckPlan:
        phase = payload.get("phase")
        rows = payload.get("rows")
        if phase not in {"source", "candidate"} or not isinstance(rows, list):
            raise DomainCheckPlanError("malformed domain check plan")
        parsed: list[DomainCheckPlanRow] = []
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise DomainCheckPlanError("plan row must be an object")
            try:
                parsed.append(
                    DomainCheckPlanRow(
                        identity=str(raw["identity"]),
                        version=int(raw["version"]),
                        owner_operation=str(raw["owner_operation"]),
                        phase=raw["phase"],  # type: ignore[arg-type]
                        denominator=str(raw["denominator"]),
                        target_bindings=tuple(str(item) for item in raw["target_bindings"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise DomainCheckPlanError(f"malformed plan row: {exc}") from exc
        plan = compile_domain_check_plan(parsed, phase=phase)  # type: ignore[arg-type]
        if plan.to_dict() != json_document(dict(payload)):
            raise DomainCheckPlanError("plan is not canonical")
        return plan


def _declaration_row(declaration: DomainCheckDeclaration, phase: PlanPhase) -> DomainCheckPlanRow:
    if declaration.phase != phase:
        raise DomainCheckPlanError(
            f"phase/target mismatch for {declaration.identity!r}: {declaration.phase} != {phase}"
        )
    return DomainCheckPlanRow(
        identity=declaration.identity,
        version=declaration.version,
        owner_operation=declaration.owner_operation,
        phase=phase,
        denominator=declaration.denominator,
        target_bindings=tuple(sorted(declaration.target_bindings)),
    )


def compile_domain_check_plan(
    declarations: Iterable[DomainCheckDeclaration | DomainCheckPlanRow],
    *,
    phase: PlanPhase,
) -> DomainCheckPlan:
    """Validate and compile declarations independently of declaration order."""

    rows: list[DomainCheckPlanRow] = []
    seen: set[tuple[str, int]] = set()
    for declaration in declarations:
        if isinstance(declaration, DomainCheckPlanRow):
            row = declaration
            if row.phase != phase:
                raise DomainCheckPlanError(f"phase/target mismatch for {row.identity!r}")
        else:
            if not declaration.identity.strip() or declaration.version <= 0:
                raise DomainCheckPlanError("check identity and positive version are required")
            if declaration.owner_operation not in _KNOWN_OPERATIONS:
                raise DomainCheckPlanError(f"unknown owning operation: {declaration.owner_operation}")
            if not declaration.production_route.strip() or not declaration.oracle_reference.strip():
                raise DomainCheckPlanError(f"check {declaration.identity!r} has no production owner/oracle")
            if not declaration.denominator.strip():
                raise DomainCheckPlanError(f"check {declaration.identity!r} has no denominator")
            if not declaration.target_bindings:
                raise DomainCheckPlanError(f"check {declaration.identity!r} has no target binding")
            if declaration.candidate_applicability == "not-applicable" and phase == "candidate":
                continue
            if declaration.candidate_applicability not in {"required", "not-applicable"}:
                raise DomainCheckPlanError(f"invalid non-applicability for {declaration.identity!r}")
            row = _declaration_row(declaration, phase)
        key = (row.identity, row.version)
        if not row.identity.strip() or row.version <= 0:
            raise DomainCheckPlanError("check identity and positive version are required")
        if key in seen:
            raise DomainCheckPlanError(f"duplicate check identity/version: {row.identity}@{row.version}")
        if row.owner_operation not in _KNOWN_OPERATIONS:
            raise DomainCheckPlanError(f"unknown owning operation: {row.owner_operation}")
        if not row.denominator.strip() or not row.target_bindings:
            raise DomainCheckPlanError(f"plan row {row.identity!r} has a weak denominator or target")
        seen.add(key)
        rows.append(row)
    rows.sort(key=lambda row: (row.identity, row.version, row.owner_operation, row.phase, row.target_bindings))
    return DomainCheckPlan(phase=phase, rows=tuple(rows))


def declarations_from_outcome_owners(
    owners: Iterable[object], *, phase: PlanPhase
) -> tuple[DomainCheckDeclaration, ...]:
    """Adapt existing domain-owned outcome declarations without storing them."""
    result: list[DomainCheckDeclaration] = []
    for owner in owners:
        name = str(getattr(owner, "name", ""))
        routes = getattr(owner, "applicable_routes", frozenset())
        candidate_routes = {
            "reindex-index-candidate",
            "reindex-cross-tier-candidate",
            "reindex-canary-candidate",
            "corpus-fidelity",
        }
        if phase == "candidate" and not candidate_routes.intersection(routes):
            continue
        result.append(
            DomainCheckDeclaration(
                identity=name,
                version=1,
                owner_operation="candidate-build",
                phase=phase,
                denominator="|".join(str(item) for item in getattr(owner, "population", ())),
                target_bindings=tuple(
                    sorted(str(route) for route in routes if phase == "source" or route in candidate_routes)
                ),
                production_route=str(getattr(owner, "production_route", "")),
                oracle_reference=str(getattr(owner, "owned_reference", "")),
            )
        )
    return tuple(result)


__all__ = [
    "DomainCheckDeclaration",
    "DomainCheckPlan",
    "DomainCheckPlanError",
    "DomainCheckPlanRow",
    "compile_domain_check_plan",
    "declarations_from_outcome_owners",
]

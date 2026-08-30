"""Read-only operations used to inspect an inactive candidate.

The four operations in this module are deliberately concrete.  They share
only small value objects for identities and do not form a proof/result
hierarchy or persist execution state.  Each operation validates the result
shape it owns before returning it to a caller.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from polylogue.core.outcomes import OutcomeCheck, OutcomeStatus
from polylogue.maintenance.assertion_transition import (
    AssertionTransitionPlan,
    IdentityMigrationMap,
    SourceIdentityClaims,
    TransitionBinding,
    reconcile_object_refs,
)


class CandidateProofError(ValueError):
    """A candidate operation received incomplete or mismatched evidence."""


class _NamedOwner(Protocol):
    name: str


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateCheckPlan:
    """Exact compiled membership and owner dispatch for candidate checks."""

    members: tuple[str, ...]
    digest: str

    @classmethod
    def compile(cls, owners: Iterable[_NamedOwner]) -> CandidateCheckPlan:
        members = tuple(owner.name for owner in owners)
        if not members or len(members) != len(set(members)):
            raise CandidateProofError("candidate check plan must contain unique domain members")
        return cls(members, _digest(members))

    def validate(self, names: Iterable[str]) -> None:
        observed = tuple(names)
        if observed != self.members:
            raise CandidateProofError("candidate results do not exactly match the compiled check plan")
        if self.digest != _digest(self.members):
            raise CandidateProofError("candidate check plan digest is stale")


@dataclass(frozen=True, slots=True)
class CandidateSemanticRequest:
    candidate_root: Path
    plan: CandidateCheckPlan


@dataclass(frozen=True, slots=True)
class CandidateSemanticResult:
    candidate_root: Path
    plan_digest: str
    checks: tuple[OutcomeCheck, ...]

    def validate(self, plan: CandidateCheckPlan) -> CandidateSemanticResult:
        if self.plan_digest != plan.digest:
            raise CandidateProofError("candidate result is bound to a different check plan")
        plan.validate(check.name for check in self.checks)
        if any(not isinstance(check.status, OutcomeStatus) for check in self.checks):
            raise CandidateProofError("candidate check has an unknown status")
        return self


def run_candidate_semantic(request: CandidateSemanticRequest, *, sample_limit: int = 10) -> CandidateSemanticResult:
    """Run precisely the owner checks named by ``request.plan``."""

    # Keep the pure operation contract importable without optional parser
    # dependencies; opening an archive happens only in this execution route.
    from polylogue.maintenance.archive_verification import archive_verification_domain_adapters

    owners = archive_verification_domain_adapters(
        request.candidate_root,
        sample_limit=sample_limit,
        index_path_override=request.candidate_root,
        active_index_context="unavailable_for_candidate",
    )
    by_name = {owner.name: owner for owner in owners}
    applicable = {name for name, owner in by_name.items() if owner.candidate_check is not None}
    if applicable != set(request.plan.members):
        raise CandidateProofError("compiled candidate plan is not the current domain-owner plan")
    checks: list[OutcomeCheck] = []
    for name in request.plan.members:
        owner = by_name[name]
        if owner.candidate_check is None:
            raise CandidateProofError(f"candidate check is not applicable: {name}")
        result = owner.candidate_check()
        if result.name != name:
            raise CandidateProofError(f"candidate check returned mismatched name: {result.name!r}")
        checks.append(result)
    return CandidateSemanticResult(request.candidate_root, request.plan.digest, tuple(checks)).validate(request.plan)


@dataclass(frozen=True, slots=True)
class ConservationTerm:
    name: str
    source_count: int
    candidate_count: int

    def __post_init__(self) -> None:
        if min(self.source_count, self.candidate_count) < 0 or not self.name:
            raise CandidateProofError("conservation terms require a name and non-negative counts")


@dataclass(frozen=True, slots=True)
class CandidateFidelityRequest:
    source_manifest_digest: str
    source_items: tuple[str, ...]
    candidate_items: tuple[str, ...]
    terms: tuple[ConservationTerm, ...]


@dataclass(frozen=True, slots=True)
class CandidateFidelityResult:
    source_manifest_digest: str
    terms: tuple[ConservationTerm, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]

    @property
    def balanced(self) -> bool:
        return not self.missing and not self.unexpected and all(t.source_count == t.candidate_count for t in self.terms)


def check_candidate_fidelity(request: CandidateFidelityRequest) -> CandidateFidelityResult:
    source = tuple(request.source_items)
    candidate = tuple(request.candidate_items)
    if len(source) != len(set(source)) or len(candidate) != len(set(candidate)):
        raise CandidateProofError("source and candidate identities must be unique")
    if not request.source_manifest_digest:
        raise CandidateProofError("source manifest digest is required")
    if len(request.terms) != len({term.name for term in request.terms}):
        raise CandidateProofError("conservation terms must be unique")
    if sum(term.source_count for term in request.terms) != len(source) or sum(
        term.candidate_count for term in request.terms
    ) != len(candidate):
        raise CandidateProofError("conservation terms do not cover the sealed source and candidate denominators")
    missing = tuple(sorted(set(source) - set(candidate)))
    unexpected = tuple(sorted(set(candidate) - set(source)))
    return CandidateFidelityResult(request.source_manifest_digest, request.terms, missing, unexpected)


@dataclass(frozen=True, slots=True)
class PopulationCoverageRequest:
    archive_root: Path


@dataclass(frozen=True, slots=True)
class PopulationCoverageRow:
    population: str
    owner: str
    disposition: str
    check: str


@dataclass(frozen=True, slots=True)
class PopulationCoverageResult:
    rows: tuple[PopulationCoverageRow, ...]

    def validate(self) -> PopulationCoverageResult:
        keys = [row.population for row in self.rows]
        if len(keys) != len(set(keys)) or any(
            not row.owner or not row.disposition or not row.check for row in self.rows
        ):
            raise CandidateProofError("population coverage must have one owner, disposition, and check per population")
        return self


def reverse_population_coverage(request: PopulationCoverageRequest) -> PopulationCoverageResult:
    """Enumerate physical schema populations from the canonical tier DDL."""

    from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS

    rows: list[PopulationCoverageRow] = []
    for tier, spec in ARCHIVE_TIER_SPECS.items():
        path = request.archive_root / spec.filename
        if not path.exists():
            continue
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            objects = conn.execute(
                "SELECT type, name FROM sqlite_master WHERE type IN ('table','view') ORDER BY type,name"
            ).fetchall()
        finally:
            conn.close()
        for kind, name in objects:
            population = f"{tier.value}.{name}"
            rows.append(PopulationCoverageRow(population, f"storage.{tier.value}", kind, "schema-declaration"))
    return PopulationCoverageResult(tuple(rows)).validate()


@dataclass(frozen=True, slots=True)
class TransitionPlanningRequest:
    user_refs: tuple[str, ...]
    candidate_refs: tuple[str, ...]
    source_claims: SourceIdentityClaims
    binding: TransitionBinding
    predecessor_refs: tuple[str, ...] = ()
    migration_map: IdentityMigrationMap | None = None


def plan_candidate_transition(request: TransitionPlanningRequest) -> AssertionTransitionPlan:
    """Produce an immutable ObjectRef transition plan without writing tiers."""

    return reconcile_object_refs(
        request.user_refs,
        candidate_refs=request.candidate_refs,
        source_claims=request.source_claims,
        predecessor_refs=request.predecessor_refs,
        migration_map=request.migration_map,
        binding=request.binding,
    )


__all__ = [
    "CandidateCheckPlan",
    "CandidateFidelityRequest",
    "CandidateFidelityResult",
    "CandidateProofError",
    "CandidateSemanticRequest",
    "CandidateSemanticResult",
    "ConservationTerm",
    "PopulationCoverageRequest",
    "PopulationCoverageResult",
    "PopulationCoverageRow",
    "TransitionPlanningRequest",
    "check_candidate_fidelity",
    "plan_candidate_transition",
    "reverse_population_coverage",
    "run_candidate_semantic",
]

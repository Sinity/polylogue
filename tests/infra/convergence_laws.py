"""Architecture-neutral convergence laws and their shared experiment input.

The workload is made from the existing archive pathology composer.  The oracle
only sees those authoritative ``Session`` values.  It does not inspect a
derived table, a route implementation, a generation directory, or a write
receipt.  Variant runners in the convergence experiment provide observations
through the small :class:`ConvergenceRoute` protocol below.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from polylogue.archive.models import Session
from polylogue.core.enums import Provider
from polylogue.core.identity_law import block_id, message_id
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.scenarios import BudgetMeasure, WorkloadBudget, WorkloadEnvelopeSpec, WorkloadInputRef
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.pathology_composer import ComposedPathology


class ConvergenceLaw(StrEnum):
    """Semantic laws shared by both experimental routes."""

    PERMUTATION = "permutation"
    BATCHING = "batching"
    IDEMPOTENCE = "idempotence"
    LOCALITY = "locality"


class ConvergenceCase(StrEnum):
    """Admissible schedule and fault cases in the experiment contract."""

    CLEAN = "clean"
    INCREMENTAL = "incremental"
    PERMUTATION = "permutation"
    BATCHING = "batching"
    REPLACEMENT = "replacement"
    DELETION = "deletion"
    VALID_EMPTY = "valid-empty"
    MISSING = "missing"
    STALE = "stale"
    EXCESS = "excess"
    DUPLICATE = "duplicate"
    POISON_SIBLING = "poison-sibling"
    BOUNDED_YIELD = "bounded-yield"
    CRASH_BEFORE_PUBLICATION = "crash-before-publication"
    CRASH_AFTER_PUBLICATION = "crash-after-publication"
    RESTART = "restart"
    GENERATION_MISMATCH = "generation-mismatch"
    UNCHANGED_SECOND_PASS = "unchanged-second-pass"


class ConvergenceMutant(StrEnum):
    """Required controlled production-seam mutants."""

    ORDER_SENSITIVE_OVERWRITE = "order-sensitive-overwrite"
    OMITTED_BATCH_MEMBER = "omitted-batch-member"
    UNCONDITIONAL_REWRITE = "unconditional-rewrite-publication"
    STALE_EXCESS_RETENTION = "stale-excess-retention"
    OVER_BROAD_INVALIDATION = "over-broad-invalidation"


_PROVIDER = Provider.CODEX
_PROBE_TERMS = ("revision", "shared", "orphaned", "lineage")
_LAW_SET = tuple(law.value for law in ConvergenceLaw)
_CASE_SET = tuple(case.value for case in ConvergenceCase)
_MUTANT_SET = tuple(mutant.value for mutant in ConvergenceMutant)
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class AuthoritativeMessage:
    """The input fields used by the independent semantic oracle."""

    native_id: str
    role: str
    text: str


@dataclass(frozen=True, slots=True)
class AuthoritativeSession:
    """A selected logical session revision, independent of storage rows."""

    native_id: str
    revision: int
    messages: tuple[AuthoritativeMessage, ...]
    parent_native_id: str | None = None

    @property
    def session_id(self) -> str:
        return str(make_session_id(_PROVIDER, self.native_id))


@dataclass(frozen=True, slots=True)
class SemanticProjection:
    """The only route output compared by the convergence laws."""

    # FTS membership is a document set for each probe term.  Block ids are
    # public semantic identities here, not opaque derivation addresses.
    fts_membership: tuple[tuple[str, tuple[str, ...]], ...]
    # One exact aggregate over message-role partitions.
    role_counts: tuple[tuple[str, int], ...]
    affected_partitions: tuple[str, ...] = ()

    def without_locality(self) -> tuple[object, object]:
        return self.fts_membership, self.role_counts


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    """Non-semantic execution envelope recorded in the input contract."""

    concurrency: int = 1
    batch_size: int = 1
    yield_limit: int | None = None
    crash_points: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RouteObservation:
    """Smallest common test seam beyond typed semantic reads."""

    projection: SemanticProjection
    publication_count: int = 0


@dataclass(frozen=True, slots=True)
class GeneratedConvergenceWorkload:
    """One immutable, generated workload over all six archive tiers."""

    workload_id: str
    seed: int
    pathology: ComposedPathology
    tiers: tuple[str, ...]
    probe_terms: tuple[str, ...]
    laws: tuple[str, ...]
    cases: tuple[str, ...]
    mutants: tuple[str, ...]

    @property
    def workload_spec(self) -> WorkloadEnvelopeSpec:
        return WorkloadEnvelopeSpec(
            workload_id=self.workload_id,
            family_id="polylogue-convergence-laws-v1",
            version=1,
            inputs=(
                WorkloadInputRef(
                    input_id=f"generated:{self.workload_id}",
                    corpus_id=self.pathology.name,
                    profile_id="convergence-laws-six-tier",
                    seed=self.seed,
                    distribution_refs=("tests.infra.pathology_composer",),
                ),
            ),
            phases=("generate", "clean", "incremental", "faults", "read"),
            concurrency=1,
            budgets=(WorkloadBudget(measure=BudgetMeasure.WALL_MS, maximum=300_000),),
        )

    @property
    def authoritative_sessions(self) -> tuple[AuthoritativeSession, ...]:
        """Select the highest declared revision for each logical session."""
        selected: dict[str, Session] = {}
        revisions: dict[str, int] = {}
        for session in self.pathology.sessions:
            native_id = str(session.id)
            revision_value = session.metadata.get("revision_index", 0)
            revision = revision_value if isinstance(revision_value, int) else 0
            if native_id not in selected or revision >= revisions[native_id]:
                selected[native_id] = session
                revisions[native_id] = revision
        result: list[AuthoritativeSession] = []
        for native_id in sorted(selected):
            session = selected[native_id]
            result.append(
                AuthoritativeSession(
                    native_id=native_id,
                    revision=revisions[native_id],
                    parent_native_id=None if session.parent_id is None else str(session.parent_id),
                    messages=tuple(
                        AuthoritativeMessage(
                            native_id=str(message.id),
                            role=str(message.role),
                            text="" if message.text is None else str(message.text),
                        )
                        for message in session.messages
                    ),
                )
            )
        return tuple(result)

    @property
    def digest(self) -> str:
        payload = {
            "workload_id": self.workload_id,
            "seed": self.seed,
            "tiers": self.tiers,
            "probe_terms": self.probe_terms,
            "sessions": [
                {
                    "native_id": session.native_id,
                    "revision": session.revision,
                    "parent_native_id": session.parent_native_id,
                    "messages": [
                        {
                            "native_id": message.native_id,
                            "role": message.role,
                            "text": message.text,
                        }
                        for message in session.messages
                    ],
                }
                for session in self.authoritative_sessions
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ExperimentInputContract:
    """Compact handoff contract consumed by either convergence variant."""

    workload_digest: str
    law_set: tuple[str, ...]
    route_identities: tuple[str, ...]
    expected_projections: tuple[SemanticProjection, ...]
    resource_profile: ResourceProfile
    mutant_set: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.workload_digest) != 64:
            raise ValueError("workload_digest must be a SHA-256 hex digest")
        if not self.route_identities or any(not item for item in self.route_identities):
            raise ValueError("at least one route identity is required")
        if set(self.law_set) != set(_LAW_SET):
            raise ValueError("contract law_set must contain the four shared laws")
        if set(self.mutant_set) != set(_MUTANT_SET):
            raise ValueError("contract mutant_set must contain every required mutant")

    def to_payload(self) -> dict[str, object]:
        return {
            "contract": "polylogue-convergence-experiment/v1",
            "workload_digest": self.workload_digest,
            "law_set": list(self.law_set),
            "route_identities": list(self.route_identities),
            "expected_projections": [
                {
                    "fts_membership": [[term, list(ids)] for term, ids in projection.fts_membership],
                    "role_counts": [[key, count] for key, count in projection.role_counts],
                    "affected_partitions": list(projection.affected_partitions),
                }
                for projection in self.expected_projections
            ],
            "resource_profile": {
                "concurrency": self.resource_profile.concurrency,
                "batch_size": self.resource_profile.batch_size,
                "yield_limit": self.resource_profile.yield_limit,
                "crash_points": list(self.resource_profile.crash_points),
            },
            "mutant_set": list(self.mutant_set),
        }


class ConvergenceRoute(Protocol):
    """Small common seam implemented by both 04r9f variants."""

    route_id: str

    def run(self, workload: GeneratedConvergenceWorkload, *, case: ConvergenceCase) -> SemanticProjection:
        """Run one declared case and return typed semantic reads."""


def _session_block_id(session: AuthoritativeSession, message: AuthoritativeMessage, position: int) -> str:
    return block_id(message_id(session.session_id, message.native_id, position=0), position=position)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.casefold()))


def semantic_oracle(
    sessions: Sequence[AuthoritativeSession],
    *,
    probe_terms: Sequence[str] = _PROBE_TERMS,
    affected_partitions: Sequence[str] = (),
) -> SemanticProjection:
    """Derive expected FTS membership and role counts from authoritative input."""
    session_by_native_id = {session.native_id: session for session in sessions}
    logical_messages: dict[str, tuple[AuthoritativeMessage, ...]] = {}
    for session in sessions:
        messages = session.messages
        parent = session_by_native_id.get(session.parent_native_id or "")
        if parent is not None:
            shared = 0
            for child_message, parent_message in zip(messages, parent.messages, strict=False):
                if child_message != parent_message:
                    break
                shared += 1
            messages = messages[shared:]
        logical_messages[session.native_id] = messages

    fts: list[tuple[str, tuple[str, ...]]] = []
    for term in probe_terms:
        members: set[str] = set()
        normalized_term = term.casefold()
        for session in sessions:
            for message in logical_messages[session.native_id]:
                if normalized_term not in _tokens(message.text):
                    continue
                members.add(_session_block_id(session, message, 0))
        fts.append((term, tuple(sorted(members))))
    counts = Counter(message.role for session in sessions for message in logical_messages[session.native_id])
    return SemanticProjection(
        fts_membership=tuple(fts),
        role_counts=tuple(sorted((role, count) for role, count in counts.items())),
        affected_partitions=tuple(sorted(set(affected_partitions))),
    )


def expected_projection(workload: GeneratedConvergenceWorkload) -> SemanticProjection:
    return semantic_oracle(workload.authoritative_sessions, probe_terms=workload.probe_terms)


def read_semantic_projection(
    archive_root: str | Path,
    *,
    probe_terms: Sequence[str] = _PROBE_TERMS,
    affected_partitions: Sequence[str] = (),
) -> SemanticProjection:
    """Read the two semantic projections through ordinary archive APIs."""
    from polylogue.archive.query.predicate import QueryBoolPredicate
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = Path(archive_root).resolve()
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        fts_membership = tuple((term, tuple(sorted(archive.search_blocks(term)))) for term in probe_terms)
        rows = archive.query_unit_counts(
            "message",
            QueryBoolPredicate(op="and", children=()),
            group_by="role",
            sort="key",
            sort_direction="asc",
            limit=100_000,
        )
    role_counts = tuple(sorted((str(row.group_key), int(row.count)) for row in rows if row.group_key is not None))
    return SemanticProjection(
        fts_membership=fts_membership,
        role_counts=role_counts,
        affected_partitions=tuple(sorted(set(affected_partitions))),
    )


def changed_partitions(before: SemanticProjection, after: SemanticProjection) -> tuple[str, ...]:
    before_counts = dict(before.role_counts)
    after_counts = dict(after.role_counts)
    return tuple(
        sorted(
            role for role in set(before_counts) | set(after_counts) if before_counts.get(role) != after_counts.get(role)
        )
    )


def assert_permutation_law(projections: Sequence[SemanticProjection]) -> None:
    _require_nonempty(projections, "permutation")
    expected = projections[0].without_locality()
    if any(projection.without_locality() != expected for projection in projections[1:]):
        raise AssertionError("permutation law failed: typed FTS membership or role counts differ")


def assert_batching_law(bulk: SemanticProjection, batches: Sequence[SemanticProjection]) -> None:
    _require_nonempty(batches, "batching")
    if any(batch.without_locality() != bulk.without_locality() for batch in batches):
        raise AssertionError("batching law failed: typed FTS membership or role counts differ")


def assert_idempotence_law(first: SemanticProjection, second: SemanticProjection) -> None:
    if first.without_locality() != second.without_locality():
        raise AssertionError("idempotence law failed: unchanged repetition changed semantic output")


def assert_locality_law(
    before: SemanticProjection,
    after: SemanticProjection,
    declared_affected_partitions: Sequence[str],
) -> None:
    actual = set(changed_partitions(before, after))
    declared = set(declared_affected_partitions)
    if actual != declared:
        raise AssertionError(
            "locality law failed: declared role partitions are not exact "
            f"(changed={sorted(actual)}, declared={sorted(declared)})"
        )


def assert_unchanged_publication_law(first: RouteObservation, second: RouteObservation) -> None:
    if first.publication_count != second.publication_count:
        raise AssertionError(
            "unchanged repetition law failed: route published work despite unchanged semantic input "
            f"(first={first.publication_count}, second={second.publication_count})"
        )


def build_experiment_contract(
    workload: GeneratedConvergenceWorkload,
    *,
    route_identities: Sequence[str] = ("04r9f.variant-a", "04r9f.variant-b"),
    resource_profile: ResourceProfile = ResourceProfile(),
) -> ExperimentInputContract:
    return ExperimentInputContract(
        workload_digest=workload.digest,
        law_set=workload.laws,
        route_identities=tuple(route_identities),
        expected_projections=(expected_projection(workload),),
        resource_profile=resource_profile,
        mutant_set=workload.mutants,
    )


@lru_cache(maxsize=1)
def generated_convergence_workload() -> GeneratedConvergenceWorkload:
    """Build the sole deterministic workload used by the experiment."""
    from tests.infra.convergence_harness import rich_convergence_pathology

    return GeneratedConvergenceWorkload(
        workload_id="convergence-laws-six-tier-v1",
        seed=20260825,
        pathology=rich_convergence_pathology(),
        tiers=tuple(tier.value for tier in ArchiveTier),
        probe_terms=_PROBE_TERMS,
        laws=_LAW_SET,
        cases=_CASE_SET,
        mutants=_MUTANT_SET,
    )


def _require_nonempty(values: Sequence[object], law: str) -> None:
    if not values:
        raise AssertionError(f"{law} law requires at least one projection")


__all__ = [
    "AuthoritativeMessage",
    "AuthoritativeSession",
    "ConvergenceCase",
    "ConvergenceLaw",
    "ConvergenceMutant",
    "ConvergenceRoute",
    "ExperimentInputContract",
    "GeneratedConvergenceWorkload",
    "ResourceProfile",
    "RouteObservation",
    "SemanticProjection",
    "assert_batching_law",
    "assert_idempotence_law",
    "assert_locality_law",
    "assert_permutation_law",
    "assert_unchanged_publication_law",
    "build_experiment_contract",
    "changed_partitions",
    "expected_projection",
    "generated_convergence_workload",
    "read_semantic_projection",
    "semantic_oracle",
]

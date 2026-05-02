"""Typed proof-obligation primitives.

These models are intentionally runner-agnostic. They describe what can be
verified, which claim applies, how evidence should be produced, and the shape
of the resulting evidence without executing any runner.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from polylogue.core.json import JSONDocument, JSONValue, require_json_document, require_json_value
from polylogue.core.outcomes import OutcomeStatus

ClaimSeverity = Literal["info", "serious"]
CostTier = Literal["static", "unit", "integration", "live"]
EvidenceClass = Literal["smoke", "semantic", "structural", "trace", "performance", "workflow"]
NetworkPolicy = Literal["none", "optional", "required"]
TrustLevel = Literal["authored", "generated", "external"]

Oracle = Literal[
    "proof",
    "drift_check",
    "construction_sanity",
    "smoke",
    "manual_review",
    "ceremonial",
]
OracleKind = Literal[
    "proof",
    "drift_check",
    "construction_sanity",
    "smoke",
    "manual_review",
    "ceremonial",
]
EvidenceSource = Literal[
    "proof_catalog",
    "repo_static_analysis",
    "generated_fixture",
    "unit_test",
    "integration_test",
    "manual_review",
    "same_source_manifest",
]
IndependenceLevel = Literal[
    "independent",
    "cross_checked",
    "same_source",
    "self_attesting",
    "ceremonial",
]

AssuranceDomain = Literal[
    "operational_resilience",
    "surface_parity",
    "docs_media",
    "security_privacy",
    "distribution",
    "test_quality",
    "schema_correctness",
    "parser_correctness",
    "pipeline_correctness",
    "storage_correctness",
    "search_correctness",
    "site_publication",
    "mcp_surface",
    "cli_surface",
    "api_surface",
    "spec_completeness",
    "spec_accuracy",
    "migration_safety",
    "provider_coverage",
    "archive_integrity",
    "performance",
    "dependency_closure",
    "scenario_coverage",
    "mutation_coverage",
    "benchmark_coverage",
    "architecture_discipline",
    "unclassified",
]


def _json_mapping(items: dict[str, object]) -> JSONDocument:
    return {key: require_json_value(value, context=key) for key, value in items.items()}


def _json_value_tuple(values: Iterable[JSONValue]) -> tuple[JSONValue, ...]:
    return tuple(require_json_value(value, context="query value") for value in values)


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Source provenance for a discovered subject or generated evidence."""

    path: str
    line: int | None = None
    symbol: str | None = None

    @property
    def present(self) -> bool:
        return bool(self.path.strip())

    def to_payload(self) -> JSONDocument:
        return _json_mapping(
            {
                "path": self.path,
                "line": self.line,
                "symbol": self.symbol,
            }
        )


@dataclass(frozen=True, slots=True)
class SubjectRef:
    """A verifiable object discovered from code, schema, or an insight registry."""

    kind: str
    id: str
    attrs: JSONDocument = field(default_factory=dict)
    source_span: SourceSpan | None = None

    def attr(self, name: str) -> JSONValue | None:
        return self.attrs.get(name)

    def to_payload(self) -> JSONDocument:
        return _json_mapping(
            {
                "kind": self.kind,
                "id": self.id,
                "attrs": dict(self.attrs),
                "source_span": self.source_span.to_payload() if self.source_span is not None else None,
            }
        )


class SubjectQuery:
    """Serializable subject-query AST node."""

    op: str

    def matches(self, subject: SubjectRef) -> bool:
        raise NotImplementedError

    def to_payload(self) -> JSONDocument:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Kind(SubjectQuery):
    kind: str
    op: str = "kind"

    def matches(self, subject: SubjectRef) -> bool:
        return subject.kind == self.kind

    def to_payload(self) -> JSONDocument:
        return _json_mapping({"op": self.op, "kind": self.kind})


@dataclass(frozen=True, slots=True)
class AttrEq(SubjectQuery):
    attr: str
    value: JSONValue
    op: str = "attr_eq"

    def matches(self, subject: SubjectRef) -> bool:
        return subject.attr(self.attr) == self.value

    def to_payload(self) -> JSONDocument:
        return _json_mapping({"op": self.op, "attr": self.attr, "value": self.value})


@dataclass(frozen=True, slots=True)
class AttrIn(SubjectQuery):
    attr: str
    values: tuple[JSONValue, ...]
    op: str = "attr_in"

    def __init__(self, attr: str, values: Iterable[JSONValue]) -> None:
        object.__setattr__(self, "attr", attr)
        object.__setattr__(self, "values", _json_value_tuple(values))
        object.__setattr__(self, "op", "attr_in")

    def matches(self, subject: SubjectRef) -> bool:
        return subject.attr(self.attr) in set(self.values)

    def to_payload(self) -> JSONDocument:
        return _json_mapping({"op": self.op, "attr": self.attr, "values": list(self.values)})


@dataclass(frozen=True, slots=True)
class AttrContains(SubjectQuery):
    attr: str
    value: JSONValue
    op: str = "attr_contains"

    def matches(self, subject: SubjectRef) -> bool:
        current = subject.attr(self.attr)
        if isinstance(current, str) and isinstance(self.value, str):
            return self.value in current
        if isinstance(current, list):
            return self.value in current
        if isinstance(current, dict) and isinstance(self.value, str):
            return self.value in current
        return False

    def to_payload(self) -> JSONDocument:
        return _json_mapping({"op": self.op, "attr": self.attr, "value": self.value})


@dataclass(frozen=True, slots=True)
class And(SubjectQuery):
    children: tuple[SubjectQuery, ...]
    op: str = "and"

    def __init__(self, children: Iterable[SubjectQuery]) -> None:
        object.__setattr__(self, "children", tuple(children))
        object.__setattr__(self, "op", "and")

    def matches(self, subject: SubjectRef) -> bool:
        return all(child.matches(subject) for child in self.children)

    def to_payload(self) -> JSONDocument:
        return _json_mapping({"op": self.op, "children": [child.to_payload() for child in self.children]})


@dataclass(frozen=True, slots=True)
class Or(SubjectQuery):
    children: tuple[SubjectQuery, ...]
    op: str = "or"

    def __init__(self, children: Iterable[SubjectQuery]) -> None:
        object.__setattr__(self, "children", tuple(children))
        object.__setattr__(self, "op", "or")

    def matches(self, subject: SubjectRef) -> bool:
        return any(child.matches(subject) for child in self.children)

    def to_payload(self) -> JSONDocument:
        return _json_mapping({"op": self.op, "children": [child.to_payload() for child in self.children]})


@dataclass(frozen=True, slots=True)
class Not(SubjectQuery):
    child: SubjectQuery
    op: str = "not"

    def matches(self, subject: SubjectRef) -> bool:
        return not self.child.matches(subject)

    def to_payload(self) -> JSONDocument:
        return _json_mapping({"op": self.op, "child": self.child.to_payload()})


def subject_query_from_payload(payload: JSONDocument) -> SubjectQuery:
    """Deserialize a subject query payload."""
    op = payload.get("op")
    if op == "kind":
        return Kind(str(payload["kind"]))
    if op == "attr_eq":
        return AttrEq(str(payload["attr"]), require_json_value(payload.get("value"), context="attr_eq value"))
    if op == "attr_in":
        values = payload.get("values")
        if not isinstance(values, list):
            values = []
        return AttrIn(str(payload["attr"]), [require_json_value(value, context="attr_in value") for value in values])
    if op == "attr_contains":
        return AttrContains(
            str(payload["attr"]),
            require_json_value(payload.get("value"), context="attr_contains value"),
        )
    if op in {"and", "or"}:
        children_value = payload.get("children")
        children_payloads = children_value if isinstance(children_value, list) else []
        children = [
            subject_query_from_payload(require_json_document(child, context=f"{op} child"))
            for child in children_payloads
        ]
        return And(children) if op == "and" else Or(children)
    if op == "not":
        return Not(subject_query_from_payload(require_json_document(payload.get("child"), context="not child")))
    raise ValueError(f"unknown subject query op: {op}")


@dataclass(frozen=True, slots=True)
class BreakerMetadata:
    """How a claim is expected to fail when the protected contract regresses."""

    description: str
    issue: str | None = None
    command: tuple[str, ...] = ()

    def to_payload(self) -> JSONDocument:
        return _json_mapping(
            {
                "description": self.description,
                "issue": self.issue,
                "command": list(self.command),
            }
        )


@dataclass(frozen=True, slots=True)
class Claim:
    """Semantic property over one or more subject queries.

    Every claim declares an Oracle classification (how evidence is produced)
    and an AssuranceDomain (which confidence area the claim supports). Together
    they let the proof-pack surface answer \"what confidence does this change
    affect?\" with oracle-aware quality rather than an undifferentiated count.
    """

    id: str
    description: str
    subject_query: SubjectQuery
    evidence_schema: JSONDocument
    oracle: Oracle = "construction_sanity"
    oracle_kind: OracleKind | None = None
    assertion_source: EvidenceSource = "proof_catalog"
    observation_source: EvidenceSource = "repo_static_analysis"
    independence_level: IndependenceLevel = "independent"
    assurance_domain: AssuranceDomain = "unclassified"
    bug_classes: tuple[str, ...] = ()
    breaker: BreakerMetadata | None = None
    tracked_exception: str | None = None
    runner_classes: tuple[str, ...] = ()
    observed_facts: tuple[str, ...] = ()
    staleness_conditions: tuple[str, ...] = ()
    severity: ClaimSeverity = "serious"
    abstract: bool = False

    def matches(self, subject: SubjectRef) -> bool:
        return self.subject_query.matches(subject)

    def to_payload(self) -> JSONDocument:
        return _json_mapping(
            {
                "id": self.id,
                "description": self.description,
                "subject_query": self.subject_query.to_payload(),
                "evidence_schema": dict(self.evidence_schema),
                "oracle": self.oracle,
                "oracle_kind": self.oracle_kind or self.oracle,
                "assertion_source": self.assertion_source,
                "observation_source": self.observation_source,
                "independence_level": self.independence_level,
                "assurance_domain": self.assurance_domain,
                "bug_classes": list(self.bug_classes),
                "breaker": self.breaker.to_payload() if self.breaker is not None else None,
                "tracked_exception": self.tracked_exception,
                "runner_classes": list(self.runner_classes),
                "observed_facts": list(self.observed_facts),
                "staleness_conditions": list(self.staleness_conditions),
                "severity": self.severity,
                "abstract": self.abstract,
            }
        )


@dataclass(frozen=True, slots=True)
class EnvironmentContract:
    """Runner prerequisites that must be preserved for evidence to be meaningful."""

    required_commands: tuple[str, ...] = ()
    required_env: tuple[str, ...] = ()
    controlled_dimensions: tuple[str, ...] = ()
    uncontrolled_dimensions: tuple[str, ...] = ()
    network: NetworkPolicy = "none"
    live_archive: bool = False
    notes: tuple[str, ...] = ()

    def to_payload(self) -> JSONDocument:
        return _json_mapping(
            {
                "required_commands": list(self.required_commands),
                "required_env": list(self.required_env),
                "controlled_dimensions": list(self.controlled_dimensions),
                "uncontrolled_dimensions": list(self.uncontrolled_dimensions),
                "network": self.network,
                "live_archive": self.live_archive,
                "notes": list(self.notes),
            }
        )


@dataclass(frozen=True, slots=True)
class TrustMetadata:
    """Authorship, freshness, and trust boundary metadata for evidence or runners."""

    producer: str
    reviewed_at: str
    level: TrustLevel = "authored"
    expires_at: str | None = None
    privacy: str = "repo-local metadata only"
    code_revision: str | None = None
    dirty_state: bool | None = None
    schema_version: int | None = None
    provider_schema_digest: str | None = None
    input_fingerprint: str | None = None
    environment_fingerprint: str | None = None
    runner_version: str | None = None
    freshness: str | None = None
    origin: str | None = None

    def expires_before(self, now: datetime) -> bool:
        if self.expires_at is None:
            return False
        try:
            expires_at = datetime.fromisoformat(self.expires_at)
        except ValueError:
            return True
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return expires_at < now

    def to_payload(self) -> JSONDocument:
        return _json_mapping(
            {
                "producer": self.producer,
                "reviewed_at": self.reviewed_at,
                "level": self.level,
                "expires_at": self.expires_at,
                "privacy": self.privacy,
                "code_revision": self.code_revision,
                "dirty_state": self.dirty_state,
                "schema_version": self.schema_version,
                "provider_schema_digest": self.provider_schema_digest,
                "input_fingerprint": self.input_fingerprint,
                "environment_fingerprint": self.environment_fingerprint,
                "runner_version": self.runner_version,
                "freshness": self.freshness,
                "origin": self.origin,
            }
        )


@dataclass(frozen=True, slots=True)
class RunnerBinding:
    """A runner contract bound to a claim."""

    id: str
    claim_id: str
    runner: str
    evidence_class: EvidenceClass
    cost_tier: CostTier
    freshness_policy: str
    environment: EnvironmentContract
    trust: TrustMetadata

    def to_payload(self) -> JSONDocument:
        return _json_mapping(
            {
                "id": self.id,
                "claim_id": self.claim_id,
                "runner": self.runner,
                "evidence_class": self.evidence_class,
                "cost_tier": self.cost_tier,
                "freshness_policy": self.freshness_policy,
                "environment": self.environment.to_payload(),
                "trust": self.trust.to_payload(),
            }
        )


@dataclass(frozen=True, slots=True)
class ProofObligation:
    """Compiled `(subject, claim, runner)` verification instance."""

    id: str
    subject: SubjectRef
    claim: Claim
    runner: RunnerBinding

    def to_payload(self) -> JSONDocument:
        return _json_mapping(
            {
                "id": self.id,
                "subject": self.subject.to_payload(),
                "claim_id": self.claim.id,
                "runner_id": self.runner.id,
            }
        )


@dataclass(frozen=True, slots=True)
class EvidenceEnvelope:
    """Result envelope produced by a runner for one proof obligation."""

    obligation_id: str
    status: OutcomeStatus
    evidence: JSONDocument
    counterexample: JSONDocument | None
    reproducer: tuple[str, ...]
    artifacts: tuple[str, ...]
    environment: JSONDocument
    provenance: SourceSpan | None
    fingerprint: str
    trust: TrustMetadata

    @classmethod
    def build(
        cls,
        *,
        obligation_id: str,
        status: OutcomeStatus,
        evidence: JSONDocument,
        trust: TrustMetadata,
        counterexample: JSONDocument | None = None,
        reproducer: tuple[str, ...] = (),
        artifacts: tuple[str, ...] = (),
        environment: JSONDocument | None = None,
        provenance: SourceSpan | None = None,
    ) -> EvidenceEnvelope:
        environment_payload = environment or {}
        fingerprint = _fingerprint(
            {
                "obligation_id": obligation_id,
                "status": status.value,
                "evidence": evidence,
                "counterexample": counterexample,
                "reproducer": list(reproducer),
                "artifacts": list(artifacts),
                "environment": environment_payload,
                "provenance": provenance.to_payload() if provenance is not None else None,
                "trust": trust.to_payload(),
            }
        )
        return cls(
            obligation_id=obligation_id,
            status=status,
            evidence=evidence,
            counterexample=counterexample,
            reproducer=reproducer,
            artifacts=artifacts,
            environment=environment_payload,
            provenance=provenance,
            fingerprint=fingerprint,
            trust=trust,
        )

    def to_payload(self) -> JSONDocument:
        return _json_mapping(
            {
                "obligation_id": self.obligation_id,
                "status": self.status.value,
                "evidence": dict(self.evidence),
                "counterexample": self.counterexample,
                "reproducer": list(self.reproducer),
                "artifacts": list(self.artifacts),
                "environment": dict(self.environment),
                "provenance": self.provenance.to_payload() if self.provenance is not None else None,
                "fingerprint": self.fingerprint,
                "trust": self.trust.to_payload(),
            }
        )


def _fingerprint(payload: JSONDocument) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "And",
    "AssuranceDomain",
    "AttrContains",
    "AttrEq",
    "AttrIn",
    "BreakerMetadata",
    "Claim",
    "ClaimSeverity",
    "CostTier",
    "EnvironmentContract",
    "EvidenceEnvelope",
    "EvidenceClass",
    "EvidenceSource",
    "IndependenceLevel",
    "Kind",
    "NetworkPolicy",
    "Not",
    "Or",
    "Oracle",
    "OracleKind",
    "ProofObligation",
    "RunnerBinding",
    "SourceSpan",
    "SubjectQuery",
    "SubjectRef",
    "TrustLevel",
    "TrustMetadata",
    "subject_query_from_payload",
]

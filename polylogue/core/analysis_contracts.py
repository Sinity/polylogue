"""Storage-free contracts shared by evidence and analysis consumers.

The archive has several typed analysis definitions (queries, metrics, cohorts,
experiments, and loops), but they share a small *protocol*, not a universal
row.  This module contains that protocol and its fail-closed laws.  Owners may
persist their own definitions and receipts in the tier appropriate to their
lifecycle; these DTOs do not create another ledger or executor.

The important distinction in a result is three-dimensional:

``enumeration``
    How completely members were enumerated (an exact enumeration is not a
    claim that the intended frame was completely captured).
``coverage``
    Which intended frame was observed and supported.
``measurement_authority``
    Who or what established the value (structural, provider-reported,
    model-derived, or judged).

The dimensions are deliberately independent and survive serialization.  A
single ``exact`` or ``confidence`` flag cannot represent that contract.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from polylogue.core.digest import nfc
from polylogue.core.enums import AssertionStatus
from polylogue.core.evidence_integrity import EvidenceIntegrityStatus
from polylogue.core.evidence_value import FrameCoverage
from polylogue.core.hashing import hash_payload
from polylogue.core.refs import ActorRef, ExecutionContextRef, ObjectRef


class AnalysisContractError(ValueError):
    """Raised when an analysis contract would overstate or lose evidence."""


class UnsupportedProtocolError(AnalysisContractError):
    """Raised before a definition or receipt is evaluated under an unknown protocol."""


class IncompatibleRelationError(AnalysisContractError):
    """Raised when relations cannot be safely composed."""


JsonValue: TypeAlias = str | int | float | bool | None | Mapping[str, "JsonValue"] | Sequence["JsonValue"]
DefinitionKind: TypeAlias = Literal[
    "query",
    "metric",
    "pattern",
    "cohort",
    "experiment",
    "improvement-loop",
    "ranker",
    "context-policy",
]
PrivacyClass: TypeAlias = Literal["public", "private", "sensitive", "secret"]
DurabilityTier: TypeAlias = Literal["ephemeral", "ops", "user", "audit"]
EnumerationStatus: TypeAlias = Literal[
    "exact",
    "capped",
    "sampled",
    "estimated",
    "inferred-partial",
    "not-applicable",
]
MeasurementAuthority: TypeAlias = Literal[
    "structural",
    "provider-reported",
    "catalog-derived",
    "rule-derived",
    "model-derived",
    "agent-declared",
    "judged",
]

_DEFINITION_KINDS = frozenset(
    {"query", "metric", "pattern", "cohort", "experiment", "improvement-loop", "ranker", "context-policy"}
)
_ENUMERATIONS = frozenset({"exact", "capped", "sampled", "estimated", "inferred-partial", "not-applicable"})
_AUTHORITIES = frozenset(
    {
        "structural",
        "provider-reported",
        "catalog-derived",
        "rule-derived",
        "model-derived",
        "agent-declared",
        "judged",
    }
)
_DURABILITY = frozenset({"ephemeral", "ops", "user", "audit"})
_PRIVACY = frozenset({"public", "private", "sensitive", "secret"})


def _canonical(value: object) -> object:
    """Canonicalize JSON-ish values without importing the analysis package."""

    if isinstance(value, Mapping):
        return {nfc(str(k)): _canonical(v) for k, v in sorted(value.items(), key=lambda item: nfc(str(item[0])))}
    if isinstance(value, (set, frozenset)):
        values = [_canonical(item) for item in value]
        return sorted(
            values, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        )
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, str):
        return nfc(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    raise TypeError(f"analysis contract value is not JSON-compatible: {type(value)!r}")


def _text(value: str, field_name: str) -> str:
    value = nfc(value).strip()
    if not value:
        raise AnalysisContractError(f"{field_name} cannot be empty")
    return value


def _refs(values: Iterable[ObjectRef | str], *, field_name: str) -> tuple[ObjectRef, ...]:
    parsed: list[ObjectRef] = []
    for value in values:
        try:
            ref = value if isinstance(value, ObjectRef) else ObjectRef.parse(value)
        except ValueError as exc:
            raise AnalysisContractError(f"{field_name} contains an invalid object ref") from exc
        parsed.append(ref)
    unique = {ref.format(): ref for ref in parsed}
    return tuple(unique[key] for key in sorted(unique))


def _ordered_refs(values: Iterable[ObjectRef | str], *, field_name: str) -> tuple[ObjectRef, ...]:
    """Normalize refs while retaining declared rank/order for relation members."""

    result: list[ObjectRef] = []
    seen: set[str] = set()
    for value in values:
        try:
            ref = value if isinstance(value, ObjectRef) else ObjectRef.parse(value)
        except ValueError as exc:
            raise AnalysisContractError(f"{field_name} contains an invalid object ref") from exc
        text = ref.format()
        if text in seen:
            raise AnalysisContractError(f"{field_name} must not contain duplicate refs")
        seen.add(text)
        result.append(ref)
    return tuple(result)


def _protocol_version(value: str) -> str:
    version = _text(value, "protocol_version")
    if not version.rsplit(".v", 1)[-1].isdigit() or ".v" not in version:
        raise UnsupportedProtocolError(f"protocol version must end in .vN: {version!r}")
    return version


@dataclass(frozen=True, slots=True)
class DefinitionIdentity:
    """Content-addressed identity for one typed analysis definition.

    ``protocol_version`` participates in the digest.  A semantic-version
    change therefore cannot silently reuse an old definition identity.
    """

    kind: DefinitionKind
    protocol_version: str
    content: Mapping[str, JsonValue]
    privacy_class: PrivacyClass = "private"
    durability: DurabilityTier = "ephemeral"
    promoted: bool = False
    cited: bool = False
    retention_policy: Mapping[str, JsonValue] | None = None
    excision_link: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _DEFINITION_KINDS:
            raise AnalysisContractError(f"unsupported definition kind: {self.kind!r}")
        object.__setattr__(self, "protocol_version", _protocol_version(self.protocol_version))
        if not isinstance(self.content, Mapping) or not self.content:
            raise AnalysisContractError("definition content must be a non-empty mapping")
        object.__setattr__(self, "content", _canonical(self.content))
        if self.privacy_class not in _PRIVACY:
            raise AnalysisContractError(f"unsupported privacy class: {self.privacy_class!r}")
        if self.durability not in _DURABILITY:
            raise AnalysisContractError(f"unsupported durability tier: {self.durability!r}")
        if (
            self.durability in {"user", "audit"}
            and (self.privacy_class in {"sensitive", "secret"})
            and not (self.promoted or self.cited)
        ):
            raise AnalysisContractError("sensitive durable definitions require explicit promotion or citation")
        if (
            self.durability in {"user", "audit"}
            and (self.promoted or self.cited)
            and (not self.retention_policy or not self.excision_link)
        ):
            raise AnalysisContractError("promoted/cited durable definitions require retention and excision")
        if self.excision_link is not None:
            _text(self.excision_link, "excision_link")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "protocol_version": self.protocol_version,
            "content": self.content,
        }

    @property
    def digest(self) -> str:
        return hash_payload(self.canonical_payload)

    @property
    def ref(self) -> ObjectRef:
        return ObjectRef(kind=self.kind, object_id=self.digest)  # type: ignore[arg-type]

    @property
    def ref_text(self) -> str:
        return self.ref.format()

    def compatible_with(self, other: DefinitionIdentity) -> bool:
        """Definitions are compatible only when kind, protocol, and content agree."""

        return self.canonical_payload == other.canonical_payload

    def require_compatible_with(self, other: DefinitionIdentity) -> None:
        if not self.compatible_with(other):
            raise UnsupportedProtocolError(
                f"incompatible definitions: {self.ref_text} and {other.ref_text} do not share a protocol/content"
            )

    @property
    def excisable(self) -> bool:
        return bool(self.excision_link)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.canonical_payload,
            "ref": self.ref_text,
            "privacy_class": self.privacy_class,
            "durability": self.durability,
            "promoted": self.promoted,
            "cited": self.cited,
            "retention_policy": None if self.retention_policy is None else _canonical(self.retention_policy),
            "excision_link": self.excision_link,
        }


@dataclass(frozen=True, slots=True)
class EvaluationWorld:
    """The exact world in which a claim-supporting execution was evaluated."""

    source_generation: str
    user_generation: str
    index_generation: str
    runtime_build_ref: str
    frame_ref: ObjectRef
    resolved_bounds: Mapping[str, JsonValue] = field(default_factory=dict)
    embedding_refs: tuple[ObjectRef, ...] = ()
    model_refs: tuple[ObjectRef, ...] = ()
    actor: ActorRef | None = None
    execution_context: ExecutionContextRef | None = None
    world_protocol_version: str = "polylogue.evaluation-world.v1"

    def __post_init__(self) -> None:
        for name in ("source_generation", "user_generation", "index_generation", "runtime_build_ref"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        _protocol_version(self.world_protocol_version)
        object.__setattr__(self, "resolved_bounds", _canonical(self.resolved_bounds))
        object.__setattr__(self, "embedding_refs", _refs(self.embedding_refs, field_name="embedding_refs"))
        object.__setattr__(self, "model_refs", _refs(self.model_refs, field_name="model_refs"))

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "source_generation": self.source_generation,
            "user_generation": self.user_generation,
            "index_generation": self.index_generation,
            "runtime_build_ref": self.runtime_build_ref,
            "frame_ref": self.frame_ref.format(),
            "resolved_bounds": self.resolved_bounds,
            "embedding_refs": [ref.format() for ref in self.embedding_refs],
            "model_refs": [ref.format() for ref in self.model_refs],
            "actor_ref": None if self.actor is None else self.actor.format(),
            "execution_context_id": None if self.execution_context is None else self.execution_context.context_id,
            "world_protocol_version": self.world_protocol_version,
        }

    @property
    def world_id(self) -> str:
        return f"evaluation-world:{hash_payload(self.canonical_payload)}"

    def differs_from(self, other: EvaluationWorld) -> bool:
        return self.canonical_payload != other.canonical_payload

    def to_dict(self) -> dict[str, object]:
        return {**self.canonical_payload, "world_id": self.world_id}


@dataclass(frozen=True, slots=True)
class RelationManifest:
    """Versioned relation metadata independent of relation persistence."""

    relation_ref: ObjectRef
    definition_ref: ObjectRef
    evaluation_world: EvaluationWorld
    grain: str
    coverage: FrameCoverage
    enumeration: EnumerationStatus
    measurement_authority: tuple[MeasurementAuthority, ...]
    member_refs: tuple[ObjectRef, ...] = ()
    frame_ref: ObjectRef | None = None

    def __post_init__(self) -> None:
        if self.relation_ref.kind not in {"relation", "result-set", "cohort", "query-run", "match-set"}:
            raise AnalysisContractError("relation_ref must identify a relation-shaped object")
        _text(self.grain, "grain")
        if self.enumeration not in _ENUMERATIONS:
            raise AnalysisContractError(f"unsupported enumeration status: {self.enumeration!r}")
        authorities = tuple(dict.fromkeys(self.measurement_authority))
        if not authorities or any(authority not in _AUTHORITIES for authority in authorities):
            raise AnalysisContractError("relation measurement authority must use the closed vocabulary")
        object.__setattr__(self, "measurement_authority", tuple(sorted(authorities)))
        object.__setattr__(self, "member_refs", _ordered_refs(self.member_refs, field_name="member_refs"))

    @property
    def exact_enumeration(self) -> bool:
        return self.enumeration == "exact"

    @property
    def frame_complete(self) -> bool:
        return self.coverage.complete is True

    @property
    def authority(self) -> MeasurementAuthority:
        # The tuple is sorted for deterministic wire output.  ``authority``
        # is only a convenience and never replaces the complete tuple.
        return self.measurement_authority[0]

    def compatible_with(self, other: RelationManifest) -> bool:
        return self.grain == other.grain and self.definition_ref == other.definition_ref

    def require_compatible_with(self, other: RelationManifest) -> None:
        if not self.compatible_with(other):
            raise IncompatibleRelationError(
                f"relations are incompatible: grain {self.grain!r}/{other.grain!r} or definition "
                f"{self.definition_ref.format()!r}/{other.definition_ref.format()!r} differs"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "relation_ref": self.relation_ref.format(),
            "definition_ref": self.definition_ref.format(),
            "evaluation_world": self.evaluation_world.to_dict(),
            "grain": self.grain,
            "coverage": self.coverage.to_dict(),
            "enumeration": self.enumeration,
            "measurement_authority": list(self.measurement_authority),
            "member_refs": [ref.format() for ref in self.member_refs],
            "frame_ref": None if self.frame_ref is None else self.frame_ref.format(),
        }


@dataclass(frozen=True, slots=True)
class ResultEnvelope:
    """Public result contract preserving independent evidence axes."""

    result_ref: ObjectRef
    definition_ref: ObjectRef
    evaluation_world: EvaluationWorld
    relation: RelationManifest
    value_state: Literal["known", "unknown", "unavailable", "redacted"]
    value: object | None = None
    sampling_interval: Mapping[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if self.result_ref.kind not in {"result-set", "query-run", "relation", "analysis-run"}:
            raise AnalysisContractError("result_ref must identify a result-shaped object")
        if self.relation.definition_ref != self.definition_ref:
            raise AnalysisContractError("result definition and relation definition differ")
        if self.relation.evaluation_world.world_id != self.evaluation_world.world_id:
            raise AnalysisContractError("result relation was evaluated in a different world")
        if self.value_state == "known" and self.value is None:
            raise AnalysisContractError("known result values cannot be null")
        if self.sampling_interval is not None and self.relation.enumeration == "exact":
            raise AnalysisContractError("sampling intervals are not valid for exact enumeration")

    @property
    def enumeration(self) -> EnumerationStatus:
        return self.relation.enumeration

    @property
    def exact_enumeration(self) -> bool:
        return self.enumeration == "exact"

    @property
    def frame_complete(self) -> bool:
        return self.coverage.complete is True

    @property
    def model_derived(self) -> bool:
        return "model-derived" in self.measurement_authority

    @property
    def coverage(self) -> FrameCoverage:
        return self.relation.coverage

    @property
    def measurement_authority(self) -> tuple[MeasurementAuthority, ...]:
        return self.relation.measurement_authority

    def to_dict(self) -> dict[str, object]:
        return {
            "result_ref": self.result_ref.format(),
            "definition_ref": self.definition_ref.format(),
            "evaluation_world": self.evaluation_world.to_dict(),
            "relation": self.relation.to_dict(),
            "value_state": self.value_state,
            "value": self.value,
            "enumeration": self.enumeration,
            "coverage": self.coverage.to_dict(),
            "measurement_authority": list(self.measurement_authority),
            "sampling_interval": None if self.sampling_interval is None else _canonical(self.sampling_interval),
        }


@dataclass(frozen=True, slots=True)
class TypedReceiptEnvelope:
    """Execution receipt without imposing a persistence tier or table."""

    object_ref: ObjectRef
    definition: DefinitionIdentity
    evaluation_world: EvaluationWorld
    durability: DurabilityTier = "ops"
    privacy_class: PrivacyClass = "private"
    promoted: bool = False
    cited: bool = False
    retention_policy: Mapping[str, JsonValue] | None = None
    excision_link: str | None = None
    evidence_refs: tuple[ObjectRef, ...] = ()

    def __post_init__(self) -> None:
        if self.durability not in _DURABILITY or self.privacy_class not in _PRIVACY:
            raise AnalysisContractError("receipt has unsupported privacy or durability")
        if (
            self.durability in {"user", "audit"}
            and self.privacy_class in {"sensitive", "secret"}
            and not (self.promoted or self.cited)
        ):
            raise AnalysisContractError("sensitive receipt requires explicit promotion or citation")
        if (self.promoted or self.cited) and (not self.retention_policy or not self.excision_link):
            raise AnalysisContractError("promoted/cited receipt requires retention and excision")
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs, field_name="evidence_refs"))

    @property
    def excisable(self) -> bool:
        return bool(self.excision_link)

    def to_dict(self) -> dict[str, object]:
        return {
            "object_ref": self.object_ref.format(),
            "definition": self.definition.to_dict(),
            "evaluation_world": self.evaluation_world.to_dict(),
            "durability": self.durability,
            "privacy_class": self.privacy_class,
            "promoted": self.promoted,
            "cited": self.cited,
            "retention_policy": None if self.retention_policy is None else _canonical(self.retention_policy),
            "excision_link": self.excision_link,
            "evidence_refs": [ref.format() for ref in self.evidence_refs],
        }


@dataclass(frozen=True, slots=True)
class ExperimentReceipt:
    """Receipt needed before an analysis may make a causal claim."""

    definition_ref: ObjectRef
    assignment_ref: ObjectRef | None
    exposure_ref: ObjectRef | None
    frame_ref: ObjectRef | None
    exclusion_ref: ObjectRef | None
    stopping_ref: ObjectRef | None
    outcome_refs: tuple[ObjectRef, ...]
    evaluation_world: EvaluationWorld

    @property
    def causal_ready(self) -> bool:
        return bool(
            self.definition_ref.kind == "experiment"
            and self.assignment_ref
            and self.exposure_ref
            and self.frame_ref
            and self.exclusion_ref
            and self.stopping_ref
            and self.outcome_refs
        )

    def require_causal(self) -> None:
        if not self.causal_ready:
            raise AnalysisContractError(
                "causal claims require an ExperimentDefinition receipt with assignment, exposure, frame, "
                "stopping, and outcome refs"
            )


def claim_class_for(
    *, causal_requested: bool, experiment: ExperimentReceipt | None
) -> Literal["causal", "observational"]:
    """Return claim class, refusing causal language without a full receipt."""

    if not causal_requested:
        return "observational"
    if experiment is None:
        raise AnalysisContractError("causal claims require an ExperimentDefinition receipt")
    experiment.require_causal()
    return "causal"


@dataclass(frozen=True, slots=True)
class FindingRecord:
    """Finding projection over the canonical assertion lifecycle."""

    finding_ref: ObjectRef
    status: AssertionStatus
    evidence_refs: tuple[ObjectRef, ...]
    integrity_status: EvidenceIntegrityStatus
    definition_ref: ObjectRef
    frame_ref: ObjectRef
    judgment_ref: ObjectRef | None = None

    def __post_init__(self) -> None:
        if self.finding_ref.kind != "finding":
            raise AnalysisContractError("finding_ref must have finding kind")
        if not self.evidence_refs:
            raise AnalysisContractError("finding requires evidence refs")
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs, field_name="evidence_refs"))

    @property
    def current_supported(self) -> bool:
        return self.status in {AssertionStatus.ACTIVE, AssertionStatus.ACCEPTED} and self.integrity_status in {
            EvidenceIntegrityStatus.SUPPORTED,
            EvidenceIntegrityStatus.PARTIALLY_SUPPORTED,
        }

    @property
    def context_injectable(self) -> bool:
        return self.current_supported and self.integrity_status is EvidenceIntegrityStatus.SUPPORTED

    def with_status(self, status: AssertionStatus) -> FindingRecord:
        return FindingRecord(
            finding_ref=self.finding_ref,
            status=status,
            evidence_refs=self.evidence_refs,
            integrity_status=self.integrity_status,
            definition_ref=self.definition_ref,
            frame_ref=self.frame_ref,
            judgment_ref=self.judgment_ref,
        )


def claims_view(findings: Iterable[FindingRecord]) -> tuple[FindingRecord, ...]:
    """Project current-supported findings without creating a claims store.

    The assertion row remains the lifecycle authority.  This deterministic
    view is intentionally derived on demand and therefore cannot drift into a
    duplicate claim ledger.
    """

    return tuple(
        sorted(
            (finding for finding in findings if finding.current_supported), key=lambda item: item.finding_ref.format()
        )
    )


@dataclass(frozen=True, slots=True)
class BasketPointer:
    """Workspace pointer to versioned evidence; it never owns a second member store."""

    workspace_ref: ObjectRef
    relation_ref: ObjectRef
    relation_version: str
    evidence_refs: tuple[ObjectRef, ...] = ()

    def __post_init__(self) -> None:
        if self.workspace_ref.kind != "workspace":
            raise AnalysisContractError("basket workspace_ref must have workspace kind")
        _text(self.relation_version, "relation_version")
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs, field_name="evidence_refs"))

    @property
    def ref(self) -> ObjectRef:
        return ObjectRef(kind="basket", object_id=hash_payload(self.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "workspace_ref": self.workspace_ref.format(),
            "relation_ref": self.relation_ref.format(),
            "relation_version": self.relation_version,
            "evidence_refs": [ref.format() for ref in self.evidence_refs],
        }


@dataclass(frozen=True, slots=True)
class ImprovementLoopContract:
    """One scheduler/state contract reusable by multiple loop pilots."""

    loop_ref: ObjectRef
    protocol_version: str
    scheduler_ref: ObjectRef
    state_ref: ObjectRef
    pilot_key: str

    def __post_init__(self) -> None:
        if self.loop_ref.kind != "improvement-loop":
            raise AnalysisContractError("loop_ref must have improvement-loop kind")
        _protocol_version(self.protocol_version)
        if self.scheduler_ref.kind != "run" or self.state_ref.kind not in {"assertion", "workspace", "run"}:
            raise AnalysisContractError("loop scheduler/state refs do not use the shared loop contract")
        _text(self.pilot_key, "pilot_key")

    def compatible_with(self, other: ImprovementLoopContract) -> bool:
        return (
            self.protocol_version == other.protocol_version
            and self.scheduler_ref == other.scheduler_ref
            and self.state_ref.kind == other.state_ref.kind
        )


def require_shared_loop_contract(*loops: ImprovementLoopContract) -> None:
    """Reject per-loop scheduler/state forks before a second pilot activates."""

    if not loops:
        raise AnalysisContractError("at least one loop contract is required")
    first = loops[0]
    for loop in loops[1:]:
        if not first.compatible_with(loop):
            raise AnalysisContractError("improvement loops must share one scheduler/state contract")


__all__ = [
    "AnalysisContractError",
    "BasketPointer",
    "DefinitionIdentity",
    "DurabilityTier",
    "EnumerationStatus",
    "EvaluationWorld",
    "ExperimentReceipt",
    "FindingRecord",
    "ImprovementLoopContract",
    "IncompatibleRelationError",
    "JsonValue",
    "MeasurementAuthority",
    "PrivacyClass",
    "RelationManifest",
    "ResultEnvelope",
    "TypedReceiptEnvelope",
    "UnsupportedProtocolError",
    "claim_class_for",
    "claims_view",
    "require_shared_loop_contract",
]

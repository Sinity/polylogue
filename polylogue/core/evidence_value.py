"""Storage-free evidence values and conservative quantitative composition.

``EvidenceValue`` is a wire/domain protocol embedded by owning models. It does
not own persistence, fact lifecycle, or a universal registry. The helpers here
only validate declared axes and compose already-owned observations without
losing unknowns, contradictions, duplicate identity, or weak provenance.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Literal, TypeAlias, TypeVar, cast

from polylogue.core.refs import EvidenceRef, ObjectRef, PublicRef
from polylogue.core.temporal import (
    TemporalSource,
    TimeConfidence,
    time_confidence_for_source,
    weakest_of,
)
from polylogue.declarations import (
    CompatibilityKey,
    CompletenessEdge,
    DeclarationRegistry,
    DeclarationSpec,
    ExampleSpec,
    HandlerBinding,
    OutputSpec,
    validate_registry,
)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
N = TypeVar("N", int, float)
Numeric: TypeAlias = int | float

ValueState = Literal[
    "known",
    "unknown",
    "unavailable",
    "skipped",
    "not_applicable",
    "redacted",
]
MeasurementAuthority = Literal[
    "structural",
    "provider-reported",
    "catalog-derived",
    "rule-derived",
    "model-derived",
    "agent-declared",
    "judged",
]
EnumerationKind = Literal["census", "sample", "inferred-partial", "not-applicable"]
FreshnessState = Literal["fresh", "stale", "timed-out", "unavailable", "degraded"]
EvidenceAxis = Literal[
    "value_state",
    "measurement_authority",
    "evidence_refs",
    "definition_ref",
    "temporal",
    "enumeration",
    "coverage",
    "freshness",
    "calibrated_confidence",
]
ValueSchema = Literal["integer", "number", "string", "boolean", "object", "array"]

_VALUE_STATES: frozenset[ValueState] = frozenset(
    {"known", "unknown", "unavailable", "skipped", "not_applicable", "redacted"}
)
_AUTHORITIES: frozenset[MeasurementAuthority] = frozenset(
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
_ENUMERATIONS: frozenset[EnumerationKind] = frozenset({"census", "sample", "inferred-partial", "not-applicable"})
_FRESHNESS_STATES: frozenset[FreshnessState] = frozenset({"fresh", "stale", "timed-out", "unavailable", "degraded"})
_EVIDENCE_AXES: frozenset[EvidenceAxis] = frozenset(
    {
        "value_state",
        "measurement_authority",
        "evidence_refs",
        "definition_ref",
        "temporal",
        "enumeration",
        "coverage",
        "freshness",
        "calibrated_confidence",
    }
)
_FRESHNESS_WEAKNESS: Mapping[FreshnessState, int] = {
    "fresh": 0,
    "stale": 1,
    "degraded": 2,
    "timed-out": 3,
    "unavailable": 4,
}
_ENUMERATION_WEAKNESS: Mapping[EnumerationKind, int] = {
    "census": 0,
    "sample": 1,
    "inferred-partial": 2,
    "not-applicable": 3,
}


class EvidenceInvariantError(ValueError):
    """An evidence value or composition would violate a survivor law."""


@dataclass(frozen=True, slots=True)
class CoverageExclusion:
    """One explicitly excluded member of an intended frame."""

    subject_ref: ObjectRef
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise EvidenceInvariantError("coverage exclusion reason cannot be empty")

    def to_dict(self) -> dict[str, str]:
        return {"subject_ref": self.subject_ref.format(), "reason": self.reason}


@dataclass(frozen=True, slots=True)
class FrameCoverage:
    """Population/grain coverage independent of value and authority."""

    intended_frame: str
    grain: str
    denominator: str
    intended_count: int | None = None
    observed_count: int | None = None
    supported_count: int | None = None
    complete: bool | None = None
    intended_refs: tuple[ObjectRef, ...] = ()
    observed_refs: tuple[ObjectRef, ...] = ()
    exclusions: tuple[CoverageExclusion, ...] = ()

    def __post_init__(self) -> None:
        for name in ("intended_frame", "grain", "denominator"):
            if not str(getattr(self, name)).strip():
                raise EvidenceInvariantError(f"coverage {name} cannot be empty")
        for name in ("intended_count", "observed_count", "supported_count"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value < 0):
                raise EvidenceInvariantError(f"coverage {name} must be a non-negative integer")
        if (
            self.intended_count is not None
            and self.observed_count is not None
            and self.observed_count > self.intended_count
        ):
            raise EvidenceInvariantError("coverage observed_count cannot exceed intended_count")
        if (
            self.observed_count is not None
            and self.supported_count is not None
            and self.supported_count > self.observed_count
        ):
            raise EvidenceInvariantError("coverage supported_count cannot exceed observed_count")
        intended_refs = _normalize_object_refs(self.intended_refs)
        observed_refs = _normalize_object_refs(self.observed_refs)
        if intended_refs and any(ref not in set(intended_refs) for ref in observed_refs):
            raise EvidenceInvariantError("coverage observed_refs must belong to intended_refs")
        exclusions = tuple(
            sorted(
                _dedupe_by_key(self.exclusions, lambda item: (item.subject_ref.format(), item.reason)),
                key=lambda item: (item.subject_ref.format(), item.reason),
            )
        )
        object.__setattr__(self, "intended_refs", intended_refs)
        object.__setattr__(self, "observed_refs", observed_refs)
        object.__setattr__(self, "exclusions", exclusions)

    @property
    def ratio(self) -> float | None:
        intended_count = self.intended_count
        if intended_count is None or intended_count == 0 or self.observed_count is None:
            return None
        return self.observed_count / intended_count

    def to_dict(self) -> dict[str, object]:
        return {
            "intended_frame": self.intended_frame,
            "grain": self.grain,
            "denominator": self.denominator,
            "intended_count": self.intended_count,
            "observed_count": self.observed_count,
            "supported_count": self.supported_count,
            "complete": self.complete,
            "coverage_ratio": self.ratio,
            "intended_refs": [ref.format() for ref in self.intended_refs],
            "observed_refs": [ref.format() for ref in self.observed_refs],
            "exclusions": [item.to_dict() for item in self.exclusions],
        }


@dataclass(frozen=True, slots=True)
class TemporalProvenance:
    """Observation time plus the independently declared source-clock quality."""

    observed_at: str | None
    time_source: TemporalSource | None
    time_confidence: TimeConfidence
    frame_start: str | None = None
    frame_end: str | None = None

    def __post_init__(self) -> None:
        for name in ("observed_at", "frame_start", "frame_end"):
            value = getattr(self, name)
            if value is not None:
                _require_aware_iso8601(name, value)
        expected = time_confidence_for_source(self.time_source)
        if self.time_confidence != expected:
            raise EvidenceInvariantError(
                f"time_confidence {self.time_confidence!r} overstates source {self.time_source!r}; expected {expected!r}"
            )
        if (
            self.frame_start is not None
            and self.frame_end is not None
            and _parse_aware_iso8601(self.frame_start) > _parse_aware_iso8601(self.frame_end)
        ):
            raise EvidenceInvariantError("temporal frame_start cannot be after frame_end")

    @classmethod
    def from_source(
        cls,
        *,
        observed_at: str | None,
        time_source: TemporalSource | None,
        frame_start: str | None = None,
        frame_end: str | None = None,
    ) -> TemporalProvenance:
        return cls(
            observed_at=observed_at,
            time_source=time_source,
            time_confidence=time_confidence_for_source(time_source),
            frame_start=frame_start,
            frame_end=frame_end,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "observed_at": self.observed_at,
            "time_source": self.time_source,
            "time_confidence": self.time_confidence,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
        }


@dataclass(frozen=True, slots=True)
class FreshnessProvenance:
    """Current degradation state with an explainable last-good anchor."""

    state: FreshnessState
    evaluated_at: str | None
    cause: str | None = None
    last_good_at: str | None = None
    last_good_evidence_refs: tuple[PublicRef, ...] = ()

    def __post_init__(self) -> None:
        if self.state not in _FRESHNESS_STATES:
            raise EvidenceInvariantError(f"unsupported freshness state: {self.state!r}")
        if self.evaluated_at is not None:
            _require_aware_iso8601("freshness evaluated_at", self.evaluated_at)
        if self.last_good_at is not None:
            _require_aware_iso8601("freshness last_good_at", self.last_good_at)
        if self.state != "fresh" and not (self.cause or "").strip():
            raise EvidenceInvariantError(f"freshness state {self.state!r} requires a degradation cause")
        object.__setattr__(self, "last_good_evidence_refs", _normalize_public_refs(self.last_good_evidence_refs))

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "evaluated_at": self.evaluated_at,
            "cause": self.cause,
            "last_good_at": self.last_good_at,
            "last_good_evidence_refs": [_format_public_ref(ref) for ref in self.last_good_evidence_refs],
        }


@dataclass(frozen=True, slots=True)
class CalibratedConfidence:
    """Optional calibrated probability; never an unreferenced generic float."""

    value: float
    calibration_ref: ObjectRef
    definition_ref: ObjectRef

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or not 0.0 <= self.value <= 1.0:
            raise EvidenceInvariantError("calibrated confidence must be finite and within [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "calibration_ref": self.calibration_ref.format(),
            "definition_ref": self.definition_ref.format(),
        }


@dataclass(frozen=True, slots=True)
class EvidenceObservation(Generic[T_co]):
    """One independently identifiable observation used by reconciliation/sums."""

    fact_ref: ObjectRef
    value_state: ValueState
    value: T_co | None
    measurement_authority: tuple[MeasurementAuthority, ...]
    evidence_refs: tuple[PublicRef, ...]
    temporal: TemporalProvenance
    enumeration: EnumerationKind
    coverage: FrameCoverage
    freshness: FreshnessProvenance

    def __post_init__(self) -> None:
        _validate_value_state(self.value_state, self.value)
        _validate_authorities(self.measurement_authority)
        if self.enumeration not in _ENUMERATIONS:
            raise EvidenceInvariantError(f"unsupported enumeration: {self.enumeration!r}")
        object.__setattr__(self, "measurement_authority", _normalize_authorities(self.measurement_authority))
        object.__setattr__(self, "evidence_refs", _normalize_public_refs(self.evidence_refs))

    def to_dict(self) -> dict[str, object]:
        return {
            "fact_ref": self.fact_ref.format(),
            "value_state": self.value_state,
            "value": self.value,
            "measurement_authority": list(self.measurement_authority),
            "evidence_refs": [_format_public_ref(ref) for ref in self.evidence_refs],
            "temporal": self.temporal.to_dict(),
            "enumeration": self.enumeration,
            "coverage": self.coverage.to_dict(),
            "freshness": self.freshness.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EvidenceConflict(Generic[T_co]):
    """Order-independent contradictory observations for one fact identity."""

    fact_ref: ObjectRef
    reason: str
    observations: tuple[EvidenceObservation[T_co], ...]

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise EvidenceInvariantError("evidence conflict reason cannot be empty")
        if len(self.observations) < 2:
            raise EvidenceInvariantError("evidence conflict requires at least two observations")
        if any(observation.fact_ref != self.fact_ref for observation in self.observations):
            raise EvidenceInvariantError("conflict observations must share fact_ref")
        observations = tuple(sorted(self.observations, key=_observation_sort_key))
        object.__setattr__(self, "observations", observations)

    def to_dict(self) -> dict[str, object]:
        return {
            "fact_ref": self.fact_ref.format(),
            "reason": self.reason,
            "observations": [observation.to_dict() for observation in self.observations],
        }


@dataclass(frozen=True, slots=True)
class EvidenceValue(Generic[T_co]):
    """Typed public claim carrying independent evidence/provenance axes."""

    family: str
    fact_ref: ObjectRef
    value_state: ValueState
    value: T_co | None
    measurement_authority: tuple[MeasurementAuthority, ...]
    evidence_refs: tuple[PublicRef, ...]
    definition_ref: ObjectRef
    temporal: TemporalProvenance
    enumeration: EnumerationKind
    coverage: FrameCoverage
    freshness: FreshnessProvenance
    weakest_measurement_authority: MeasurementAuthority | None = None
    calibrated_confidence: CalibratedConfidence | None = None
    contributions: tuple[EvidenceObservation[T_co], ...] = ()
    conflicts: tuple[EvidenceConflict[T_co], ...] = ()

    def __post_init__(self) -> None:
        if not self.family.strip():
            raise EvidenceInvariantError("evidence family cannot be empty")
        _validate_value_state(self.value_state, self.value)
        _validate_authorities(self.measurement_authority)
        if self.enumeration not in _ENUMERATIONS:
            raise EvidenceInvariantError(f"unsupported enumeration: {self.enumeration!r}")
        authorities = _normalize_authorities(self.measurement_authority)
        if self.weakest_measurement_authority is not None and self.weakest_measurement_authority not in authorities:
            raise EvidenceInvariantError("weakest_measurement_authority must occur in measurement_authority")
        object.__setattr__(self, "measurement_authority", authorities)
        object.__setattr__(self, "evidence_refs", _normalize_public_refs(self.evidence_refs))
        object.__setattr__(
            self,
            "contributions",
            tuple(sorted(_dedupe_by_key(self.contributions, _observation_identity), key=_observation_sort_key)),
        )
        object.__setattr__(
            self,
            "conflicts",
            tuple(sorted(_dedupe_by_key(self.conflicts, _conflict_identity), key=lambda item: item.fact_ref.format())),
        )

    def as_observation(self) -> EvidenceObservation[T_co]:
        return EvidenceObservation(
            fact_ref=self.fact_ref,
            value_state=self.value_state,
            value=self.value,
            measurement_authority=self.measurement_authority,
            evidence_refs=self.evidence_refs,
            temporal=self.temporal,
            enumeration=self.enumeration,
            coverage=self.coverage,
            freshness=self.freshness,
        )

    def leaf_observations(self) -> tuple[EvidenceObservation[T_co], ...]:
        if not self.contributions and not self.conflicts:
            return (self.as_observation(),)
        observations = (
            *self.contributions,
            *(observation for conflict in self.conflicts for observation in conflict.observations),
        )
        return tuple(
            sorted(
                _dedupe_by_key(observations, _observation_identity),
                key=_observation_sort_key,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "fact_ref": self.fact_ref.format(),
            "value_state": self.value_state,
            "value": self.value,
            "measurement_authority": list(self.measurement_authority),
            "weakest_measurement_authority": self.weakest_measurement_authority,
            "evidence_refs": [_format_public_ref(ref) for ref in self.evidence_refs],
            "definition_ref": self.definition_ref.format(),
            "temporal": self.temporal.to_dict(),
            "enumeration": self.enumeration,
            "coverage": self.coverage.to_dict(),
            "freshness": self.freshness.to_dict(),
            "calibrated_confidence": (
                None if self.calibrated_confidence is None else self.calibrated_confidence.to_dict()
            ),
            "contributions": [observation.to_dict() for observation in self.contributions],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
        }


@dataclass(frozen=True, slots=True)
class FactFamilySpec:
    """Minimal evidence-family declaration, not a universal declaration kernel."""

    family: str
    owner: str
    source_adapter: str
    public_field: str
    renderer_label: str
    value_schema: ValueSchema
    unit: str
    grain: str
    denominator: str
    definition_ref: ObjectRef
    required_axes: frozenset[EvidenceAxis]
    allowed_states: frozenset[ValueState]
    allowed_authorities: frozenset[MeasurementAuthority]
    authority_precedence: tuple[MeasurementAuthority, ...] = ()
    requires_last_good_when_degraded: bool = False

    def __post_init__(self) -> None:
        for name in (
            "family",
            "owner",
            "source_adapter",
            "public_field",
            "renderer_label",
            "unit",
            "grain",
            "denominator",
        ):
            if not str(getattr(self, name)).strip():
                raise EvidenceInvariantError(f"fact family {name} cannot be empty")
        unknown_axes = set(self.required_axes) - _EVIDENCE_AXES
        if unknown_axes:
            raise EvidenceInvariantError(f"fact family has unsupported evidence axes: {sorted(unknown_axes)}")
        if not self.allowed_states:
            raise EvidenceInvariantError("fact family must allow at least one value state")
        if not self.allowed_authorities:
            raise EvidenceInvariantError("fact family must allow at least one authority")
        unknown_states = set(self.allowed_states) - _VALUE_STATES
        if unknown_states:
            raise EvidenceInvariantError(f"fact family has unsupported states: {sorted(unknown_states)}")
        unknown_authorities = set(self.allowed_authorities) - _AUTHORITIES
        if unknown_authorities:
            raise EvidenceInvariantError(f"fact family has unsupported authorities: {sorted(unknown_authorities)}")
        if len(set(self.authority_precedence)) != len(self.authority_precedence):
            raise EvidenceInvariantError("authority_precedence cannot contain duplicates")
        if set(self.authority_precedence) != set(self.allowed_authorities):
            raise EvidenceInvariantError("authority_precedence must enumerate every allowed authority exactly once")

    def validate(self, value: EvidenceValue[object]) -> tuple[str, ...]:
        diagnostics: list[str] = []
        if value.family != self.family:
            diagnostics.append(f"family expected {self.family!r}, got {value.family!r}")
        if value.definition_ref != self.definition_ref:
            diagnostics.append(
                f"definition_ref expected {self.definition_ref.format()!r}, got {value.definition_ref.format()!r}"
            )
        if value.value_state not in self.allowed_states:
            diagnostics.append(f"value_state {value.value_state!r} is not allowed")
        disallowed_authorities = set(value.measurement_authority) - set(self.allowed_authorities)
        if disallowed_authorities:
            diagnostics.append(f"measurement_authority not allowed: {sorted(disallowed_authorities)}")
        for axis in sorted(self.required_axes):
            if axis == "definition_ref" and not value.definition_ref.object_id:
                diagnostics.append("missing required definition_ref")
            elif axis == "temporal" and value.temporal.observed_at is None:
                diagnostics.append("missing required temporal observed_at")
            elif axis == "coverage" and not value.coverage.intended_frame:
                diagnostics.append("missing required coverage")
            elif axis == "calibrated_confidence" and value.calibrated_confidence is None:
                diagnostics.append("missing required calibrated_confidence")
        if value.coverage.grain != self.grain:
            diagnostics.append(f"coverage grain expected {self.grain!r}, got {value.coverage.grain!r}")
        if value.coverage.denominator != self.denominator:
            diagnostics.append(
                f"coverage denominator expected {self.denominator!r}, got {value.coverage.denominator!r}"
            )
        if (
            self.requires_last_good_when_degraded
            and value.freshness.state in {"stale", "timed-out", "degraded"}
            and not value.freshness.last_good_evidence_refs
        ):
            diagnostics.append("degraded value is missing last_good_evidence_refs")
        if value.value_state == "known" and not value.measurement_authority:
            diagnostics.append("known value is missing measurement_authority")
        if value.value_state == "known" and not value.evidence_refs:
            diagnostics.append("known value is missing evidence_refs")
        return tuple(diagnostics)

    def require(self, value: EvidenceValue[object]) -> None:
        diagnostics = self.validate(value)
        if diagnostics:
            raise EvidenceInvariantError(f"{self.family}: " + "; ".join(diagnostics))

    def public_schema(self) -> dict[str, object]:
        required = [
            "family",
            "fact_ref",
            "value_state",
            "value",
            "measurement_authority",
            "weakest_measurement_authority",
            "evidence_refs",
            "definition_ref",
            "temporal",
            "enumeration",
            "coverage",
            "freshness",
            "calibrated_confidence",
            "contributions",
            "conflicts",
        ]
        return {
            "family": self.family,
            "owner": self.owner,
            "source_adapter": self.source_adapter,
            "public_field": self.public_field,
            "renderer_label": self.renderer_label,
            "type": "object",
            "required": required,
            "value_schema": {"type": [self.value_schema, "null"]},
            "allowed_states": sorted(self.allowed_states),
            "allowed_authorities": sorted(self.allowed_authorities),
            "unit": self.unit,
            "grain": self.grain,
            "denominator": self.denominator,
            "definition_ref": self.definition_ref.format(),
        }

    def declaration(self) -> DeclarationSpec:
        """Project this domain declaration through the shared kernel.

        ``FactFamilySpec`` owns evidence-specific axes, while the generic
        declaration kernel owns deterministic registration, derivation, and
        source-locatable completeness diagnostics.  Keeping this as a
        projection prevents evidence families from growing a second registry.
        """

        owner_path = _owner_path_for_adapter(self.source_adapter)
        declaration_id = f"evidence.{self.family}"
        return DeclarationSpec(
            declaration_id=declaration_id,
            family_id=declaration_id,
            public_name=self.public_field,
            owner_path=owner_path,
            compatibility=CompatibilityKey(
                identity=self.family,
                lifecycle="owner-retained",
                authority="evidence-value",
                access_result_shape=self.value_schema,
                durability="owner-derived",
            ),
            producer=self.source_adapter,
            role_gate="public-read",
            schema_ref=f"EvidenceValue[{self.value_schema}]",
            discovery_text=f"{self.renderer_label}: {self.family}",
            repair_command="devtools verify --quick",
            handlers=(
                HandlerBinding(
                    surface="domain",
                    owner_path=owner_path,
                    symbol=self.source_adapter,
                    binding_key=declaration_id,
                ),
            ),
            outputs=(
                OutputSpec(
                    name=self.public_field,
                    kind="evidence-value",
                    schema_ref=f"EvidenceValue[{self.value_schema}]",
                    target_path=owner_path,
                ),
            ),
            examples=(
                ExampleSpec(
                    name="family-schema",
                    summary=f"Discover the {self.family} evidence family.",
                    arguments=(("family", self.family),),
                ),
            ),
            completeness_edges=(
                CompletenessEdge(
                    producer=self.source_adapter,
                    consumer=self.public_field,
                    kind="evidence-family-projection",
                    owner_path=owner_path,
                ),
            ),
        )

    def authority_rank(self, authority: MeasurementAuthority) -> int:
        try:
            return self.authority_precedence.index(authority)
        except ValueError as exc:
            raise EvidenceInvariantError(f"authority {authority!r} is not declared for {self.family}") from exc

    def weakest_authority(self, authorities: Sequence[MeasurementAuthority]) -> MeasurementAuthority | None:
        if not authorities:
            return None
        weakest = authorities[0]
        for authority in authorities[1:]:
            if self.authority_rank(authority) > self.authority_rank(weakest):
                weakest = authority
        return weakest


@dataclass(frozen=True, slots=True)
class FactFamilyDiagnostic:
    """Source-locatable completeness failure for an evidence-family inventory."""

    family: str
    owner: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"family": self.family, "owner": self.owner, "message": self.message}


def audit_fact_family_completeness(
    specs: Iterable[FactFamilySpec],
    values: Iterable[EvidenceValue[T]],
    *,
    required_families: Iterable[str] = (),
) -> tuple[FactFamilyDiagnostic, ...]:
    """Audit declarations and values without creating a global registry."""

    diagnostics: list[FactFamilyDiagnostic] = []
    by_family: dict[str, FactFamilySpec] = {}
    for spec in specs:
        if spec.family in by_family:
            diagnostics.append(FactFamilyDiagnostic(spec.family, spec.owner, "duplicate fact-family declaration"))
        else:
            by_family[spec.family] = spec
    registry = DeclarationRegistry()
    for family in sorted(by_family):
        registry.register(by_family[family].declaration())
    diagnostics.extend(
        FactFamilyDiagnostic(
            family=diagnostic.declaration_id.removeprefix("evidence."),
            owner=diagnostic.owner_path,
            message=diagnostic.message,
        )
        for diagnostic in validate_registry(registry)
    )
    for family in sorted(set(required_families)):
        if family not in by_family:
            diagnostics.append(FactFamilyDiagnostic(family, "unknown", "required fact-family declaration missing"))
    for value in values:
        family_spec = by_family.get(value.family)
        if family_spec is None:
            diagnostics.append(
                FactFamilyDiagnostic(value.family, "unknown", "public value has no fact-family declaration")
            )
            continue
        diagnostics.extend(
            FactFamilyDiagnostic(value.family, family_spec.owner, message) for message in family_spec.validate(value)
        )
    return tuple(sorted(diagnostics, key=lambda item: (item.family, item.owner, item.message)))


def refine_evidence_value(
    current: EvidenceValue[T],
    candidate: EvidenceValue[T],
    *,
    spec: FactFamilySpec,
) -> EvidenceValue[T]:
    """Refine one fact while conserving all support and explicit conflicts.

    A stronger authority claim must contribute evidence not already present in
    the current value. Contradictory equal-authority observations remain
    unresolved and order-independent; a declared strictly stronger authority
    may supersede while conserving the weaker observation's support refs.
    """

    spec.require(current)
    spec.require(candidate)
    if current.family != candidate.family or current.fact_ref != candidate.fact_ref:
        raise EvidenceInvariantError("refinement requires the same family and fact_ref")
    current_floor = spec.weakest_authority(current.measurement_authority)
    candidate_floor = spec.weakest_authority(candidate.measurement_authority)
    if (
        current_floor is not None
        and candidate_floor is not None
        and spec.authority_rank(candidate_floor) < spec.authority_rank(current_floor)
        and set(candidate.evidence_refs) <= set(current.evidence_refs)
    ):
        raise EvidenceInvariantError("measurement authority cannot strengthen without new evidence")
    return _reconcile_same_fact(
        (*current.leaf_observations(), *candidate.leaf_observations()),
        spec=spec,
    )


def sum_evidence_values(
    values: Iterable[EvidenceValue[N]],
    *,
    spec: FactFamilySpec,
    fact_ref: ObjectRef,
    observed_at: str,
    intended_frame: str,
    expected_fact_refs: Sequence[ObjectRef] | None = None,
) -> EvidenceValue[N]:
    """Sum disjoint quantitative facts without zero-coercion or double count.

    Nested aggregates are flattened to their original contributions. Repeated
    identical fact identities deduplicate. Contradictory duplicates, missing
    expected facts, or non-known contributions make the total unknown while
    preserving every observation and conflict witness.
    """

    _require_aware_iso8601("aggregate observed_at", observed_at)
    materialized = tuple(values)
    for value in materialized:
        spec.require(value)
        if value.family != spec.family:
            raise EvidenceInvariantError("sum inputs must share the declared fact family")

    observations = tuple(observation for value in materialized for observation in value.leaf_observations())
    by_ref: dict[str, list[EvidenceObservation[N]]] = {}
    for observation in observations:
        by_ref.setdefault(observation.fact_ref.format(), []).append(observation)

    resolved: list[EvidenceObservation[N]] = []
    conflicts: list[EvidenceConflict[N]] = []
    for ref_text in sorted(by_ref):
        group = tuple(by_ref[ref_text])
        resolution = _resolve_observation_group(group, spec=spec)
        if isinstance(resolution, EvidenceConflict):
            conflicts.append(resolution)
        else:
            resolved.append(resolution)

    observed_refs = _normalize_object_refs(tuple(item.fact_ref for item in observations))
    expected = observed_refs if expected_fact_refs is None else _normalize_object_refs(tuple(expected_fact_refs))
    expected_set = set(expected)
    observed_set = set(observed_refs)
    missing_refs = tuple(ref for ref in expected if ref not in observed_set)
    unexpected_refs = tuple(ref for ref in observed_refs if ref not in expected_set)
    if unexpected_refs:
        raise EvidenceInvariantError(
            "aggregate received facts outside its intended frame: " + ", ".join(ref.format() for ref in unexpected_refs)
        )

    known = [item for item in resolved if item.value_state == "known"]
    unsupported = [item for item in resolved if item.value_state != "known"]
    numeric_values: list[N] = []
    for item in known:
        raw_value: object = item.value
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise EvidenceInvariantError("quantitative aggregation requires int or float values")
        if isinstance(raw_value, float) and not math.isfinite(raw_value):
            raise EvidenceInvariantError("quantitative aggregation requires finite values")
        numeric_values.append(cast(N, raw_value))

    unresolved = bool(not observations or conflicts or missing_refs or unsupported)
    if unresolved:
        value_state: ValueState = "unknown"
        total: N | None = None
    else:
        value_state = "known"
        total = cast(N, sum(numeric_values))

    authorities = _normalize_authorities(
        tuple(authority for item in observations for authority in item.measurement_authority)
    )
    evidence_refs = _normalize_public_refs(tuple(ref for item in observations for ref in item.evidence_refs))
    temporal = _aggregate_temporal(tuple(item.temporal for item in observations), observed_at=observed_at)
    enumeration = _weakest_enumeration(tuple(item.enumeration for item in observations))
    conflict_refs = _normalize_object_refs(tuple(conflict.fact_ref for conflict in conflicts))
    incomplete_resolved = tuple(item for item in resolved if item.coverage.complete is not True)
    coverage_exclusion_items: list[CoverageExclusion] = [
        exclusion for item in observations for exclusion in item.coverage.exclusions
    ]
    coverage_exclusion_items.extend(CoverageExclusion(ref, "missing-contribution") for ref in missing_refs)
    coverage_exclusion_items.extend(CoverageExclusion(ref, "contradictory-contribution") for ref in conflict_refs)
    coverage_exclusion_items.extend(
        CoverageExclusion(item.fact_ref, f"unsupported-contribution:{item.value_state}") for item in unsupported
    )
    coverage_exclusion_items.extend(
        CoverageExclusion(item.fact_ref, "contribution-coverage-incomplete") for item in incomplete_resolved
    )
    coverage_exclusions = tuple(coverage_exclusion_items)
    supported_refs = {item.fact_ref for item in known}
    coverage = FrameCoverage(
        intended_frame=intended_frame,
        grain=spec.grain,
        denominator=spec.denominator,
        intended_count=len(expected),
        observed_count=len(observed_set & expected_set),
        supported_count=len(supported_refs & expected_set),
        complete=(
            bool(observations)
            and not conflicts
            and not missing_refs
            and not unsupported
            and not incomplete_resolved
            and observed_set == expected_set
        ),
        intended_refs=expected,
        observed_refs=tuple(ref for ref in observed_refs if ref in expected_set),
        exclusions=coverage_exclusions,
    )
    freshness = _weakest_freshness(tuple(item.freshness for item in observations), observed_at=observed_at)
    if missing_refs and freshness.state == "fresh":
        freshness = FreshnessProvenance(
            state="degraded",
            evaluated_at=observed_at,
            cause="missing-contribution",
            last_good_at=freshness.last_good_at,
            last_good_evidence_refs=evidence_refs,
        )
    result: EvidenceValue[N] = EvidenceValue(
        family=spec.family,
        fact_ref=fact_ref,
        value_state=value_state,
        value=total,
        measurement_authority=authorities,
        weakest_measurement_authority=spec.weakest_authority(authorities),
        evidence_refs=evidence_refs,
        definition_ref=spec.definition_ref,
        temporal=temporal,
        enumeration=enumeration,
        coverage=coverage,
        freshness=freshness,
        contributions=observations,
        conflicts=tuple(conflicts),
    )
    spec.require(result)
    return result


def _reconcile_same_fact(
    observations: Sequence[EvidenceObservation[T]],
    *,
    spec: FactFamilySpec,
) -> EvidenceValue[T]:
    resolution = _resolve_observation_group(tuple(observations), spec=spec)
    all_evidence = _normalize_public_refs(tuple(ref for item in observations for ref in item.evidence_refs))
    all_authorities = _normalize_authorities(
        tuple(authority for item in observations for authority in item.measurement_authority)
    )
    temporal = _aggregate_temporal(tuple(item.temporal for item in observations))
    coverage = _merge_same_fact_coverage(tuple(item.coverage for item in observations))
    freshness = _weakest_freshness(tuple(item.freshness for item in observations), observed_at=temporal.observed_at)
    enumeration = _weakest_enumeration(tuple(item.enumeration for item in observations))
    conflicts: tuple[EvidenceConflict[T], ...]
    if isinstance(resolution, EvidenceConflict):
        value_state: ValueState = "unknown"
        value: T | None = None
        conflicts = (resolution,)
    else:
        value_state = resolution.value_state
        value = resolution.value
        conflicts = ()
    result = EvidenceValue(
        family=spec.family,
        fact_ref=observations[0].fact_ref,
        value_state=value_state,
        value=value,
        measurement_authority=all_authorities,
        weakest_measurement_authority=spec.weakest_authority(all_authorities),
        evidence_refs=all_evidence,
        definition_ref=spec.definition_ref,
        temporal=temporal,
        enumeration=enumeration,
        coverage=coverage,
        freshness=freshness,
        contributions=tuple(observations),
        conflicts=conflicts,
    )
    spec.require(result)
    return result


def _resolve_observation_group(
    observations: tuple[EvidenceObservation[T], ...],
    *,
    spec: FactFamilySpec,
) -> EvidenceObservation[T] | EvidenceConflict[T]:
    if not observations:
        raise EvidenceInvariantError("cannot resolve an empty observation group")
    fact_ref = observations[0].fact_ref
    if any(item.fact_ref != fact_ref for item in observations):
        raise EvidenceInvariantError("observation group must share fact_ref")
    unique = tuple(sorted(_dedupe_by_key(observations, _observation_identity), key=_observation_sort_key))
    known_by_value: dict[str, list[EvidenceObservation[T]]] = {}
    for item in unique:
        if item.value_state == "known":
            known_by_value.setdefault(_stable_value_key(item.value), []).append(item)
    if not known_by_value:
        return _merge_equivalent_observations(unique)
    if len(known_by_value) == 1:
        selected = tuple(next(iter(known_by_value.values())))
        return _merge_equivalent_observations((*selected, *(item for item in unique if item.value_state != "known")))

    candidates: list[tuple[int, EvidenceObservation[T]]] = []
    for item in unique:
        if item.value_state != "known":
            continue
        floor = spec.weakest_authority(item.measurement_authority)
        if floor is None:
            continue
        candidates.append((spec.authority_rank(floor), item))
    if candidates:
        best_rank = min(rank for rank, _ in candidates)
        strongest = [item for rank, item in candidates if rank == best_rank]
        strongest_values = {_stable_value_key(item.value) for item in strongest}
        if len(strongest_values) == 1 and all(rank > best_rank for rank, item in candidates if item not in strongest):
            return _merge_equivalent_observations(tuple(strongest))
    return EvidenceConflict(
        fact_ref=fact_ref,
        reason="contradictory-observations",
        observations=unique,
    )


def _merge_equivalent_observations(observations: Sequence[EvidenceObservation[T]]) -> EvidenceObservation[T]:
    if not observations:
        raise EvidenceInvariantError("cannot merge empty observations")
    known = [item for item in observations if item.value_state == "known"]
    selected = known[0] if known else min(observations, key=lambda item: _state_weakness(item.value_state))
    return EvidenceObservation(
        fact_ref=selected.fact_ref,
        value_state=selected.value_state,
        value=selected.value,
        measurement_authority=_normalize_authorities(
            tuple(authority for item in observations for authority in item.measurement_authority)
        ),
        evidence_refs=_normalize_public_refs(tuple(ref for item in observations for ref in item.evidence_refs)),
        temporal=_aggregate_temporal(tuple(item.temporal for item in observations)),
        enumeration=_weakest_enumeration(tuple(item.enumeration for item in observations)),
        coverage=_merge_same_fact_coverage(tuple(item.coverage for item in observations)),
        freshness=_weakest_freshness(tuple(item.freshness for item in observations)),
    )


def _aggregate_temporal(
    values: Sequence[TemporalProvenance],
    *,
    observed_at: str | None = None,
) -> TemporalProvenance:
    sources = [item.time_source for item in values if item.time_source is not None]
    weakest = weakest_of(cast(Sequence[TemporalSource], sources))
    timestamps = [item.observed_at for item in values if item.observed_at is not None]
    chosen_observed_at = observed_at or (max(timestamps) if timestamps else None)
    frame_starts = [item.frame_start for item in values if item.frame_start is not None]
    frame_ends = [item.frame_end for item in values if item.frame_end is not None]
    return TemporalProvenance.from_source(
        observed_at=chosen_observed_at,
        time_source=weakest,
        frame_start=min(frame_starts) if frame_starts else None,
        frame_end=max(frame_ends) if frame_ends else None,
    )


def _weakest_freshness(
    values: Sequence[FreshnessProvenance],
    *,
    observed_at: str | None = None,
) -> FreshnessProvenance:
    if not values:
        return FreshnessProvenance(
            state="unavailable",
            evaluated_at=observed_at,
            cause="no-freshness-evidence",
        )
    weakest = max(values, key=lambda item: _FRESHNESS_WEAKNESS[item.state])
    refs = _normalize_public_refs(tuple(ref for item in values for ref in item.last_good_evidence_refs))
    last_good_times = [item.last_good_at for item in values if item.last_good_at is not None]
    evaluated_times = [item.evaluated_at for item in values if item.evaluated_at is not None]
    return FreshnessProvenance(
        state=weakest.state,
        evaluated_at=observed_at or (max(evaluated_times) if evaluated_times else None),
        cause=weakest.cause,
        last_good_at=min(last_good_times) if last_good_times else None,
        last_good_evidence_refs=refs,
    )


def _weakest_enumeration(values: Sequence[EnumerationKind]) -> EnumerationKind:
    if not values:
        return "inferred-partial"
    return max(values, key=lambda item: _ENUMERATION_WEAKNESS[item])


def _merge_same_fact_coverage(values: Sequence[FrameCoverage]) -> FrameCoverage:
    if not values:
        raise EvidenceInvariantError("cannot merge empty coverage")
    first = values[0]
    if any(
        (item.intended_frame, item.grain, item.denominator) != (first.intended_frame, first.grain, first.denominator)
        for item in values
    ):
        raise EvidenceInvariantError("same-fact refinement cannot change frame, grain, or denominator")
    intended_counts = [item.intended_count for item in values if item.intended_count is not None]
    observed_counts = [item.observed_count for item in values if item.observed_count is not None]
    supported_counts = [item.supported_count for item in values if item.supported_count is not None]
    return FrameCoverage(
        intended_frame=first.intended_frame,
        grain=first.grain,
        denominator=first.denominator,
        intended_count=max(intended_counts) if intended_counts else None,
        observed_count=max(observed_counts) if observed_counts else None,
        supported_count=max(supported_counts) if supported_counts else None,
        complete=any(item.complete is True for item in values)
        if any(item.complete is not None for item in values)
        else None,
        intended_refs=_normalize_object_refs(tuple(ref for item in values for ref in item.intended_refs)),
        observed_refs=_normalize_object_refs(tuple(ref for item in values for ref in item.observed_refs)),
        exclusions=tuple(exclusion for item in values for exclusion in item.exclusions),
    )


def _state_weakness(state: ValueState) -> int:
    return {
        "known": 0,
        "not_applicable": 1,
        "skipped": 2,
        "unknown": 3,
        "unavailable": 4,
        "redacted": 5,
    }[state]


def _validate_value_state(state: ValueState, value: object) -> None:
    if state not in _VALUE_STATES:
        raise EvidenceInvariantError(f"unsupported value state: {state!r}")
    if state == "known" and value is None:
        raise EvidenceInvariantError("known evidence value requires a typed value")
    if state != "known" and value is not None:
        raise EvidenceInvariantError(f"{state} evidence value cannot carry a numeric/string sentinel")


def _validate_authorities(authorities: Sequence[MeasurementAuthority]) -> None:
    unknown = set(authorities) - _AUTHORITIES
    if unknown:
        raise EvidenceInvariantError(f"unsupported measurement authority: {sorted(unknown)}")


def _normalize_authorities(
    authorities: Sequence[MeasurementAuthority],
) -> tuple[MeasurementAuthority, ...]:
    _validate_authorities(authorities)
    return tuple(sorted(set(authorities)))


def _normalize_public_refs(refs: Sequence[PublicRef]) -> tuple[PublicRef, ...]:
    return tuple(sorted(_dedupe_by_key(refs, _public_ref_identity), key=_public_ref_sort_key))


def _normalize_object_refs(refs: Sequence[ObjectRef]) -> tuple[ObjectRef, ...]:
    return tuple(sorted(_dedupe_by_key(refs, lambda ref: ref.format()), key=lambda ref: ref.format()))


def _format_public_ref(ref: PublicRef) -> str:
    return ref.format()


def _owner_path_for_adapter(adapter: str) -> str:
    """Translate a dotted owner symbol into the kernel's source-path field."""

    module, _, _symbol = adapter.rpartition(".")
    if not module:
        return adapter
    return module.replace(".", "/") + ".py"


def _public_ref_identity(ref: PublicRef) -> tuple[str, str]:
    return ("evidence" if isinstance(ref, EvidenceRef) else "object", ref.format())


def _public_ref_sort_key(ref: PublicRef) -> tuple[str, str]:
    return _public_ref_identity(ref)


def _observation_identity(observation: EvidenceObservation[object]) -> tuple[object, ...]:
    return (
        observation.fact_ref.format(),
        observation.value_state,
        _stable_value_key(observation.value),
        tuple(observation.measurement_authority),
        tuple(_public_ref_identity(ref) for ref in observation.evidence_refs),
        repr(observation.temporal.to_dict()),
        observation.enumeration,
        repr(observation.coverage.to_dict()),
        repr(observation.freshness.to_dict()),
    )


def _observation_sort_key(observation: EvidenceObservation[object]) -> tuple[str, str, str]:
    return (observation.fact_ref.format(), observation.value_state, _stable_value_key(observation.value))


def _conflict_identity(conflict: EvidenceConflict[object]) -> tuple[object, ...]:
    return (
        conflict.fact_ref.format(),
        conflict.reason,
        tuple(_observation_identity(item) for item in conflict.observations),
    )


def _stable_value_key(value: object) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceInvariantError("evidence values must be finite")
        return f"float:{value.hex()}"
    return f"{type(value).__name__}:{value!r}"


def _dedupe_by_key(items: Iterable[T], key: Callable[[T], Hashable]) -> tuple[T, ...]:
    result: list[T] = []
    seen: set[Hashable] = set()
    for item in items:
        identity = key(item)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(item)
    return tuple(result)


def _parse_aware_iso8601(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceInvariantError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise EvidenceInvariantError(f"timestamp must be timezone-aware: {value!r}")
    return parsed


def _require_aware_iso8601(name: str, value: str) -> None:
    try:
        _parse_aware_iso8601(value)
    except EvidenceInvariantError as exc:
        raise EvidenceInvariantError(f"{name}: {exc}") from exc


__all__ = [
    "CalibratedConfidence",
    "CoverageExclusion",
    "EnumerationKind",
    "EvidenceAxis",
    "EvidenceConflict",
    "EvidenceInvariantError",
    "EvidenceObservation",
    "EvidenceValue",
    "FactFamilyDiagnostic",
    "FactFamilySpec",
    "FrameCoverage",
    "FreshnessProvenance",
    "FreshnessState",
    "MeasurementAuthority",
    "TemporalProvenance",
    "ValueSchema",
    "ValueState",
    "audit_fact_family_completeness",
    "refine_evidence_value",
    "sum_evidence_values",
]

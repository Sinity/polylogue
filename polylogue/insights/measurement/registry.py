"""Construct-valid measures over the archive query algebra."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import cast

from polylogue.core.evidence_value import EvidenceAxis, EvidenceValue
from polylogue.insights.measurement.canon import content_ref
from polylogue.insights.measurement.metric import MetricDefinition


class MeasureValidityError(ValueError):
    """A measure declaration or composition would overstate its evidence."""


@dataclass(frozen=True, slots=True)
class MeasureSpec:
    """A registered analytic definition with its construct-validity envelope."""

    name: str
    metric: MetricDefinition
    evidence_tier: str
    sample_frame: str
    confounds: tuple[str, ...]
    required_axes: frozenset[EvidenceAxis] = frozenset(
        {"value_state", "measurement_authority", "evidence_refs", "definition_ref", "enumeration", "coverage"}
    )
    coverage_preconditions: tuple[str, ...] = ()
    non_claim: str = ""

    def __post_init__(self) -> None:
        missing: list[str] = []
        if not self.name.strip():
            missing.append("name")
        if not self.evidence_tier.strip():
            missing.append("evidence_tier")
        if not self.sample_frame.strip():
            missing.append("sample_frame")
        if not self.confounds:
            missing.append("confounds")
        if not self.metric.required_frame:
            missing.append("frame requirement")
        if not self.metric.null_policy:
            missing.append("null policy")
        if not self.metric.formula_version:
            missing.append("formula version")
        if not self.metric.measurement_authority:
            missing.append("measurement authority")
        if "coverage" not in self.required_axes:
            missing.append("required EvidenceValue axis: coverage")
        if any(not item.strip() for item in self.coverage_preconditions):
            missing.append("coverage preconditions")
        if missing:
            raise MeasureValidityError("measure is missing " + ", ".join(dict.fromkeys(missing)))

    @property
    def ref(self) -> str:
        return content_ref("measure", {"name": self.name, "metric_ref": self.metric.ref})

    @property
    def tier_footnote(self) -> str:
        return f"Evidence tier: {self.evidence_tier}; sample frame: {self.sample_frame}; confounds: {', '.join(self.confounds)}."


@dataclass(frozen=True, slots=True)
class MeasurePlan:
    """The query-algebra dimensions for one measure evaluation."""

    measure_ref: str
    group_by: str | None = None
    window: str | None = None
    comparison: str | None = None
    uncertainty: str = "none"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "measure_ref": self.measure_ref,
            "group_by": self.group_by,
            "window": self.window,
            "comparison": self.comparison,
            "uncertainty": self.uncertainty,
        }

    @property
    def query_ref(self) -> str:
        return content_ref("query", self.canonical_payload())


@dataclass(frozen=True, slots=True)
class MeasureResult:
    """A value plus provenance needed to drill back to its members."""

    measure_ref: str
    query_ref: str
    result_ref: str
    value: object
    value_state: str
    group: str | None
    member_refs: tuple[str, ...]
    tier_footnote: str
    diagnostic: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "measure_ref": self.measure_ref,
            "query_ref": self.query_ref,
            "result_ref": self.result_ref,
            "value": self.value,
            "value_state": self.value_state,
            "group": self.group,
            "member_refs": list(self.member_refs),
            "tier_footnote": self.tier_footnote,
            "diagnostic": self.diagnostic,
        }

    def render(self) -> str:
        group = f" [{self.group}]" if self.group is not None else ""
        diagnostic = f" ({self.diagnostic})" if self.diagnostic else ""
        return f"{self.value_state}: {self.value}{group}{diagnostic}\n{self.tier_footnote}"


@dataclass(slots=True)
class MeasureRegistry:
    """Registry keyed by friendly name and canonical measure identity."""

    _by_name: dict[str, MeasureSpec] = field(default_factory=dict)
    _by_ref: dict[str, MeasureSpec] = field(default_factory=dict)

    def register(self, spec: MeasureSpec) -> str:
        previous = self._by_name.get(spec.name)
        if previous is not None and previous.ref != spec.ref:
            raise MeasureValidityError(f"measure name {spec.name!r} is already bound to {previous.ref!r}")
        previous_ref = self._by_ref.get(spec.ref)
        if previous_ref is not None and previous_ref != spec:
            raise MeasureValidityError(f"measure ref {spec.ref!r} is already bound to a different definition")
        self._by_name[spec.name] = spec
        self._by_ref[spec.ref] = spec
        return spec.ref

    def get(self, ref_or_name: str) -> MeasureSpec | None:
        return self._by_ref.get(ref_or_name) or self._by_name.get(ref_or_name)

    def require(self, ref_or_name: str) -> MeasureSpec:
        spec = self.get(ref_or_name)
        if spec is None:
            raise MeasureValidityError(f"unknown measure {ref_or_name!r}")
        return spec

    def __iter__(self) -> Iterator[MeasureSpec]:
        return iter(self._by_name.values())

    def __len__(self) -> int:
        return len(self._by_name)


def compose_measure(
    registry: MeasureRegistry,
    plan: MeasurePlan,
    rows: Iterable[Mapping[str, object]],
    *,
    value_field: str = "value",
    ref_field: str = "ref",
    group_field: str | None = None,
) -> tuple[MeasureResult, ...]:
    """Compose basic count/sum/mean values without coercing unknown to zero."""
    spec = registry.require(plan.measure_ref)
    groups: dict[str | None, list[Mapping[str, object]]] = {}
    for row in rows:
        _validate_composition_row(spec, row)
        group = None if group_field is None else str(row.get(group_field, "unknown"))
        groups.setdefault(group, []).append(row)
    results: list[MeasureResult] = []
    for group, members in groups.items():
        known = [
            row[value_field]
            for row in members
            if row.get("value_state", "known") == "known" and row.get(value_field) is not None
        ]
        refs = tuple(str(row[ref_field]) for row in members if ref_field in row)
        aggregate = spec.metric.aggregation.lower()
        value: object
        state: str
        if not known:
            value, state = None, "unknown"
        elif aggregate == "count":
            value, state = len(known), "known"
        elif aggregate == "sum":
            numeric = [cast(float | int, item) for item in known]
            value, state = sum(numeric), "known"
        elif aggregate in {"mean", "average"}:
            numeric = [cast(float | int, item) for item in known]
            value, state = sum(numeric) / len(numeric), "known"
        else:
            raise MeasureValidityError(f"unsupported basic aggregate {spec.metric.aggregation!r}")
        result_ref = content_ref("result-set", {"query_ref": plan.query_ref, "group": group, "members": refs})
        results.append(
            MeasureResult(spec.ref, plan.query_ref, result_ref, value, state, group, refs, spec.tier_footnote)
        )
    return tuple(results)


def _validate_composition_row(spec: MeasureSpec, row: Mapping[str, object]) -> None:
    """Apply the declaration's evidence and frame gates before aggregation.

    Legacy callers may still provide the small mapping shape used by the
    original aggregate helper.  New callers can attach the canonical
    ``EvidenceValue`` under ``evidence``; when present, all declared axes are
    checked here so an aggregate cannot hide an incomplete frame.
    """

    evidence = row.get("evidence")
    if evidence is not None and not isinstance(evidence, EvidenceValue):
        raise MeasureValidityError("composition evidence must be an EvidenceValue")
    for precondition in spec.coverage_preconditions:
        if precondition in {"coverage_complete", "frame_complete"}:
            complete = evidence.coverage.complete if evidence is not None else row.get("coverage_complete")
            if complete is not True:
                raise MeasureValidityError(
                    f"measure {spec.name!r} requires complete frame coverage; "
                    "composition was suppressed because coverage is incomplete or undeclared"
                )
        elif precondition == "enumeration_exact":
            enumeration = evidence.enumeration if evidence is not None else row.get("enumeration")
            if enumeration not in {"census", "exact"}:
                raise MeasureValidityError(f"measure {spec.name!r} requires exact enumeration, got {enumeration!r}")
        elif precondition == "authority_compatible":
            authorities = (
                set(evidence.measurement_authority)
                if evidence is not None
                else set(cast(tuple[str, ...], row.get("measurement_authority", ())))
            )
            required = set(spec.metric.measurement_authority)
            if not authorities or (required and not authorities & required):
                raise MeasureValidityError(
                    f"measure {spec.name!r} requires compatible measurement authority; got {sorted(authorities)}"
                )
        else:
            raise MeasureValidityError(f"measure {spec.name!r} has unsupported coverage precondition {precondition!r}")

    if evidence is None:
        return
    missing_axes: list[str] = []
    for axis in spec.required_axes:
        value = getattr(evidence, axis, None)
        if axis == "coverage":
            value = evidence.coverage.intended_frame
        if axis == "definition_ref":
            value = evidence.definition_ref.object_id
        if axis == "evidence_refs" and evidence.value_state == "known":
            value = evidence.evidence_refs
        if axis == "measurement_authority" and evidence.value_state == "known":
            value = evidence.measurement_authority
        if value in (None, (), ""):
            missing_axes.append(axis)
    if missing_axes:
        raise MeasureValidityError(
            f"measure {spec.name!r} composition is missing EvidenceValue axes: {', '.join(sorted(missing_axes))}"
        )
    if spec.metric.required_frame and evidence.coverage.intended_frame != spec.metric.required_frame:
        raise MeasureValidityError(
            f"measure {spec.name!r} requires frame {spec.metric.required_frame!r}, "
            f"got {evidence.coverage.intended_frame!r}"
        )
    if spec.metric.measurement_authority and not (
        set(evidence.measurement_authority) & set(spec.metric.measurement_authority)
    ):
        raise MeasureValidityError(
            f"measure {spec.name!r} has incompatible measurement authority {list(evidence.measurement_authority)!r}"
        )


__all__ = ["MeasurePlan", "MeasureRegistry", "MeasureResult", "MeasureSpec", "MeasureValidityError", "compose_measure"]

"""Construct-valid measures over the archive query algebra."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from polylogue.analysis.measurement.canon import content_ref
from polylogue.analysis.measurement.metric import MetricDefinition
from polylogue.core.evidence_value import EvidenceAxis


class MeasureValidityError(ValueError):
    """A measure declaration would overstate its evidence."""


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


__all__ = ["MeasureRegistry", "MeasureSpec", "MeasureValidityError"]

"""Structured validation-lane family models compiled into composite lanes."""

from __future__ import annotations

from dataclasses import dataclass

from devtools.lane_models import LaneEntry
from devtools.validation_lane_base import composite_lane
from polylogue.scenarios import NamedScenarioSource, ScenarioProjectionSourceKind


@dataclass(frozen=True, kw_only=True)
class ValidationLaneStageSpec:
    """One authored family stage compiled into a named composite lane."""

    suffix: str
    description: str
    timeout_s: int
    members: tuple[str, ...] = ()
    member_stages: tuple[str, ...] = ()
    origin: str = "authored.validation-lane.composite-family"
    path_targets: tuple[str, ...] = ()
    artifact_targets: tuple[str, ...] = ()
    operation_targets: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def compile(self, *, family_name: str) -> ValidationLaneCompositeSpec:
        resolved_members = (
            *self.members,
            *(f"{family_name}-{stage}" for stage in self.member_stages),
        )
        return ValidationLaneCompositeSpec(
            name=f"{family_name}-{self.suffix}",
            description=self.description,
            timeout_s=self.timeout_s,
            members=resolved_members,
            origin=self.origin,
            path_targets=self.path_targets,
            artifact_targets=self.artifact_targets,
            operation_targets=self.operation_targets,
            tags=self.tags,
        )


@dataclass(frozen=True, kw_only=True)
class ValidationLaneCompositeSpec:
    """One composite lane declared as part of a validation family."""

    name: str
    description: str
    timeout_s: int
    members: tuple[str, ...]
    origin: str = "authored.validation-lane.composite-family"
    path_targets: tuple[str, ...] = ()
    artifact_targets: tuple[str, ...] = ()
    operation_targets: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def compile(self, *, family: str) -> LaneEntry:
        return composite_lane(
            self.name,
            self.description,
            self.timeout_s,
            *self.members,
            category="composite",
            family=family,
            origin=self.origin,
            path_targets=self.path_targets,
            artifact_targets=self.artifact_targets,
            operation_targets=self.operation_targets,
            tags=self.tags,
        )


@dataclass(frozen=True, kw_only=True)
class ValidationLaneFamily(NamedScenarioSource):
    """One named validation-lane family compiled into composite lane entries."""

    lanes: tuple[ValidationLaneCompositeSpec, ...]

    @classmethod
    def from_stages(
        cls,
        *,
        name: str,
        description: str,
        stages: tuple[ValidationLaneStageSpec, ...],
        origin: str = "authored",
        path_targets: tuple[str, ...] = (),
        artifact_targets: tuple[str, ...] = (),
        operation_targets: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
    ) -> ValidationLaneFamily:
        return cls(
            name=name,
            description=description,
            lanes=tuple(stage.compile(family_name=name) for stage in stages),
            origin=origin,
            path_targets=path_targets,
            artifact_targets=artifact_targets,
            operation_targets=operation_targets,
            tags=tags,
        )

    def compile_entries(self) -> tuple[LaneEntry, ...]:
        return tuple(lane.compile(family=self.name) for lane in self.lanes)

    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.VALIDATION_FAMILY

    def projection_source_payload(self) -> dict[str, object]:
        payload = self.scenario_payload()
        payload["lanes"] = [lane.name for lane in self.lanes]
        return payload


def compile_validation_lane_families(
    families: tuple[ValidationLaneFamily, ...],
) -> dict[str, LaneEntry]:
    compiled: dict[str, LaneEntry] = {}
    for family in families:
        for lane in family.compile_entries():
            compiled[lane.name] = lane
    return compiled


__all__ = [
    "compile_validation_lane_families",
    "ValidationLaneCompositeSpec",
    "ValidationLaneFamily",
    "ValidationLaneStageSpec",
]

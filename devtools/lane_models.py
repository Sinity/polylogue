"""Shared control-plane lane metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

from polylogue.scenarios.assertions import AssertionSpec
from polylogue.scenarios.execution import ExecutionSpec
from polylogue.scenarios.metadata import ScenarioMetadata
from polylogue.scenarios.projections import ScenarioProjectionEntry, ScenarioProjectionSourceKind


@dataclass(frozen=True, kw_only=True)
class LaneEntry:
    """One named control-plane lane."""

    name: str
    description: str
    timeout_s: int
    category: str
    execution: ExecutionSpec | None = None
    assertion: AssertionSpec = field(default_factory=AssertionSpec)
    family: str | None = None

    # Metadata fields kept as direct attributes for ScenarioMetadata.from_object
    # compatibility and for _build_lane_entry merge logic.
    origin: str = "authored.validation-lane"
    path_targets: tuple[str, ...] = ()
    artifact_targets: tuple[str, ...] = ()
    operation_targets: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def is_composite(self) -> bool:
        return self.execution is not None and self.execution.is_composite

    @property
    def sub_lanes(self) -> tuple[str, ...]:
        if self.execution is None:
            return ()
        return self.execution.members

    # ------------------------------------------------------------------
    # Projection protocol (replaces ExecutableScenario/NamedScenarioSource inheritance)
    # ------------------------------------------------------------------

    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.VALIDATION_LANE

    @property
    def projection_name(self) -> str:
        return self.name

    @property
    def projection_description(self) -> str:
        return self.description

    def projection_source_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "timeout_s": self.timeout_s,
            "category": self.category,
        }
        if self.family is not None:
            payload["family"] = self.family
        return payload

    def to_projection_entry(self) -> ScenarioProjectionEntry:
        metadata = ScenarioMetadata.from_object(self)
        return ScenarioProjectionEntry(
            source_kind=self.projection_source_kind,
            name=self.name,
            description=self.description,
            origin=metadata.origin,
            path_targets=metadata.path_targets,
            artifact_targets=metadata.artifact_targets,
            operation_targets=metadata.operation_targets,
            tags=metadata.tags,
            docs_role=metadata.docs_role,
            caption=metadata.caption,
            narrative_order=metadata.narrative_order,
            audience=metadata.audience,
            demonstrates=metadata.demonstrates,
            privacy_level=metadata.privacy_level,
            media=metadata.media,
            visual_style=metadata.visual_style,
            source_payload=dict(self.projection_source_payload()),
        )


__all__ = ["LaneEntry"]

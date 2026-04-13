"""Typed validation-lane catalog shared across control-plane surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from devtools.execution_specs import ExecutionSpec
from polylogue.scenarios import ScenarioMetadata

from .validation_lane_base import LaneConfig
from .validation_lane_catalog_composites import COMPOSITE_LANES
from .validation_lane_catalog_contracts import CONTRACT_LANES
from .validation_lane_catalog_live import LIVE_LANES


@dataclass(frozen=True)
class ValidationLaneEntry(ScenarioMetadata):
    name: str
    description: str
    timeout_s: int
    category: str
    execution: ExecutionSpec

    @property
    def is_composite(self) -> bool:
        return self.execution.is_composite

    @property
    def command(self) -> list[str] | None:
        command = self.execution.command
        return list(command) if command is not None else None

    @property
    def sub_lanes(self) -> tuple[str, ...]:
        return self.execution.members


ALL_VALIDATION_LANES: dict[str, LaneConfig] = {
    **CONTRACT_LANES,
    **LIVE_LANES,
    **COMPOSITE_LANES,
}

LANE_CATEGORIES: dict[str, str] = {
    **dict.fromkeys(CONTRACT_LANES, "contract"),
    **dict.fromkeys(LIVE_LANES, "live"),
    **dict.fromkeys(COMPOSITE_LANES, "composite"),
}


def _merge_unique(*groups: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                merged.append(item)
    return tuple(merged)


def _build_lane_entry(
    lane_name: str,
    *,
    cache: dict[str, ValidationLaneEntry],
    visiting: set[str],
) -> ValidationLaneEntry:
    if lane_name in cache:
        return cache[lane_name]
    if lane_name in visiting:
        raise ValueError(f"Cyclic validation lane metadata dependency: {lane_name}")

    visiting.add(lane_name)
    lane = ALL_VALIDATION_LANES[lane_name]
    child_entries = tuple(_build_lane_entry(child, cache=cache, visiting=visiting) for child in lane.sub_lanes)
    entry = ValidationLaneEntry(
        name=lane.name,
        description=lane.description,
        timeout_s=lane.timeout_s,
        category=LANE_CATEGORIES[lane_name],
        execution=lane.execution,
        origin=lane.origin,
        path_targets=_merge_unique(lane.path_targets, *(child.path_targets for child in child_entries)),
        artifact_targets=_merge_unique(lane.artifact_targets, *(child.artifact_targets for child in child_entries)),
        operation_targets=_merge_unique(lane.operation_targets, *(child.operation_targets for child in child_entries)),
        tags=_merge_unique(lane.tags, *(child.tags for child in child_entries)),
    )
    visiting.remove(lane_name)
    cache[lane_name] = entry
    return entry


def build_validation_lane_entries() -> tuple[ValidationLaneEntry, ...]:
    cache: dict[str, ValidationLaneEntry] = {}
    return tuple(
        sorted(
            (
                _build_lane_entry(name, cache=cache, visiting=set())
                for name in ALL_VALIDATION_LANES
            ),
            key=lambda item: item.name,
        )
    )


def _category_entries(category: str) -> tuple[ValidationLaneEntry, ...]:
    return tuple(entry for entry in build_validation_lane_entries() if entry.category == category)


def build_contract_lane_entries() -> tuple[ValidationLaneEntry, ...]:
    return _category_entries("contract")


def build_live_lane_entries() -> tuple[ValidationLaneEntry, ...]:
    return _category_entries("live")


def build_composite_lane_entries() -> tuple[ValidationLaneEntry, ...]:
    return _category_entries("composite")


__all__ = [
    "build_validation_lane_entries",
    "ValidationLaneEntry",
    "build_composite_lane_entries",
    "build_contract_lane_entries",
    "build_live_lane_entries",
]

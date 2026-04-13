"""Typed validation-lane catalog shared across control-plane surfaces."""

from __future__ import annotations

from dataclasses import replace

from .lane_models import LaneEntry
from .validation_lane_catalog_composites import COMPOSITE_LANES
from .validation_lane_catalog_contracts import CONTRACT_LANES
from .validation_lane_catalog_live import LIVE_LANES

ALL_VALIDATION_LANES: dict[str, LaneEntry] = {
    **CONTRACT_LANES,
    **LIVE_LANES,
    **COMPOSITE_LANES,
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
    cache: dict[str, LaneEntry],
    visiting: set[str],
) -> LaneEntry:
    if lane_name in cache:
        return cache[lane_name]
    if lane_name in visiting:
        raise ValueError(f"Cyclic validation lane metadata dependency: {lane_name}")

    visiting.add(lane_name)
    lane = ALL_VALIDATION_LANES[lane_name]
    child_entries = tuple(_build_lane_entry(child, cache=cache, visiting=visiting) for child in lane.sub_lanes)
    entry = replace(
        lane,
        path_targets=_merge_unique(lane.path_targets, *(child.path_targets for child in child_entries)),
        artifact_targets=_merge_unique(lane.artifact_targets, *(child.artifact_targets for child in child_entries)),
        operation_targets=_merge_unique(lane.operation_targets, *(child.operation_targets for child in child_entries)),
        tags=_merge_unique(lane.tags, *(child.tags for child in child_entries)),
    )
    visiting.remove(lane_name)
    cache[lane_name] = entry
    return entry


def build_validation_lane_entries() -> tuple[LaneEntry, ...]:
    cache: dict[str, LaneEntry] = {}
    return tuple(
        sorted(
            (
                _build_lane_entry(name, cache=cache, visiting=set())
                for name in ALL_VALIDATION_LANES
            ),
            key=lambda item: item.name,
        )
    )


def _category_entries(category: str) -> tuple[LaneEntry, ...]:
    return tuple(entry for entry in build_validation_lane_entries() if entry.category == category)


def build_contract_lane_entries() -> tuple[LaneEntry, ...]:
    return _category_entries("contract")


def build_live_lane_entries() -> tuple[LaneEntry, ...]:
    return _category_entries("live")


def build_composite_lane_entries() -> tuple[LaneEntry, ...]:
    return _category_entries("composite")


__all__ = [
    "build_validation_lane_entries",
    "LaneEntry",
    "build_composite_lane_entries",
    "build_contract_lane_entries",
    "build_live_lane_entries",
]

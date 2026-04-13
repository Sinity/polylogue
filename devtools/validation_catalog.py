"""Typed validation-lane catalog shared across control-plane surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from .validation_lane_base import LaneConfig
from .validation_lane_catalog_composites import COMPOSITE_LANES
from .validation_lane_catalog_contracts import CONTRACT_LANES
from .validation_lane_catalog_live import LIVE_LANES


@dataclass(frozen=True)
class ValidationLaneEntry:
    name: str
    description: str
    timeout_s: int
    category: str
    sub_lanes: tuple[str, ...] = ()


def _lane_entries(category: str, lanes: dict[str, LaneConfig]) -> tuple[ValidationLaneEntry, ...]:
    entries = [
        ValidationLaneEntry(
            name=lane.name,
            description=lane.description,
            timeout_s=lane.timeout_s,
            category=category,
            sub_lanes=tuple(lane.sub_lanes),
        )
        for lane in lanes.values()
    ]
    return tuple(sorted(entries, key=lambda item: item.name))


def build_contract_lane_entries() -> tuple[ValidationLaneEntry, ...]:
    return _lane_entries("contract", CONTRACT_LANES)


def build_live_lane_entries() -> tuple[ValidationLaneEntry, ...]:
    return _lane_entries("live", LIVE_LANES)


def build_composite_lane_entries() -> tuple[ValidationLaneEntry, ...]:
    return _lane_entries("composite", COMPOSITE_LANES)


__all__ = [
    "ValidationLaneEntry",
    "build_composite_lane_entries",
    "build_contract_lane_entries",
    "build_live_lane_entries",
]

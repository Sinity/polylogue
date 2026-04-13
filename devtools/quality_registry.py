"""Shared registry for validation lanes and durable quality campaigns."""

from __future__ import annotations

from dataclasses import dataclass

from devtools.benchmark_catalog import (
    BenchmarkCampaignEntry,
    build_benchmark_entries,
    build_synthetic_benchmark_entries,
)
from devtools.mutmut_campaign import CAMPAIGNS as MUTATION_CAMPAIGNS
from devtools.scenario_projection_catalog import build_scenario_projection_entries
from devtools.validation_lane_base import LaneConfig
from devtools.validation_lane_catalog_composites import COMPOSITE_LANES
from devtools.validation_lane_catalog_contracts import CONTRACT_LANES
from devtools.validation_lane_catalog_live import LIVE_LANES
from polylogue.scenarios import ScenarioProjectionEntry


@dataclass(frozen=True)
class ValidationLaneEntry:
    name: str
    description: str
    timeout_s: int
    category: str
    sub_lanes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MutationCampaignEntry:
    name: str
    description: str
    paths_to_mutate: tuple[str, ...]
    tests: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualityRegistry:
    contract_lanes: tuple[ValidationLaneEntry, ...]
    live_lanes: tuple[ValidationLaneEntry, ...]
    composite_lanes: tuple[ValidationLaneEntry, ...]
    mutation_campaigns: tuple[MutationCampaignEntry, ...]
    benchmark_campaigns: tuple[BenchmarkCampaignEntry, ...]
    synthetic_benchmark_campaigns: tuple[BenchmarkCampaignEntry, ...]
    scenario_projections: tuple[ScenarioProjectionEntry, ...]


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


def _mutation_entries() -> tuple[MutationCampaignEntry, ...]:
    entries = [
        MutationCampaignEntry(
            name=campaign.name,
            description=campaign.description,
            paths_to_mutate=tuple(campaign.paths_to_mutate),
            tests=tuple(campaign.tests),
            notes=tuple(campaign.notes),
        )
        for campaign in MUTATION_CAMPAIGNS.values()
    ]
    return tuple(sorted(entries, key=lambda item: item.name))


def build_quality_registry() -> QualityRegistry:
    return QualityRegistry(
        contract_lanes=_lane_entries("contract", CONTRACT_LANES),
        live_lanes=_lane_entries("live", LIVE_LANES),
        composite_lanes=_lane_entries("composite", COMPOSITE_LANES),
        mutation_campaigns=_mutation_entries(),
        benchmark_campaigns=build_benchmark_entries(),
        synthetic_benchmark_campaigns=build_synthetic_benchmark_entries(),
        scenario_projections=build_scenario_projection_entries(),
    )


__all__ = [
    "MutationCampaignEntry",
    "QualityRegistry",
    "ValidationLaneEntry",
    "build_quality_registry",
]

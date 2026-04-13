"""Shared registry for validation lanes and durable quality campaigns."""

from __future__ import annotations

from dataclasses import dataclass

from devtools.benchmark_campaign import CAMPAIGNS as BENCHMARK_CAMPAIGNS
from devtools.benchmark_campaigns import SYNTHETIC_BENCHMARK_SCENARIOS
from devtools.mutmut_campaign import CAMPAIGNS as MUTATION_CAMPAIGNS
from devtools.scenario_projection_catalog import build_scenario_projection_entries
from devtools.validation_lane_base import LaneConfig
from devtools.validation_lane_catalog_composites import COMPOSITE_LANES
from devtools.validation_lane_catalog_contracts import CONTRACT_LANES
from devtools.validation_lane_catalog_live import LIVE_LANES
from polylogue.scenarios import (
    ScenarioMetadata,
    ScenarioProjectionEntry,
    ScenarioProjectionSourceKind,
)


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
class BenchmarkCampaignEntry(ScenarioMetadata):
    name: str
    description: str
    tests: tuple[str, ...]
    notes: tuple[str, ...] = ()
    warn_pct: float = 0.0
    fail_pct: float = 0.0


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


def _benchmark_entries() -> tuple[BenchmarkCampaignEntry, ...]:
    entries = [
        BenchmarkCampaignEntry(
            name=campaign.name,
            description=campaign.description,
            tests=tuple(campaign.tests),
            notes=tuple(campaign.notes),
            warn_pct=campaign.warn_pct,
            fail_pct=campaign.fail_pct,
            origin=campaign.origin,
            path_targets=tuple(campaign.path_targets),
            artifact_targets=tuple(campaign.artifact_targets),
            operation_targets=tuple(campaign.operation_targets),
            tags=tuple(campaign.tags),
        )
        for campaign in BENCHMARK_CAMPAIGNS.values()
    ]
    return tuple(sorted(entries, key=lambda item: item.name))


def _synthetic_benchmark_entries() -> tuple[BenchmarkCampaignEntry, ...]:
    entries = [
        BenchmarkCampaignEntry(
            name=scenario.scenario_id,
            description=scenario.description,
            tests=(),
            notes=(),
            origin=scenario.origin,
            path_targets=scenario.path_targets,
            artifact_targets=scenario.artifact_targets,
            operation_targets=scenario.operation_targets,
            tags=scenario.tags,
        )
        for scenario in SYNTHETIC_BENCHMARK_SCENARIOS
    ]
    return tuple(sorted(entries, key=lambda item: item.name))


def build_quality_registry() -> QualityRegistry:
    return QualityRegistry(
        contract_lanes=_lane_entries("contract", CONTRACT_LANES),
        live_lanes=_lane_entries("live", LIVE_LANES),
        composite_lanes=_lane_entries("composite", COMPOSITE_LANES),
        mutation_campaigns=_mutation_entries(),
        benchmark_campaigns=_benchmark_entries(),
        synthetic_benchmark_campaigns=_synthetic_benchmark_entries(),
        scenario_projections=build_scenario_projection_entries(),
    )


__all__ = [
    "BenchmarkCampaignEntry",
    "MutationCampaignEntry",
    "QualityRegistry",
    "ScenarioProjectionEntry",
    "ScenarioProjectionSourceKind",
    "ValidationLaneEntry",
    "build_quality_registry",
]

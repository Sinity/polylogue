"""Shared registry for validation lanes and durable quality campaigns."""

from __future__ import annotations

from dataclasses import dataclass

from devtools.benchmark_catalog import (
    BenchmarkCampaignEntry,
    build_benchmark_entries,
    build_synthetic_benchmark_entries,
)
from devtools.mutation_catalog import MutationCampaignEntry, build_mutation_entries
from devtools.scenario_projection_catalog import build_scenario_projection_entries
from devtools.validation_catalog import (
    ValidationLaneEntry,
    build_composite_lane_entries,
    build_contract_lane_entries,
    build_live_lane_entries,
)
from polylogue.scenarios import ScenarioProjectionEntry


@dataclass(frozen=True)
class QualityRegistry:
    contract_lanes: tuple[ValidationLaneEntry, ...]
    live_lanes: tuple[ValidationLaneEntry, ...]
    composite_lanes: tuple[ValidationLaneEntry, ...]
    mutation_campaigns: tuple[MutationCampaignEntry, ...]
    benchmark_campaigns: tuple[BenchmarkCampaignEntry, ...]
    synthetic_benchmark_campaigns: tuple[BenchmarkCampaignEntry, ...]
    scenario_projections: tuple[ScenarioProjectionEntry, ...]


def build_quality_registry() -> QualityRegistry:
    return QualityRegistry(
        contract_lanes=build_contract_lane_entries(),
        live_lanes=build_live_lane_entries(),
        composite_lanes=build_composite_lane_entries(),
        mutation_campaigns=build_mutation_entries(),
        benchmark_campaigns=build_benchmark_entries(),
        synthetic_benchmark_campaigns=build_synthetic_benchmark_entries(),
        scenario_projections=build_scenario_projection_entries(),
    )


__all__ = [
    "QualityRegistry",
    "build_quality_registry",
]

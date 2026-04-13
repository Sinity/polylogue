"""Shared registry for validation lanes and durable quality campaigns."""

from __future__ import annotations

from dataclasses import dataclass

from devtools.benchmark_catalog import (
    BenchmarkCampaignEntry,
    build_benchmark_entries,
    build_synthetic_benchmark_entries,
)
from devtools.lane_models import LaneEntry
from devtools.mutation_catalog import MutationCampaignEntry, build_mutation_entries
from devtools.scenario_projection_catalog import build_scenario_projection_entries
from devtools.validation_catalog import build_validation_lane_entries
from polylogue.scenarios import CorpusScenario, ScenarioProjectionEntry
from polylogue.schemas.operator_inference import list_inferred_corpus_scenarios


@dataclass(frozen=True)
class QualityRegistry:
    contract_lanes: tuple[LaneEntry, ...]
    live_lanes: tuple[LaneEntry, ...]
    composite_lanes: tuple[LaneEntry, ...]
    mutation_campaigns: tuple[MutationCampaignEntry, ...]
    benchmark_campaigns: tuple[BenchmarkCampaignEntry, ...]
    synthetic_benchmark_campaigns: tuple[BenchmarkCampaignEntry, ...]
    inferred_corpus_scenarios: tuple[CorpusScenario, ...]
    scenario_projections: tuple[ScenarioProjectionEntry, ...]


def build_quality_registry() -> QualityRegistry:
    validation_lanes = build_validation_lane_entries()
    contract_lanes = tuple(entry for entry in validation_lanes if entry.category == "contract")
    live_lanes = tuple(entry for entry in validation_lanes if entry.category == "live")
    composite_lanes = tuple(entry for entry in validation_lanes if entry.category == "composite")
    mutation_campaigns = build_mutation_entries()
    benchmark_campaigns = build_benchmark_entries()
    synthetic_benchmark_campaigns = build_synthetic_benchmark_entries()
    inferred_corpus_scenarios = list_inferred_corpus_scenarios()
    return QualityRegistry(
        contract_lanes=contract_lanes,
        live_lanes=live_lanes,
        composite_lanes=composite_lanes,
        mutation_campaigns=mutation_campaigns,
        benchmark_campaigns=benchmark_campaigns,
        synthetic_benchmark_campaigns=synthetic_benchmark_campaigns,
        inferred_corpus_scenarios=inferred_corpus_scenarios,
        scenario_projections=build_scenario_projection_entries(
            validation_lanes=validation_lanes,
            mutation_campaigns=mutation_campaigns,
            benchmark_campaigns=benchmark_campaigns,
            synthetic_benchmark_campaigns=synthetic_benchmark_campaigns,
            inferred_corpus_scenarios=inferred_corpus_scenarios,
        ),
    )


__all__ = [
    "QualityRegistry",
    "build_quality_registry",
]

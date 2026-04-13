"""Shared registry for validation lanes and durable quality campaigns."""

from __future__ import annotations

from dataclasses import dataclass

from devtools.authored_scenario_catalog import AuthoredScenarioCatalog, build_authored_scenario_catalog
from devtools.benchmark_catalog import BenchmarkCampaignEntry
from devtools.lane_models import LaneEntry
from devtools.mutation_catalog import MutationCampaignEntry
from devtools.scenario_projection_catalog import build_scenario_projection_entries
from polylogue.scenarios import CorpusScenario, ScenarioProjectionEntry
from polylogue.showcase.scenario_models import ExerciseScenario


@dataclass(frozen=True)
class QualityRegistry:
    catalog: AuthoredScenarioCatalog
    scenario_projections: tuple[ScenarioProjectionEntry, ...]

    @property
    def exercise_scenarios(self) -> tuple[ExerciseScenario, ...]:
        return self.catalog.exercise_scenarios

    @property
    def qa_extra_scenarios(self) -> tuple[ExerciseScenario, ...]:
        return self.catalog.qa_extra_scenarios

    @property
    def contract_lanes(self) -> tuple[LaneEntry, ...]:
        return self.catalog.contract_lanes

    @property
    def live_lanes(self) -> tuple[LaneEntry, ...]:
        return self.catalog.live_lanes

    @property
    def composite_lanes(self) -> tuple[LaneEntry, ...]:
        return self.catalog.composite_lanes

    @property
    def mutation_campaigns(self) -> tuple[MutationCampaignEntry, ...]:
        return self.catalog.mutation_campaigns

    @property
    def benchmark_campaigns(self) -> tuple[BenchmarkCampaignEntry, ...]:
        return self.catalog.benchmark_campaigns

    @property
    def synthetic_benchmark_campaigns(self) -> tuple[BenchmarkCampaignEntry, ...]:
        return self.catalog.synthetic_benchmark_campaigns

    @property
    def inferred_corpus_scenarios(self) -> tuple[CorpusScenario, ...]:
        return self.catalog.inferred_corpus_scenarios


def build_quality_registry() -> QualityRegistry:
    catalog = build_authored_scenario_catalog()
    return QualityRegistry(
        catalog=catalog,
        scenario_projections=build_scenario_projection_entries(catalog=catalog),
    )


__all__ = [
    "QualityRegistry",
    "build_quality_registry",
]

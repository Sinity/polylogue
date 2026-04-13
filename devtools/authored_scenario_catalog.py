"""Central authored catalog for scenario-bearing verification sources."""

from __future__ import annotations

from dataclasses import dataclass

from devtools.benchmark_catalog import (
    BenchmarkCampaignEntry,
    build_benchmark_entries,
    build_synthetic_benchmark_entries,
)
from devtools.lane_models import LaneEntry
from devtools.mutation_catalog import MutationCampaignEntry, build_mutation_entries
from devtools.validation_catalog import build_validation_family_entries, build_validation_lane_entries
from devtools.validation_family_models import ValidationLaneFamily
from polylogue.scenarios import (
    CorpusScenario,
    ScenarioProjectionEntry,
    ScenarioProjectionSource,
    compile_projection_entries,
)
from polylogue.schemas.operator_inference import list_inferred_corpus_scenarios
from polylogue.showcase.exercise_models import Exercise
from polylogue.showcase.exercises import EXERCISE_SCENARIOS, QA_EXTRA_SCENARIOS


@dataclass(frozen=True)
class AuthoredScenarioCatalog:
    exercise_scenarios: tuple[Exercise, ...]
    qa_extra_scenarios: tuple[Exercise, ...]
    validation_families: tuple[ValidationLaneFamily, ...]
    validation_lanes: tuple[LaneEntry, ...]
    mutation_campaigns: tuple[MutationCampaignEntry, ...]
    benchmark_campaigns: tuple[BenchmarkCampaignEntry, ...]
    synthetic_benchmark_campaigns: tuple[BenchmarkCampaignEntry, ...]
    inferred_corpus_scenarios: tuple[CorpusScenario, ...]

    @property
    def contract_lanes(self) -> tuple[LaneEntry, ...]:
        return tuple(entry for entry in self.validation_lanes if entry.category == "contract")

    @property
    def live_lanes(self) -> tuple[LaneEntry, ...]:
        return tuple(entry for entry in self.validation_lanes if entry.category == "live")

    @property
    def composite_lanes(self) -> tuple[LaneEntry, ...]:
        return tuple(entry for entry in self.validation_lanes if entry.category == "composite")

    def projection_sources(self) -> tuple[ScenarioProjectionSource, ...]:
        return (
            *self.exercise_scenarios,
            *self.qa_extra_scenarios,
            *self.validation_families,
            *self.validation_lanes,
            *self.mutation_campaigns,
            *self.benchmark_campaigns,
            *self.synthetic_benchmark_campaigns,
            *self.inferred_corpus_scenarios,
        )

    def compile_projection_entries(self) -> tuple[ScenarioProjectionEntry, ...]:
        return tuple(
            sorted(
                compile_projection_entries(self.projection_sources()),
                key=lambda item: (item.source_kind.value, item.name),
            )
        )


def build_authored_scenario_catalog() -> AuthoredScenarioCatalog:
    return AuthoredScenarioCatalog(
        exercise_scenarios=EXERCISE_SCENARIOS,
        qa_extra_scenarios=QA_EXTRA_SCENARIOS,
        validation_families=build_validation_family_entries(),
        validation_lanes=build_validation_lane_entries(),
        mutation_campaigns=build_mutation_entries(),
        benchmark_campaigns=build_benchmark_entries(),
        synthetic_benchmark_campaigns=build_synthetic_benchmark_entries(),
        inferred_corpus_scenarios=list_inferred_corpus_scenarios(),
    )


__all__ = ["AuthoredScenarioCatalog", "build_authored_scenario_catalog"]

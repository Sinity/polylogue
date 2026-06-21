"""Central authored catalog for scenario-bearing verification sources."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from devtools.benchmark_catalog import (
    BenchmarkCampaignEntry,
    build_benchmark_entries,
    build_synthetic_benchmark_entries,
)
from devtools.lane_models import LaneEntry
from devtools.mutation_catalog import MutationCampaignEntry, build_mutation_entries
from devtools.validation_catalog import build_validation_lane_entries
from polylogue.scenarios import (
    CorpusScenario,
    ScenarioProjectionEntry,
    ScenarioProjectionSource,
    compile_projection_entries,
)
from polylogue.schemas.operator.inference import list_inferred_corpus_scenarios
from polylogue.showcase.exercise_models import Exercise
from polylogue.showcase.exercises import EXERCISE_SCENARIOS, SUPPLEMENTAL_SCENARIOS


@dataclass(frozen=True)
class AuthoredScenarioCatalog:
    exercise_scenarios: tuple[Exercise, ...]
    supplemental_scenarios: tuple[Exercise, ...]
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
        result: list[ScenarioProjectionSource] = []
        result.extend(self.exercise_scenarios)
        result.extend(self.supplemental_scenarios)
        result.extend(self.validation_lanes)  # type: ignore[arg-type]
        result.extend(self.mutation_campaigns)
        result.extend(self.benchmark_campaigns)
        result.extend(self.synthetic_benchmark_campaigns)
        result.extend(self.inferred_corpus_scenarios)
        return tuple(result)

    def validation_lane_index(self) -> dict[str, LaneEntry]:
        return {entry.name: entry for entry in self.validation_lanes}

    def mutation_campaign_index(self) -> dict[str, MutationCampaignEntry]:
        return {entry.name: entry for entry in self.mutation_campaigns}

    def benchmark_campaign_index(self) -> dict[str, BenchmarkCampaignEntry]:
        return {entry.name: entry for entry in self.benchmark_campaigns}

    def synthetic_benchmark_campaign_index(self) -> dict[str, BenchmarkCampaignEntry]:
        return {entry.name: entry for entry in self.synthetic_benchmark_campaigns}

    def compile_projection_entries(self) -> tuple[ScenarioProjectionEntry, ...]:
        return tuple(
            sorted(
                compile_projection_entries(self.projection_sources()),
                key=lambda item: (item.source_kind.value, item.name),
            )
        )


@lru_cache(maxsize=1)
def get_authored_scenario_catalog() -> AuthoredScenarioCatalog:
    return AuthoredScenarioCatalog(
        exercise_scenarios=EXERCISE_SCENARIOS,
        supplemental_scenarios=SUPPLEMENTAL_SCENARIOS,
        validation_lanes=build_validation_lane_entries(),
        mutation_campaigns=build_mutation_entries(),
        benchmark_campaigns=build_benchmark_entries(),
        synthetic_benchmark_campaigns=build_synthetic_benchmark_entries(),
        inferred_corpus_scenarios=list_inferred_corpus_scenarios(),
    )


__all__ = ["AuthoredScenarioCatalog", "get_authored_scenario_catalog"]

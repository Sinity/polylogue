"""Live catalog of authored scenario-bearing verification projections."""

from __future__ import annotations

from devtools.benchmark_catalog import (
    BenchmarkCampaignEntry,
    build_benchmark_entries,
    build_synthetic_benchmark_entries,
)
from devtools.lane_models import LaneEntry
from devtools.mutation_catalog import MutationCampaignEntry, build_mutation_entries
from devtools.validation_catalog import build_validation_lane_entries
from polylogue.scenarios import (
    ScenarioProjectionEntry,
    compile_projection_entries,
)
from polylogue.schemas.operator_inference import list_inferred_corpus_specs
from polylogue.showcase.exercises import EXERCISE_SCENARIOS, QA_EXTRA_SCENARIOS


def build_scenario_projection_entries(
    *,
    validation_lanes: tuple[LaneEntry, ...] | None = None,
    mutation_campaigns: tuple[MutationCampaignEntry, ...] | None = None,
    benchmark_campaigns: tuple[BenchmarkCampaignEntry, ...] | None = None,
    synthetic_benchmark_campaigns: tuple[BenchmarkCampaignEntry, ...] | None = None,
    inferred_corpus_specs: tuple[object, ...] | None = None,
) -> tuple[ScenarioProjectionEntry, ...]:
    lane_entries = validation_lanes or build_validation_lane_entries()
    mutation_entries = mutation_campaigns or build_mutation_entries()
    benchmark_entries = benchmark_campaigns or build_benchmark_entries()
    synthetic_benchmark_entries = synthetic_benchmark_campaigns or build_synthetic_benchmark_entries()
    inferred_specs = inferred_corpus_specs or list_inferred_corpus_specs()
    entries = list(
        compile_projection_entries(
            (
                *EXERCISE_SCENARIOS,
                *QA_EXTRA_SCENARIOS,
                *lane_entries,
                *mutation_entries,
                *benchmark_entries,
                *synthetic_benchmark_entries,
                *inferred_specs,
            )
        )
    )
    return tuple(sorted(entries, key=lambda item: (item.source_kind.value, item.name)))


__all__ = ["build_scenario_projection_entries"]

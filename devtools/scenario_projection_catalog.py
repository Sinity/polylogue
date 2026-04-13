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
    ScenarioProjectionSourceKind,
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
    entries = [
        ScenarioProjectionEntry.from_object(
            source_kind=ScenarioProjectionSourceKind.EXERCISE,
            name=scenario.scenario_id,
            description=scenario.description,
            obj=scenario,
        )
        for scenario in EXERCISE_SCENARIOS
    ]
    entries.extend(
        ScenarioProjectionEntry.from_object(
            source_kind=ScenarioProjectionSourceKind.EXERCISE,
            name=scenario.scenario_id,
            description=scenario.description,
            obj=scenario,
        )
        for scenario in QA_EXTRA_SCENARIOS
    )
    entries.extend(
        ScenarioProjectionEntry.from_object(
            source_kind=ScenarioProjectionSourceKind.VALIDATION_LANE,
            name=lane.name,
            description=lane.description,
            obj=lane,
        )
        for lane in lane_entries
    )
    entries.extend(
        ScenarioProjectionEntry.from_object(
            source_kind=ScenarioProjectionSourceKind.MUTATION_CAMPAIGN,
            name=campaign.name,
            description=campaign.description,
            obj=campaign,
        )
        for campaign in mutation_entries
    )
    entries.extend(
        ScenarioProjectionEntry.from_object(
            source_kind=ScenarioProjectionSourceKind.BENCHMARK_CAMPAIGN,
            name=entry.name,
            description=entry.description,
            obj=entry,
        )
        for entry in benchmark_entries
    )
    entries.extend(
        ScenarioProjectionEntry.from_object(
            source_kind=ScenarioProjectionSourceKind.SYNTHETIC_BENCHMARK,
            name=entry.name,
            description=entry.description,
            obj=entry,
        )
        for entry in synthetic_benchmark_entries
    )
    entries.extend(
        spec.to_projection_entry()
        for spec in inferred_specs
    )
    return tuple(sorted(entries, key=lambda item: (item.source_kind.value, item.name)))


__all__ = ["build_scenario_projection_entries"]

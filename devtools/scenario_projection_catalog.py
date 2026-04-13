"""Live catalog of authored scenario-bearing verification projections."""

from __future__ import annotations

from polylogue.scenarios import (
    ScenarioProjectionEntry,
    ScenarioProjectionSourceKind,
)
from polylogue.showcase.exercises import EXERCISE_SCENARIOS, QA_EXTRA_SCENARIOS

from .benchmark_catalog import build_benchmark_entries, build_synthetic_benchmark_entries
from .mutation_catalog import build_mutation_entries
from .validation_catalog import build_validation_lane_entries


def build_scenario_projection_entries() -> tuple[ScenarioProjectionEntry, ...]:
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
        for lane in build_validation_lane_entries()
    )
    entries.extend(
        ScenarioProjectionEntry.from_object(
            source_kind=ScenarioProjectionSourceKind.MUTATION_CAMPAIGN,
            name=campaign.name,
            description=campaign.description,
            obj=campaign,
        )
        for campaign in build_mutation_entries()
    )
    entries.extend(
        ScenarioProjectionEntry.from_object(
            source_kind=ScenarioProjectionSourceKind.BENCHMARK_CAMPAIGN,
            name=entry.name,
            description=entry.description,
            obj=entry,
        )
        for entry in build_benchmark_entries()
    )
    entries.extend(
        ScenarioProjectionEntry.from_object(
            source_kind=ScenarioProjectionSourceKind.SYNTHETIC_BENCHMARK,
            name=entry.name,
            description=entry.description,
            obj=entry,
        )
        for entry in build_synthetic_benchmark_entries()
    )
    return tuple(sorted(entries, key=lambda item: (item.source_kind.value, item.name)))


__all__ = ["build_scenario_projection_entries"]

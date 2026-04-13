"""Live catalog of authored scenario-bearing verification projections."""

from __future__ import annotations

from polylogue.scenarios import (
    ScenarioProjectionEntry,
    ScenarioProjectionSourceKind,
)
from polylogue.showcase.exercises import EXERCISE_SCENARIOS

from .benchmark_scenario_catalog import BENCHMARK_CAMPAIGNS
from .synthetic_benchmark_catalog import SYNTHETIC_BENCHMARK_SCENARIOS


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
            source_kind=ScenarioProjectionSourceKind.BENCHMARK_CAMPAIGN,
            name=campaign.name,
            description=campaign.description,
            obj=campaign,
        )
        for campaign in BENCHMARK_CAMPAIGNS.values()
    )
    entries.extend(
        ScenarioProjectionEntry.from_object(
            source_kind=ScenarioProjectionSourceKind.SYNTHETIC_BENCHMARK,
            name=scenario.scenario_id,
            description=scenario.description,
            obj=scenario,
        )
        for scenario in SYNTHETIC_BENCHMARK_SCENARIOS
    )
    return tuple(sorted(entries, key=lambda item: (item.source_kind.value, item.name)))


__all__ = ["build_scenario_projection_entries"]

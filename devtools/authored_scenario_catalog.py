"""Central authored catalog for verification lanes and campaigns."""

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


@dataclass(frozen=True)
class AuthoredScenarioCatalog:
    validation_lanes: tuple[LaneEntry, ...]
    mutation_campaigns: tuple[MutationCampaignEntry, ...]
    benchmark_campaigns: tuple[BenchmarkCampaignEntry, ...]
    synthetic_benchmark_campaigns: tuple[BenchmarkCampaignEntry, ...]

    def validation_lane_index(self) -> dict[str, LaneEntry]:
        return {entry.name: entry for entry in self.validation_lanes}

    def mutation_campaign_index(self) -> dict[str, MutationCampaignEntry]:
        return {entry.name: entry for entry in self.mutation_campaigns}

    def benchmark_campaign_index(self) -> dict[str, BenchmarkCampaignEntry]:
        return {entry.name: entry for entry in self.benchmark_campaigns}

    def synthetic_benchmark_campaign_index(self) -> dict[str, BenchmarkCampaignEntry]:
        return {entry.name: entry for entry in self.synthetic_benchmark_campaigns}


@lru_cache(maxsize=1)
def get_authored_scenario_catalog() -> AuthoredScenarioCatalog:
    return AuthoredScenarioCatalog(
        validation_lanes=build_validation_lane_entries(),
        mutation_campaigns=build_mutation_entries(),
        benchmark_campaigns=build_benchmark_entries(),
        synthetic_benchmark_campaigns=build_synthetic_benchmark_entries(),
    )


__all__ = ["AuthoredScenarioCatalog", "get_authored_scenario_catalog"]

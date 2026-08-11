"""Central authored catalog for executable benchmark and mutation campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from devtools.benchmark_catalog import (
    BenchmarkCampaignEntry,
    build_benchmark_entries,
    build_synthetic_benchmark_entries,
)
from devtools.mutation_catalog import MutationCampaignEntry, build_mutation_entries


@dataclass(frozen=True)
class AuthoredScenarioCatalog:
    mutation_campaigns: tuple[MutationCampaignEntry, ...]
    benchmark_campaigns: tuple[BenchmarkCampaignEntry, ...]
    synthetic_benchmark_campaigns: tuple[BenchmarkCampaignEntry, ...]

    def mutation_campaign_index(self) -> dict[str, MutationCampaignEntry]:
        return {entry.name: entry for entry in self.mutation_campaigns}

    def benchmark_campaign_index(self) -> dict[str, BenchmarkCampaignEntry]:
        return {entry.name: entry for entry in self.benchmark_campaigns}

    def synthetic_benchmark_campaign_index(self) -> dict[str, BenchmarkCampaignEntry]:
        return {entry.name: entry for entry in self.synthetic_benchmark_campaigns}


@lru_cache(maxsize=1)
def get_authored_scenario_catalog() -> AuthoredScenarioCatalog:
    return AuthoredScenarioCatalog(
        mutation_campaigns=build_mutation_entries(),
        benchmark_campaigns=build_benchmark_entries(),
        synthetic_benchmark_campaigns=build_synthetic_benchmark_entries(),
    )


__all__ = ["AuthoredScenarioCatalog", "get_authored_scenario_catalog"]

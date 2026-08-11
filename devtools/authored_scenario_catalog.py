"""Compatibility lookup for executable mutation campaigns."""

from __future__ import annotations

from dataclasses import dataclass

from devtools.mutation_catalog import MutationCampaignEntry, build_mutation_entries


@dataclass(frozen=True)
class AuthoredScenarioCatalog:
    mutation_campaigns: tuple[MutationCampaignEntry, ...]

    def mutation_campaign_index(self) -> dict[str, MutationCampaignEntry]:
        return {entry.name: entry for entry in self.mutation_campaigns}


def get_authored_scenario_catalog() -> AuthoredScenarioCatalog:
    return AuthoredScenarioCatalog(
        mutation_campaigns=build_mutation_entries(),
    )


__all__ = ["AuthoredScenarioCatalog", "get_authored_scenario_catalog"]

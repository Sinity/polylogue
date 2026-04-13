"""Mutation-campaign catalog shared across control-plane surfaces."""

from __future__ import annotations

from .mutation_scenario_catalog import MUTATION_CAMPAIGNS, MutationCampaign

MutationCampaignEntry = MutationCampaign


def build_mutation_entries() -> tuple[MutationCampaignEntry, ...]:
    return tuple(sorted(MUTATION_CAMPAIGNS.values(), key=lambda item: item.name))


__all__ = [
    "MutationCampaignEntry",
    "build_mutation_entries",
]

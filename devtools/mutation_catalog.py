"""Typed mutation-campaign catalog shared across control-plane surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from .mutmut_campaign import CAMPAIGNS as MUTATION_CAMPAIGNS


@dataclass(frozen=True)
class MutationCampaignEntry:
    name: str
    description: str
    paths_to_mutate: tuple[str, ...]
    tests: tuple[str, ...]
    notes: tuple[str, ...] = ()


def build_mutation_entries() -> tuple[MutationCampaignEntry, ...]:
    entries = [
        MutationCampaignEntry(
            name=campaign.name,
            description=campaign.description,
            paths_to_mutate=tuple(campaign.paths_to_mutate),
            tests=tuple(campaign.tests),
            notes=tuple(campaign.notes),
        )
        for campaign in MUTATION_CAMPAIGNS.values()
    ]
    return tuple(sorted(entries, key=lambda item: item.name))


__all__ = [
    "MutationCampaignEntry",
    "build_mutation_entries",
]

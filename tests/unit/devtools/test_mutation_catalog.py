from __future__ import annotations

from devtools.mutation_catalog import build_mutation_entries
from devtools.mutation_scenario_catalog import MUTATION_CAMPAIGNS
from polylogue.scenarios import ScenarioProjectionSourceKind


def test_build_mutation_entries_tracks_authored_catalog() -> None:
    entries = build_mutation_entries()

    assert {entry.name for entry in entries} == set(MUTATION_CAMPAIGNS)
    filters = next(entry for entry in entries if entry.name == "filters")
    assert filters.origin == "authored.mutation-campaign"
    assert filters.description == "ConversationFilter semantics and summary/picker contracts"
    assert filters.paths_to_mutate == ("polylogue/lib/filters.py",)
    assert filters.tests == (
        "tests/unit/core/test_filters_schemas.py",
        "tests/unit/core/test_filters_props.py",
    )
    assert filters.tags == ("mutation",)


def test_mutation_campaign_compiles_its_own_projection_entry() -> None:
    filters = MUTATION_CAMPAIGNS["filters"]

    projection = filters.to_projection_entry()

    assert projection.source_kind is ScenarioProjectionSourceKind.MUTATION_CAMPAIGN
    assert projection.name == "filters"
    assert projection.description == "ConversationFilter semantics and summary/picker contracts"
    assert projection.tags == ("mutation",)

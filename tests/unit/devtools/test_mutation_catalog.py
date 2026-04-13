from __future__ import annotations

from devtools.mutation_catalog import build_mutation_entries
from devtools.mutation_scenario_catalog import MUTATION_CAMPAIGNS


def test_build_mutation_entries_tracks_authored_catalog() -> None:
    entries = build_mutation_entries()

    assert {entry.name for entry in entries} == set(MUTATION_CAMPAIGNS)
    filters = next(entry for entry in entries if entry.name == "filters")
    assert filters.description == "ConversationFilter semantics and summary/picker contracts"
    assert filters.paths_to_mutate == ("polylogue/lib/filters.py",)
    assert filters.tests == (
        "tests/unit/core/test_filters_schemas.py",
        "tests/unit/core/test_filters_props.py",
    )

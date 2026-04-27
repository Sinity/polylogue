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
    assert filters.paths_to_mutate == ("polylogue/lib/filter/filters.py",)
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


def test_cli_run_campaign_carries_render_runtime_metadata() -> None:
    cli_run = MUTATION_CAMPAIGNS["cli-run"]

    assert cli_run.path_targets == ("conversation-render-loop",)
    assert cli_run.artifact_targets == (
        "conversation_render_projection",
        "rendered_conversation_artifacts",
    )
    assert cli_run.operation_targets == ("render-conversations",)
    assert cli_run.tags == ("mutation", "run", "render")


def test_site_builder_campaign_carries_publication_runtime_metadata() -> None:
    site_builder = MUTATION_CAMPAIGNS["site-builder"]

    assert site_builder.path_targets == ("site-publication-loop",)
    assert site_builder.artifact_targets == (
        "conversation_render_projection",
        "site_conversation_pages",
        "site_publication_manifest",
        "publication_records",
    )
    assert site_builder.operation_targets == ("publish-site",)
    assert site_builder.tags == ("mutation", "site", "publication")

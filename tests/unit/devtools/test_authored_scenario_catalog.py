from __future__ import annotations

from devtools.authored_scenario_catalog import build_authored_scenario_catalog


def test_build_authored_scenario_catalog_exposes_all_authored_families() -> None:
    catalog = build_authored_scenario_catalog()

    assert any(entry.name == "json-doctor-action-event-preview" for entry in catalog.exercise_scenarios)
    assert any(entry.name == "gen-schema-list" for entry in catalog.qa_extra_scenarios)
    assert any(entry.name == "machine-contract" for entry in catalog.contract_lanes)
    assert any(entry.name == "frontier-local" for entry in catalog.composite_lanes)
    assert any(entry.name == "filters" for entry in catalog.mutation_campaigns)
    assert any(entry.name == "search-filters" for entry in catalog.benchmark_campaigns)
    assert any(entry.name == "startup-health" for entry in catalog.synthetic_benchmark_campaigns)
    assert any(entry.projection_name == "chatgpt:v1" for entry in catalog.inferred_corpus_scenarios)


def test_authored_scenario_catalog_compiles_projection_entries_from_shared_sources() -> None:
    catalog = build_authored_scenario_catalog()

    projections = catalog.compile_projection_entries()

    assert any(entry.source_kind.value == "exercise" and entry.name == "json-doctor-action-event-preview" for entry in projections)
    assert any(entry.source_kind.value == "validation-lane" and entry.name == "machine-contract" for entry in projections)
    assert any(entry.source_kind.value == "mutation-campaign" and entry.name == "filters" for entry in projections)
    assert any(entry.source_kind.value == "benchmark-campaign" and entry.name == "search-filters" for entry in projections)
    assert any(entry.source_kind.value == "synthetic-benchmark" and entry.name == "startup-health" for entry in projections)
    assert any(entry.source_kind.value == "inferred-corpus-scenario" and entry.name == "chatgpt:v1" for entry in projections)

from __future__ import annotations

from devtools.authored_scenario_catalog import get_authored_scenario_catalog


def test_authored_scenario_catalog_builds_runtime_lookup_indexes() -> None:
    catalog = get_authored_scenario_catalog()

    assert catalog.validation_lane_index()["machine-contract"].name == "machine-contract"
    assert catalog.mutation_campaign_index()["filters"].name == "filters"
    assert catalog.benchmark_campaign_index()["search-filters"].name == "search-filters"
    assert (
        catalog.synthetic_benchmark_campaign_index()["action-event-materialization"].name
        == "action-event-materialization"
    )


def test_authored_scenario_catalog_is_cached_singleton() -> None:
    assert get_authored_scenario_catalog() is get_authored_scenario_catalog()

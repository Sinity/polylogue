"""Live catalog of authored scenario-bearing verification projections."""

from __future__ import annotations

from devtools.authored_scenario_catalog import AuthoredScenarioCatalog, get_authored_scenario_catalog
from polylogue.scenarios import ScenarioProjectionEntry


def build_scenario_projection_entries(
    *,
    catalog: AuthoredScenarioCatalog | None = None,
) -> tuple[ScenarioProjectionEntry, ...]:
    authored_catalog = catalog or get_authored_scenario_catalog()
    return authored_catalog.compile_projection_entries()


__all__ = ["build_scenario_projection_entries"]

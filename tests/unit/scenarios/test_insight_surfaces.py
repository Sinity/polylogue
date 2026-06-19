from __future__ import annotations

from polylogue.scenarios import build_insight_contract_surfaces, build_live_insight_surface_lanes


def test_build_insight_contract_surfaces_compiles_canonical_json_contract_entries() -> None:
    surfaces = {surface.name: surface for surface in build_insight_contract_surfaces()}

    assert surfaces["json-insights-profiles"].args == ("ops", "insights", "profiles", "--format", "json")
    assert surfaces["json-insights-profiles"].tags == ("insights", "session-profiles")
    assert surfaces["json-insights-coverage"].args == ("ops", "insights", "coverage", "--format", "json")


def test_build_live_insight_surface_lanes_compiles_live_variants() -> None:
    surfaces = {surface.name: surface for surface in build_live_insight_surface_lanes()}

    assert surfaces["live-insights-status"].args == ("ops", "insights", "status", "--format", "json")
    assert surfaces["live-insights-profiles-evidence"].args == (
        "ops",
        "insights",
        "profiles",
        "--tier",
        "evidence",
        "--limit",
        "3",
        "--format",
        "json",
    )
    assert surfaces["live-insights-coverage-day"].args == (
        "--provider",
        "claude-code",
        "--since",
        "2026-03-01",
        "ops",
        "insights",
        "coverage",
        "--group-by",
        "day",
        "--limit",
        "14",
        "--format",
        "json",
    )
    assert surfaces["live-insights-debt"].tags == ("insights", "debt")

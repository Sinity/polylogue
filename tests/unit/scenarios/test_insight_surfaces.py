from __future__ import annotations

from polylogue.scenarios import build_insight_contract_surfaces, build_live_insight_surface_lanes


def test_build_insight_contract_surfaces_compiles_canonical_json_contract_entries() -> None:
    surfaces = {surface.name: surface for surface in build_insight_contract_surfaces()}

    assert surfaces["json-insights-profiles"].args == ("insights", "profiles", "--format", "json")
    assert surfaces["json-insights-profiles"].tags == ("insights", "session-profiles")
    assert surfaces["json-insights-week-summaries"].args == ("insights", "week-summaries", "--format", "json")
    assert surfaces["json-insights-analytics"].args == ("insights", "analytics", "--format", "json")


def test_build_live_insight_surface_lanes_compiles_live_variants() -> None:
    surfaces = {surface.name: surface for surface in build_live_insight_surface_lanes()}

    assert surfaces["live-insights-status"].args == ("insights", "status", "--format", "json")
    assert surfaces["live-insights-profiles-evidence"].args == (
        "insights",
        "profiles",
        "--tier",
        "evidence",
        "--limit",
        "3",
        "--format",
        "json",
    )
    assert surfaces["live-insights-day-summaries"].args == (
        "--provider",
        "claude-code",
        "--since",
        "2026-03-01",
        "insights",
        "day-summaries",
        "--limit",
        "14",
        "--format",
        "json",
    )
    assert surfaces["live-insights-debt"].tags == ("insights", "debt")

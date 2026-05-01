from __future__ import annotations

from polylogue.scenarios import (
    build_live_operational_surface_lanes,
    build_memory_budget_operational_surface_lanes,
    build_operational_contract_surfaces,
)


def test_build_operational_contract_surfaces_compiles_runtime_aligned_json_contracts() -> None:
    surfaces = {surface.name: surface for surface in build_operational_contract_surfaces()}

    assert surfaces["json-doctor"].args == ("doctor", "--format", "json")
    assert surfaces["json-doctor"].tags == ("maintenance", "readiness")
    assert surfaces["json-doctor-action-event-preview"].args == (
        "doctor",
        "--format",
        "json",
        "--repair",
        "--preview",
        "--target",
        "action_event_read_model",
    )
    assert surfaces["json-doctor-action-event-preview"].tags == ("maintenance", "action-events")
    assert surfaces["json-doctor-session-insights-preview"].args == (
        "doctor",
        "--format",
        "json",
        "--repair",
        "--preview",
        "--target",
        "session_products",
    )
    assert surfaces["json-doctor-session-insights-preview"].tags == ("maintenance", "session-insights")


def test_build_live_operational_surface_lanes_compiles_live_variants() -> None:
    surfaces = {surface.name: surface for surface in build_live_operational_surface_lanes()}

    assert surfaces["live-readiness-json"].args == ("doctor", "--format", "json")
    assert surfaces["live-readiness-json"].tags == ("maintenance", "readiness")
    assert surfaces["live-retrieval-checks"].args == (
        "--provider",
        "claude-code",
        "--since",
        "2026-01-01",
        "--stats-by",
        "action",
        "--format",
        "json",
        "--limit",
        "50",
    )
    assert surfaces["live-retrieval-checks"].tags == ("live", "retrieval", "readiness")
    assert surfaces["live-project-stats"].args == (
        "--provider",
        "claude-code",
        "--since",
        "2026-01-01",
        "--stats-by",
        "project",
        "--format",
        "json",
        "--limit",
        "50",
    )


def test_build_memory_budget_operational_surface_lanes_compiles_budget_variants() -> None:
    surfaces = {surface.name: surface for surface in build_memory_budget_operational_surface_lanes()}

    assert surfaces["memory-budget"].args == (
        "--provider",
        "claude-code",
        "--since",
        "2026-01-01",
        "--stats-by",
        "action",
        "--format",
        "json",
        "--limit",
        "50",
    )
    assert surfaces["memory-budget"].max_rss_mb == 1536
    assert surfaces["maintenance-memory-budget"].args == (
        "doctor",
        "--format",
        "json",
        "--repair",
        "--cleanup",
        "--preview",
    )
    assert surfaces["maintenance-memory-budget"].max_rss_mb == 1024

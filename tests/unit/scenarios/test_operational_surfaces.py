from __future__ import annotations

from polylogue.scenarios import (
    build_live_operational_surface_lanes,
    build_memory_budget_operational_surface_lanes,
    build_operational_contract_surfaces,
)


def test_build_operational_contract_surfaces_compiles_runtime_aligned_json_contracts() -> None:
    surfaces = {surface.name: surface for surface in build_operational_contract_surfaces()}

    assert surfaces["json-doctor"].args == ("doctor", "--json")
    assert surfaces["json-doctor"].tags == ("maintenance", "health")
    assert surfaces["json-doctor-action-event-preview"].args == (
        "doctor",
        "--json",
        "--repair",
        "--preview",
        "--target",
        "action_event_read_model",
    )
    assert surfaces["json-doctor-action-event-preview"].tags == ("maintenance", "action-events")
    assert surfaces["json-doctor-session-products-preview"].args == (
        "doctor",
        "--json",
        "--repair",
        "--preview",
        "--target",
        "session_products",
    )
    assert surfaces["json-doctor-session-products-preview"].tags == ("maintenance", "session-products")


def test_build_live_operational_surface_lanes_compiles_live_variants() -> None:
    surfaces = {surface.name: surface for surface in build_live_operational_surface_lanes()}

    assert surfaces["live-health-json"].args == ("doctor", "--json")
    assert surfaces["live-health-json"].tags == ("maintenance", "health")
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
    assert surfaces["live-retrieval-checks"].tags == ("live", "retrieval", "health")
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
        "--json",
        "--repair",
        "--cleanup",
        "--preview",
    )
    assert surfaces["maintenance-memory-budget"].max_rss_mb == 1024

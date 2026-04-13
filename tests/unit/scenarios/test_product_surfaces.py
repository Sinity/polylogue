from __future__ import annotations

from polylogue.scenarios import build_live_product_surface_lanes, build_product_contract_surfaces


def test_build_product_contract_surfaces_compiles_canonical_json_contract_entries() -> None:
    surfaces = {surface.name: surface for surface in build_product_contract_surfaces()}

    assert surfaces["json-products-profiles"].args == ("products", "profiles", "--json")
    assert surfaces["json-products-profiles"].tags == ("products", "session-profiles")
    assert surfaces["json-products-week-summaries"].args == ("products", "week-summaries", "--json")
    assert surfaces["json-products-analytics"].args == ("products", "analytics", "--json")


def test_build_live_product_surface_lanes_compiles_live_variants() -> None:
    surfaces = {surface.name: surface for surface in build_live_product_surface_lanes()}

    assert surfaces["live-products-status"].args == ("products", "status", "--json")
    assert surfaces["live-products-profiles-evidence"].args == (
        "products",
        "profiles",
        "--tier",
        "evidence",
        "--limit",
        "3",
        "--json",
    )
    assert surfaces["live-products-day-summaries"].args == (
        "--provider",
        "claude-code",
        "--since",
        "2026-03-01",
        "products",
        "day-summaries",
        "--limit",
        "14",
        "--json",
    )
    assert surfaces["live-products-debt"].tags == ("products", "debt")

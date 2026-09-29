"""Continuity must observe the complete production viewport-profile inventory."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from devtools.continuity_replay import compare_observed_facts, project_fact
from devtools.continuity_scenarios import continuity_scenario
from polylogue import Polylogue
from polylogue.archive.viewport import profiles
from polylogue.core.json import JSONDocument, require_json_document
from tests.infra.continuity import load_continuity_catalog
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock


@pytest.mark.asyncio
async def test_self_inspection_detects_removed_viewport_profile_on_mcp_route(
    mcp_server: MCPServerUnderTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting a shipped profile must fail the oracle even when capability total is unchanged."""
    poly = make_polylogue_mock()
    poly.stats = AsyncMock(return_value=SimpleNamespace(session_count=0, message_count=0))

    async def real_profiles() -> list[JSONDocument]:
        # Execute the existing facade method, not a mock list derived from the oracle.
        return await Polylogue.list_read_view_profiles(poly)

    poly.list_read_view_profiles = AsyncMock(side_effect=real_profiles)
    catalog = load_continuity_catalog()
    oracles = require_json_document(catalog["oracles"], context="continuity oracles")
    oracle = require_json_document(oracles["self-inspection"], context="self-inspection oracle")
    expected = require_json_document(oracle["facts"], context="self-inspection facts")["read_view_ids"]

    async def explain_profiles() -> JSONDocument:
        with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
            text = await invoke_surface_async(
                mcp_server._tool_manager._tools["explain"].fn,
                subject="capability",
                limit=1,
            )
        return require_json_document(json.loads(text), context="capability response")

    before = await explain_profiles()
    assert "read_view_profiles" in before
    scenario = continuity_scenario("self-inspection")
    projection = next(fact for fact in scenario.fact_projections if fact.name == "read_view_ids")
    observed = project_fact(projection, {"read-views": before})
    assert observed == expected
    assert isinstance(observed, list)
    assert "chronicle" in observed
    assert len(observed) > 1  # The query-catalog page size must not truncate profile identities.

    monkeypatch.setattr(
        profiles,
        "READ_VIEW_PROFILES",
        tuple(profile for profile in profiles.READ_VIEW_PROFILES if profile.view_id != "chronicle"),
    )
    after = await explain_profiles()
    assert after["total"] == before["total"]
    changed = project_fact(projection, {"read-views": after})
    assert changed != expected
    diagnostics = compare_observed_facts(
        expected={"read_view_ids": expected},
        observed={"read_view_ids": changed},
        source_refs=("source:polylogue/archive/viewport/profiles.py",),
    )
    assert any(item["kind"] == "fact_mismatch" and item["fact"] == "read_view_ids" for item in diagnostics)

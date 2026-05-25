from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.insights.archive import SessionProfileInsight
from polylogue.insights.archive_models import SessionEvidencePayload, SessionInferencePayload
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock
from tests.unit.mcp.test_tool_contracts import _inference_provenance, _provenance


def _profile() -> SessionProfileInsight:
    return SessionProfileInsight(
        conversation_id="conv-1",
        provider_name="claude-code",
        title="Profiled Session",
        semantic_tier="merged",
        provenance=_provenance(),
        evidence=SessionEvidencePayload(
            canonical_session_date="2026-03-24",
            message_count=2,
            cwd_paths=("/realm/project/polylogue",),
            terminal_state_evidence={"message_id": "u1"},
        ),
        inference_provenance=_inference_provenance(),
        inference=SessionInferencePayload(
            engaged_duration_ms=120000,
            workflow_shape="agentic_loop",
            workflow_shape_confidence=0.86,
            terminal_state="question_left",
            terminal_state_confidence=0.72,
        ),
    )


@pytest.mark.asyncio
async def test_workflow_shape_distribution_groups_materialized_profiles(
    mcp_server: MCPServerUnderTest,
) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(return_value=[_profile()])
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["workflow_shape_distribution"].fn,
            group_by="provider",
        )

    payload = json.loads(raw)
    assert payload["buckets"]["claude-code"]["agentic_loop"] == 1


@pytest.mark.asyncio
async def test_find_abandoned_sessions_filters_and_cites_terminal_evidence(
    mcp_server: MCPServerUnderTest,
) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(return_value=[_profile()])
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["find_abandoned_sessions"].fn,
            repo_path="polylogue",
            min_severity="question_left",
            limit=5,
        )

    payload = json.loads(raw)
    assert payload["total"] == 1
    assert payload["items"][0]["terminal_state"] == "question_left"
    assert payload["items"][0]["evidence"] == {"message_id": "u1"}

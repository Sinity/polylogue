"""Tests for the aggregate_sessions MCP tool (#1691)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.insights.archive import SessionProfileInsight
from polylogue.insights.archive_models import (
    SessionEvidencePayload,
    SessionInferencePayload,
)
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock
from tests.unit.mcp.test_tool_contracts import _inference_provenance, _provenance


def _make_profile(
    session_id: str,
    source_name: str = "claude-code",
    workflow_shape: str = "agentic_loop",
    terminal_state: str = "resolved",
) -> SessionProfileInsight:
    """Build a minimal session profile for aggregate testing."""
    return SessionProfileInsight(
        session_id=session_id,
        logical_session_id=session_id,
        source_name=source_name,
        title=f"Session {session_id}",
        provenance=_provenance(),
        semantic_tier="merged",
        evidence=SessionEvidencePayload(message_count=5),
        inference_provenance=_inference_provenance(),
        inference=SessionInferencePayload(
            workflow_shape=workflow_shape,
            terminal_state=terminal_state,
        ),
    )


# ── GROUP BY workflow_shape ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_aggregate_by_workflow_shape(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(
            return_value=[
                _make_profile("c1", workflow_shape="chat"),
                _make_profile("c2", workflow_shape="chat"),
                _make_profile("c3", workflow_shape="agentic_loop"),
            ]
        )
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["aggregate_sessions"].fn,
            group_by="workflow_shape",
        )

    payload = json.loads(raw)
    assert payload["group_by"] == "workflow_shape"
    assert payload["total_sessions"] == 3
    assert payload["buckets"] == {"chat": 2, "agentic_loop": 1}


@pytest.mark.asyncio
async def test_aggregate_by_workflow_shape_unknown_when_null_inference(
    mcp_server: MCPServerUnderTest,
) -> None:
    profile = SessionProfileInsight(
        session_id="c-null",
        logical_session_id="c-null",
        source_name="codex",
        provenance=_provenance(),
        semantic_tier="merged",
        evidence=SessionEvidencePayload(message_count=1),
        inference_provenance=None,
        inference=None,
    )
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(return_value=[profile])
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["aggregate_sessions"].fn,
            group_by="workflow_shape",
        )

    payload = json.loads(raw)
    assert payload["buckets"] == {"unknown": 1}


# ── GROUP BY terminal_state ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_aggregate_by_terminal_state(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(
            return_value=[
                _make_profile("c1", terminal_state="resolved"),
                _make_profile("c2", terminal_state="resolved"),
                _make_profile("c3", terminal_state="question_left"),
                _make_profile("c4", terminal_state="question_left"),
                _make_profile("c5", terminal_state="error_left"),
            ]
        )
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["aggregate_sessions"].fn,
            group_by="terminal_state",
        )

    payload = json.loads(raw)
    assert payload["group_by"] == "terminal_state"
    assert payload["total_sessions"] == 5
    assert payload["buckets"] == {
        "resolved": 2,
        "question_left": 2,
        "error_left": 1,
    }


# ── GROUP BY origin ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_aggregate_by_origin(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(
            return_value=[
                _make_profile("c1", source_name="claude-code"),
                _make_profile("c2", source_name="claude-code"),
                _make_profile("c3", source_name="claude-code"),
                _make_profile("c4", source_name="chatgpt"),
                _make_profile("c5", source_name="codex"),
            ]
        )
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["aggregate_sessions"].fn,
            group_by="origin",
        )

    payload = json.loads(raw)
    assert payload["group_by"] == "origin"
    assert payload["total_sessions"] == 5
    assert payload["buckets"] == {
        "claude-code-session": 3,
        "chatgpt-export": 1,
        "codex-session": 1,
    }


@pytest.mark.asyncio
async def test_aggregate_by_origin_unknown_when_empty_source(
    mcp_server: MCPServerUnderTest,
) -> None:
    profile = SessionProfileInsight(
        session_id="c-unk",
        logical_session_id="c-unk",
        source_name="",
        provenance=_provenance(),
        semantic_tier="merged",
        evidence=SessionEvidencePayload(message_count=1),
        inference_provenance=_inference_provenance(),
        inference=SessionInferencePayload(),
    )
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(return_value=[profile])
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["aggregate_sessions"].fn,
            group_by="origin",
        )

    payload = json.loads(raw)
    assert payload["buckets"] == {"unknown": 1}


# ── Invalid group_by ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_aggregate_with_invalid_group_by_returns_error(
    mcp_server: MCPServerUnderTest,
) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(return_value=[_make_profile("c1")])
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["aggregate_sessions"].fn,
            group_by="invalid_key",
        )

    payload = json.loads(raw)
    assert "message" in payload
    assert "invalid_key" in payload["message"]


# ── Empty archive ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_aggregate_with_empty_archive_returns_empty_buckets(
    mcp_server: MCPServerUnderTest,
) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(return_value=[])
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["aggregate_sessions"].fn,
            group_by="workflow_shape",
        )

    payload = json.loads(raw)
    assert payload["group_by"] == "workflow_shape"
    assert payload["total_sessions"] == 0
    assert payload["buckets"] == {}


# ── Single session ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_aggregate_with_single_session_counts_one(
    mcp_server: MCPServerUnderTest,
) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(return_value=[_make_profile("only-one")])
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["aggregate_sessions"].fn,
            group_by="workflow_shape",
        )

    payload = json.loads(raw)
    assert payload["total_sessions"] == 1
    assert payload["buckets"] == {"agentic_loop": 1}


# ── Default group_by ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_aggregate_defaults_to_workflow_shape(
    mcp_server: MCPServerUnderTest,
) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(return_value=[_make_profile("c1", workflow_shape="chat")])
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["aggregate_sessions"].fn,
        )

    payload = json.loads(raw)
    assert payload["group_by"] == "workflow_shape"
    assert payload["buckets"] == {"chat": 1}

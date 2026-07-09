"""Thin-wrapper tests for session analysis primitive MCP tools (#1691 / polylogue-9e5.24).

Covers compare_sessions, find_similar_sessions, and correlate_sessions.
The actual math (set-diff, metadata-similarity heuristic, Pearson
correlation) moved to polylogue.insights.session_analytics and is pinned
directly (no MCP/mock scaffolding) in
tests/unit/insights/test_session_analytics.py. This module only proves
each MCP tool's argument parsing/validation and delegation to the facade
method; tests/unit/mcp/test_analysis_primitives_facade_parity.py is the
concrete facade-vs-MCP equivalence proof.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock

# ── compare_sessions ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compare_sessions_delegates_to_facade_with_parsed_ids(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.compare_sessions = AsyncMock(
            return_value={
                "sessions": [{"id": "c1"}, {"id": "c2"}],
                "differences": {},
                "not_found": [],
                "total_requested": 2,
                "total_found": 2,
            }
        )
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["compare_sessions"].fn,
            session_ids="c1, c2",
        )

    payload = json.loads(raw)
    assert payload["total_found"] == 2
    mock_poly.compare_sessions.assert_awaited_once_with(["c1", "c2"])


@pytest.mark.asyncio
async def test_compare_sessions_rejects_empty_input_without_calling_facade(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.compare_sessions = AsyncMock()
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["compare_sessions"].fn,
            session_ids=None,
        )

    payload = json.loads(raw)
    assert "message" in payload or "code" in payload
    mock_poly.compare_sessions.assert_not_awaited()


@pytest.mark.asyncio
async def test_compare_sessions_rejects_single_id_without_calling_facade(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.compare_sessions = AsyncMock()
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["compare_sessions"].fn,
            session_ids="c1",
        )

    payload = json.loads(raw)
    assert "message" in payload or "code" in payload
    mock_poly.compare_sessions.assert_not_awaited()


@pytest.mark.asyncio
async def test_compare_sessions_rejects_too_many_ids_without_calling_facade(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.compare_sessions = AsyncMock()
        mock_get_polylogue.return_value = mock_poly

        ids = ",".join(f"c{i}" for i in range(1, 12))
        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["compare_sessions"].fn,
            session_ids=ids,
        )

    payload = json.loads(raw)
    assert "Too many" in str(payload)
    mock_poly.compare_sessions.assert_not_awaited()


# ── find_similar_sessions ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_find_similar_sessions_metadata_delegates_to_facade(
    mcp_server: MCPServerUnderTest,
) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.config = MagicMock(embedding_enabled=False)
        mock_poly.find_similar_sessions_by_metadata = AsyncMock(
            return_value={"source_session_id": "ref", "method": "metadata", "similar": [{"session_id": "sim"}]}
        )
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["find_similar_sessions"].fn,
            session_id="ref",
            similarity_dimension="metadata",
            limit=5,
        )

    payload = json.loads(raw)
    assert payload["method"] == "metadata"
    assert payload["similar"][0]["session_id"] == "sim"
    mock_poly.find_similar_sessions_by_metadata.assert_awaited_once()
    call_kwargs = mock_poly.find_similar_sessions_by_metadata.await_args.kwargs
    assert call_kwargs["limit"] == 5


@pytest.mark.asyncio
async def test_find_similar_sessions_handles_not_found(
    mcp_server: MCPServerUnderTest,
) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.config = MagicMock(embedding_enabled=False)
        mock_poly.find_similar_sessions_by_metadata = AsyncMock(return_value=None)
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["find_similar_sessions"].fn,
            session_id="nonexistent",
            similarity_dimension="metadata",
        )

    payload = json.loads(raw)
    assert "message" in payload or "code" in payload


@pytest.mark.asyncio
async def test_find_similar_sessions_rejects_invalid_dimension(
    mcp_server: MCPServerUnderTest,
) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.config = MagicMock(embedding_enabled=False)
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["find_similar_sessions"].fn,
            session_id="c1",
            similarity_dimension="color",
        )

    payload = json.loads(raw)
    assert "message" in payload or "code" in payload


@pytest.mark.asyncio
async def test_find_similar_sessions_prefers_embedding_neighbors_when_enabled(
    mcp_server: MCPServerUnderTest,
) -> None:
    neighbor = MagicMock()
    neighbor.session_id = "n1"
    neighbor.score = 0.9
    neighbor.rank = 1
    neighbor.reasons = []
    neighbor.summary = MagicMock(title="Neighbor", origin="claude-code-session")

    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.config = MagicMock(embedding_enabled=True)
        mock_poly.neighbor_candidates = AsyncMock(return_value=[neighbor])
        mock_poly.find_similar_sessions_by_metadata = AsyncMock()
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["find_similar_sessions"].fn,
            session_id="ref",
            similarity_dimension="auto",
        )

    payload = json.loads(raw)
    assert payload["method"] == "embedding"
    mock_poly.find_similar_sessions_by_metadata.assert_not_awaited()


# ── correlate_sessions ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_correlate_sessions_delegates_to_facade(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.correlate_sessions = AsyncMock(
            return_value={
                "metric_x": "message_count",
                "metric_y": "word_count",
                "pearson_r": 1.0,
                "sample_count": 3,
                "interpretation": "strong positive correlation (r=1.000)",
            }
        )
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["correlate_sessions"].fn,
            metric_x="message_count",
            metric_y="word_count",
            origin="claude-code-session",
        )

    payload = json.loads(raw)
    assert payload["sample_count"] == 3
    assert payload["pearson_r"] == 1.0
    mock_poly.correlate_sessions.assert_awaited_once_with(
        metric_x="message_count",
        metric_y="word_count",
        provider="claude-code",
        since=None,
        until=None,
    )


@pytest.mark.asyncio
async def test_correlate_sessions_maps_invalid_metric_value_error_to_error_envelope(
    mcp_server: MCPServerUnderTest,
) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.correlate_sessions = AsyncMock(side_effect=ValueError("Unknown metric_x: 'favorite_color'"))
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["correlate_sessions"].fn,
            metric_x="favorite_color",
            metric_y="word_count",
        )

    payload = json.loads(raw)
    assert "Unknown metric_x" in str(payload)
    assert "Supported metrics" in str(payload)

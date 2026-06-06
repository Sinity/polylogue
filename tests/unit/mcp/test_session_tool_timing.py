"""Tests for the session_tool_timing MCP tool (#1686)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.archive.models import Session
from polylogue.insights.otlp_correlation import (
    SessionToolTiming,
    ToolTimingEntry,
)
from tests.infra.builders import make_conv, make_msg
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock


def _make_session() -> Session:
    """Build a mock session for testing."""
    return make_conv(
        id="test-session-1",
        provider="claude-code",
        title="Test Session",
        messages=[
            make_msg(id="msg-1", role="user", text="Read the file"),
            make_msg(id="msg-2", role="assistant", text="I'll read it."),
        ],
    )


def _make_tool_timing_otlp() -> SessionToolTiming:
    """Create a SessionToolTiming with OTLP evidence."""
    return SessionToolTiming(
        session_id="test-session-1",
        tool_timings=[
            ToolTimingEntry(
                tool_name="Read",
                start_time="2024-01-15T10:00:00",
                end_time="2024-01-15T10:00:01",
                duration_ms=1000,
                status="ok",
                evidence_source="otlp_span",
                span_id="span-1",
            ),
            ToolTimingEntry(
                tool_name="Write",
                start_time="2024-01-15T10:00:02",
                end_time="2024-01-15T10:00:03",
                duration_ms=500,
                status="error",
                evidence_source="otlp_span",
                span_id="span-2",
            ),
        ],
        evidence_available=True,
        total_tools_with_otlp=2,
        total_tools_total=2,
    )


def _make_tool_timing_fallback() -> SessionToolTiming:
    """Create a SessionToolTiming with message-gap estimates."""
    return SessionToolTiming(
        session_id="test-session-1",
        tool_timings=[
            ToolTimingEntry(
                tool_name="Read",
                start_time="2024-01-15T10:00:00",
                end_time="2024-01-15T10:00:01",
                duration_ms=1000,
                evidence_source="message_gap_estimate",
            ),
        ],
        evidence_available=False,
        total_tools_with_otlp=0,
        total_tools_total=1,
    )


@pytest.mark.asyncio
async def test_session_tool_timing_not_found(
    mcp_server: MCPServerUnderTest,
) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.get_session = AsyncMock(return_value=None)
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["session_tool_timing"].fn,
            session_id="nonexistent",
        )

    payload = json.loads(raw)
    assert "error" in payload or "code" in payload


@pytest.mark.asyncio
async def test_session_tool_timing_with_otlp_data(
    mcp_server: MCPServerUnderTest,
) -> None:
    conv = _make_session()
    timing = _make_tool_timing_otlp()

    with (
        patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        patch(
            "polylogue.insights.otlp_correlation.get_session_tool_timing",
            return_value=timing,
        ),
    ):
        mock_poly = make_polylogue_mock()
        mock_poly.get_session = AsyncMock(return_value=conv)
        mock_poly.db_path = ":memory:"
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["session_tool_timing"].fn,
            session_id="test-session-1",
        )

    payload = json.loads(raw)
    root = payload.get("root", payload)
    assert root["session_id"] == "test-session-1"
    assert root["evidence_available"] is True
    assert root["total_tools_with_otlp"] == 2
    assert len(root["tool_timings"]) == 2

    tool1 = root["tool_timings"][0]
    assert tool1["tool_name"] == "Read"
    assert tool1["evidence_source"] == "otlp_span"
    assert tool1["span_id"] == "span-1"
    assert tool1["status"] == "ok"

    tool2 = root["tool_timings"][1]
    assert tool2["tool_name"] == "Write"
    assert tool2["evidence_source"] == "otlp_span"
    assert tool2["status"] == "error"


@pytest.mark.asyncio
async def test_session_tool_timing_fallback_to_message_gaps(
    mcp_server: MCPServerUnderTest,
) -> None:
    conv = _make_session()
    timing = _make_tool_timing_fallback()

    with (
        patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        patch(
            "polylogue.insights.otlp_correlation.get_session_tool_timing",
            return_value=timing,
        ),
    ):
        mock_poly = make_polylogue_mock()
        mock_poly.get_session = AsyncMock(return_value=conv)
        mock_poly.db_path = ":memory:"
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["session_tool_timing"].fn,
            session_id="test-session-1",
        )

    payload = json.loads(raw)
    root = payload.get("root", payload)
    assert root["session_id"] == "test-session-1"
    assert root["evidence_available"] is False
    assert root["total_tools_with_otlp"] == 0
    assert len(root["tool_timings"]) == 1
    assert root["tool_timings"][0]["tool_name"] == "Read"
    assert root["tool_timings"][0]["evidence_source"] == "message_gap_estimate"

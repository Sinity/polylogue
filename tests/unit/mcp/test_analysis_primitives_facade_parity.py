"""Facade/MCP parity proof for the sunk session-analysis primitives.

polylogue-9e5.24 moved the math behind correlate_sessions,
find_similar_sessions's metadata lane, aggregate_sessions,
workflow_shape_distribution, find_abandoned_sessions,
tool_call_latency_distribution, and compare_sessions out of the MCP-only
surface (mcp/server_insight_tools.py) into polylogue/insights/ + the async
Polylogue facade (polylogue/api/insights.py). The math itself is unit
tested directly against the pure functions in
tests/unit/insights/test_archive_rollups.py and
tests/unit/insights/test_session_analytics.py.

This module is the concrete "no behavior change" proof the bead's
acceptance criteria calls for: the same fixture profiles fetched through
the real ``Polylogue`` facade method, and through the MCP tool's
underlying handler function (called directly, not over the wire, against
that same facade instance), produce byte-identical JSON output -- because
both paths now share one insights-layer implementation instead of two.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from polylogue import Polylogue
from polylogue.insights.archive import SessionLatencyProfileInsight, SessionProfileInsight
from polylogue.insights.archive_models import (
    SessionEvidencePayload,
    SessionInferencePayload,
    SessionLatencyProfilePayload,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async
from tests.unit.mcp.test_tool_contracts import _inference_provenance, _provenance


def _archive(tmp_path: Path) -> Polylogue:
    """Construct a real Polylogue facade against an isolated tmp archive."""
    with ArchiveStore(tmp_path):
        pass
    return Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")


def _profile(
    session_id: str,
    *,
    source_name: str = "claude-code",
    workflow_shape: str = "agentic_loop",
    terminal_state: str = "resolved",
    message_count: int = 10,
    word_count: int = 500,
    tool_use_count: int = 5,
    tool_active_duration_ms: int = 45_000,
    canonical_session_date: str = "2026-05-01",
    tags: tuple[str, ...] = (),
) -> SessionProfileInsight:
    return SessionProfileInsight(
        session_id=session_id,
        logical_session_id=session_id,
        origin=source_name,
        title=f"Session {session_id}",
        provenance=_provenance(),
        semantic_tier="merged",
        evidence=SessionEvidencePayload(
            message_count=message_count,
            word_count=word_count,
            tool_use_count=tool_use_count,
            tool_active_duration_ms=tool_active_duration_ms,
            canonical_session_date=canonical_session_date,
            tags=tags,
        ),
        inference_provenance=_inference_provenance(),
        inference=SessionInferencePayload(
            workflow_shape=workflow_shape,
            terminal_state=terminal_state,
            engaged_duration_ms=120_000,
        ),
    )


def _latency_profile(
    session_id: str, *, median_ms: int, p90_ms: int, max_ms: int, stuck: int = 0
) -> SessionLatencyProfileInsight:
    return SessionLatencyProfileInsight(
        session_id=session_id,
        origin="claude-code",
        title=f"Session {session_id}",
        provenance=_provenance(),
        latency=SessionLatencyProfilePayload(
            median_tool_call_ms=median_ms,
            p90_tool_call_ms=p90_ms,
            max_tool_call_ms=max_ms,
            stuck_tool_count=stuck,
        ),
    )


_FIXTURE_PROFILES = [
    _profile(
        "c1",
        workflow_shape="chat",
        terminal_state="resolved",
        message_count=10,
        word_count=100,
    ),
    _profile(
        "c2",
        workflow_shape="agentic_loop",
        terminal_state="question_left",
        message_count=20,
        word_count=200,
        canonical_session_date="2026-05-08",
        tags=("python",),
    ),
    _profile(
        "c3",
        source_name="chatgpt",
        workflow_shape="agentic_loop",
        terminal_state="agent_hanging",
        message_count=30,
        word_count=300,
        canonical_session_date="2026-01-15",
        tags=("python", "api"),
    ),
]

_FIXTURE_LATENCY_PROFILES = [
    _latency_profile("c1", median_ms=1000, p90_ms=4000, max_ms=9000, stuck=1),
    _latency_profile("c2", median_ms=2000, p90_ms=5000, max_ms=8000),
    _latency_profile("c3", median_ms=1500, p90_ms=6000, max_ms=12000, stuck=2),
]


def _fixture_polylogue(tmp_path: Path) -> Polylogue:
    poly = _archive(tmp_path)
    poly.list_session_profile_insights = AsyncMock(return_value=list(_FIXTURE_PROFILES))  # type: ignore[method-assign]
    poly.list_session_latency_profile_insights = AsyncMock(  # type: ignore[method-assign]
        return_value=list(_FIXTURE_LATENCY_PROFILES)
    )
    by_id = {p.session_id: p for p in _FIXTURE_PROFILES}
    poly.get_session_profile_insight = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda session_id, tier="merged": by_id.get(session_id)
    )
    return poly


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "kwargs"),
    [
        ("aggregate_sessions", {"group_by": "workflow_shape"}),
        ("aggregate_sessions", {"group_by": "origin"}),
        ("aggregate_sessions", {"group_by": "terminal_state"}),
        ("workflow_shape_distribution", {"group_by": "week"}),
        ("workflow_shape_distribution", {"group_by": "origin"}),
        ("find_abandoned_sessions", {"min_severity": "question_left", "limit": 5}),
        ("tool_call_latency_distribution", {"limit": 5}),
        ("correlate_sessions", {"metric_x": "message_count", "metric_y": "word_count"}),
    ],
)
async def test_facade_and_mcp_tool_agree(
    tmp_path: Path,
    mcp_server: MCPServerUnderTest,
    tool_name: str,
    kwargs: dict[str, object],
) -> None:
    """The facade method and the MCP tool handler return identical payloads.

    Both delegate to the same insights-layer pure function over the same
    fetched profiles -- the concrete proof there is no separate math
    living in the MCP surface anymore.
    """
    poly = _fixture_polylogue(tmp_path)

    facade_method = getattr(poly, tool_name)
    facade_result = await facade_method(**kwargs)

    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        raw = await invoke_surface_async(mcp_server._tool_manager._tools[tool_name].fn, **kwargs)

    assert json.loads(raw) == facade_result


@pytest.mark.asyncio
async def test_compare_sessions_facade_and_mcp_tool_agree(
    tmp_path: Path,
    mcp_server: MCPServerUnderTest,
) -> None:
    poly = _fixture_polylogue(tmp_path)

    facade_result = await poly.compare_sessions(["c1", "c2", "c3"])

    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["compare_sessions"].fn,
            session_ids="c1,c2,c3",
        )

    assert json.loads(raw) == facade_result


@pytest.mark.asyncio
async def test_find_similar_sessions_metadata_facade_and_mcp_tool_agree(
    tmp_path: Path,
    mcp_server: MCPServerUnderTest,
) -> None:
    poly = _fixture_polylogue(tmp_path)

    facade_result = await poly.find_similar_sessions_by_metadata("c1", limit=5)
    assert facade_result is not None

    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["find_similar_sessions"].fn,
            session_id="c1",
            similarity_dimension="metadata",
            limit=5,
        )

    assert json.loads(raw) == facade_result

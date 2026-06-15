"""Tests for session analysis primitive MCP tools (#1691).

Covers compare_sessions, find_similar_sessions, and correlate_sessions.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

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
    message_count: int = 10,
    word_count: int = 500,
    tool_use_count: int = 5,
    engaged_duration_ms: int = 120_000,
    tool_active_duration_ms: int = 45_000,
    canonical_session_date: str = "2026-05-01",
    tags: tuple[str, ...] = (),
) -> SessionProfileInsight:
    return SessionProfileInsight(
        session_id=session_id,
        logical_session_id=session_id,
        source_name=source_name,
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
            engaged_duration_ms=engaged_duration_ms,
        ),
    )


# ── compare_sessions ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compare_sessions_side_by_side(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.get_session_profile_insight = AsyncMock(
            side_effect=lambda cid: {
                "c1": _make_profile("c1", source_name="claude-code", workflow_shape="chat", message_count=20),
                "c2": _make_profile("c2", source_name="claude-code", workflow_shape="agentic_loop", message_count=5),
            }.get(cid)
        )
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["compare_sessions"].fn,
            session_ids="c1, c2",
        )

    payload = json.loads(raw)
    assert payload["total_requested"] == 2
    assert payload["total_found"] == 2
    assert payload["not_found"] == []
    assert len(payload["sessions"]) == 2
    assert payload["sessions"][0]["origin"] == "claude-code-session"
    assert set(payload["differences"]["workflow_shape"]) == {"agentic_loop", "chat"}
    assert set(payload["differences"]["message_count"]) == {5, 20}


@pytest.mark.asyncio
async def test_compare_sessions_handles_missing(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.get_session_profile_insight = AsyncMock(
            side_effect=lambda cid: _make_profile(cid) if cid == "c1" else None
        )
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["compare_sessions"].fn,
            session_ids="c1, c2, c3",
        )

    payload = json.loads(raw)
    assert payload["total_requested"] == 3
    assert payload["total_found"] == 1
    assert payload["not_found"] == ["c2", "c3"]
    assert len(payload["sessions"]) == 1


@pytest.mark.asyncio
async def test_compare_sessions_rejects_single_id(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.get_session_profile_insight = AsyncMock(return_value=_make_profile("c1"))
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["compare_sessions"].fn,
            session_ids="c1",
        )

    payload = json.loads(raw)
    assert "message" in payload or "not_found" in payload
    assert payload.get("total_found", 1) <= 1


@pytest.mark.asyncio
async def test_compare_sessions_rejects_empty_input(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["compare_sessions"].fn,
            session_ids=None,
        )

    payload = json.loads(raw)
    assert "message" in payload or "code" in payload


@pytest.mark.asyncio
async def test_compare_sessions_rejects_too_many_ids(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_get_polylogue.return_value = mock_poly

        ids = ",".join(f"c{i}" for i in range(1, 12))
        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["compare_sessions"].fn,
            session_ids=ids,
        )

    payload = json.loads(raw)
    assert "message" in payload or "code" in payload
    assert "Too many" in str(payload)


@pytest.mark.asyncio
async def test_compare_sessions_no_differences_when_identical(
    mcp_server: MCPServerUnderTest,
) -> None:
    p1 = _make_profile("c1", source_name="claude-code", workflow_shape="chat", message_count=10)
    p2 = _make_profile("c2", source_name="claude-code", workflow_shape="chat", message_count=10)
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.get_session_profile_insight = AsyncMock(side_effect=[p1, p2])
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["compare_sessions"].fn,
            session_ids="c1, c2",
        )

    payload = json.loads(raw)
    assert payload["total_found"] == 2
    # origin, workflow_shape, terminal_state all identical => no differences
    assert "workflow_shape" not in payload["differences"]
    assert "origin" not in payload["differences"]


@pytest.mark.asyncio
async def test_compare_sessions_includes_all_found_sessions(
    mcp_server: MCPServerUnderTest,
) -> None:
    profiles = {
        "c1": _make_profile("c1", source_name="claude-code"),
        "c2": _make_profile("c2", source_name="chatgpt"),
        "c3": _make_profile("c3", source_name="codex"),
    }
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.get_session_profile_insight = AsyncMock(side_effect=lambda cid: profiles.get(cid))
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["compare_sessions"].fn,
            session_ids="c1, c2, c3",
        )

    payload = json.loads(raw)
    assert payload["total_requested"] == 3
    assert payload["total_found"] == 3
    ids = [s["id"] for s in payload["sessions"]]
    assert ids == ["c1", "c2", "c3"]
    assert [s["origin"] for s in payload["sessions"]] == [
        "claude-code-session",
        "chatgpt-export",
        "codex-session",
    ]


# ── find_similar_sessions ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_find_similar_sessions_via_metadata(
    mcp_server: MCPServerUnderTest,
) -> None:
    ref = _make_profile(
        "ref",
        source_name="claude-code",
        workflow_shape="agentic_loop",
        canonical_session_date="2026-05-01",
        tags=("python", "api"),
    )
    similar = _make_profile(
        "sim",
        source_name="claude-code",
        workflow_shape="agentic_loop",
        canonical_session_date="2026-05-02",
        tags=("python",),
    )
    different = _make_profile(
        "diff",
        source_name="chatgpt",
        workflow_shape="chat",
        canonical_session_date="2026-01-15",
    )

    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.config = MagicMock(embedding_enabled=False)
        mock_poly.get_session_profile_insight = AsyncMock(return_value=ref)
        mock_poly.list_session_profile_insights = AsyncMock(return_value=[ref, similar, different])
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["find_similar_sessions"].fn,
            session_id="ref",
            similarity_dimension="metadata",
            limit=5,
        )

    payload = json.loads(raw)
    assert payload["method"] == "metadata"
    assert payload["source_session_id"] == "ref"
    assert len(payload["similar"]) >= 1
    # The most similar should be 'sim' (same source + same shape + close date)
    top_id = payload["similar"][0]["session_id"]
    assert top_id == "sim"
    assert payload["similar"][0]["origin"] == "claude-code-session"


@pytest.mark.asyncio
async def test_find_similar_sessions_handles_not_found(
    mcp_server: MCPServerUnderTest,
) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.config = MagicMock(embedding_enabled=False)
        mock_poly.get_session_profile_insight = AsyncMock(return_value=None)
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
async def test_find_similar_sessions_excludes_self(
    mcp_server: MCPServerUnderTest,
) -> None:
    ref = _make_profile("ref", source_name="claude-code", workflow_shape="chat")
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.config = MagicMock(embedding_enabled=False)
        mock_poly.get_session_profile_insight = AsyncMock(return_value=ref)
        mock_poly.list_session_profile_insights = AsyncMock(return_value=[ref])
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["find_similar_sessions"].fn,
            session_id="ref",
            similarity_dimension="metadata",
        )

    payload = json.loads(raw)
    # Self should be excluded from results
    assert len(payload["similar"]) == 0


# ── correlate_sessions ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_correlate_sessions_perfect_positive(
    mcp_server: MCPServerUnderTest,
) -> None:
    profiles = [
        _make_profile("c1", message_count=10, word_count=100),
        _make_profile("c2", message_count=20, word_count=200),
        _make_profile("c3", message_count=30, word_count=300),
    ]
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(return_value=profiles)
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["correlate_sessions"].fn,
            metric_x="message_count",
            metric_y="word_count",
        )

    payload = json.loads(raw)
    assert payload["sample_count"] == 3
    assert payload["pearson_r"] == pytest.approx(1.0, abs=0.01)
    assert "strong positive" in payload["interpretation"]


@pytest.mark.asyncio
async def test_correlate_sessions_negative_correlation(
    mcp_server: MCPServerUnderTest,
) -> None:
    profiles = [
        _make_profile("c1", message_count=10, tool_active_duration_ms=100_000),
        _make_profile("c2", message_count=50, tool_active_duration_ms=75_000),
        _make_profile("c3", message_count=100, tool_active_duration_ms=50_000),
    ]
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(return_value=profiles)
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["correlate_sessions"].fn,
            metric_x="message_count",
            metric_y="tool_active_duration_ms",
        )

    payload = json.loads(raw)
    assert payload["sample_count"] == 3
    assert payload["pearson_r"] == pytest.approx(-1.0, abs=0.01)
    assert "strong negative" in payload["interpretation"]


@pytest.mark.asyncio
async def test_correlate_insufficient_data(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(return_value=[_make_profile("c1"), _make_profile("c2")])
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["correlate_sessions"].fn,
            metric_x="message_count",
            metric_y="word_count",
        )

    payload = json.loads(raw)
    assert payload["pearson_r"] is None
    assert "insufficient data" in payload["interpretation"]


@pytest.mark.asyncio
async def test_correlate_constant_metric_zero_variance(
    mcp_server: MCPServerUnderTest,
) -> None:
    profiles = [
        _make_profile("c1", message_count=10, word_count=100),
        _make_profile("c2", message_count=10, word_count=200),
        _make_profile("c3", message_count=10, word_count=300),
    ]
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(return_value=profiles)
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["correlate_sessions"].fn,
            metric_x="message_count",  # constant 10
            metric_y="word_count",  # varies
        )

    payload = json.loads(raw)
    assert payload["pearson_r"] is None
    assert "constant metric" in payload["interpretation"]


@pytest.mark.asyncio
async def test_correlate_invalid_metric_rejected(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(return_value=[])
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["correlate_sessions"].fn,
            metric_x="favorite_color",
            metric_y="word_count",
        )

    payload = json.loads(raw)
    assert "Unknown metric" in str(payload)


@pytest.mark.asyncio
async def test_correlate_skips_profiles_with_missing_metrics(
    mcp_server: MCPServerUnderTest,
) -> None:
    # Profile with no inference (engaged_duration_ms won't be available)
    incomplete = SessionProfileInsight(
        session_id="c-incomplete",
        logical_session_id="c-incomplete",
        source_name="codex",
        provenance=_provenance(),
        semantic_tier="merged",
        evidence=SessionEvidencePayload(message_count=7, word_count=350),
        inference_provenance=None,
        inference=None,
    )
    profiles = [
        _make_profile("c1", message_count=10, word_count=100),
        incomplete,
        _make_profile("c3", message_count=30, word_count=300),
    ]
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(return_value=profiles)
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["correlate_sessions"].fn,
            metric_x="message_count",
            metric_y="word_count",
        )

    payload = json.loads(raw)
    # Should use all 3 (incomplete still has message_count/word_count from evidence)
    assert payload["sample_count"] == 3


@pytest.mark.asyncio
async def test_correlate_weak_correlation(mcp_server: MCPServerUnderTest) -> None:
    # Near-zero correlation data
    profiles = [
        _make_profile("c1", message_count=10, tool_active_duration_ms=100),
        _make_profile("c2", message_count=50, tool_active_duration_ms=120),
        _make_profile("c3", message_count=30, tool_active_duration_ms=80),
        _make_profile("c4", message_count=15, tool_active_duration_ms=110),
        _make_profile("c5", message_count=40, tool_active_duration_ms=90),
    ]
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(return_value=profiles)
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["correlate_sessions"].fn,
            metric_x="message_count",
            metric_y="tool_active_duration_ms",
        )

    payload = json.loads(raw)
    assert payload["sample_count"] == 5
    assert payload["pearson_r"] is not None
    # This set should produce a weak/negligible correlation
    assert "weak" in payload["interpretation"] or "negligible" in payload["interpretation"]


@pytest.mark.asyncio
async def test_correlate_with_origin_filter(mcp_server: MCPServerUnderTest) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.list_session_profile_insights = AsyncMock(
            return_value=[
                _make_profile("c1", message_count=10, word_count=100),
                _make_profile("c2", message_count=20, word_count=200),
                _make_profile("c3", message_count=30, word_count=300),
            ]
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
    # Verify the origin filter maps to the storage-facing provider query.
    assert mock_poly.list_session_profile_insights.called
    call_kwargs = mock_poly.list_session_profile_insights.call_args
    query = call_kwargs[0][0]
    assert query.provider == "claude-code"

"""Facade-level contracts for the sunk session-analysis primitives (#1691 / polylogue-9e5.24).

These pin I/O-boundary behaviors that live in the fetch-then-reduce facade
methods themselves (polylogue/api/insights.py), not in the pure reducers
(tests/unit/insights/test_archive_rollups.py,
tests/unit/insights/test_session_analytics.py) or in the MCP thin
wrappers (tests/unit/mcp/). In particular: the facade methods must fetch
the *full* matched scope (``limit=None``) rather than any surface's page
size, and correlate_sessions must validate metric names before paying for
a fetch.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from polylogue import Polylogue
from polylogue.insights.archive import ArchiveInferenceProvenance, ArchiveInsightProvenance, SessionProfileInsight
from polylogue.insights.archive_models import SessionEvidencePayload, SessionInferencePayload
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _archive(tmp_path: Path) -> Polylogue:
    with ArchiveStore(tmp_path):
        pass
    return Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")


def _provenance() -> ArchiveInsightProvenance:
    return ArchiveInsightProvenance(materializer_version=1, materialized_at="2026-03-24T10:00:00+00:00")


def _inference_provenance() -> ArchiveInferenceProvenance:
    return ArchiveInferenceProvenance(
        materializer_version=1,
        materialized_at="2026-03-24T10:00:00+00:00",
        inference_version=1,
        inference_family="heuristic_session_semantics",
    )


def _profile(session_id: str, *, message_count: int = 10, word_count: int = 100) -> SessionProfileInsight:
    return SessionProfileInsight(
        session_id=session_id,
        logical_session_id=session_id,
        source_name="claude-code",
        title=session_id,
        provenance=_provenance(),
        semantic_tier="merged",
        evidence=SessionEvidencePayload(message_count=message_count, word_count=word_count),
        inference_provenance=_inference_provenance(),
        inference=SessionInferencePayload(workflow_shape="chat", terminal_state="resolved"),
    )


@pytest.mark.asyncio
async def test_aggregate_sessions_fetches_full_scope_not_a_page_limit(tmp_path: Path) -> None:
    poly = _archive(tmp_path)
    profiles = [_profile(f"c{i}") for i in range(1005)]
    fetch_mock = AsyncMock(return_value=profiles)
    poly.list_session_profile_insights = fetch_mock  # type: ignore[method-assign]

    result = await poly.aggregate_sessions(group_by="workflow_shape")

    assert result["total_sessions"] == 1005
    assert fetch_mock.await_args is not None
    assert fetch_mock.await_args.args[0].limit is None


@pytest.mark.asyncio
async def test_aggregate_sessions_maps_provider_filter_through(tmp_path: Path) -> None:
    poly = _archive(tmp_path)
    fetch_mock = AsyncMock(return_value=[])
    poly.list_session_profile_insights = fetch_mock  # type: ignore[method-assign]

    await poly.aggregate_sessions(group_by="workflow_shape", origin="claude-code")

    assert fetch_mock.await_args is not None
    query = fetch_mock.await_args.args[0]
    assert query.origin == "claude-code"


@pytest.mark.asyncio
async def test_correlate_sessions_fetches_full_scope_not_a_page_limit(tmp_path: Path) -> None:
    poly = _archive(tmp_path)
    profiles = [_profile(f"c{i}", message_count=i, word_count=i * 2) for i in range(1, 1006)]
    fetch_mock = AsyncMock(return_value=profiles)
    poly.list_session_profile_insights = fetch_mock  # type: ignore[method-assign]

    result = await poly.correlate_sessions(metric_x="message_count", metric_y="word_count")

    assert result["sample_count"] == 1005
    assert fetch_mock.await_args is not None
    assert fetch_mock.await_args.args[0].limit is None


@pytest.mark.asyncio
async def test_correlate_sessions_validates_metrics_before_fetching(tmp_path: Path) -> None:
    poly = _archive(tmp_path)
    poly.list_session_profile_insights = AsyncMock(return_value=[])  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="Unknown metric_x"):
        await poly.correlate_sessions(metric_x="favorite_color", metric_y="word_count")

    poly.list_session_profile_insights.assert_not_awaited()


@pytest.mark.asyncio
async def test_compare_sessions_rejects_out_of_range_counts(tmp_path: Path) -> None:
    poly = _archive(tmp_path)

    with pytest.raises(ValueError, match="Need at least 2"):
        await poly.compare_sessions(["only-one"])

    with pytest.raises(ValueError, match="Too many"):
        await poly.compare_sessions([f"c{i}" for i in range(11)])


@pytest.mark.asyncio
async def test_find_similar_sessions_by_metadata_returns_none_for_unknown_session(tmp_path: Path) -> None:
    poly = _archive(tmp_path)
    poly.get_session_profile_insight = AsyncMock(return_value=None)  # type: ignore[method-assign]

    result = await poly.find_similar_sessions_by_metadata("nonexistent")

    assert result is None

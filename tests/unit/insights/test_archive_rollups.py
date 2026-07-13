"""Unit tests for the session-analysis reducers in archive_rollups.py (#1691).

These pin the math that used to live inline in
mcp/server_insight_tools.py (polylogue-9e5.24): GROUP BY aggregation,
ISO-week bucketing, the abandonment severity rank, and the nearest-rank
percentile reused from portfolio.py. Pure-function tests -- no MCP/mock
scaffolding needed; the facade/MCP parity proof lives in
tests/unit/mcp/test_analysis_primitives_facade_parity.py.
"""

from __future__ import annotations

import pytest

from polylogue.archive.semantic.pricing import CostEstimatePayload
from polylogue.insights.archive import (
    ArchiveInferenceProvenance,
    ArchiveInsightProvenance,
    SessionCostInsight,
    SessionLatencyProfileInsight,
    SessionProfileInsight,
)
from polylogue.insights.archive_models import (
    SessionEvidencePayload,
    SessionInferencePayload,
    SessionLatencyProfilePayload,
)
from polylogue.insights.archive_rollups import (
    ABANDONMENT_SEVERITY_RANK,
    abandoned_session_items,
    aggregate_cost_rollup_insights,
    aggregate_session_profiles_by_dimension,
    iso_week_bucket_key,
    tool_call_latency_distribution_payload,
    workflow_shape_distribution_buckets,
)


def _provenance() -> ArchiveInsightProvenance:
    return ArchiveInsightProvenance(materializer_version=1, materialized_at="2026-03-24T10:00:00+00:00")


def _inference_provenance() -> ArchiveInferenceProvenance:
    return ArchiveInferenceProvenance(
        materializer_version=1,
        materialized_at="2026-03-24T10:00:00+00:00",
        inference_version=1,
        inference_family="heuristic_session_semantics",
    )


def _profile(
    session_id: str,
    *,
    source_name: str = "claude-code",
    workflow_shape: str | None = "agentic_loop",
    terminal_state: str | None = "resolved",
    canonical_session_date: str | None = "2026-05-01",
    cwd_paths: tuple[str, ...] = (),
    no_inference: bool = False,
) -> SessionProfileInsight:
    return SessionProfileInsight(
        session_id=session_id,
        logical_session_id=session_id,
        source_name=source_name,
        title=f"Session {session_id}",
        provenance=_provenance(),
        semantic_tier="merged",
        evidence=SessionEvidencePayload(
            message_count=1,
            canonical_session_date=canonical_session_date,
            cwd_paths=cwd_paths,
        ),
        inference_provenance=None if no_inference else _inference_provenance(),
        inference=(
            None
            if no_inference
            else SessionInferencePayload(
                workflow_shape=workflow_shape or "unknown",
                terminal_state=terminal_state or "unknown",
            )
        ),
    )


def _latency(
    session_id: str,
    *,
    median_ms: int,
    p90_ms: int,
    max_ms: int,
    stuck: int = 0,
    tool_call_count_by_category: dict[str, int] | None = None,
) -> SessionLatencyProfileInsight:
    return SessionLatencyProfileInsight(
        session_id=session_id,
        source_name="claude-code",
        title=session_id,
        provenance=_provenance(),
        latency=SessionLatencyProfilePayload(
            median_tool_call_ms=median_ms,
            p90_tool_call_ms=p90_ms,
            max_tool_call_ms=max_ms,
            stuck_tool_count=stuck,
            tool_call_count_by_category=tool_call_count_by_category or {},
        ),
    )


# ── iso_week_bucket_key ──────────────────────────────────────────────


def test_iso_week_bucket_key_parses_iso_date() -> None:
    assert iso_week_bucket_key("2026-05-04") == "2026-W19"


def test_iso_week_bucket_key_undated_when_none() -> None:
    assert iso_week_bucket_key(None) == "undated"


def test_iso_week_bucket_key_undated_when_empty() -> None:
    assert iso_week_bucket_key("") == "undated"


def test_iso_week_bucket_key_falls_back_on_unparseable_date() -> None:
    assert iso_week_bucket_key("2026-05") == "2026-05"


# ── aggregate_session_profiles_by_dimension ──────────────────────────


def test_aggregate_by_workflow_shape_counts_buckets() -> None:
    profiles = [
        _profile("c1", workflow_shape="chat"),
        _profile("c2", workflow_shape="chat"),
        _profile("c3", workflow_shape="agentic_loop"),
    ]
    assert aggregate_session_profiles_by_dimension(profiles, "workflow_shape") == {"chat": 2, "agentic_loop": 1}


def test_aggregate_by_terminal_state_counts_buckets() -> None:
    profiles = [
        _profile("c1", terminal_state="resolved"),
        _profile("c2", terminal_state="question_left"),
        _profile("c3", terminal_state="question_left"),
    ]
    assert aggregate_session_profiles_by_dimension(profiles, "terminal_state") == {
        "resolved": 1,
        "question_left": 2,
    }


def test_aggregate_by_origin_maps_source_name_to_public_origin() -> None:
    profiles = [_profile("c1", source_name="claude-code"), _profile("c2", source_name="chatgpt")]
    buckets = aggregate_session_profiles_by_dimension(profiles, "origin")
    assert buckets == {"claude-code-session": 1, "chatgpt-export": 1}


def test_aggregate_unknown_when_no_inference() -> None:
    profiles = [_profile("c1", no_inference=True)]
    assert aggregate_session_profiles_by_dimension(profiles, "workflow_shape") == {"unknown": 1}


def test_aggregate_rejects_unknown_group_by() -> None:
    with pytest.raises(ValueError, match="Unknown group_by"):
        aggregate_session_profiles_by_dimension([_profile("c1")], "not_a_dimension")


def test_aggregate_empty_profiles_returns_empty_buckets() -> None:
    assert aggregate_session_profiles_by_dimension([], "workflow_shape") == {}


def test_cost_rollup_confidence_is_none_without_priced_sessions() -> None:
    unavailable = SessionCostInsight(
        session_id="c1",
        source_name="claude-code",
        estimate=CostEstimatePayload(source_name="claude-code", status="unavailable"),
        provenance=_provenance(),
    )

    [rollup] = aggregate_cost_rollup_insights([unavailable], materialized_at="2026-05-01T00:00:00+00:00")

    assert rollup.priced_session_count == 0
    assert rollup.confidence is None


# ── workflow_shape_distribution_buckets ──────────────────────────────


def test_workflow_shape_distribution_by_week() -> None:
    profiles = [
        _profile("c1", workflow_shape="chat", canonical_session_date="2026-05-04"),
        _profile("c2", workflow_shape="chat", canonical_session_date="2026-05-05"),
    ]
    buckets = workflow_shape_distribution_buckets(profiles, "week")
    assert buckets == {"2026-W19": {"chat": 2}}


def test_workflow_shape_distribution_by_origin() -> None:
    profiles = [_profile("c1", source_name="claude-code", workflow_shape="agentic_loop")]
    buckets = workflow_shape_distribution_buckets(profiles, "origin")
    assert buckets == {"claude-code-session": {"agentic_loop": 1}}


def test_workflow_shape_distribution_by_project_unattributed_without_cwd() -> None:
    profiles = [_profile("c1", workflow_shape="chat", cwd_paths=())]
    buckets = workflow_shape_distribution_buckets(profiles, "project")
    assert buckets == {"unattributed": {"chat": 1}}


def test_workflow_shape_distribution_by_project_uses_cwd_paths() -> None:
    profiles = [_profile("c1", workflow_shape="chat", cwd_paths=("/realm/project/polylogue",))]
    buckets = workflow_shape_distribution_buckets(profiles, "project")
    assert buckets == {"/realm/project/polylogue": {"chat": 1}}


def test_workflow_shape_distribution_rejects_invalid_group_by() -> None:
    with pytest.raises(ValueError, match="group_by must be one of"):
        workflow_shape_distribution_buckets([_profile("c1")], "invalid")


# ── abandoned_session_items ──────────────────────────────────────────


def test_abandonment_severity_rank_is_the_canonical_vocabulary() -> None:
    assert ABANDONMENT_SEVERITY_RANK == {
        "question_left": 1,
        "error_left": 2,
        "tool_left": 3,
        "agent_hanging": 4,
    }


def test_abandoned_session_items_filters_by_min_severity() -> None:
    profiles = [
        _profile("c1", terminal_state="resolved"),
        _profile("c2", terminal_state="question_left"),
        _profile("c3", terminal_state="agent_hanging"),
    ]
    items = abandoned_session_items(profiles, min_severity="tool_left")
    assert [item["session_id"] for item in items] == ["c3"]


def test_abandoned_session_items_sorts_by_date_descending() -> None:
    profiles = [
        _profile("c1", terminal_state="question_left", canonical_session_date="2026-05-01"),
        _profile("c2", terminal_state="question_left", canonical_session_date="2026-05-10"),
    ]
    items = abandoned_session_items(profiles, min_severity="question_left")
    assert [item["session_id"] for item in items] == ["c2", "c1"]


def test_abandoned_session_items_filters_by_repo_path() -> None:
    profiles = [
        _profile("c1", terminal_state="question_left", cwd_paths=("/realm/project/polylogue",)),
        _profile("c2", terminal_state="question_left", cwd_paths=("/realm/project/sinnix",)),
    ]
    items = abandoned_session_items(profiles, min_severity="question_left", repo_path="polylogue")
    assert [item["session_id"] for item in items] == ["c1"]


def test_abandoned_session_items_rejects_invalid_min_severity() -> None:
    with pytest.raises(ValueError, match="min_severity must be one of"):
        abandoned_session_items([_profile("c1")], min_severity="not_a_severity")


# ── tool_call_latency_distribution_payload ───────────────────────────


def test_tool_call_latency_distribution_nearest_rank_percentiles() -> None:
    insights = [
        _latency("c1", median_ms=1000, p90_ms=4000, max_ms=9000, stuck=1),
        _latency("c2", median_ms=2000, p90_ms=5000, max_ms=8000),
        _latency("c3", median_ms=1500, p90_ms=6000, max_ms=12000, stuck=2),
    ]
    payload = tool_call_latency_distribution_payload(insights)
    assert payload["total_sessions"] == 3
    assert payload["median_tool_call_ms"] == 1500
    assert payload["p90_tool_call_ms"] == 6000
    assert payload["max_tool_call_ms"] == 12000
    assert payload["stuck_tool_count"] == 3


def test_tool_call_latency_distribution_filters_by_tool_category() -> None:
    matching = _latency("c1", median_ms=1000, p90_ms=4000, max_ms=9000, tool_call_count_by_category={"shell": 3})
    non_matching = _latency("c2", median_ms=5000, p90_ms=9000, max_ms=20000)
    payload = tool_call_latency_distribution_payload([matching, non_matching], tool_category="shell")
    assert payload["total_sessions"] == 1
    assert payload["median_tool_call_ms"] == 1000


def test_tool_call_latency_distribution_empty_is_all_zero() -> None:
    payload = tool_call_latency_distribution_payload([])
    assert payload["total_sessions"] == 0
    assert payload["median_tool_call_ms"] == 0
    assert payload["p90_tool_call_ms"] == 0
    assert payload["max_tool_call_ms"] == 0
    assert payload["stuck_tool_count"] == 0

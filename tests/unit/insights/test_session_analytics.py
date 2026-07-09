"""Unit tests for insights/session_analytics.py (#1691 / polylogue-9e5.24).

Pins the Pearson correlation, metadata-similarity heuristic, and per-key
set-diff math that used to live inline in mcp/server_insight_tools.py.
Pure-function tests -- the facade/MCP parity proof lives in
tests/unit/mcp/test_analysis_primitives_facade_parity.py.
"""

from __future__ import annotations

import pytest

from polylogue.insights.archive import ArchiveInferenceProvenance, ArchiveInsightProvenance, SessionProfileInsight
from polylogue.insights.archive_models import SessionEvidencePayload, SessionInferencePayload
from polylogue.insights.session_analytics import (
    CORRELATABLE_SESSION_METRICS,
    build_session_comparison_row,
    compute_metadata_similarity_candidates,
    diff_session_comparison_rows,
    ensure_known_session_metric,
    pearson_session_correlation,
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
    workflow_shape: str = "agentic_loop",
    terminal_state: str = "resolved",
    message_count: int = 10,
    word_count: int = 500,
    tool_use_count: int = 5,
    engaged_duration_ms: int = 120_000,
    tool_active_duration_ms: int = 45_000,
    canonical_session_date: str | None = "2026-05-01",
    tags: tuple[str, ...] = (),
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
            message_count=message_count,
            word_count=word_count,
            tool_use_count=tool_use_count,
            tool_active_duration_ms=tool_active_duration_ms,
            canonical_session_date=canonical_session_date,
            tags=tags,
        ),
        inference_provenance=None if no_inference else _inference_provenance(),
        inference=(
            None
            if no_inference
            else SessionInferencePayload(
                workflow_shape=workflow_shape,
                terminal_state=terminal_state,
                engaged_duration_ms=engaged_duration_ms,
            )
        ),
    )


# ── ensure_known_session_metric / pearson_session_correlation ───────


def test_ensure_known_session_metric_accepts_every_correlatable_metric() -> None:
    for metric in CORRELATABLE_SESSION_METRICS:
        ensure_known_session_metric(metric, "metric_x")  # no raise


def test_ensure_known_session_metric_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown metric_x: 'favorite_color'"):
        ensure_known_session_metric("favorite_color", "metric_x")


def test_pearson_correlation_perfect_positive() -> None:
    profiles = [
        _profile("c1", message_count=10, word_count=100),
        _profile("c2", message_count=20, word_count=200),
        _profile("c3", message_count=30, word_count=300),
    ]
    result = pearson_session_correlation(profiles, metric_x="message_count", metric_y="word_count")
    assert result["sample_count"] == 3
    assert result["pearson_r"] == pytest.approx(1.0, abs=0.01)
    assert "strong positive" in str(result["interpretation"])


def test_pearson_correlation_perfect_negative() -> None:
    profiles = [
        _profile("c1", message_count=10, tool_active_duration_ms=100_000),
        _profile("c2", message_count=50, tool_active_duration_ms=75_000),
        _profile("c3", message_count=100, tool_active_duration_ms=50_000),
    ]
    result = pearson_session_correlation(profiles, metric_x="message_count", metric_y="tool_active_duration_ms")
    assert result["pearson_r"] == pytest.approx(-1.0, abs=0.01)
    assert "strong negative" in str(result["interpretation"])


def test_pearson_correlation_insufficient_data() -> None:
    profiles = [_profile("c1"), _profile("c2")]
    result = pearson_session_correlation(profiles, metric_x="message_count", metric_y="word_count")
    assert result["pearson_r"] is None
    assert "insufficient data" in str(result["interpretation"])
    assert result["sample_count"] == 2


def test_pearson_correlation_constant_metric_zero_variance() -> None:
    profiles = [
        _profile("c1", message_count=10, word_count=100),
        _profile("c2", message_count=10, word_count=200),
        _profile("c3", message_count=10, word_count=300),
    ]
    result = pearson_session_correlation(profiles, metric_x="message_count", metric_y="word_count")
    assert result["pearson_r"] is None
    assert "constant metric" in str(result["interpretation"])


def test_pearson_correlation_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="Unknown metric_y"):
        pearson_session_correlation([], metric_x="message_count", metric_y="favorite_color")


def test_pearson_correlation_skips_profiles_missing_a_metric() -> None:
    incomplete = _profile("c-incomplete", no_inference=True)
    profiles = [
        _profile("c1", message_count=10, word_count=100),
        incomplete,
        _profile("c3", message_count=30, word_count=300),
    ]
    # message_count/word_count both come from evidence, present even without inference.
    result = pearson_session_correlation(profiles, metric_x="message_count", metric_y="word_count")
    assert result["sample_count"] == 3


# ── build_session_comparison_row / diff_session_comparison_rows ─────


def test_build_session_comparison_row_shape() -> None:
    profile = _profile("c1", source_name="claude-code", workflow_shape="chat", message_count=20, tags=("a", "b"))
    row = build_session_comparison_row(profile)
    assert row["id"] == "c1"
    assert row["origin"] == "claude-code-session"
    assert row["workflow_shape"] == "chat"
    assert row["message_count"] == 20
    assert row["tags"] == ["a", "b"]


def test_build_session_comparison_row_defaults_when_no_inference() -> None:
    profile = _profile("c1", no_inference=True)
    row = build_session_comparison_row(profile)
    assert row["workflow_shape"] == "unknown"
    assert row["terminal_state"] == "unknown"
    assert row["engaged_duration_ms"] == 0
    assert row["auto_tags"] == []


def test_diff_session_comparison_rows_reports_only_varying_keys() -> None:
    rows = [
        build_session_comparison_row(_profile("c1", workflow_shape="chat", message_count=10)),
        build_session_comparison_row(_profile("c2", workflow_shape="agentic_loop", message_count=5)),
    ]
    diff = diff_session_comparison_rows(rows)
    assert set(diff["workflow_shape"]) == {"agentic_loop", "chat"}
    assert set(diff["message_count"]) == {5, 10}
    # Both rows share the same origin -- no diff entry expected.
    assert "origin" not in diff


def test_diff_session_comparison_rows_empty_when_identical() -> None:
    rows = [
        build_session_comparison_row(_profile("c1", workflow_shape="chat", message_count=10)),
        build_session_comparison_row(_profile("c2", workflow_shape="chat", message_count=10)),
    ]
    diff = diff_session_comparison_rows(rows)
    assert diff == {}


# ── compute_metadata_similarity_candidates ───────────────────────────


def test_metadata_similarity_ranks_same_shape_and_recent_date_highest() -> None:
    ref = _profile(
        "ref",
        source_name="claude-code",
        workflow_shape="agentic_loop",
        canonical_session_date="2026-05-01",
        tags=("python", "api"),
    )
    similar = _profile(
        "sim",
        source_name="claude-code",
        workflow_shape="agentic_loop",
        canonical_session_date="2026-05-02",
        tags=("python",),
    )
    different = _profile(
        "diff",
        source_name="chatgpt",
        workflow_shape="chat",
        canonical_session_date="2026-01-15",
    )
    scored = compute_metadata_similarity_candidates(ref, [ref, similar, different], exclude_session_id="ref")
    assert [item["session_id"] for item in scored][0] == "sim"
    assert all(item["session_id"] != "ref" for item in scored)


def test_metadata_similarity_excludes_self_even_if_present_in_candidates() -> None:
    ref = _profile("ref", workflow_shape="chat")
    scored = compute_metadata_similarity_candidates(ref, [ref], exclude_session_id="ref")
    assert scored == []


def test_metadata_similarity_zero_score_candidates_excluded() -> None:
    ref = _profile(
        "ref",
        source_name="claude-code",
        workflow_shape="agentic_loop",
        canonical_session_date="2026-05-01",
        tags=(),
    )
    unrelated = _profile(
        "unrelated",
        source_name="chatgpt",
        workflow_shape="chat",
        canonical_session_date="2020-01-01",
        tags=("unrelated-tag",),
    )
    scored = compute_metadata_similarity_candidates(ref, [unrelated], exclude_session_id="ref")
    assert scored == []


def test_metadata_similarity_scores_tag_overlap() -> None:
    # Isolate tag-overlap scoring: different workflow_shape (no +3), different
    # source_name (no +1), no date on either side (no date scoring) -- only
    # the 2 shared tags should contribute (+1 each).
    ref = _profile(
        "ref",
        source_name="claude-code",
        workflow_shape="agentic_loop",
        canonical_session_date=None,
        tags=("python", "api", "backend"),
    )
    candidate = _profile(
        "cand",
        source_name="chatgpt",
        workflow_shape="chat",
        canonical_session_date=None,
        tags=("python", "api"),
    )
    scored = compute_metadata_similarity_candidates(ref, [candidate], exclude_session_id="ref")
    assert len(scored) == 1
    assert scored[0]["similarity_score"] == 2
    assert "2 overlapping tags" in str(scored[0]["similarity_reasons"])

"""Unit tests for the session-analysis reducers in archive_rollups.py (#1691).

These pin ISO-week bucketing, abandonment severity, cost confidence, and
latency percentiles. Scalar profile analytics are exercised through the real
facade in tests/unit/api/test_session_analytics_facade.py.
"""

from __future__ import annotations

from polylogue.analysis.archive import (
    ArchiveInsightProvenance,
    SessionCostInsight,
    SessionLatencyProfileInsight,
)
from polylogue.analysis.archive_models import (
    SessionLatencyProfilePayload,
)
from polylogue.analysis.archive_rollups import (
    ABANDONMENT_SEVERITY_RANK,
    aggregate_cost_rollup_insights,
    iso_week_bucket_key,
    tool_call_latency_distribution_payload,
)
from polylogue.archive.semantic.pricing import CostEstimatePayload
from polylogue.core.enums import TERMINAL_STATE_VALUES


def _provenance() -> ArchiveInsightProvenance:
    return ArchiveInsightProvenance(materializer_version=1, materialized_at="2026-03-24T10:00:00+00:00")


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
        origin="claude-code",
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


def test_cost_rollup_confidence_is_none_without_priced_sessions() -> None:
    unavailable = SessionCostInsight(
        session_id="c1",
        origin="claude-code",
        estimate=CostEstimatePayload(origin="claude-code-session", status="unavailable"),
        provenance=_provenance(),
    )

    [rollup] = aggregate_cost_rollup_insights([unavailable], materialized_at="2026-05-01T00:00:00+00:00")

    assert rollup.priced_session_count == 0
    assert rollup.confidence is None


def test_abandonment_severity_rank_is_the_canonical_vocabulary() -> None:
    assert ABANDONMENT_SEVERITY_RANK == {
        "unknown": 0,
        "question_left": 1,
        "refused": 2,
        "error_left": 3,
        "tool_left": 4,
        "truncated": 5,
    }
    assert set(ABANDONMENT_SEVERITY_RANK) == TERMINAL_STATE_VALUES


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

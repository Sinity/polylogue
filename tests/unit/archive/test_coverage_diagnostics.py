from __future__ import annotations

from datetime import datetime, timezone

from polylogue.archive.coverage import analyze_coverage
from polylogue.archive.session.domain_models import SessionSummary
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.core.types import SessionId


def _summary(
    session_id: str,
    *,
    provider: Provider,
    updated_at: datetime | None,
    message_count: int | None,
) -> SessionSummary:
    return SessionSummary(
        id=SessionId(session_id),
        origin=origin_from_provider(provider),
        updated_at=updated_at,
        message_count=message_count,
    )


def test_analyze_coverage_handles_empty_archive() -> None:
    coverage = analyze_coverage(())

    assert coverage.origin_ranges == ()
    assert coverage.origin_counts == {}
    assert coverage.gaps == ()
    assert coverage.truncated_sessions == 0
    assert coverage.total_sessions == 0
    assert coverage.total_messages == 0
    assert coverage.date_range == (None, None)


def test_analyze_coverage_reports_origin_ranges_gaps_and_truncation() -> None:
    coverage = analyze_coverage(
        (
            _summary(
                "chatgpt:first",
                provider=Provider.CHATGPT,
                updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                message_count=1,
            ),
            _summary(
                "chatgpt:second",
                provider=Provider.CHATGPT,
                updated_at=datetime(2026, 1, 4, tzinfo=timezone.utc),
                message_count=5,
            ),
            _summary(
                "claude-ai:first",
                provider=Provider.CLAUDE_AI,
                updated_at=datetime(2026, 1, 4, 12, tzinfo=timezone.utc),
                message_count=None,
            ),
        )
    )

    assert coverage.origin_counts == {"chatgpt-export": 2, "claude-ai-export": 1}
    assert [
        (item.origin, item.first_date.isoformat(), item.last_date.isoformat(), item.count)
        for item in coverage.origin_ranges
    ] == [
        ("chatgpt-export", "2026-01-01", "2026-01-04", 2),
        ("claude-ai-export", "2026-01-04", "2026-01-04", 1),
    ]
    assert [(gap.start_date.isoformat(), gap.end_date.isoformat(), gap.days) for gap in coverage.gaps] == [
        ("2026-01-02", "2026-01-03", 2)
    ]
    assert coverage.truncated_sessions == 2
    assert coverage.total_sessions == 3
    assert coverage.total_messages == 6
    assert tuple(item.isoformat() for item in coverage.date_range if item is not None) == ("2026-01-01", "2026-01-04")

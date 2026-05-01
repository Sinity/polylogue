from __future__ import annotations

from datetime import datetime, timezone

from polylogue.archive.conversation.models import ConversationSummary
from polylogue.archive.coverage import analyze_coverage
from polylogue.types import ConversationId, Provider


def _summary(
    conversation_id: str,
    *,
    provider: Provider,
    updated_at: datetime | None,
    message_count: int | None,
) -> ConversationSummary:
    return ConversationSummary(
        id=ConversationId(conversation_id),
        provider=provider,
        updated_at=updated_at,
        message_count=message_count,
    )


def test_analyze_coverage_handles_empty_archive() -> None:
    coverage = analyze_coverage(())

    assert coverage.provider_ranges == ()
    assert coverage.provider_counts == {}
    assert coverage.gaps == ()
    assert coverage.truncated_sessions == 0
    assert coverage.total_conversations == 0
    assert coverage.total_messages == 0
    assert coverage.date_range == (None, None)


def test_analyze_coverage_reports_provider_ranges_gaps_and_truncation() -> None:
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

    assert coverage.provider_counts == {"chatgpt": 2, "claude-ai": 1}
    assert [
        (item.provider, item.first_date.isoformat(), item.last_date.isoformat(), item.count)
        for item in coverage.provider_ranges
    ] == [
        ("chatgpt", "2026-01-01", "2026-01-04", 2),
        ("claude-ai", "2026-01-04", "2026-01-04", 1),
    ]
    assert [(gap.start_date.isoformat(), gap.end_date.isoformat(), gap.days) for gap in coverage.gaps] == [
        ("2026-01-02", "2026-01-03", 2)
    ]
    assert coverage.truncated_sessions == 2
    assert coverage.total_conversations == 3
    assert coverage.total_messages == 6
    assert tuple(item.isoformat() for item in coverage.date_range if item is not None) == ("2026-01-01", "2026-01-04")

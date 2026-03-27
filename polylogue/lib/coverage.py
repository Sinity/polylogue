"""Archive coverage diagnostics.

Analyzes the completeness and consistency of the conversation archive,
identifying provider date ranges, gaps, and truncated sessions.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.lib.models import ConversationSummary


@dataclass(frozen=True)
class ProviderRange:
    """Date range covered by a single provider."""

    provider: str
    first_date: date
    last_date: date
    count: int


@dataclass(frozen=True)
class CoverageGap:
    """A gap of >1 day with no conversations from any provider."""

    start_date: date
    end_date: date
    days: int


@dataclass(frozen=True)
class ArchiveCoverage:
    """Completeness diagnostics for the conversation archive."""

    provider_ranges: tuple[ProviderRange, ...]
    provider_counts: dict[str, int]
    gaps: tuple[CoverageGap, ...]
    truncated_sessions: int
    total_conversations: int
    total_messages: int
    date_range: tuple[date | None, date | None]


def analyze_coverage(summaries: Sequence[ConversationSummary]) -> ArchiveCoverage:
    """Analyze archive coverage from conversation summaries."""
    if not summaries:
        return ArchiveCoverage(
            provider_ranges=(),
            provider_counts={},
            gaps=(),
            truncated_sessions=0,
            total_conversations=0,
            total_messages=0,
            date_range=(None, None),
        )

    provider_dates: dict[str, list[date]] = defaultdict(list)
    provider_counts: dict[str, int] = defaultdict(int)
    all_dates: set[date] = set()
    total_messages = 0
    truncated = 0

    for summary in summaries:
        provider = str(summary.provider)
        provider_counts[provider] += 1
        total_messages += summary.message_count or 0

        dt = summary.updated_at or summary.created_at
        if dt:
            d = dt.date() if isinstance(dt, datetime) else dt
            provider_dates[provider].append(d)
            all_dates.add(d)

        # Heuristic for truncated: very few messages suggest incomplete export
        if (summary.message_count or 0) <= 2:
            truncated += 1

    # Build provider ranges
    ranges: list[ProviderRange] = []
    for provider, dates in sorted(provider_dates.items()):
        if dates:
            ranges.append(ProviderRange(
                provider=provider,
                first_date=min(dates),
                last_date=max(dates),
                count=len(dates),
            ))

    # Find gaps (>1 day without any conversation)
    gaps: list[CoverageGap] = []
    if all_dates:
        sorted_dates = sorted(all_dates)
        for i in range(len(sorted_dates) - 1):
            gap_days = (sorted_dates[i + 1] - sorted_dates[i]).days
            if gap_days > 1:
                gaps.append(CoverageGap(
                    start_date=sorted_dates[i] + timedelta(days=1),
                    end_date=sorted_dates[i + 1] - timedelta(days=1),
                    days=gap_days - 1,
                ))

    date_range: tuple[date | None, date | None] = (None, None)
    if all_dates:
        date_range = (min(all_dates), max(all_dates))

    return ArchiveCoverage(
        provider_ranges=tuple(ranges),
        provider_counts=dict(provider_counts),
        gaps=tuple(gaps),
        truncated_sessions=truncated,
        total_conversations=len(summaries),
        total_messages=total_messages,
        date_range=date_range,
    )

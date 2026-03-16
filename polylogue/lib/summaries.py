"""Hierarchical session summary builder.

Aggregates SessionProfile objects into day and week summaries
for reporting and context injection.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Sequence

from polylogue.lib.session_profile import SessionProfile


@dataclass(frozen=True)
class DaySessionSummary:
    """Summary of all sessions on a single day."""

    date: date
    session_count: int
    total_cost_usd: float
    total_duration_ms: int
    total_wall_duration_ms: int
    total_messages: int
    total_words: int
    work_event_breakdown: dict[str, int]
    projects_active: tuple[str, ...]
    providers: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "date": self.date.isoformat(),
            "session_count": self.session_count,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_duration_ms": self.total_duration_ms,
            "total_wall_duration_ms": self.total_wall_duration_ms,
            "total_messages": self.total_messages,
            "total_words": self.total_words,
            "work_event_breakdown": self.work_event_breakdown,
            "projects_active": list(self.projects_active),
            "providers": self.providers,
        }


@dataclass(frozen=True)
class WeekSessionSummary:
    """Summary of all sessions in an ISO week."""

    iso_week: str
    day_summaries: tuple[DaySessionSummary, ...]
    session_count: int
    total_cost_usd: float
    total_duration_ms: int
    total_messages: int

    def to_dict(self) -> dict[str, object]:
        return {
            "iso_week": self.iso_week,
            "day_summaries": [d.to_dict() for d in self.day_summaries],
            "session_count": self.session_count,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_duration_ms": self.total_duration_ms,
            "total_messages": self.total_messages,
        }


def _profile_date(profile: SessionProfile) -> Optional[date]:
    """Best available date for a profile."""
    ts = profile.first_message_at or profile.created_at
    if ts is None:
        return None
    return ts.date() if isinstance(ts, datetime) else ts


def summarize_day(
    profiles: Sequence[SessionProfile],
    target_date: date,
) -> DaySessionSummary:
    """Build a day summary from session profiles."""
    work_events: Counter[str] = Counter()
    projects: set[str] = set()
    providers: Counter[str] = Counter()
    total_cost = 0.0
    total_duration = 0
    total_wall = 0
    total_messages = 0
    total_words = 0

    for profile in profiles:
        total_cost += profile.total_cost_usd
        total_duration += profile.total_duration_ms
        total_wall += profile.wall_duration_ms
        total_messages += profile.message_count
        total_words += profile.word_count
        providers[profile.provider] += 1
        for we in profile.work_events:
            kind = we.kind.value if hasattr(we.kind, "value") else str(we.kind)
            work_events[kind] += 1
        for repo in profile.repo_paths:
            projects.add(repo)

    return DaySessionSummary(
        date=target_date,
        session_count=len(profiles),
        total_cost_usd=total_cost,
        total_duration_ms=total_duration,
        total_wall_duration_ms=total_wall,
        total_messages=total_messages,
        total_words=total_words,
        work_event_breakdown=dict(work_events),
        projects_active=tuple(sorted(projects)),
        providers=dict(providers),
    )


def summarize_week(
    day_summaries: Sequence[DaySessionSummary],
) -> WeekSessionSummary:
    """Build a week summary from day summaries."""
    if not day_summaries:
        return WeekSessionSummary(
            iso_week="",
            day_summaries=(),
            session_count=0,
            total_cost_usd=0.0,
            total_duration_ms=0,
            total_messages=0,
        )

    first_date = day_summaries[0].date
    iso = first_date.isocalendar()
    iso_week = f"{iso[0]}-W{iso[1]:02d}"

    return WeekSessionSummary(
        iso_week=iso_week,
        day_summaries=tuple(day_summaries),
        session_count=sum(d.session_count for d in day_summaries),
        total_cost_usd=sum(d.total_cost_usd for d in day_summaries),
        total_duration_ms=sum(d.total_duration_ms for d in day_summaries),
        total_messages=sum(d.total_messages for d in day_summaries),
    )

"""Time-bucket summaries over semantic session profiles."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

from polylogue.lib.repo_identity import normalize_repo_names
from polylogue.lib.session.session_profile import SessionProfile


@dataclass(frozen=True)
class DaySessionSummary:
    date: date
    session_count: int
    total_cost_usd: float
    total_duration_ms: int
    total_wall_duration_ms: int
    total_messages: int
    total_words: int
    work_event_breakdown: dict[str, int]
    repos_active: tuple[str, ...]
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
            "repos_active": list(self.repos_active),
            "providers": self.providers,
        }


@dataclass(frozen=True)
class WeekSessionSummary:
    iso_week: str
    day_summaries: tuple[DaySessionSummary, ...]
    session_count: int
    total_cost_usd: float
    total_duration_ms: int
    total_messages: int

    def to_dict(self) -> dict[str, object]:
        return {
            "iso_week": self.iso_week,
            "day_summaries": [day.to_dict() for day in self.day_summaries],
            "session_count": self.session_count,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_duration_ms": self.total_duration_ms,
            "total_messages": self.total_messages,
        }


def _profile_date(profile: SessionProfile) -> date | None:
    if profile.canonical_session_date is not None:
        return profile.canonical_session_date
    timestamp = profile.first_message_at or profile.created_at
    if timestamp is None:
        return None
    return timestamp.date() if isinstance(timestamp, datetime) else timestamp


def summarize_day(
    profiles: Sequence[SessionProfile],
    target_date: date,
) -> DaySessionSummary:
    work_events: Counter[str] = Counter()
    repos: set[str] = set()
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
        work_events.update(
            event.kind.value if hasattr(event.kind, "value") else str(event.kind) for event in profile.work_events
        )
        repos.update(profile.repo_names or normalize_repo_names(repo_paths=profile.repo_paths))
    return DaySessionSummary(
        date=target_date,
        session_count=len(profiles),
        total_cost_usd=total_cost,
        total_duration_ms=total_duration,
        total_wall_duration_ms=total_wall,
        total_messages=total_messages,
        total_words=total_words,
        work_event_breakdown=dict(work_events),
        repos_active=tuple(sorted(repos)),
        providers=dict(providers),
    )


def summarize_week(day_summaries: Sequence[DaySessionSummary]) -> WeekSessionSummary:
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
    return WeekSessionSummary(
        iso_week=f"{iso[0]}-W{iso[1]:02d}",
        day_summaries=tuple(day_summaries),
        session_count=sum(day.session_count for day in day_summaries),
        total_cost_usd=sum(day.total_cost_usd for day in day_summaries),
        total_duration_ms=sum(day.total_duration_ms for day in day_summaries),
        total_messages=sum(day.total_messages for day in day_summaries),
    )


def summarize_days(
    profiles: Sequence[SessionProfile],
) -> tuple[DaySessionSummary, ...]:
    by_day: dict[date, list[SessionProfile]] = {}
    for profile in profiles:
        profile_day = _profile_date(profile)
        if profile_day is None:
            continue
        by_day.setdefault(profile_day, []).append(profile)
    return tuple(summarize_day(by_day[target_day], target_day) for target_day in sorted(by_day.keys(), reverse=True))


def summarize_weeks(
    day_summaries: Sequence[DaySessionSummary],
) -> tuple[WeekSessionSummary, ...]:
    by_week: dict[str, list[DaySessionSummary]] = {}
    for day_summary in day_summaries:
        iso = day_summary.date.isocalendar()
        week_key = f"{iso[0]}-W{iso[1]:02d}"
        by_week.setdefault(week_key, []).append(day_summary)
    return tuple(
        summarize_week(tuple(sorted(by_week[week_key], key=lambda item: item.date)))
        for week_key in sorted(by_week.keys(), reverse=True)
    )


__all__ = [
    "DaySessionSummary",
    "WeekSessionSummary",
    "summarize_day",
    "summarize_days",
    "summarize_week",
    "summarize_weeks",
]

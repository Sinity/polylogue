"""Day/week summary builders and aggregate reducers for archive insights."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from polylogue.archive.session.session_profile import SessionProfile
from polylogue.archive.session.session_summaries import DaySessionSummary, summarize_day, summarize_week
from polylogue.insights.archive import (
    DaySessionSummaryInsight,
    WeekSessionSummaryInsight,
    profile_bucket_day,
    profile_timestamp_values,
    records_provenance,
)
from polylogue.insights.archive_models import DaySessionSummaryPayload, WeekSessionSummaryPayload
from polylogue.insights.temporal_source import TemporalSource, classify_aggregate_hwm_source
from polylogue.storage.runtime import DaySessionSummaryRecord


@dataclass(slots=True)
class _DayAggregateBucket:
    session_count: int = 0
    logical_session_ids: set[str] = field(default_factory=set)
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    total_tool_active_duration_ms: int = 0
    total_wall_duration_ms: int = 0
    total_messages: int = 0
    total_words: int = 0
    work_event_breakdown: Counter[str] = field(default_factory=Counter)
    repos_active: set[str] = field(default_factory=set)
    origins: Counter[str] = field(default_factory=Counter)
    rows: list[DaySessionSummaryRecord] = field(default_factory=list)


def build_day_session_summary_records(
    profiles: Iterable[SessionProfile],
    *,
    materialized_at: str | None = None,
) -> list[DaySessionSummaryRecord]:
    built_at = materialized_at or datetime.now(UTC).isoformat()
    by_provider_day: dict[tuple[str, date], list[SessionProfile]] = defaultdict(list)
    for profile in profiles:
        bucket_day = profile_bucket_day(profile)
        if bucket_day is None:
            continue
        by_provider_day[(profile.origin, bucket_day)].append(profile)

    rows: list[DaySessionSummaryRecord] = []
    for (source_name, bucket_day), day_profiles in sorted(
        by_provider_day.items(),
        key=lambda item: (item[0][1], item[0][0]),
        reverse=True,
    ):
        summary = summarize_day(day_profiles, bucket_day)
        source_updates: list[str] = []
        source_sorts: list[float] = []
        contributor_sources: list[TemporalSource] = []
        for profile in day_profiles:
            iso_values, sort_values = profile_timestamp_values(profile)
            source_updates.extend(iso_values)
            source_sorts.extend(sort_values)
            if iso_values:
                # profile_timestamp_values draws from four raw provider-parsed
                # fields (updated_at, last_message_at, first_message_at,
                # created_at); classify_profile_hwm_source only checks
                # updated_at specifically, which can disagree with which
                # field actually won the aggregate's max() below. Any
                # non-empty result here is genuinely provider-sourced —
                # tag it directly rather than re-deriving from one field.
                contributor_sources.append("provider_ts")
        search_text = " \n".join(
            part
            for part in (
                source_name,
                bucket_day.isoformat(),
                *summary.repos_active,
                *summary.work_event_breakdown.keys(),
            )
            if part
        )
        hwm = max(source_updates) if source_updates else None
        rows.append(
            DaySessionSummaryRecord(
                day=bucket_day.isoformat(),
                source_name=source_name,
                materialized_at=built_at,
                source_updated_at=hwm,
                source_sort_key=max(source_sorts) if source_sorts else None,
                input_high_water_mark=hwm,
                input_high_water_mark_source=classify_aggregate_hwm_source(contributor_sources),
                input_row_count=summary.session_count,
                session_count=summary.session_count,
                logical_session_count=summary.logical_session_count,
                logical_session_ids=summary.logical_session_ids,
                total_cost_usd=summary.total_cost_usd,
                total_duration_ms=summary.total_duration_ms,
                total_tool_active_duration_ms=summary.total_tool_active_duration_ms,
                total_wall_duration_ms=summary.total_wall_duration_ms,
                total_messages=summary.total_messages,
                total_words=summary.total_words,
                work_event_breakdown=summary.work_event_breakdown,
                repos_active=summary.repos_active,
                payload=DaySessionSummaryPayload.model_validate(summary.to_dict()),
                search_text=search_text or bucket_day.isoformat(),
            )
        )
    return rows


def _aggregate_day_buckets(
    rows: Sequence[DaySessionSummaryRecord],
) -> dict[str, _DayAggregateBucket]:
    grouped: dict[str, _DayAggregateBucket] = {}
    for row in rows:
        bucket = grouped.setdefault(row.day, _DayAggregateBucket())
        bucket.session_count += row.session_count
        bucket.logical_session_ids.update(row.logical_session_ids)
        bucket.total_cost_usd += row.total_cost_usd
        bucket.total_duration_ms += row.total_duration_ms
        bucket.total_tool_active_duration_ms += row.total_tool_active_duration_ms
        bucket.total_wall_duration_ms += row.total_wall_duration_ms
        bucket.total_messages += row.total_messages
        bucket.total_words += row.total_words
        bucket.work_event_breakdown.update(row.work_event_breakdown)
        bucket.repos_active.update(str(name) for name in row.repos_active if str(name).strip())
        bucket.origins[row.source_name] += row.session_count
        bucket.rows.append(row)
    return grouped


def _aggregate_day_summaries(
    rows: Sequence[DaySessionSummaryRecord],
) -> list[tuple[str, DaySessionSummary, list[DaySessionSummaryRecord]]]:
    grouped = _aggregate_day_buckets(rows)
    summaries: list[tuple[str, DaySessionSummary, list[DaySessionSummaryRecord]]] = []
    for day in sorted(grouped.keys(), reverse=True):
        bucket = grouped[day]
        summary = DaySessionSummary(
            date=datetime.fromisoformat(day).date(),
            session_count=bucket.session_count,
            logical_session_count=len(bucket.logical_session_ids),
            logical_session_ids=tuple(sorted(bucket.logical_session_ids)),
            total_cost_usd=bucket.total_cost_usd,
            total_duration_ms=bucket.total_duration_ms,
            total_tool_active_duration_ms=bucket.total_tool_active_duration_ms,
            total_wall_duration_ms=bucket.total_wall_duration_ms,
            total_messages=bucket.total_messages,
            total_words=bucket.total_words,
            work_event_breakdown=dict(bucket.work_event_breakdown),
            repos_active=tuple(sorted(bucket.repos_active)),
            origins=dict(bucket.origins),
        )
        summaries.append((day, summary, bucket.rows))
    return summaries


def aggregate_day_session_summary_insights(
    rows: Sequence[DaySessionSummaryRecord],
) -> list[DaySessionSummaryInsight]:
    return [
        DaySessionSummaryInsight(
            date=day,
            provenance=records_provenance(record_rows),
            summary=DaySessionSummaryPayload.model_validate(summary.to_dict()),
        )
        for day, summary, record_rows in _aggregate_day_summaries(rows)
    ]


def aggregate_week_session_summary_insights(
    rows: Sequence[DaySessionSummaryRecord],
) -> list[WeekSessionSummaryInsight]:
    day_entries = _aggregate_day_summaries(rows)
    by_week: dict[str, list[DaySessionSummary]] = defaultdict(list)
    provenance_rows: dict[str, list[DaySessionSummaryRecord]] = defaultdict(list)
    for _day, day_summary, record_rows in day_entries:
        iso = day_summary.date.isocalendar()
        week_key = f"{iso[0]}-W{iso[1]:02d}"
        by_week[week_key].append(day_summary)
        provenance_rows[week_key].extend(record_rows)

    insights: list[WeekSessionSummaryInsight] = []
    for iso_week in sorted(by_week.keys(), reverse=True):
        week_summary = summarize_week(tuple(sorted(by_week[iso_week], key=lambda item: item.date)))
        insights.append(
            WeekSessionSummaryInsight(
                iso_week=iso_week,
                provenance=records_provenance(provenance_rows[iso_week]),
                summary=WeekSessionSummaryPayload.model_validate(week_summary.to_dict()),
            )
        )
    return insights


__all__ = [
    "aggregate_day_session_summary_insights",
    "aggregate_week_session_summary_insights",
    "build_day_session_summary_records",
]

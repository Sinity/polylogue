"""Day/week summary builders and aggregate reducers for archive products."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from polylogue.archive_products import (
    DaySessionSummaryProduct,
    WeekSessionSummaryProduct,
    profile_bucket_day,
    profile_timestamp_values,
    records_provenance,
)
from polylogue.lib.session_profile import SessionProfile
from polylogue.lib.session_summaries import DaySessionSummary, summarize_day, summarize_week
from polylogue.storage.store import DaySessionSummaryRecord


@dataclass(slots=True)
class _DayAggregateBucket:
    session_count: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    total_wall_duration_ms: int = 0
    total_messages: int = 0
    total_words: int = 0
    work_event_breakdown: Counter[str] = field(default_factory=Counter)
    repos_active: set[str] = field(default_factory=set)
    providers: Counter[str] = field(default_factory=Counter)
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
        by_provider_day[(profile.provider, bucket_day)].append(profile)

    rows: list[DaySessionSummaryRecord] = []
    for (provider_name, bucket_day), day_profiles in sorted(
        by_provider_day.items(),
        key=lambda item: (item[0][1], item[0][0]),
        reverse=True,
    ):
        summary = summarize_day(day_profiles, bucket_day)
        source_updates: list[str] = []
        source_sorts: list[float] = []
        for profile in day_profiles:
            iso_values, sort_values = profile_timestamp_values(profile)
            source_updates.extend(iso_values)
            source_sorts.extend(sort_values)
        search_text = " \n".join(
            part
            for part in (
                provider_name,
                bucket_day.isoformat(),
                *summary.repos_active,
                *summary.work_event_breakdown.keys(),
            )
            if part
        )
        rows.append(
            DaySessionSummaryRecord(
                day=bucket_day.isoformat(),
                provider_name=provider_name,
                materialized_at=built_at,
                source_updated_at=max(source_updates) if source_updates else None,
                source_sort_key=max(source_sorts) if source_sorts else None,
                conversation_count=summary.session_count,
                total_cost_usd=summary.total_cost_usd,
                total_duration_ms=summary.total_duration_ms,
                total_wall_duration_ms=summary.total_wall_duration_ms,
                total_messages=summary.total_messages,
                total_words=summary.total_words,
                work_event_breakdown=summary.work_event_breakdown,
                repos_active=summary.repos_active,
                payload=summary.to_dict(),
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
        bucket.session_count += row.conversation_count
        bucket.total_cost_usd += row.total_cost_usd
        bucket.total_duration_ms += row.total_duration_ms
        bucket.total_wall_duration_ms += row.total_wall_duration_ms
        bucket.total_messages += row.total_messages
        bucket.total_words += row.total_words
        bucket.work_event_breakdown.update(row.work_event_breakdown)
        bucket.repos_active.update(str(name) for name in row.repos_active if str(name).strip())
        bucket.providers[row.provider_name] += row.conversation_count
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
            total_cost_usd=bucket.total_cost_usd,
            total_duration_ms=bucket.total_duration_ms,
            total_wall_duration_ms=bucket.total_wall_duration_ms,
            total_messages=bucket.total_messages,
            total_words=bucket.total_words,
            work_event_breakdown=dict(bucket.work_event_breakdown),
            repos_active=tuple(sorted(bucket.repos_active)),
            providers=dict(bucket.providers),
        )
        summaries.append((day, summary, bucket.rows))
    return summaries


def aggregate_day_session_summary_products(
    rows: Sequence[DaySessionSummaryRecord],
) -> list[DaySessionSummaryProduct]:
    return [
        DaySessionSummaryProduct(
            date=day,
            provenance=records_provenance(record_rows),
            summary=summary.to_dict(),
        )
        for day, summary, record_rows in _aggregate_day_summaries(rows)
    ]


def aggregate_week_session_summary_products(
    rows: Sequence[DaySessionSummaryRecord],
) -> list[WeekSessionSummaryProduct]:
    day_entries = _aggregate_day_summaries(rows)
    by_week: dict[str, list[DaySessionSummary]] = defaultdict(list)
    provenance_rows: dict[str, list[DaySessionSummaryRecord]] = defaultdict(list)
    for _day, day_summary, record_rows in day_entries:
        iso = day_summary.date.isocalendar()
        week_key = f"{iso[0]}-W{iso[1]:02d}"
        by_week[week_key].append(day_summary)
        provenance_rows[week_key].extend(record_rows)

    products: list[WeekSessionSummaryProduct] = []
    for iso_week in sorted(by_week.keys(), reverse=True):
        week_summary = summarize_week(tuple(sorted(by_week[iso_week], key=lambda item: item.date)))
        products.append(
            WeekSessionSummaryProduct(
                iso_week=iso_week,
                provenance=records_provenance(provenance_rows[iso_week]),
                summary=week_summary.to_dict(),
            )
        )
    return products


__all__ = [
    "aggregate_day_session_summary_products",
    "aggregate_week_session_summary_products",
    "build_day_session_summary_records",
]

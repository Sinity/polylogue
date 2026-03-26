"""Day/week summary builders and aggregate reducers for archive products."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime

from polylogue.archive_product_support import (
    date_from_iso,
    profile_bucket_day,
    profile_timestamp_values,
    records_provenance,
)
from polylogue.archive_products import DaySessionSummaryProduct, WeekSessionSummaryProduct
from polylogue.lib.project_normalization import normalize_project_names
from polylogue.lib.session_profile import SessionProfile
from polylogue.lib.session_summaries import DaySessionSummary, summarize_day, summarize_week
from polylogue.storage.store import DaySessionSummaryRecord


def build_day_session_summary_records(
    profiles: Sequence[SessionProfile],
    *,
    materialized_at: str | None = None,
) -> list[DaySessionSummaryRecord]:
    built_at = materialized_at or datetime.now(UTC).isoformat()
    by_provider_day: dict[tuple[str, object], list[SessionProfile]] = defaultdict(list)
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
                *summary.projects_active,
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
                projects_active=summary.projects_active,
                payload=summary.to_dict(),
                search_text=search_text or bucket_day.isoformat(),
            )
        )
    return rows


def aggregate_day_session_summary_products(
    rows: Sequence[DaySessionSummaryRecord],
) -> list[DaySessionSummaryProduct]:
    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        bucket = grouped.setdefault(
            row.day,
            {
                "session_count": 0,
                "total_cost_usd": 0.0,
                "total_duration_ms": 0,
                "total_wall_duration_ms": 0,
                "total_messages": 0,
                "total_words": 0,
                "work_event_breakdown": Counter(),
                "projects_active": set(),
                "providers": Counter(),
                "rows": [],
            },
        )
        bucket["session_count"] = int(bucket["session_count"]) + row.conversation_count
        bucket["total_cost_usd"] = float(bucket["total_cost_usd"]) + row.total_cost_usd
        bucket["total_duration_ms"] = int(bucket["total_duration_ms"]) + row.total_duration_ms
        bucket["total_wall_duration_ms"] = int(bucket["total_wall_duration_ms"]) + row.total_wall_duration_ms
        bucket["total_messages"] = int(bucket["total_messages"]) + row.total_messages
        bucket["total_words"] = int(bucket["total_words"]) + row.total_words
        work_event_breakdown = bucket["work_event_breakdown"]
        projects_active = bucket["projects_active"]
        providers = bucket["providers"]
        record_rows = bucket["rows"]
        assert isinstance(work_event_breakdown, Counter)
        assert isinstance(projects_active, set)
        assert isinstance(providers, Counter)
        assert isinstance(record_rows, list)
        work_event_breakdown.update(row.work_event_breakdown)
        projects_active.update(normalize_project_names(row.projects_active))
        providers[row.provider_name] += row.conversation_count
        record_rows.append(row)

    products: list[DaySessionSummaryProduct] = []
    for day, bucket in sorted(grouped.items(), reverse=True):
        work_event_breakdown = bucket["work_event_breakdown"]
        projects_active = bucket["projects_active"]
        providers = bucket["providers"]
        record_rows = bucket["rows"]
        assert isinstance(work_event_breakdown, Counter)
        assert isinstance(projects_active, set)
        assert isinstance(providers, Counter)
        assert isinstance(record_rows, list)
        summary = DaySessionSummary(
            date=date_from_iso(day),
            session_count=int(bucket["session_count"]),
            total_cost_usd=float(bucket["total_cost_usd"]),
            total_duration_ms=int(bucket["total_duration_ms"]),
            total_wall_duration_ms=int(bucket["total_wall_duration_ms"]),
            total_messages=int(bucket["total_messages"]),
            total_words=int(bucket["total_words"]),
            work_event_breakdown=dict(work_event_breakdown),
            projects_active=tuple(sorted(str(path) for path in projects_active)),
            providers=dict(providers),
        )
        products.append(
            DaySessionSummaryProduct(
                date=day,
                provenance=records_provenance(record_rows),
                summary=summary.to_dict(),
            )
        )
    return products


def aggregate_week_session_summary_products(
    rows: Sequence[DaySessionSummaryRecord],
) -> list[WeekSessionSummaryProduct]:
    day_products = aggregate_day_session_summary_products(rows)
    day_summaries = [
        DaySessionSummary(
            date=date_from_iso(product.date),
            session_count=int(product.summary["session_count"]),
            total_cost_usd=float(product.summary["total_cost_usd"]),
            total_duration_ms=int(product.summary["total_duration_ms"]),
            total_wall_duration_ms=int(product.summary["total_wall_duration_ms"]),
            total_messages=int(product.summary["total_messages"]),
            total_words=int(product.summary["total_words"]),
            work_event_breakdown=dict(product.summary["work_event_breakdown"]),
            projects_active=tuple(product.summary["projects_active"]),
            providers=dict(product.summary["providers"]),
        )
        for product in day_products
    ]
    by_week: dict[str, list[DaySessionSummary]] = defaultdict(list)
    provenance_rows: dict[str, list[DaySessionSummaryRecord]] = defaultdict(list)
    for day_summary in day_summaries:
        iso = day_summary.date.isocalendar()
        week_key = f"{iso[0]}-W{iso[1]:02d}"
        by_week[week_key].append(day_summary)
        day_key = day_summary.date.isoformat()
        provenance_rows[week_key].extend([row for row in rows if row.day == day_key])

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

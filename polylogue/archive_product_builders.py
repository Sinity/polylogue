"""Builders for durable archive-product rows and public aggregate payloads."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, timedelta

from polylogue.archive_products import (
    ArchiveProductProvenance,
    DaySessionSummaryProduct,
    SessionTagRollupProduct,
    WeekSessionSummaryProduct,
)
from polylogue.lib.session_profile import SessionProfile
from polylogue.lib.session_summaries import DaySessionSummary, summarize_day, summarize_week
from polylogue.storage.store import (
    SESSION_PRODUCT_MATERIALIZER_VERSION,
    DaySessionSummaryRecord,
    SessionTagRollupRecord,
)


def _profile_bucket_day(profile: SessionProfile) -> date | None:
    timestamp = profile.first_message_at or profile.created_at or profile.updated_at or profile.last_message_at
    if timestamp is None:
        return None
    return timestamp.date() if isinstance(timestamp, datetime) else timestamp


def _profile_timestamp_values(profile: SessionProfile) -> tuple[list[str], list[float]]:
    timestamps = [
        timestamp
        for timestamp in (
            profile.updated_at,
            profile.last_message_at,
            profile.first_message_at,
            profile.created_at,
        )
        if timestamp is not None
    ]
    return (
        [timestamp.isoformat() for timestamp in timestamps],
        [timestamp.timestamp() for timestamp in timestamps],
    )


def _records_provenance(
    rows: Iterable[object],
    *,
    materialized_at_attr: str = "materialized_at",
    source_updated_at_attr: str = "source_updated_at",
    source_sort_key_attr: str = "source_sort_key",
) -> ArchiveProductProvenance:
    row_list = list(rows)
    materialized_at = max(
        (
            str(getattr(row, materialized_at_attr))
            for row in row_list
            if getattr(row, materialized_at_attr, None)
        ),
        default="1970-01-01T00:00:00+00:00",
    )
    source_updated_at = max(
        (
            str(getattr(row, source_updated_at_attr))
            for row in row_list
            if getattr(row, source_updated_at_attr, None)
        ),
        default=None,
    )
    source_sort_key = max(
        (
            float(getattr(row, source_sort_key_attr))
            for row in row_list
            if getattr(row, source_sort_key_attr, None) is not None
        ),
        default=None,
    )
    return ArchiveProductProvenance(
        materializer_version=SESSION_PRODUCT_MATERIALIZER_VERSION,
        materialized_at=materialized_at,
        source_updated_at=source_updated_at,
        source_sort_key=source_sort_key,
    )


def build_session_tag_rollup_records(
    profiles: Sequence[SessionProfile],
    *,
    materialized_at: str | None = None,
) -> list[SessionTagRollupRecord]:
    built_at = materialized_at or datetime.now(UTC).isoformat()
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for profile in profiles:
        bucket_day = _profile_bucket_day(profile)
        if bucket_day is None:
            continue
        explicit_tags = {tag for tag in profile.tags if tag}
        auto_tags = {tag for tag in profile.auto_tags if tag}
        all_tags = explicit_tags | auto_tags
        if not all_tags:
            continue
        iso_timestamps, sort_keys = _profile_timestamp_values(profile)
        for tag in all_tags:
            key = (profile.provider, bucket_day.isoformat(), tag)
            bucket = grouped.setdefault(
                key,
                {
                    "conversation_count": 0,
                    "explicit_count": 0,
                    "auto_count": 0,
                    "projects": Counter(),
                    "source_updated_at": [],
                    "source_sort_key": [],
                },
            )
            bucket["conversation_count"] = int(bucket["conversation_count"]) + 1
            if tag in explicit_tags:
                bucket["explicit_count"] = int(bucket["explicit_count"]) + 1
            if tag in auto_tags:
                bucket["auto_count"] = int(bucket["auto_count"]) + 1
            cast_projects = bucket["projects"]
            assert isinstance(cast_projects, Counter)
            cast_projects.update(profile.canonical_projects)
            cast_updates = bucket["source_updated_at"]
            cast_sorts = bucket["source_sort_key"]
            assert isinstance(cast_updates, list)
            assert isinstance(cast_sorts, list)
            cast_updates.extend(iso_timestamps)
            cast_sorts.extend(sort_keys)

    rows: list[SessionTagRollupRecord] = []
    for (provider_name, bucket_day, tag), bucket in sorted(grouped.items()):
        projects = bucket["projects"]
        source_updates = bucket["source_updated_at"]
        source_sorts = bucket["source_sort_key"]
        assert isinstance(projects, Counter)
        assert isinstance(source_updates, list)
        assert isinstance(source_sorts, list)
        search_text = " \n".join(
            part for part in (tag, provider_name, *sorted(projects.keys())) if part
        )
        rows.append(
            SessionTagRollupRecord(
                tag=tag,
                bucket_day=bucket_day,
                provider_name=provider_name,
                materialized_at=built_at,
                source_updated_at=max(source_updates) if source_updates else None,
                source_sort_key=max(source_sorts) if source_sorts else None,
                conversation_count=int(bucket["conversation_count"]),
                explicit_count=int(bucket["explicit_count"]),
                auto_count=int(bucket["auto_count"]),
                project_breakdown=dict(projects),
                search_text=search_text or tag,
            )
        )
    return rows


def build_day_session_summary_records(
    profiles: Sequence[SessionProfile],
    *,
    materialized_at: str | None = None,
) -> list[DaySessionSummaryRecord]:
    built_at = materialized_at or datetime.now(UTC).isoformat()
    by_provider_day: dict[tuple[str, date], list[SessionProfile]] = defaultdict(list)
    for profile in profiles:
        bucket_day = _profile_bucket_day(profile)
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
            iso_values, sort_values = _profile_timestamp_values(profile)
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


def aggregate_session_tag_rollup_products(
    rows: Sequence[SessionTagRollupRecord],
) -> list[SessionTagRollupProduct]:
    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        bucket = grouped.setdefault(
            row.tag,
            {
                "conversation_count": 0,
                "explicit_count": 0,
                "auto_count": 0,
                "provider_breakdown": Counter(),
                "project_breakdown": Counter(),
                "rows": [],
            },
        )
        bucket["conversation_count"] = int(bucket["conversation_count"]) + row.conversation_count
        bucket["explicit_count"] = int(bucket["explicit_count"]) + row.explicit_count
        bucket["auto_count"] = int(bucket["auto_count"]) + row.auto_count
        provider_breakdown = bucket["provider_breakdown"]
        project_breakdown = bucket["project_breakdown"]
        record_rows = bucket["rows"]
        assert isinstance(provider_breakdown, Counter)
        assert isinstance(project_breakdown, Counter)
        assert isinstance(record_rows, list)
        provider_breakdown[row.provider_name] += row.conversation_count
        project_breakdown.update(row.project_breakdown)
        record_rows.append(row)

    products: list[SessionTagRollupProduct] = []
    for tag, bucket in sorted(grouped.items(), key=lambda item: (-int(item[1]["conversation_count"]), item[0])):
        provider_breakdown = bucket["provider_breakdown"]
        project_breakdown = bucket["project_breakdown"]
        record_rows = bucket["rows"]
        assert isinstance(provider_breakdown, Counter)
        assert isinstance(project_breakdown, Counter)
        assert isinstance(record_rows, list)
        products.append(
            SessionTagRollupProduct(
                tag=tag,
                conversation_count=int(bucket["conversation_count"]),
                explicit_count=int(bucket["explicit_count"]),
                auto_count=int(bucket["auto_count"]),
                provider_breakdown=dict(provider_breakdown),
                project_breakdown=dict(project_breakdown),
                provenance=_records_provenance(record_rows),
            )
        )
    return products


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
        projects_active.update(row.projects_active)
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
                provenance=_records_provenance(record_rows),
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
    rows_by_day = {row.day: row for row in rows}
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
                provenance=_records_provenance(provenance_rows[iso_week]),
                summary=week_summary.to_dict(),
            )
        )
    return products


def day_after(iso_day: str) -> str:
    return (date_from_iso(iso_day) + timedelta(days=1)).isoformat()


def date_from_iso(value: str) -> date:
    return date.fromisoformat(value)


__all__ = [
    "aggregate_day_session_summary_products",
    "aggregate_session_tag_rollup_products",
    "aggregate_week_session_summary_products",
    "build_day_session_summary_records",
    "build_session_tag_rollup_records",
    "date_from_iso",
    "day_after",
]

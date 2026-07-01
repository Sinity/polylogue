"""Tag-rollup builders and aggregate reducers for archive insights."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from polylogue.archive.semantic.pricing import (
    CostBasisPayload,
    CostModelBreakdown,
    CostUsagePayload,
)
from polylogue.archive.session.repo_identity import normalize_repo_names
from polylogue.archive.session.session_profile import SessionProfile
from polylogue.insights.archive import (
    ArchiveInsightProvenance,
    CostRollupInsight,
    SessionCostInsight,
    SessionTagRollupInsight,
    profile_bucket_day,
    profile_timestamp_values,
    records_provenance,
)
from polylogue.insights.temporal_source import classify_aggregate_hwm_source
from polylogue.storage.runtime import SessionTagRollupRecord
from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION


@dataclass(slots=True)
class _TagRollupBucket:
    session_count: int = 0
    logical_session_ids: set[str] = field(default_factory=set)
    explicit_count: int = 0
    auto_count: int = 0
    repos: Counter[str] = field(default_factory=Counter)
    source_updated_at: list[str] = field(default_factory=list)
    source_sort_key: list[float] = field(default_factory=list)


@dataclass(slots=True)
class _TagAggregateBucket:
    session_count: int = 0
    logical_session_ids: set[str] = field(default_factory=set)
    explicit_count: int = 0
    auto_count: int = 0
    origin_breakdown: Counter[str] = field(default_factory=Counter)
    repo_breakdown: Counter[str] = field(default_factory=Counter)
    rows: list[SessionTagRollupRecord] = field(default_factory=list)


def build_session_tag_rollup_records(
    profiles: Iterable[SessionProfile],
    *,
    materialized_at: str | None = None,
) -> list[SessionTagRollupRecord]:
    built_at = materialized_at or datetime.now(UTC).isoformat()
    grouped: dict[tuple[str, str, str], _TagRollupBucket] = {}
    for profile in profiles:
        bucket_day = profile_bucket_day(profile)
        if bucket_day is None:
            continue
        explicit_tags = {tag for tag in profile.tags if tag}
        auto_tags = {tag for tag in profile.auto_tags if tag}
        all_tags = explicit_tags | auto_tags
        if not all_tags:
            continue

        iso_timestamps, sort_keys = profile_timestamp_values(profile)
        repo_names = profile.repo_names or normalize_repo_names(repo_paths=profile.repo_paths)
        bucket_day_text = bucket_day.isoformat()
        for tag in all_tags:
            key = (profile.origin, bucket_day_text, tag)
            bucket = grouped.setdefault(key, _TagRollupBucket())
            bucket.session_count += 1
            bucket.logical_session_ids.add(profile.logical_session_id or profile.session_id)
            if tag in explicit_tags:
                bucket.explicit_count += 1
            if tag in auto_tags:
                bucket.auto_count += 1
            bucket.repos.update(repo_names)
            bucket.source_updated_at.extend(iso_timestamps)
            bucket.source_sort_key.extend(sort_keys)

    rows: list[SessionTagRollupRecord] = []
    for (source_name, bucket_day_text, tag), bucket in sorted(grouped.items()):
        search_text = " \n".join(part for part in (tag, source_name, *sorted(bucket.repos.keys())) if part)
        hwm = max(bucket.source_updated_at) if bucket.source_updated_at else None
        rows.append(
            SessionTagRollupRecord(
                tag=tag,
                bucket_day=bucket_day_text,
                source_name=source_name,
                materialized_at=built_at,
                source_updated_at=hwm,
                source_sort_key=max(bucket.source_sort_key) if bucket.source_sort_key else None,
                input_high_water_mark=hwm,
                input_high_water_mark_source=classify_aggregate_hwm_source(bucket.source_updated_at),
                input_row_count=bucket.session_count,
                session_count=bucket.session_count,
                logical_session_count=len(bucket.logical_session_ids),
                logical_session_ids=tuple(sorted(bucket.logical_session_ids)),
                explicit_count=bucket.explicit_count,
                auto_count=bucket.auto_count,
                repo_breakdown=dict(bucket.repos),
                search_text=search_text or tag,
            )
        )
    return rows


def aggregate_session_tag_rollup_insights(
    rows: Sequence[SessionTagRollupRecord],
) -> list[SessionTagRollupInsight]:
    grouped: dict[str, _TagAggregateBucket] = {}
    for row in rows:
        bucket = grouped.setdefault(row.tag, _TagAggregateBucket())
        bucket.session_count += row.session_count
        bucket.logical_session_ids.update(row.logical_session_ids)
        bucket.explicit_count += row.explicit_count
        bucket.auto_count += row.auto_count
        bucket.origin_breakdown[row.source_name] += row.session_count
        bucket.repo_breakdown.update(row.repo_breakdown)
        bucket.rows.append(row)

    insights: list[SessionTagRollupInsight] = []
    for tag, bucket in sorted(grouped.items(), key=lambda item: (-item[1].session_count, item[0])):
        insights.append(
            SessionTagRollupInsight(
                tag=tag,
                session_count=bucket.session_count,
                logical_session_count=len(bucket.logical_session_ids),
                explicit_count=bucket.explicit_count,
                auto_count=bucket.auto_count,
                origin_breakdown=dict(bucket.origin_breakdown),
                repo_breakdown=dict(bucket.repo_breakdown),
                provenance=records_provenance(bucket.rows),
            )
        )
    return insights


def _merge_per_model_breakdown(
    accumulator: dict[tuple[str | None, str | None], CostModelBreakdown],
    entry: CostModelBreakdown,
) -> None:
    """Merge ``entry`` into ``accumulator`` keyed by ``(model_name, normalized_model)``.

    Each merge bumps ``session_count`` by one — entries come from individual
    sessions, so merging is the "another session contributed this model" event.
    """

    key = (entry.model_name, entry.normalized_model)
    existing = accumulator.get(key)
    if existing is None:
        accumulator[key] = CostModelBreakdown(
            model_name=entry.model_name,
            normalized_model=entry.normalized_model,
            usage=entry.usage,
            basis=entry.basis,
            total_usd=entry.total_usd,
            session_count=1,
        )
        return
    accumulator[key] = CostModelBreakdown(
        model_name=existing.model_name,
        normalized_model=existing.normalized_model,
        usage=existing.usage.plus(entry.usage),
        basis=existing.basis.plus(entry.basis),
        total_usd=existing.total_usd + entry.total_usd,
        session_count=existing.session_count + 1,
    )


def _session_per_model_entries(insight: SessionCostInsight) -> Sequence[CostModelBreakdown]:
    """Return the per-model rows to contribute from one session.

    Prefer the session-level breakdown when populated (mixed-model
    sessions). Fall back to a single synthesized row from the dominant
    model so provider-reported-only sessions still surface per-model rows
    in the rollup.
    """

    entries = insight.estimate.per_model_breakdown
    if entries:
        return entries
    estimate = insight.estimate
    if estimate.model_name or estimate.normalized_model:
        return (
            CostModelBreakdown(
                model_name=estimate.model_name,
                normalized_model=estimate.normalized_model,
                usage=estimate.usage,
                basis=estimate.basis,
                total_usd=estimate.total_usd,
                session_count=1,
            ),
        )
    return ()


def aggregate_cost_rollup_insights(
    session_costs: Sequence[SessionCostInsight],
    *,
    materialized_at: str,
) -> list[CostRollupInsight]:
    """Group ``SessionCostInsight`` rows into ``CostRollupInsight`` rows.

    Grouping is keyed by ``(source_name, normalized_model_or_model_name)``.
    Each row aggregates the basis axes, ``unavailable_reason_counts``, and a
    per-model breakdown across the sessions in the group.
    """

    grouped: dict[tuple[str, str | None], list[SessionCostInsight]] = {}
    for insight in session_costs:
        key = (insight.source_name, insight.estimate.normalized_model or insight.estimate.model_name)
        grouped.setdefault(key, []).append(insight)

    rollups: list[CostRollupInsight] = []
    for (source_name, normalized_model), insights in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1] or ""),
    ):
        usage = CostUsagePayload()
        basis = CostBasisPayload()
        status_counts: Counter[str] = Counter()
        unavailable_reason_counts: Counter[str] = Counter()
        total_usd = 0.0
        priced_count = 0
        confidence_total = 0.0
        per_model_acc: dict[tuple[str | None, str | None], CostModelBreakdown] = {}
        source_updated_at = max(
            (
                insight.provenance.source_updated_at
                for insight in insights
                if insight.provenance.source_updated_at is not None
            ),
            default=None,
        )
        source_sort_key = max(
            (
                insight.provenance.source_sort_key
                for insight in insights
                if insight.provenance.source_sort_key is not None
            ),
            default=None,
        )
        model_names = Counter(insight.estimate.model_name for insight in insights if insight.estimate.model_name)
        for insight in insights:
            estimate = insight.estimate
            usage = usage.plus(estimate.usage)
            basis = basis.plus(estimate.basis)
            status_counts[estimate.status] += 1
            total_usd += estimate.total_usd
            if estimate.unavailable_reason is not None:
                unavailable_reason_counts[estimate.unavailable_reason] += 1
            if estimate.priced:
                priced_count += 1
                confidence_total += estimate.confidence
            for entry in _session_per_model_entries(insight):
                _merge_per_model_breakdown(per_model_acc, entry)
        per_model_breakdown = tuple(sorted(per_model_acc.values(), key=lambda entry: entry.total_usd, reverse=True))
        rollups.append(
            CostRollupInsight(
                source_name=source_name,
                model_name=model_names.most_common(1)[0][0] if model_names else None,
                normalized_model=normalized_model,
                session_count=len(insights),
                priced_session_count=priced_count,
                unavailable_session_count=status_counts["unavailable"],
                status_counts=dict(sorted(status_counts.items())),
                total_usd=total_usd,
                basis=basis,
                unavailable_reason_counts=dict(sorted(unavailable_reason_counts.items())),
                per_model_breakdown=per_model_breakdown,
                usage=usage,
                confidence=(confidence_total / priced_count if priced_count else 0.0),
                provenance=ArchiveInsightProvenance(
                    materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
                    materialized_at=materialized_at,
                    source_updated_at=source_updated_at,
                    source_sort_key=source_sort_key,
                ),
            )
        )
    rollups.sort(key=lambda insight: insight.total_usd, reverse=True)
    return rollups


__all__ = [
    "aggregate_cost_rollup_insights",
    "aggregate_session_tag_rollup_insights",
    "build_session_tag_rollup_records",
]

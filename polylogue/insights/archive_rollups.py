"""Tag-rollup builders and aggregate reducers for archive insights."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from polylogue.archive.semantic.pricing import (
    CostBasisPayload,
    CostModelBreakdown,
    CostUsagePayload,
)
from polylogue.archive.session.repo_identity import normalize_repo_names
from polylogue.archive.session.session_profile import SessionProfile
from polylogue.core.sources import source_name_to_origin
from polylogue.insights.archive import (
    ArchiveInsightProvenance,
    CostRollupInsight,
    SessionCostInsight,
    SessionLatencyProfileInsight,
    SessionProfileInsight,
    SessionTagRollupInsight,
    profile_bucket_day,
    profile_timestamp_values,
    records_provenance,
)
from polylogue.insights.temporal_source import TemporalSource, classify_aggregate_hwm_source
from polylogue.storage.runtime import SessionTagRollupRecord
from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION

# Severity rank for "how badly did this session get left hanging" (#1691).
# Distinct vocabulary from portfolio.py's pathology _SEVERITY_RANK
# (high/medium/low) -- this ranks terminal_state values.
ABANDONMENT_SEVERITY_RANK: dict[str, int] = {
    "question_left": 1,
    "error_left": 2,
    "tool_left": 3,
    "agent_hanging": 4,
}


@dataclass(slots=True)
class _TagRollupBucket:
    session_count: int = 0
    logical_session_ids: set[str] = field(default_factory=set)
    explicit_count: int = 0
    auto_count: int = 0
    repos: Counter[str] = field(default_factory=Counter)
    source_updated_at: list[str] = field(default_factory=list)
    source_sort_key: list[float] = field(default_factory=list)
    contributor_sources: list[TemporalSource] = field(default_factory=list)


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
            if iso_timestamps:
                # See archive_summaries.py: iso_timestamps draws from four raw
                # provider-parsed fields, any of which winning the aggregate
                # max() is genuinely provider_ts — classify_profile_hwm_source
                # only checks updated_at and can disagree with the winner.
                bucket.contributor_sources.append("provider_ts")

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
                input_high_water_mark_source=classify_aggregate_hwm_source(bucket.contributor_sources),
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
                confidence=(confidence_total / priced_count if priced_count else None),
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


def iso_week_bucket_key(canonical_session_date: str | None) -> str:
    """Bucket key for one session date: ISO year-week, or a fallback.

    Returns ``"undated"`` when there is no date, ``"YYYY-Www"`` for a
    parseable ISO date, or the first 7 characters of the raw string when
    it fails to parse (matching the legacy MCP-inline behavior).
    """
    if not canonical_session_date:
        return "undated"
    try:
        parsed = date.fromisoformat(canonical_session_date)
        iso_year, iso_week, _ = parsed.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"
    except ValueError:
        return canonical_session_date[:7]


def aggregate_session_profiles_by_dimension(
    profiles: Sequence[SessionProfileInsight],
    group_by: str,
) -> dict[str, int]:
    """GROUP BY session count over workflow_shape / terminal_state / origin (#1691).

    Raises ``ValueError`` for an unsupported ``group_by``.
    """
    buckets: dict[str, int] = {}
    for profile in profiles:
        if group_by == "workflow_shape":
            key = (profile.inference.workflow_shape if profile.inference else None) or "unknown"
        elif group_by == "terminal_state":
            key = (profile.inference.terminal_state if profile.inference else None) or "unknown"
        elif group_by == "origin":
            key = source_name_to_origin(profile.source_name)
        else:
            raise ValueError(f"Unknown group_by: {group_by!r}. Supported: workflow_shape, terminal_state, origin.")
        buckets[key] = buckets.get(key, 0) + 1
    return buckets


def workflow_shape_distribution_buckets(
    profiles: Sequence[SessionProfileInsight],
    group_by: str,
) -> dict[str, dict[str, int]]:
    """Histogram session workflow shapes by week / origin / project (#1691).

    Raises ``ValueError`` when ``group_by`` is not one of
    ``week``, ``origin``, ``project``.
    """
    allowed_group_by = {"week", "origin", "project"}
    if group_by not in allowed_group_by:
        raise ValueError("group_by must be one of week, origin, project")
    buckets: dict[str, dict[str, int]] = {}
    for profile in profiles:
        evidence = profile.evidence
        inference = profile.inference
        shape = inference.workflow_shape if inference is not None else "unknown"
        keys: tuple[str, ...]
        if group_by == "origin":
            keys = (source_name_to_origin(profile.source_name),)
        elif group_by == "project":
            paths = evidence.cwd_paths if evidence is not None else ()
            keys = tuple(paths) or ("unattributed",)
        else:
            date_value = evidence.canonical_session_date if evidence is not None else None
            keys = (iso_week_bucket_key(date_value),)
        for key in keys:
            bucket = buckets.setdefault(key, {})
            bucket[shape] = bucket.get(shape, 0) + 1
    return buckets


def abandoned_session_items(
    profiles: Sequence[SessionProfileInsight],
    *,
    min_severity: str,
    repo_path: str | None = None,
) -> list[dict[str, object]]:
    """Sessions whose terminal state indicates dangling work (#1691).

    Sorted by ``canonical_session_date`` descending (uncapped -- callers
    apply their own limit). Raises ``ValueError`` for an unknown
    ``min_severity``.
    """
    if min_severity not in ABANDONMENT_SEVERITY_RANK:
        raise ValueError("min_severity must be one of question_left, error_left, tool_left, agent_hanging")
    min_rank = ABANDONMENT_SEVERITY_RANK[min_severity]
    items: list[dict[str, object]] = []
    for profile in profiles:
        inference = profile.inference
        evidence = profile.evidence
        state = inference.terminal_state if inference is not None else "unknown"
        if ABANDONMENT_SEVERITY_RANK.get(state, 0) < min_rank:
            continue
        cwd_paths = evidence.cwd_paths if evidence is not None else ()
        if repo_path and not any(repo_path in path for path in cwd_paths):
            continue
        items.append(
            {
                "session_id": profile.session_id,
                "origin": source_name_to_origin(profile.source_name),
                "title": profile.title,
                "terminal_state": state,
                "terminal_state_confidence": (inference.terminal_state_confidence if inference is not None else 0.0),
                "workflow_shape": inference.workflow_shape if inference is not None else "unknown",
                "canonical_session_date": evidence.canonical_session_date if evidence is not None else None,
                "evidence": evidence.terminal_state_evidence if evidence is not None else {},
            }
        )
    items.sort(key=lambda item: str(item.get("canonical_session_date") or ""), reverse=True)
    return items


def tool_call_latency_distribution_payload(
    insights: Sequence[SessionLatencyProfileInsight],
    *,
    tool_category: str | None = None,
) -> dict[str, object]:
    """Distribution of materialized per-session tool-call latency (#1691).

    Reuses the nearest-rank percentile from
    :mod:`polylogue.insights.portfolio` (``_percentile``/``DistributionStat``)
    rather than a second percentile algorithm.
    """
    from polylogue.insights.portfolio import _percentile

    def _nearest_rank(values: list[int], p: float) -> int:
        if not values:
            return 0
        return int(_percentile(sorted(values), p / 100.0))

    filtered = insights
    if tool_category:
        filtered = [
            insight for insight in insights if insight.latency.tool_call_count_by_category.get(tool_category, 0) > 0
        ]
    medians = [insight.latency.median_tool_call_ms for insight in filtered if insight.latency.median_tool_call_ms]
    p90s = [insight.latency.p90_tool_call_ms for insight in filtered if insight.latency.p90_tool_call_ms]
    maxes = [insight.latency.max_tool_call_ms for insight in filtered if insight.latency.max_tool_call_ms]
    return {
        "total_sessions": len(filtered),
        "tool_category": tool_category,
        "median_tool_call_ms": _nearest_rank(medians, 50),
        "p90_tool_call_ms": _nearest_rank(p90s, 90),
        "max_tool_call_ms": max(maxes) if maxes else 0,
        "stuck_tool_count": sum(insight.latency.stuck_tool_count for insight in filtered),
        "construct_boundary": (
            "distribution is over materialized per-session aggregates; "
            "agent-response time includes both LLM inference and tool execution"
        ),
    }


__all__ = [
    "ABANDONMENT_SEVERITY_RANK",
    "abandoned_session_items",
    "aggregate_cost_rollup_insights",
    "aggregate_session_profiles_by_dimension",
    "aggregate_session_tag_rollup_insights",
    "build_session_tag_rollup_records",
    "iso_week_bucket_key",
    "tool_call_latency_distribution_payload",
    "workflow_shape_distribution_buckets",
]

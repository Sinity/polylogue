"""Productivity rollups derived from materialized session insights.

This module computes productivity-style aggregates — hour-of-day activity,
project focus share, context-switch counts, and work-event outcome
breakdowns — over the already-materialized ``session_profile`` and
``session_work_events`` substrates.

The contract is intentionally cautious about subjective claims:

* Every metric is an objective count, distribution, or ratio over
  observable evidence (session timestamps, recorded ``cwd_paths``,
  classified ``work_event.kind``). No composite "productivity score" is
  produced.
* Every aggregation entry ships with two first-class fields:

  - ``evidence_inputs`` — opaque source identifiers (conversation IDs,
    work-event IDs, day keys, week keys) that contributed to the entry.
    A reader can drill into the source rows directly.
  - ``caveats`` — typed strings that name the structural reasons the
    metric may diverge from a reader's intuitive notion of
    "productivity".

The default envelope always carries a baseline disclaimer that the
rollups describe observed conversational signals only — they do not
reflect work performed off-screen, ambient thinking, or work that was
captured in unsupported tools. Surfaces (CLI, MCP, dashboard) must
preserve the ``caveats`` and ``baseline_caveats`` fields verbatim.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone

from pydantic import Field

from polylogue.insights.archive import (
    ProviderTimeWindowInsightQuery,
    SessionProfileInsight,
    SessionWorkEventInsight,
)
from polylogue.insights.archive_models import (
    ARCHIVE_INSIGHT_CONTRACT_VERSION,
    ArchiveInsightModel,
    ArchiveInsightProvenance,
)

PRODUCTIVITY_ROLLUP_INSIGHT_VERSION = 1
"""Insight materializer version for ``productivity_rollups`` envelopes."""

# Baseline disclaimers that every envelope ships with. These describe the
# structural limits of any rollup derived from archived AI conversations.
BASELINE_CAVEATS: tuple[str, ...] = (
    "Counts reflect observed AI conversation signals only — off-screen "
    "work, ambient thinking, and work captured outside Polylogue's "
    "supported sources are not represented.",
    "Hour-of-day distribution is computed from the canonical session "
    "timestamp; sessions without timestamps are excluded from the "
    "histogram and reported in `untimed_session_count`.",
    "Project focus share is derived from `cwd_paths` recorded by the "
    "agent runtime; sessions without recorded paths are reported as "
    "`unattributed`.",
    "Context-switch counts measure transitions between distinct "
    "`cwd_paths` within a single session as observed; they do not "
    "estimate cognitive context switching.",
    "Work-event outcome categories come from the existing work-event "
    "classifier and inherit its support level — see "
    "`work_event.support_level` for per-event confidence.",
)


class ProductivityRollupEntry(ArchiveInsightModel):
    """One productivity rollup entry at a chosen granularity."""

    granularity: str
    """One of ``day``, ``week``, or ``project``."""

    bucket_key: str
    """The bucket identifier (e.g. ``2026-05-12``, ``2026-W19``, ``polylogue``)."""

    session_count: int = 0
    """Number of distinct sessions attributed to this bucket."""

    work_event_count: int = 0
    """Number of work events attributed to this bucket."""

    untimed_session_count: int = 0
    """Sessions in the bucket lacking a canonical timestamp; excluded from histograms."""

    hour_of_day_histogram: dict[int, int] = Field(default_factory=dict)
    """Hour-of-day (0-23) distribution of session start timestamps within this bucket."""

    project_focus_share: dict[str, float] = Field(default_factory=dict)
    """Share of session count per project; keys are ``cwd_path`` strings or ``unattributed``."""

    context_switch_count: int = 0
    """Total transitions between distinct ``cwd_paths`` summed across sessions in this bucket."""

    outcome_breakdown: dict[str, int] = Field(default_factory=dict)
    """Count of work events by ``work_event.kind`` within this bucket."""

    evidence_inputs: tuple[str, ...] = ()
    """Opaque source identifiers (conversation IDs, work-event IDs) that fed this entry."""

    caveats: tuple[str, ...] = ()
    """Caveats specific to this entry (e.g. low evidence count)."""


class ProductivityRollupInsight(ArchiveInsightModel):
    """Envelope returned by ``list_productivity_rollup_insights``.

    Every consumer surface (CLI / MCP / API) returns this single envelope.
    The ``baseline_caveats`` field is always populated and must be
    surfaced — it states the structural limits that apply to every
    metric in the envelope.
    """

    contract_version: int = ARCHIVE_INSIGHT_CONTRACT_VERSION
    insight_kind: str = "productivity_rollup"
    materializer_version: int = PRODUCTIVITY_ROLLUP_INSIGHT_VERSION
    granularity: str
    entries: tuple[ProductivityRollupEntry, ...] = ()
    baseline_caveats: tuple[str, ...] = BASELINE_CAVEATS
    total_sessions: int = 0
    total_work_events: int = 0
    total_untimed_sessions: int = 0
    provenance: ArchiveInsightProvenance


class ProductivityRollupInsightQuery(ProviderTimeWindowInsightQuery):
    """Query parameters for ``list_productivity_rollup_insights``.

    ``granularity`` selects bucketing: ``day`` (ISO date), ``week``
    (ISO week, e.g. ``2026-W19``), or ``project`` (canonical
    ``cwd_path`` token from the session evidence). ``project`` filters
    aggregate sessions by their dominant recorded path.
    """

    granularity: str = "day"
    limit: int | None = None


_VALID_GRANULARITIES = ("day", "week", "project")
_UNATTRIBUTED_PROJECT = "unattributed"


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _session_bucket_keys(profile: SessionProfileInsight, granularity: str) -> tuple[str, ...]:
    """Return one or more bucket keys for a session at the requested granularity.

    For ``day`` / ``week`` the result is the canonical session date
    folded to ISO date or ISO week; for ``project`` a session may
    contribute to multiple buckets (one per recorded ``cwd_path``).
    """

    evidence = profile.evidence
    if evidence is None:
        return ()
    if granularity == "project":
        if evidence.cwd_paths:
            return tuple(evidence.cwd_paths)
        return (_UNATTRIBUTED_PROJECT,)
    date_str = evidence.canonical_session_date or (
        evidence.first_message_at[:10] if evidence.first_message_at else None
    )
    if date_str is None:
        return ()
    if granularity == "day":
        return (date_str,)
    # week: parse to ISO week. Fall back to (date,) on malformed input.
    try:
        parsed = datetime.fromisoformat(date_str).date()
    except ValueError:
        try:
            parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return ()
    iso_year, iso_week, _ = parsed.isocalendar()
    return (f"{iso_year}-W{iso_week:02d}",)


def _session_hour(profile: SessionProfileInsight) -> int | None:
    """Return the hour-of-day (0-23) of the session's canonical start."""

    evidence = profile.evidence
    if evidence is None:
        return None
    parsed = _parse_iso_timestamp(evidence.first_message_at)
    if parsed is None:
        return None
    return parsed.astimezone(timezone.utc).hour


def _session_context_switches(profile: SessionProfileInsight) -> int:
    """Count distinct ``cwd_path`` transitions observed within the session.

    The exact ordering of paths is not preserved in the substrate; the
    upper bound is therefore ``len(distinct_paths) - 1``, which we use
    as the reported value. This is an objective ceiling over the
    evidence, not an estimate.
    """

    evidence = profile.evidence
    if evidence is None or not evidence.cwd_paths:
        return 0
    distinct = len({path for path in evidence.cwd_paths if path})
    return max(distinct - 1, 0)


def _project_focus_share(profile_keys: Counter[str]) -> dict[str, float]:
    total = sum(profile_keys.values())
    if total == 0:
        return {}
    return {
        path: round(count / total, 4)
        for path, count in sorted(profile_keys.items(), key=lambda item: (-item[1], item[0]))
    }


def _entry_caveats(
    *,
    granularity: str,
    session_count: int,
    work_event_count: int,
    untimed_session_count: int,
    project_focus_share: dict[str, float],
) -> tuple[str, ...]:
    caveats: list[str] = []
    if session_count < 3:
        caveats.append(
            f"Low evidence: only {session_count} session(s) attributed to this bucket; "
            "metrics may not be representative."
        )
    if untimed_session_count and session_count:
        share = untimed_session_count / session_count
        if share >= 0.25:
            caveats.append(
                f"{untimed_session_count} of {session_count} sessions lack timestamps and are "
                "excluded from `hour_of_day_histogram`."
            )
    if granularity != "project" and project_focus_share.get(_UNATTRIBUTED_PROJECT, 0.0) >= 0.25:
        caveats.append(
            "At least 25% of sessions in this bucket lack recorded `cwd_paths`; "
            "`project_focus_share` understates real attribution."
        )
    if work_event_count == 0 and session_count > 0:
        caveats.append("No work events were classified for these sessions; `outcome_breakdown` is empty.")
    return tuple(caveats)


def _profile_in_window(profile: SessionProfileInsight, *, since: str | None, until: str | None) -> bool:
    if since is None and until is None:
        return True
    evidence = profile.evidence
    reference = (evidence.canonical_session_date if evidence is not None else None) or (
        evidence.first_message_at[:10] if evidence is not None and evidence.first_message_at else None
    )
    if reference is None:
        return False
    if since is not None and reference < since[:10]:
        return False
    return not (until is not None and reference > until[:10])


def build_productivity_rollup_insight(
    *,
    profiles: Sequence[SessionProfileInsight],
    work_events: Sequence[SessionWorkEventInsight],
    query: ProductivityRollupInsightQuery,
    materialized_at: str,
) -> ProductivityRollupInsight:
    """Assemble a ``ProductivityRollupInsight`` from already-hydrated insights.

    Pure function: callers (operations, tests, scenario runners) own the
    fetch of profiles and work events. Filtering by ``provider`` / time
    window happens here so the aggregation is consistent regardless of
    how the substrate was queried.
    """

    granularity = query.granularity if query.granularity in _VALID_GRANULARITIES else "day"

    filtered_profiles = [
        profile
        for profile in profiles
        if (query.provider is None or profile.provider_name == query.provider)
        and _profile_in_window(profile, since=query.since, until=query.until)
    ]

    # Map conversation_id -> set of bucket keys this conversation belongs to.
    profile_buckets: dict[str, tuple[str, ...]] = {
        profile.conversation_id: _session_bucket_keys(profile, granularity) for profile in filtered_profiles
    }

    sessions_per_bucket: dict[str, list[SessionProfileInsight]] = defaultdict(list)
    for profile in filtered_profiles:
        for bucket in profile_buckets[profile.conversation_id]:
            sessions_per_bucket[bucket].append(profile)

    work_events_per_bucket: dict[str, list[SessionWorkEventInsight]] = defaultdict(list)
    for event in work_events:
        if query.provider is not None and event.provider_name != query.provider:
            continue
        conversation_buckets = profile_buckets.get(event.conversation_id, ())
        for bucket in conversation_buckets:
            work_events_per_bucket[bucket].append(event)

    entries: list[ProductivityRollupEntry] = []
    total_sessions = 0
    total_work_events = 0
    total_untimed = 0

    for bucket_key in sorted(sessions_per_bucket):
        bucket_profiles = sessions_per_bucket[bucket_key]
        bucket_events = work_events_per_bucket.get(bucket_key, [])

        hour_histogram: Counter[int] = Counter()
        untimed = 0
        project_counter: Counter[str] = Counter()
        context_switches = 0
        evidence_ids: list[str] = []

        for profile in bucket_profiles:
            hour = _session_hour(profile)
            if hour is None:
                untimed += 1
            else:
                hour_histogram[hour] += 1
            context_switches += _session_context_switches(profile)
            paths = profile.evidence.cwd_paths if profile.evidence else ()
            if paths:
                for path in paths:
                    project_counter[path] += 1
            else:
                project_counter[_UNATTRIBUTED_PROJECT] += 1
            evidence_ids.append(profile.conversation_id)

        outcome_counter: Counter[str] = Counter()
        for event in bucket_events:
            outcome_counter[event.inference.kind] += 1
            evidence_ids.append(event.event_id)

        focus_share = _project_focus_share(project_counter)
        entry_caveats = _entry_caveats(
            granularity=granularity,
            session_count=len(bucket_profiles),
            work_event_count=len(bucket_events),
            untimed_session_count=untimed,
            project_focus_share=focus_share,
        )

        entries.append(
            ProductivityRollupEntry(
                granularity=granularity,
                bucket_key=bucket_key,
                session_count=len(bucket_profiles),
                work_event_count=len(bucket_events),
                untimed_session_count=untimed,
                hour_of_day_histogram=dict(sorted(hour_histogram.items())),
                project_focus_share=focus_share,
                context_switch_count=context_switches,
                outcome_breakdown=dict(sorted(outcome_counter.items())),
                evidence_inputs=tuple(evidence_ids),
                caveats=entry_caveats,
            )
        )
        total_sessions += len(bucket_profiles)
        total_work_events += len(bucket_events)
        total_untimed += untimed

    # Stable order: day/week ascending by bucket_key (lexicographic ISO is
    # chronological); project descending by session_count then name.
    if granularity == "project":
        entries.sort(key=lambda entry: (-entry.session_count, entry.bucket_key))

    if query.offset:
        entries = entries[query.offset :]
    if query.limit is not None:
        entries = entries[: query.limit]

    return ProductivityRollupInsight(
        granularity=granularity,
        entries=tuple(entries),
        total_sessions=total_sessions,
        total_work_events=total_work_events,
        total_untimed_sessions=total_untimed,
        provenance=ArchiveInsightProvenance(
            materializer_version=PRODUCTIVITY_ROLLUP_INSIGHT_VERSION,
            materialized_at=materialized_at,
            source_updated_at=None,
            source_sort_key=None,
        ),
    )


__all__ = [
    "BASELINE_CAVEATS",
    "PRODUCTIVITY_ROLLUP_INSIGHT_VERSION",
    "ProductivityRollupEntry",
    "ProductivityRollupInsight",
    "ProductivityRollupInsightQuery",
    "build_productivity_rollup_insight",
]

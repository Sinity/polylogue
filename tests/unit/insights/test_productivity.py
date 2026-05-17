"""Tests for the productivity rollup insight contract (#1134).

Covers four shapes:

1. Pure aggregation: ``build_productivity_rollup_insight`` correctly
   buckets sessions by day / week / project, counts hour-of-day, computes
   project focus share, counts context switches, and counts work-event
   outcomes.
2. First-class caveats: every envelope ships ``baseline_caveats``, and
   per-entry caveats fire on low evidence, untimed share, unattributed
   share, and missing work events.
3. Evidence linkage: every entry's ``evidence_inputs`` references the
   conversation_id / work-event ids that fed it, so a reader can drill
   into the source rows.
4. MCP envelope shape: registry-driven dispatch returns a single
   ``ProductivityRollupInsight`` consistent with day/week summary
   precedent (envelope with entries + total counters + provenance).
"""

from __future__ import annotations

from polylogue.insights.archive import SessionProfileInsight, SessionWorkEventInsight
from polylogue.insights.archive_models import (
    ArchiveInferenceProvenance,
    ArchiveInsightProvenance,
    SessionEvidencePayload,
    WorkEventEvidencePayload,
    WorkEventInferencePayload,
)
from polylogue.insights.productivity import (
    BASELINE_CAVEATS,
    PRODUCTIVITY_ROLLUP_INSIGHT_VERSION,
    ProductivityRollupInsight,
    ProductivityRollupInsightQuery,
    build_productivity_rollup_insight,
)
from polylogue.insights.registry import INSIGHT_REGISTRY

_DEFAULT_PROVENANCE = ArchiveInsightProvenance(
    materializer_version=1,
    materialized_at="2026-05-17T00:00:00+00:00",
    source_updated_at=None,
    source_sort_key=None,
)

_DEFAULT_INFERENCE_PROVENANCE = ArchiveInferenceProvenance(
    materializer_version=1,
    materialized_at="2026-05-17T00:00:00+00:00",
    source_updated_at=None,
    source_sort_key=None,
    inference_version=1,
    inference_family="test",
)


def _profile(
    *,
    conversation_id: str,
    first_message_at: str | None = "2026-05-12T09:30:00+00:00",
    canonical_session_date: str | None = "2026-05-12",
    cwd_paths: tuple[str, ...] = ("/repo/polylogue",),
    provider: str = "claude-code",
) -> SessionProfileInsight:
    evidence = SessionEvidencePayload(
        first_message_at=first_message_at,
        canonical_session_date=canonical_session_date,
        cwd_paths=cwd_paths,
    )
    return SessionProfileInsight(
        conversation_id=conversation_id,
        provider_name=provider,
        title=None,
        provenance=_DEFAULT_PROVENANCE,
        evidence=evidence,
    )


def _work_event(
    *,
    event_id: str,
    conversation_id: str,
    kind: str = "implementation",
    provider: str = "claude-code",
    event_index: int = 0,
) -> SessionWorkEventInsight:
    return SessionWorkEventInsight(
        event_id=event_id,
        conversation_id=conversation_id,
        provider_name=provider,
        event_index=event_index,
        provenance=_DEFAULT_PROVENANCE,
        inference_provenance=_DEFAULT_INFERENCE_PROVENANCE,
        evidence=WorkEventEvidencePayload(start_index=0, end_index=1),
        inference=WorkEventInferencePayload(kind=kind, summary="", confidence=0.5),
    )


class TestRegistryWiring:
    """Productivity rollups participate in the registry-driven dispatch."""

    def test_registered(self) -> None:
        assert "productivity_rollups" in INSIGHT_REGISTRY

    def test_registry_entry_routes_to_operations_method(self) -> None:
        entry = INSIGHT_REGISTRY["productivity_rollups"]
        assert entry.operations_method_name == "list_productivity_rollup_insights"
        assert entry.query_model is ProductivityRollupInsightQuery
        assert entry.json_key == "productivity_rollups"
        assert entry.readiness_exempt is True


class TestBaselineEnvelope:
    """Every envelope ships baseline caveats and a provenance record."""

    def test_baseline_caveats_present(self) -> None:
        insight = build_productivity_rollup_insight(
            profiles=[],
            work_events=[],
            query=ProductivityRollupInsightQuery(),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        assert isinstance(insight, ProductivityRollupInsight)
        assert insight.baseline_caveats == BASELINE_CAVEATS
        # Baseline disclaimer is not empty and references the off-screen
        # work limit explicitly — this is the contract callers depend on.
        assert any("off-screen" in caveat for caveat in insight.baseline_caveats)

    def test_provenance_uses_materializer_version(self) -> None:
        insight = build_productivity_rollup_insight(
            profiles=[],
            work_events=[],
            query=ProductivityRollupInsightQuery(),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        assert insight.materializer_version == PRODUCTIVITY_ROLLUP_INSIGHT_VERSION
        assert insight.provenance.materializer_version == PRODUCTIVITY_ROLLUP_INSIGHT_VERSION

    def test_empty_archive_yields_empty_entries(self) -> None:
        insight = build_productivity_rollup_insight(
            profiles=[],
            work_events=[],
            query=ProductivityRollupInsightQuery(),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        assert insight.entries == ()
        assert insight.total_sessions == 0
        assert insight.total_work_events == 0
        # Even with no entries the baseline disclaimer ships.
        assert insight.baseline_caveats == BASELINE_CAVEATS


class TestDayBucketAggregation:
    """Day granularity buckets by ISO date from the canonical session date."""

    def test_groups_sessions_by_day(self) -> None:
        profiles = [
            _profile(conversation_id="c1", canonical_session_date="2026-05-12"),
            _profile(conversation_id="c2", canonical_session_date="2026-05-12"),
            _profile(conversation_id="c3", canonical_session_date="2026-05-13"),
        ]
        insight = build_productivity_rollup_insight(
            profiles=profiles,
            work_events=[],
            query=ProductivityRollupInsightQuery(granularity="day"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        keys = [entry.bucket_key for entry in insight.entries]
        assert keys == ["2026-05-12", "2026-05-13"]
        by_day = {entry.bucket_key: entry for entry in insight.entries}
        assert by_day["2026-05-12"].session_count == 2
        assert by_day["2026-05-13"].session_count == 1
        assert insight.total_sessions == 3

    def test_hour_of_day_histogram_counts_distinct_hours(self) -> None:
        profiles = [
            _profile(
                conversation_id="c1",
                first_message_at="2026-05-12T09:00:00+00:00",
                canonical_session_date="2026-05-12",
            ),
            _profile(
                conversation_id="c2",
                first_message_at="2026-05-12T09:30:00+00:00",
                canonical_session_date="2026-05-12",
            ),
            _profile(
                conversation_id="c3",
                first_message_at="2026-05-12T14:00:00+00:00",
                canonical_session_date="2026-05-12",
            ),
        ]
        insight = build_productivity_rollup_insight(
            profiles=profiles,
            work_events=[],
            query=ProductivityRollupInsightQuery(granularity="day"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        entry = insight.entries[0]
        assert entry.hour_of_day_histogram == {9: 2, 14: 1}

    def test_untimed_sessions_excluded_from_histogram_but_tracked(self) -> None:
        profiles = [
            _profile(
                conversation_id="c1",
                first_message_at=None,
                canonical_session_date="2026-05-12",
            ),
            _profile(
                conversation_id="c2",
                first_message_at="2026-05-12T09:00:00+00:00",
                canonical_session_date="2026-05-12",
            ),
        ]
        insight = build_productivity_rollup_insight(
            profiles=profiles,
            work_events=[],
            query=ProductivityRollupInsightQuery(granularity="day"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        entry = insight.entries[0]
        assert entry.hour_of_day_histogram == {9: 1}
        assert entry.untimed_session_count == 1
        assert insight.total_untimed_sessions == 1


class TestProjectGranularity:
    """Project granularity buckets by recorded ``cwd_path``."""

    def test_session_with_multiple_cwd_paths_contributes_to_each(self) -> None:
        profiles = [
            _profile(
                conversation_id="c1",
                cwd_paths=("/repo/polylogue", "/repo/sinex"),
            ),
            _profile(
                conversation_id="c2",
                cwd_paths=("/repo/polylogue",),
            ),
        ]
        insight = build_productivity_rollup_insight(
            profiles=profiles,
            work_events=[],
            query=ProductivityRollupInsightQuery(granularity="project"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        keys = {entry.bucket_key: entry.session_count for entry in insight.entries}
        assert keys["/repo/polylogue"] == 2
        assert keys["/repo/sinex"] == 1

    def test_session_without_cwd_paths_buckets_as_unattributed(self) -> None:
        profiles = [_profile(conversation_id="c1", cwd_paths=())]
        insight = build_productivity_rollup_insight(
            profiles=profiles,
            work_events=[],
            query=ProductivityRollupInsightQuery(granularity="project"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        assert [entry.bucket_key for entry in insight.entries] == ["unattributed"]


class TestContextSwitchCount:
    """Context switches are an objective ceiling over distinct cwd_paths."""

    def test_no_switches_for_single_path(self) -> None:
        profiles = [_profile(conversation_id="c1", cwd_paths=("/repo/polylogue",))]
        insight = build_productivity_rollup_insight(
            profiles=profiles,
            work_events=[],
            query=ProductivityRollupInsightQuery(granularity="day"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        assert insight.entries[0].context_switch_count == 0

    def test_n_minus_one_switches_for_n_distinct_paths(self) -> None:
        profiles = [
            _profile(
                conversation_id="c1",
                cwd_paths=("/a", "/b", "/c"),
            ),
        ]
        insight = build_productivity_rollup_insight(
            profiles=profiles,
            work_events=[],
            query=ProductivityRollupInsightQuery(granularity="day"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        assert insight.entries[0].context_switch_count == 2


class TestOutcomeBreakdown:
    """Work-event kinds aggregate per-bucket and feed evidence_inputs."""

    def test_outcomes_grouped_by_kind(self) -> None:
        profiles = [
            _profile(conversation_id="c1", canonical_session_date="2026-05-12"),
        ]
        events = [
            _work_event(event_id="e1", conversation_id="c1", kind="implementation"),
            _work_event(event_id="e2", conversation_id="c1", kind="implementation", event_index=1),
            _work_event(event_id="e3", conversation_id="c1", kind="debugging", event_index=2),
        ]
        insight = build_productivity_rollup_insight(
            profiles=profiles,
            work_events=events,
            query=ProductivityRollupInsightQuery(granularity="day"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        entry = insight.entries[0]
        assert entry.outcome_breakdown == {"debugging": 1, "implementation": 2}
        assert entry.work_event_count == 3

    def test_evidence_inputs_reference_source_ids(self) -> None:
        profiles = [_profile(conversation_id="c1", canonical_session_date="2026-05-12")]
        events = [_work_event(event_id="e1", conversation_id="c1")]
        insight = build_productivity_rollup_insight(
            profiles=profiles,
            work_events=events,
            query=ProductivityRollupInsightQuery(granularity="day"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        entry = insight.entries[0]
        # Evidence inputs are the conversation_id and event_id that fed the bucket.
        assert "c1" in entry.evidence_inputs
        assert "e1" in entry.evidence_inputs


class TestCaveatInvariants:
    """Caveats are first-class — every entry carries them when applicable."""

    def test_low_evidence_caveat_fires_under_three_sessions(self) -> None:
        profiles = [_profile(conversation_id="c1", canonical_session_date="2026-05-12")]
        insight = build_productivity_rollup_insight(
            profiles=profiles,
            work_events=[],
            query=ProductivityRollupInsightQuery(granularity="day"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        caveats = insight.entries[0].caveats
        assert any("Low evidence" in caveat for caveat in caveats)

    def test_missing_work_events_caveat_fires(self) -> None:
        profiles = [_profile(conversation_id=f"c{idx}", canonical_session_date="2026-05-12") for idx in range(5)]
        insight = build_productivity_rollup_insight(
            profiles=profiles,
            work_events=[],
            query=ProductivityRollupInsightQuery(granularity="day"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        caveats = insight.entries[0].caveats
        assert any("No work events" in caveat for caveat in caveats)

    def test_unattributed_share_caveat_fires(self) -> None:
        profiles = [
            _profile(conversation_id=f"c{idx}", canonical_session_date="2026-05-12", cwd_paths=()) for idx in range(4)
        ]
        # Add a work event so the missing-work-events caveat does not fire,
        # isolating the unattributed-share caveat as the observed signal.
        events = [_work_event(event_id="e1", conversation_id="c0")]
        insight = build_productivity_rollup_insight(
            profiles=profiles,
            work_events=events,
            query=ProductivityRollupInsightQuery(granularity="day"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        caveats = insight.entries[0].caveats
        assert any("cwd_paths" in caveat for caveat in caveats)

    def test_untimed_share_caveat_fires(self) -> None:
        profiles = [
            _profile(conversation_id=f"c{idx}", canonical_session_date="2026-05-12", first_message_at=None)
            for idx in range(3)
        ]
        # Add one timed session and one work event so the bucket is not
        # entirely empty.
        profiles.append(
            _profile(
                conversation_id="ctimed",
                canonical_session_date="2026-05-12",
                first_message_at="2026-05-12T10:00:00+00:00",
            )
        )
        events = [_work_event(event_id="e1", conversation_id="ctimed")]
        insight = build_productivity_rollup_insight(
            profiles=profiles,
            work_events=events,
            query=ProductivityRollupInsightQuery(granularity="day"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        caveats = insight.entries[0].caveats
        assert any("lack timestamps" in caveat for caveat in caveats)

    def test_baseline_caveats_never_dropped_for_nonempty_envelope(self) -> None:
        profiles = [_profile(conversation_id="c1", canonical_session_date="2026-05-12")]
        insight = build_productivity_rollup_insight(
            profiles=profiles,
            work_events=[],
            query=ProductivityRollupInsightQuery(granularity="day"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        assert insight.baseline_caveats == BASELINE_CAVEATS


class TestProviderAndWindowFiltering:
    """Provider and time-window filters narrow the input set consistently."""

    def test_provider_filter_excludes_other_providers(self) -> None:
        profiles = [
            _profile(conversation_id="c1", provider="claude-code"),
            _profile(conversation_id="c2", provider="codex"),
        ]
        insight = build_productivity_rollup_insight(
            profiles=profiles,
            work_events=[],
            query=ProductivityRollupInsightQuery(granularity="day", provider="claude-code"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        assert insight.total_sessions == 1
        assert "c1" in insight.entries[0].evidence_inputs
        assert "c2" not in insight.entries[0].evidence_inputs

    def test_since_until_window(self) -> None:
        profiles = [
            _profile(conversation_id="c1", canonical_session_date="2026-05-12"),
            _profile(conversation_id="c2", canonical_session_date="2026-05-15"),
            _profile(conversation_id="c3", canonical_session_date="2026-05-20"),
        ]
        insight = build_productivity_rollup_insight(
            profiles=profiles,
            work_events=[],
            query=ProductivityRollupInsightQuery(
                granularity="day",
                since="2026-05-13",
                until="2026-05-19",
            ),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        assert insight.total_sessions == 1
        assert insight.entries[0].bucket_key == "2026-05-15"


class TestWeekGranularity:
    """Week granularity buckets by ISO week."""

    def test_iso_week_bucket_key(self) -> None:
        profiles = [
            _profile(conversation_id="c1", canonical_session_date="2026-05-11"),  # Mon, W20
            _profile(conversation_id="c2", canonical_session_date="2026-05-17"),  # Sun, W20
            _profile(conversation_id="c3", canonical_session_date="2026-05-18"),  # Mon, W21
        ]
        insight = build_productivity_rollup_insight(
            profiles=profiles,
            work_events=[],
            query=ProductivityRollupInsightQuery(granularity="week"),
            materialized_at="2026-05-17T00:00:00+00:00",
        )
        keys = {entry.bucket_key: entry.session_count for entry in insight.entries}
        assert keys == {"2026-W20": 2, "2026-W21": 1}

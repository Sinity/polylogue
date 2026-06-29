"""Shared runtime contracts for session-insight rebuild and refresh flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from typing_extensions import TypedDict

ProviderDayGroup: TypeAlias = tuple[str, str]
SessionInsightReadyFlag: TypeAlias = Literal[
    "profile_rows_ready",
    "latency_profile_rows_ready",
    "profile_merged_fts_ready",
    "profile_evidence_fts_ready",
    "profile_inference_fts_ready",
    "profile_enrichment_fts_ready",
    "work_event_inference_rows_ready",
    "work_event_inference_fts_ready",
    "phase_inference_rows_ready",
    "phase_rows_ready",
    "run_rows_ready",
    "observed_event_rows_ready",
    "context_snapshot_rows_ready",
    "threads_ready",
    "threads_fts_ready",
    "tag_rollups_ready",
]

SESSION_INSIGHT_MATERIALIZATION_TYPES: tuple[str, ...] = (
    "session_profile",
    "latency",
    "work_events",
    "phases",
    "thread",
    "runs",
    "observed_events",
    "context_snapshots",
)


class SessionInsightRefreshChunkPayload(TypedDict):
    session_count: int
    estimated_message_count: int
    max_estimated_session_messages: int
    hydrated_count: int
    profiles_written: int
    work_events_written: int
    phases_written: int
    load_ms: float
    hydrate_ms: float
    build_ms: float
    write_ms: float
    total_ms: float
    slow: bool


@dataclass(slots=True)
class SessionInsightCounts:
    profiles: int = 0
    work_events: int = 0
    phases: int = 0
    threads: int = 0
    tag_rollups: int = 0

    def add(
        self,
        *,
        profiles: int = 0,
        work_events: int = 0,
        phases: int = 0,
        threads: int = 0,
        tag_rollups: int = 0,
    ) -> None:
        self.profiles += profiles
        self.work_events += work_events
        self.phases += phases
        self.threads += threads
        self.tag_rollups += tag_rollups

    def to_dict(self) -> dict[str, int]:
        return {
            "profiles": self.profiles,
            "work_events": self.work_events,
            "phases": self.phases,
            "threads": self.threads,
            "tag_rollups": self.tag_rollups,
        }

    def total(self) -> int:
        return sum(self.to_dict().values())


@dataclass(slots=True, frozen=True)
class SessionInsightStatusSnapshot:
    """Readiness snapshot for session-insight materializations.

    Lightweight status calls may skip expensive freshness verification. In that
    mode, `root_threads` falls back to `thread_count`, and ready flags derived
    from freshness-gated counts can be approximate. Use a verified status call
    for health checks that must prove materialized tables are current.
    """

    total_sessions: int = 0
    root_threads: int = 0
    profile_row_count: int = 0
    latency_profile_row_count: int = 0
    work_event_inference_count: int = 0
    work_event_inference_fts_count: int = 0
    work_event_inference_fts_duplicate_count: int = 0
    phase_inference_count: int = 0
    run_count: int = 0
    observed_event_count: int = 0
    context_snapshot_count: int = 0
    thread_count: int = 0
    thread_fts_count: int = 0
    thread_fts_duplicate_count: int = 0
    tag_rollup_count: int = 0
    day_summary_count: int = 0
    missing_profile_row_count: int = 0
    missing_session_profile_materialization_count: int = 0
    stale_profile_row_count: int = 0
    orphan_profile_row_count: int = 0
    missing_latency_profile_row_count: int = 0
    missing_latency_materialization_count: int = 0
    stale_latency_profile_row_count: int = 0
    orphan_latency_profile_row_count: int = 0
    missing_work_event_materialization_count: int = 0
    expected_work_event_inference_count: int = 0
    stale_work_event_inference_count: int = 0
    orphan_work_event_inference_count: int = 0
    missing_phase_materialization_count: int = 0
    expected_phase_inference_count: int = 0
    stale_phase_inference_count: int = 0
    orphan_phase_inference_count: int = 0
    missing_run_materialization_count: int = 0
    missing_observed_event_materialization_count: int = 0
    missing_context_snapshot_materialization_count: int = 0
    missing_thread_materialization_count: int = 0
    stale_thread_count: int = 0
    orphan_thread_count: int = 0
    expected_tag_rollup_count: int = 0
    stale_tag_rollup_count: int = 0
    expected_day_summary_count: int = 0
    stale_day_summary_count: int = 0
    profile_evidence_fts_count: int = 0
    profile_evidence_fts_duplicate_count: int = 0
    profile_inference_fts_count: int = 0
    profile_inference_fts_duplicate_count: int = 0
    profile_enrichment_fts_count: int = 0
    profile_enrichment_fts_duplicate_count: int = 0
    profile_merged_fts_count: int = 0
    profile_merged_fts_duplicate_count: int = 0
    profile_rows_ready: bool = False
    latency_profile_rows_ready: bool = False
    profile_merged_fts_ready: bool = False
    profile_evidence_fts_ready: bool = False
    profile_inference_fts_ready: bool = False
    profile_enrichment_fts_ready: bool = False
    work_event_inference_rows_ready: bool = False
    work_event_inference_fts_ready: bool = False
    phase_inference_rows_ready: bool = False
    run_rows_ready: bool = False
    observed_event_rows_ready: bool = False
    context_snapshot_rows_ready: bool = False
    threads_ready: bool = False
    threads_fts_ready: bool = False
    tag_rollups_ready: bool = False

    def ready_flag(self, key: SessionInsightReadyFlag) -> bool:
        return bool(getattr(self, key))

    @property
    def phase_count(self) -> int:
        """Evidence-tier session phase row count.

        ``phase_inference_count`` is the historical storage/status field name.
        Public readers should prefer this alias: phases are deterministic
        time-gap intervals, not a probabilistic phase-kind inference surface.
        """

        return self.phase_inference_count

    @property
    def expected_phase_count(self) -> int:
        return self.expected_phase_inference_count

    @property
    def stale_phase_count(self) -> int:
        return self.stale_phase_inference_count

    @property
    def orphan_phase_count(self) -> int:
        return self.orphan_phase_inference_count

    @property
    def phase_rows_ready(self) -> bool:
        return self.phase_inference_rows_ready

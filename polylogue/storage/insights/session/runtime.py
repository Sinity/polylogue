"""Shared runtime contracts for session-product rebuild and refresh flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from typing_extensions import TypedDict

ProviderDayGroup: TypeAlias = tuple[str, str]
SessionInsightReadyFlag: TypeAlias = Literal[
    "profile_rows_ready",
    "profile_merged_fts_ready",
    "profile_evidence_fts_ready",
    "profile_inference_fts_ready",
    "profile_enrichment_fts_ready",
    "work_event_inference_rows_ready",
    "work_event_inference_fts_ready",
    "phase_inference_rows_ready",
    "threads_ready",
    "threads_fts_ready",
    "tag_rollups_ready",
    "day_summaries_ready",
    "week_summaries_ready",
]


class SessionInsightRefreshChunkPayload(TypedDict):
    conversation_count: int
    estimated_message_count: int
    max_estimated_conversation_messages: int
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
    day_summaries: int = 0

    def add(
        self,
        *,
        profiles: int = 0,
        work_events: int = 0,
        phases: int = 0,
        threads: int = 0,
        tag_rollups: int = 0,
        day_summaries: int = 0,
    ) -> None:
        self.profiles += profiles
        self.work_events += work_events
        self.phases += phases
        self.threads += threads
        self.tag_rollups += tag_rollups
        self.day_summaries += day_summaries

    def to_dict(self) -> dict[str, int]:
        return {
            "profiles": self.profiles,
            "work_events": self.work_events,
            "phases": self.phases,
            "threads": self.threads,
            "tag_rollups": self.tag_rollups,
            "day_summaries": self.day_summaries,
        }

    def total(self) -> int:
        return sum(self.to_dict().values())


@dataclass(slots=True, frozen=True)
class SessionInsightStatusSnapshot:
    total_conversations: int = 0
    root_threads: int = 0
    profile_row_count: int = 0
    profile_merged_fts_count: int = 0
    profile_merged_fts_duplicate_count: int = 0
    profile_evidence_fts_count: int = 0
    profile_evidence_fts_duplicate_count: int = 0
    profile_inference_fts_count: int = 0
    profile_inference_fts_duplicate_count: int = 0
    profile_enrichment_fts_count: int = 0
    profile_enrichment_fts_duplicate_count: int = 0
    work_event_inference_count: int = 0
    work_event_inference_fts_count: int = 0
    work_event_inference_fts_duplicate_count: int = 0
    phase_inference_count: int = 0
    thread_count: int = 0
    thread_fts_count: int = 0
    thread_fts_duplicate_count: int = 0
    tag_rollup_count: int = 0
    day_summary_count: int = 0
    missing_profile_row_count: int = 0
    stale_profile_row_count: int = 0
    orphan_profile_row_count: int = 0
    expected_work_event_inference_count: int = 0
    stale_work_event_inference_count: int = 0
    orphan_work_event_inference_count: int = 0
    expected_phase_inference_count: int = 0
    stale_phase_inference_count: int = 0
    orphan_phase_inference_count: int = 0
    stale_thread_count: int = 0
    orphan_thread_count: int = 0
    expected_tag_rollup_count: int = 0
    stale_tag_rollup_count: int = 0
    expected_day_summary_count: int = 0
    stale_day_summary_count: int = 0
    profile_rows_ready: bool = False
    profile_merged_fts_ready: bool = False
    profile_evidence_fts_ready: bool = False
    profile_inference_fts_ready: bool = False
    profile_enrichment_fts_ready: bool = False
    work_event_inference_rows_ready: bool = False
    work_event_inference_fts_ready: bool = False
    phase_inference_rows_ready: bool = False
    threads_ready: bool = False
    threads_fts_ready: bool = False
    tag_rollups_ready: bool = False
    day_summaries_ready: bool = False
    week_summaries_ready: bool = False

    def ready_flag(self, key: SessionInsightReadyFlag) -> bool:
        return bool(getattr(self, key))

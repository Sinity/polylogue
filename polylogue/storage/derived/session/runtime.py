"""Shared runtime contracts for session-insight rebuild and refresh flows."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from typing_extensions import TypedDict

ProviderDayGroup: TypeAlias = tuple[str, str]


def session_profile_candidates(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
    *,
    materializer_version: int,
) -> list[str]:
    """Sessions whose partition is not valid, by value-complete inspection.

    Re-exported here because this module is the daemon converger's declared
    window onto session-insight runtime; the implementation belongs to the
    domain (:mod:`polylogue.storage.derived.session.derivation`).
    """
    from polylogue.storage.derived.session.derivation import inspect_session_profiles

    statuses = inspect_session_profiles(conn, session_ids, materializer_version=materializer_version)
    return sorted(session_id for session_id, status in statuses.items() if status != "valid")


class SessionInsightRefreshChunkPayload(TypedDict):
    session_count: int
    estimated_message_count: int
    max_estimated_session_messages: int
    hydrated_count: int
    profiles_written: int
    load_ms: float
    hydrate_ms: float
    build_ms: float
    write_ms: float
    total_ms: float
    slow: bool


@dataclass(slots=True)
class SessionInsightCounts:
    profiles: int = 0
    threads: int = 0
    tag_rollups: int = 0

    def add(
        self,
        *,
        profiles: int = 0,
        threads: int = 0,
        tag_rollups: int = 0,
    ) -> None:
        self.profiles += profiles
        self.threads += threads
        self.tag_rollups += tag_rollups

    def to_dict(self) -> dict[str, int]:
        return {
            "profiles": self.profiles,
            "threads": self.threads,
            "tag_rollups": self.tag_rollups,
        }

    def total(self) -> int:
        return sum(self.to_dict().values())


@dataclass(slots=True, frozen=True)
class SessionInsightStatusSnapshot:
    """Row-count and integrity snapshot for session insight tables.

    Lightweight status calls may skip expensive freshness verification. In that
    mode, `root_threads` falls back to `thread_count`; convergence debt remains
    the authoritative readiness signal.
    """

    total_sessions: int = 0
    root_threads: int = 0
    profile_row_count: int = 0
    latency_profile_row_count: int = 0
    run_count: int = 0
    observed_event_count: int = 0
    context_snapshot_count: int = 0
    thread_count: int = 0
    tag_rollup_count: int = 0
    provider_usage_row_count: int = 0
    missing_profile_row_count: int = 0
    stale_profile_row_count: int = 0
    orphan_profile_row_count: int = 0
    missing_latency_profile_row_count: int = 0
    stale_latency_profile_row_count: int = 0
    orphan_latency_profile_row_count: int = 0
    stale_thread_count: int = 0
    orphan_thread_count: int = 0
    expected_tag_rollup_count: int = 0
    expected_provider_usage_row_count: int = 0
    missing_provider_usage_row_count: int = 0
    stale_tag_rollup_count: int = 0
    profile_evidence_fts_count: int = 0
    profile_evidence_fts_duplicate_count: int = 0
    profile_inference_fts_count: int = 0
    profile_inference_fts_duplicate_count: int = 0
    profile_enrichment_fts_count: int = 0
    profile_enrichment_fts_duplicate_count: int = 0
    profile_merged_fts_count: int = 0
    profile_merged_fts_duplicate_count: int = 0

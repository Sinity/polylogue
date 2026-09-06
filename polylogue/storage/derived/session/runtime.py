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
    """Sessions whose profile is not valid, by value-complete inspection.

    Re-exported here because this module is the daemon converger's declared
    window onto session-insight runtime; the implementation belongs to the
    domain (:mod:`polylogue.storage.derived.session.derivation`).

    This is authority. :func:`session_profile_stale_predicate` below is a cheap
    SQL prefilter over identity only, and identity does not move when a role, a
    model name, or a token count does.
    """
    from polylogue.storage.derived.session.derivation import inspect_session_profiles

    statuses = inspect_session_profiles(conn, session_ids, materializer_version=materializer_version)
    return sorted(session_id for session_id, status in statuses.items() if status != "valid")


def session_profile_stale_predicate(
    sessions_alias: str,
    profile_alias: str,
) -> str:
    """SQL boolean fragment: a cheap identity prefilter, never authority.

    Narrows candidates before the value-complete inspection in
    :func:`session_profile_candidates` reads them. On its own it cannot see a
    changed role, model, or token count, which is exactly the defect the
    value-complete binding exists to close; do not use it to certify freshness.

    True when ``profile_alias``'s cached sort key is
    stale relative to ``sessions_alias``.

    Single source of truth for the sort-key staleness comparison, shared by
    the daemon converger (``daemon/convergence_stages.py``), ops repair
    (``storage/raw_convergence.py``), and the candidate prefilter
    (``storage/derived/session/status.py``) — see polylogue-a7xr.2.

    ``sessions_alias.sort_key_ms`` is milliseconds; ``profile_alias`` caches it
    in ``source_sort_key`` seconds, so the common case compares the two within
    a microsecond epsilon after unit conversion.

    ``sort_key_ms`` can be NULL — a session with no derivable temporal sort
    key (the "timeless session" case). There is then no numeric sort key to
    compare, so staleness instead compares the profile's cached
    ``source_updated_at`` against the session's ``updated_at_ms`` (both
    reduced to whole seconds). This NULL-branch semantics is deliberately the
    converger's original choice, not repair's: repair previously and
    independently reimplemented this branch by COALESCEing the missing sort
    key to ``0.0`` and comparing it against ``source_sort_key``, which
    permanently flagged any NULL-``sort_key_ms`` session with a nonzero cached
    ``source_sort_key`` as stale — causing repair to re-flag rows the
    converger already considered fresh (repeated churn) or vice versa
    (missed rebuilds).
    """
    return (
        "(\n"
        f"    ({sessions_alias}.sort_key_ms IS NOT NULL\n"
        f"     AND ABS(COALESCE({profile_alias}.source_sort_key, 0.0) - "
        f"(CAST({sessions_alias}.sort_key_ms AS REAL) / 1000.0)) > 0.000001)\n"
        "    OR\n"
        f"    ({sessions_alias}.sort_key_ms IS NULL\n"
        f"     AND COALESCE(strftime('%s', {profile_alias}.source_updated_at), "
        f"{profile_alias}.source_updated_at, '') != "
        f"COALESCE(CAST({sessions_alias}.updated_at_ms / 1000 AS TEXT), ''))\n"
        ")"
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
    """Row-count and integrity snapshot for session insight tables.

    Lightweight status calls may skip expensive freshness verification. In that
    mode, `root_threads` falls back to `thread_count`; convergence debt remains
    the authoritative readiness signal.
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
    tag_rollup_count: int = 0
    day_summary_count: int = 0
    missing_profile_row_count: int = 0
    stale_profile_row_count: int = 0
    orphan_profile_row_count: int = 0
    missing_latency_profile_row_count: int = 0
    stale_latency_profile_row_count: int = 0
    orphan_latency_profile_row_count: int = 0
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
    profile_evidence_fts_count: int = 0
    profile_evidence_fts_duplicate_count: int = 0
    profile_inference_fts_count: int = 0
    profile_inference_fts_duplicate_count: int = 0
    profile_enrichment_fts_count: int = 0
    profile_enrichment_fts_duplicate_count: int = 0
    profile_merged_fts_count: int = 0
    profile_merged_fts_duplicate_count: int = 0

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

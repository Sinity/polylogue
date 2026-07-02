"""Fallback markers for degraded insight materialization (#1278).

When an insight is computed via a degraded path — heuristic instead of
measured evidence, default categorization instead of detected one,
session-total instead of phase-sum, etc. — the materializer records a
typed `FallbackReason` on the payload. The readiness report aggregates
these markers and reports a `degraded` verdict so downstream consumers
can distinguish a fully-evidenced row from a reconstructed one.

The closed `FallbackReason` enum is the durable taxonomy. New degradation
modes require an enum addition; ad-hoc strings are rejected at the
payload boundary.
"""

from __future__ import annotations

from enum import Enum


class FallbackReason(str, Enum):
    """Why an insight value was produced by a degraded path.

    Members are grouped by materialization scope:

    Profile inference (``SessionInferencePayload.fallback_reasons``):
      - ``ENGAGED_DURATION_SESSION_TOTAL`` — engaged duration computed
        from session totals because no phase had a positive duration.
      - ``NO_WORK_EVENTS_AND_NO_PHASES`` — the profile carries no
        work-event or phase rows, so support level uses a synthetic
        fallback floor.
      - ``ALL_WORK_EVENTS_WEAK`` — every materialized work event was
        emitted with weak/no evidence markers.

    Profile enrichment (``SessionEnrichmentPayload.fallback_reasons``):
      - ``MISSING_SESSION_ANALYSIS`` — enrichment built without a
        ``SessionAnalysis`` (no live message bands), so intent / outcome
        strings come from heuristic projection over the profile.
      - ``NO_USER_TURNS`` — enrichment had no user turns to summarize
        intent from.

    Work-event inference (``WorkEventInferencePayload.fallback_reasons``):
      - ``WORK_EVENT_NO_EVIDENCE`` — the work event was emitted with no
        evidence markers at all.
      - ``WORK_EVENT_WEAK_MARKERS`` — the work event carries only weak
        evidence markers (``weak_signal``, ``no_tools``,
        ``shell_default``).

    Large-session bounded materialization:
      - ``LARGE_SESSION_BOUNDED`` — the session exceeded the full semantic
        materialization threshold, so profile rows were built from durable
        archive counters and session metadata instead of hydrating every
        message/block into memory.

    Legacy phase compatibility markers:
      - ``ALL_PHASES_HEURISTIC`` and ``PHASE_NO_TOOL_COUNTS`` are retained
        so historical payloads can still validate. New phase rows are
        evidence intervals, not phase-kind classifier output.
    """

    ENGAGED_DURATION_SESSION_TOTAL = "engaged_duration_session_total"
    NO_WORK_EVENTS_AND_NO_PHASES = "no_work_events_and_no_phases"
    ALL_WORK_EVENTS_WEAK = "all_work_events_weak"
    ALL_PHASES_HEURISTIC = "all_phases_heuristic"
    MISSING_SESSION_ANALYSIS = "missing_session_analysis"
    NO_USER_TURNS = "no_user_turns"
    LARGE_SESSION_BOUNDED = "large_session_bounded"
    WORK_EVENT_NO_EVIDENCE = "work_event_no_evidence"
    WORK_EVENT_WEAK_MARKERS = "work_event_weak_markers"
    PHASE_NO_TOOL_COUNTS = "phase_no_tool_counts"


__all__ = ["FallbackReason"]

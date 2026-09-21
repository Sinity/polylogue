"""Fallback markers for degraded insight materialization (#1278).

When an insight is computed via a degraded path — heuristic instead of
measured evidence, default categorization instead of detected one, a
projection over the profile instead of live message bands — the materializer
records a typed `FallbackReason` on the payload. The readiness report
aggregates these markers and reports a `degraded` verdict so downstream
consumers can distinguish a fully-evidenced row from a reconstructed one.

The closed `FallbackReason` enum is the durable taxonomy. New degradation
modes require an enum addition; ad-hoc strings are rejected at the
payload boundary.
"""

from __future__ import annotations

from enum import Enum


class FallbackReason(str, Enum):
    """Why an insight value was produced by a degraded path.

    Members are grouped by materialization scope:

    Profile enrichment (``SessionEnrichmentPayload.fallback_reasons``):
      - ``MISSING_SESSION_ANALYSIS`` — enrichment built without a
        ``SessionAnalysis`` (no live message bands), so intent / outcome
        strings come from heuristic projection over the profile.
      - ``NO_USER_TURNS`` — enrichment had no user turns to summarize
        intent from.

    Large-session bounded materialization:
      - ``LARGE_SESSION_BOUNDED`` — the session exceeded the full semantic
        materialization threshold, so profile rows were built from durable
        archive counters and session metadata instead of hydrating every
        message/block into memory.

    polylogue-cuxz.7 retired the work-event/phase members
    (``ENGAGED_DURATION_SESSION_TOTAL``, ``NO_WORK_EVENTS_AND_NO_PHASES``,
    ``ALL_WORK_EVENTS_WEAK``, ``ALL_PHASES_HEURISTIC``,
    ``WORK_EVENT_NO_EVIDENCE``, ``WORK_EVENT_WEAK_MARKERS``,
    ``PHASE_NO_TOOL_COUNTS``) with the two tables they described. No
    materializer can emit them, and index.db is rebuilt from source.db at
    INDEX_SCHEMA_VERSION 104, so no stored payload carries them either.
    """

    MISSING_SESSION_ANALYSIS = "missing_session_analysis"
    NO_USER_TURNS = "no_user_turns"
    LARGE_SESSION_BOUNDED = "large_session_bounded"


__all__ = ["FallbackReason"]

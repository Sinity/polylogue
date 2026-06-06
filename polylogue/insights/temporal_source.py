"""Temporal source taxonomy for insight provenance (#1276).

Every materialized insight row records an ``input_high_water_mark`` — the
latest source change timestamp folded into that row. The HWM is itself a
timestamp, and the *clock* it was sampled from carries meaning that a
downstream reader cannot recover from the value alone. A timestamp that
came from a provider's wire payload, a hook event captured at request
time, the message sort key, the source file mtime, the moment the row
was materialized, or a synthetic fallback date are all valid HWMs, but
they have different reliability and recency semantics.

The :data:`TemporalSource` literal type names the six recognized sources.
Insight builders set the value at materialization time so the row carries
both the timestamp and an explicit answer to "which clock produced this
value?".

Recognized values:

- ``provider_ts``: a timestamp the upstream provider attached to the
  session, message, or event (e.g. ChatGPT ``update_time``, Claude
  message ``created_at``). The strongest signal — anchored to provider
  wall-clock semantics.
- ``hook_event_ts``: a timestamp captured by a session-lifecycle hook
  script at the moment the event fired (``SessionStart``,
  ``PreToolUse``, etc.). High fidelity for events the provider does not
  directly timestamp.
- ``sort_key``: the message sort key carried by the archive substrate.
  Used when the only chronological signal available is the per-message
  sort key (e.g. legacy imports where wall-clock was lost).
- ``file_mtime``: the filesystem mtime of the source artifact at parse
  time. The weakest "real" signal — survives normalization but reflects
  filesystem state, not provider intent.
- ``materialization_ts``: the wall-clock instant the insight row was
  materialized. Used only when no source-anchored timestamp is recoverable
  and the row's freshness is defined by the materializer itself.
- ``fallback_date``: a synthetic placeholder produced by deterministic
  fallback logic (e.g. canonical session date inferred from an adjacent
  signal). Tagged so readers can suppress these rows when only
  source-anchored evidence is acceptable.

This taxonomy intentionally only describes the *source clock* of a
timestamp — it is orthogonal to existing ``timing_provenance`` (which
describes the *coverage shape*: ``timestamped_range`` vs
``untimestamped`` vs ``start_timestamp_only`` vs ``end_timestamp_only``)
and to ``date_provenance`` (which describes how a canonical session date
was derived). The three axes coexist and answer different questions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, get_args

TemporalSource = Literal[
    "provider_ts",
    "hook_event_ts",
    "sort_key",
    "file_mtime",
    "materialization_ts",
    "fallback_date",
]

TEMPORAL_SOURCE_VALUES: frozenset[TemporalSource] = frozenset(get_args(TemporalSource))


def classify_profile_hwm_source(updated_at: datetime | None) -> TemporalSource:
    """Classify the temporal source of a per-session insight HWM.

    Per-session insight rows (session profiles, work events,
    phases) derive their HWM from the session's ``updated_at``,
    which comes from the provider parser. When the provider supplied an
    explicit timestamp this is ``provider_ts``; otherwise the
    materialization has nothing source-anchored to record and falls
    back to ``materialization_ts`` semantics via the materialized_at
    column. A ``None`` HWM is tagged ``fallback_date`` so the row
    explicitly states that the HWM is a placeholder.
    """

    if updated_at is None:
        return "fallback_date"
    return "provider_ts"


def classify_thread_hwm_source(end_time: datetime | None) -> TemporalSource:
    """Classify the temporal source of a work-thread insight HWM.

    Threads aggregate over their member sessions and record the latest
    member ``end_time`` as the HWM. ``end_time`` is itself sourced from
    provider timestamps when present.
    """

    if end_time is None:
        return "fallback_date"
    return "provider_ts"


def classify_aggregate_hwm_source(source_updates: list[str]) -> TemporalSource:
    """Classify the temporal source of an aggregate insight HWM.

    Aggregates (day summaries, tag rollups) compute their HWM as the
    max over the per-session ``updated_at`` values that contributed.
    Each input is itself a ``provider_ts`` (per the per-session
    classifier above), so the max is also ``provider_ts``.
    """

    if not source_updates:
        return "fallback_date"
    return "provider_ts"


def is_valid_temporal_source(value: str) -> bool:
    """Return True when *value* is a recognized temporal source token."""

    return value in TEMPORAL_SOURCE_VALUES


__all__ = [
    "TEMPORAL_SOURCE_VALUES",
    "TemporalSource",
    "classify_aggregate_hwm_source",
    "classify_profile_hwm_source",
    "classify_thread_hwm_source",
    "is_valid_temporal_source",
]

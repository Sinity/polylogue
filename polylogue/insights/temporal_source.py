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

import re
from collections.abc import Sequence
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

# Provenance lattice, strongest first. A lower rank is a stronger (more
# trustworthy/recency-anchored) signal; ``weakest_source`` picks the higher
# rank between two values. Order matches the docstring above verbatim.
_SOURCE_RANK: dict[TemporalSource, int] = {source: rank for rank, source in enumerate(get_args(TemporalSource))}


def weakest_source(a: TemporalSource, b: TemporalSource) -> TemporalSource:
    """Return the weaker (less trustworthy) of two temporal sources."""

    return a if _SOURCE_RANK[a] >= _SOURCE_RANK[b] else b


def weakest_of(sources: Sequence[TemporalSource]) -> TemporalSource | None:
    """Reduce a non-empty sequence to its weakest member; ``None`` if empty."""

    if not sources:
        return None
    result = sources[0]
    for source in sources[1:]:
        result = weakest_source(result, source)
    return result


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
    """Classify the temporal source of a thread insight HWM.

    Threads aggregate over their member sessions and record the latest
    member ``end_time`` as the HWM. ``end_time`` is itself sourced from
    provider timestamps when present.
    """

    if end_time is None:
        return "fallback_date"
    return "provider_ts"


def classify_aggregate_hwm_source(contributor_sources: Sequence[TemporalSource]) -> TemporalSource:
    """Classify the temporal source of an aggregate insight HWM.

    Aggregates (day summaries, tag rollups) fold together timestamps from
    many contributing sessions. The aggregate's HWM is only as trustworthy
    as its *weakest* contributor: a single session whose HWM fell back to
    ``fallback_date`` (or any weaker source) must not be laundered into
    ``provider_ts`` just because other contributors had real provider
    timestamps. Callers pass each contributing session's own classified
    source (e.g. ``classify_profile_hwm_source(profile.updated_at)``), not
    raw timestamp strings — this function has no way to judge provenance
    from a bare date string, which is exactly the bug this replaces.
    """

    weakest = weakest_of(list(contributor_sources))
    return weakest if weakest is not None else "fallback_date"


_LEAF_CALLER_CONTRACTS: dict[str, re.Pattern[str]] = {
    # classify_profile_hwm_source is justified ONLY when called with the raw
    # provider-parsed Session.updated_at field (never backfilled from a
    # weaker clock). classify_thread_hwm_source: same contract for end_time.
    "classify_profile_hwm_source": re.compile(r"classify_profile_hwm_source\(\s*\w+\.updated_at\s*\)"),
    "classify_thread_hwm_source": re.compile(r"classify_thread_hwm_source\(\s*\w+\.end_time\s*\)"),
}


def audit_temporal_source_leaf_callers(package_root: str) -> list[str]:
    """Flag leaf-classifier call sites that do not pass the field they claim to.

    ``classify_profile_hwm_source``/``classify_thread_hwm_source`` are
    correct only because every caller passes the exact field the docstring
    justifies (``<x>.updated_at``, ``<x>.end_time`` — raw provider-parsed
    fields, never backfilled from a weaker clock). If a future call site
    passes a different field while still expecting ``provider_ts``
    semantics, that is an unjustifiable provider_ts path. Scans every
    ``.py`` file under *package_root* (this module's own definitions are
    skipped by filename); an empty result means every call site found is
    justified. Finding zero call sites at all is itself reported — it
    means the contract can no longer be checked, not that it is satisfied.
    """

    import pathlib

    violations: list[str] = []
    call_site_counts: dict[str, int] = dict.fromkeys(_LEAF_CALLER_CONTRACTS, 0)
    for path in pathlib.Path(package_root).rglob("*.py"):
        if path.name == "temporal_source.py":
            continue
        source = path.read_text()
        for name, pattern in _LEAF_CALLER_CONTRACTS.items():
            for line in source.splitlines():
                if f"{name}(" not in line:
                    continue
                call_site_counts[name] += 1
                if not pattern.search(line):
                    violations.append(f"{name}: unjustified call in {path}: {line.strip()}")
    for name, count in call_site_counts.items():
        if count == 0:
            violations.append(f"{name}: no call sites found under {package_root} — contract unverifiable")
    return violations


def is_valid_temporal_source(value: str) -> bool:
    """Return True when *value* is a recognized temporal source token."""

    return value in TEMPORAL_SOURCE_VALUES


__all__ = [
    "TEMPORAL_SOURCE_VALUES",
    "TemporalSource",
    "audit_temporal_source_leaf_callers",
    "classify_aggregate_hwm_source",
    "classify_profile_hwm_source",
    "classify_thread_hwm_source",
    "is_valid_temporal_source",
    "weakest_of",
    "weakest_source",
]

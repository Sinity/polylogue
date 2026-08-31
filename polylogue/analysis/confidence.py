"""Shared confidence vocabulary for insight materializations (#1277).

Insight payloads (work events, phases, profiles, enrichments, threads,
classifications) have historically reported their rigor as ad-hoc string
literals — ``"strong"``, ``"moderate"``, ``"weak"``, ``"none"`` — computed
from raw confidence scores and support-signal counts by helpers scattered
across the materialization layer. The values are stable downstream
contracts: dashboards, MCP responses, and learning corrections all key
off them.

This module promotes that vocabulary to a closed enum so downstream
consumers can pattern-match on a known set of bands and so new
materialization sites cannot silently invent new spellings ("medium"
vs. "moderate", "high" vs. "strong").

Vocabulary:

- ``STRONG``   — primary evidence path; multi-signal support and
  high model/heuristic confidence. Safe to display unqualified.
- ``MODERATE`` — partial evidence; either confidence is mid-band or
  the supporting signals are thin. Display with light caveats.
- ``WEAK``     — minimal or contradictory evidence; the row exists
  but downstream consumers should treat the value as suggestive.
- ``NONE``     — explicit absence of usable evidence (used by
  ``repo_inference_strength`` when no repo names are detected at all).
  Distinct from ``WEAK``: ``WEAK`` says "we tried and it's thin",
  ``NONE`` says "we did not try".

The enum is a ``str`` subclass so JSON serialization yields the
existing string forms — storage TEXT columns and the wire contract are
unchanged. Re-ingest is not required.
"""

from __future__ import annotations

from collections.abc import Sized
from enum import Enum


class ConfidenceBand(str, Enum):
    """Closed vocabulary for the ``support_level`` / ``*_strength`` family.

    Members are string-valued (``"strong" / "moderate" / "weak" / "none"``)
    so callers can treat the enum as a drop-in for the legacy literals on
    JSON/SQLite boundaries.
    """

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


# Confidence thresholds used by ``from_signals`` — mirror the bands the
# legacy ``support_level()`` helper enforced on session-profile rebuild.
_STRONG_CONFIDENCE_FLOOR = 0.78
_MODERATE_CONFIDENCE_FLOOR = 0.55
_STRONG_SIGNAL_FLOOR = 2

# Score-only bucketing for places that only have a raw float (audit
# distributions, etc.). Bands split [0, 1] into thirds rounded to
# match the existing audit ``low / mid / high`` cutoffs (#1019 §rigor).
_SCORE_STRONG_FLOOR = 0.67
_SCORE_MODERATE_FLOOR = 0.34


def from_signals(
    confidence: float,
    *,
    support_signals: Sized,
    fallback: bool = False,
) -> ConfidenceBand:
    """Compute a band from a numeric confidence and a count of support signals.

    This is the canonical band used by ``support_level`` on every
    session-insight payload (profiles, work events, phases, enrichments,
    threads).

    Semantics:

    - ``fallback=True`` short-circuits to ``WEAK`` — the materialization
      path itself is degraded, so the band must reflect that regardless
      of the raw confidence number.
    - Confidence below ``_MODERATE_CONFIDENCE_FLOOR`` or zero supporting
      signals → ``WEAK``.
    - Confidence at or above ``_STRONG_CONFIDENCE_FLOOR`` with at least
      two distinct supporting signals → ``STRONG``.
    - Everything else → ``MODERATE``.
    """

    if fallback or confidence < _MODERATE_CONFIDENCE_FLOOR or len(support_signals) == 0:
        return ConfidenceBand.WEAK
    if confidence >= _STRONG_CONFIDENCE_FLOOR and len(support_signals) >= _STRONG_SIGNAL_FLOOR:
        return ConfidenceBand.STRONG
    return ConfidenceBand.MODERATE


def from_score(score: float) -> ConfidenceBand:
    """Bucket a raw ``[0, 1]`` confidence score with no signal context.

    Used by classification-style helpers that only know the within-classifier
    vote margin and an evidence count, without the richer support-signal
    list that the session-profile materialization path computes. Three
    thirds-of-the-range bands; values outside ``[0, 1]`` are clipped.
    """

    if score >= _SCORE_STRONG_FLOOR:
        return ConfidenceBand.STRONG
    if score >= _SCORE_MODERATE_FLOOR:
        return ConfidenceBand.MODERATE
    return ConfidenceBand.WEAK


def confidence_band_from_stored(value: str | ConfidenceBand | None) -> ConfidenceBand:
    """Coerce a stored string literal into a typed band.

    Accepts the persisted wire values verbatim. Unknown strings collapse
    to ``WEAK`` rather than raising so stored records remain readable.
    """

    if isinstance(value, ConfidenceBand):
        return value
    if value is None:
        return ConfidenceBand.WEAK
    text = str(value).strip().lower()
    try:
        return ConfidenceBand(text)
    except ValueError:
        return ConfidenceBand.WEAK


__all__ = [
    "ConfidenceBand",
    "confidence_band_from_stored",
    "from_score",
    "from_signals",
]

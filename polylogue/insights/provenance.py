"""Insight provenance helpers.

Every materialized insight row records four provenance fields:

- ``materialized_at`` (a.k.a. computed_at): wall-clock instant the row was
  produced.
- ``source_updated_at`` and ``source_sort_key``: the source change timestamp
  observed at materialization time (for aggregates this is the high-water
  mark across all sources that contributed to the row).
- ``input_high_water_mark``: explicit alias for the latest input source
  change timestamp folded into this insight row. For per-session
  insights (profiles, work events, phases, threads) this is equivalent to
  ``source_updated_at``. Aggregates compute it from the input bucket.
- ``input_row_count``: number of source rows that produced the insight.
  For ``session_profiles`` this is the session's message count; for
  timeline rows it is the events/phases produced; for aggregates it is the
  number of sessions folded into the bucket.

These fields exist so a downstream reader can answer:

- *How fresh is this insight?* — compare ``materialized_at`` against now.
- *Was this insight built from current inputs?* — compare
  ``input_high_water_mark`` against the latest archive change for the same
  scope.
- *Did the insight see all inputs we expect?* — compare ``input_row_count``
  against the expected number of inputs.

This module provides typed helpers callers can use without re-importing the
storage layer or re-deriving the comparison logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class HasProvenance(Protocol):
    """Structural type for any insight record exposing provenance fields.

    ``input_high_water_mark_source`` (added in #1276) names the temporal
    source the HWM was sampled from (see
    :mod:`polylogue.insights.temporal_source`). It is optional only
    because legacy rows materialized before that taxonomy existed do not
    carry the tag; all freshly materialized rows must set it.
    """

    materialized_at: str
    materializer_version: int
    input_high_water_mark: str | None
    input_high_water_mark_source: str | None
    input_row_count: int


@dataclass(frozen=True, slots=True)
class StalenessVerdict:
    """Outcome of comparing an insight's provenance against fresh inputs."""

    stale: bool
    reason: str
    insight_high_water_mark: str | None
    source_high_water_mark: str | None
    insight_row_count: int
    source_row_count: int | None


def is_stale(
    record: HasProvenance,
    *,
    source_high_water_mark: str | None,
    source_row_count: int | None = None,
) -> StalenessVerdict:
    """Decide whether an insight row is stale relative to its inputs.

    An insight is considered stale if any of:

    - ``source_high_water_mark`` is later than the record's
      ``input_high_water_mark`` (inputs changed after materialization);
    - ``source_row_count`` is provided and exceeds the record's
      ``input_row_count`` (more inputs exist than the record folded in).

    Records with no high-water mark recorded are treated as stale when a
    source high-water mark is known, since the contract is that fresh
    insights record provenance.
    """

    insight_hwm = record.input_high_water_mark
    insight_rows = int(record.input_row_count)

    if source_high_water_mark is not None:
        if insight_hwm is None:
            return StalenessVerdict(
                stale=True,
                reason="insight has no input_high_water_mark",
                insight_high_water_mark=None,
                source_high_water_mark=source_high_water_mark,
                insight_row_count=insight_rows,
                source_row_count=source_row_count,
            )
        if source_high_water_mark > insight_hwm:
            return StalenessVerdict(
                stale=True,
                reason="source has changed after insight materialization",
                insight_high_water_mark=insight_hwm,
                source_high_water_mark=source_high_water_mark,
                insight_row_count=insight_rows,
                source_row_count=source_row_count,
            )

    if source_row_count is not None and source_row_count > insight_rows:
        return StalenessVerdict(
            stale=True,
            reason="source row count exceeds insight input_row_count",
            insight_high_water_mark=insight_hwm,
            source_high_water_mark=source_high_water_mark,
            insight_row_count=insight_rows,
            source_row_count=source_row_count,
        )

    return StalenessVerdict(
        stale=False,
        reason="fresh",
        insight_high_water_mark=insight_hwm,
        source_high_water_mark=source_high_water_mark,
        insight_row_count=insight_rows,
        source_row_count=source_row_count,
    )


__all__ = ["HasProvenance", "StalenessVerdict", "is_stale"]

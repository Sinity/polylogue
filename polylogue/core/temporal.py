"""Shared temporal provenance vocabulary and weakest-source algebra.

This module owns only the storage-free timestamp source/confidence contract.
Domain classifiers remain in their owning packages; consumers can reduce
provenance without importing an insight or surface layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeGuard, get_args

TemporalSource = Literal[
    "provider_ts",
    "hook_event_ts",
    "sort_key",
    "file_mtime",
    "materialization_ts",
    "fallback_date",
]
TimeConfidence = Literal["recorded", "estimated", "unknown"]

TEMPORAL_SOURCE_VALUES: frozenset[TemporalSource] = frozenset(get_args(TemporalSource))
TIME_CONFIDENCE_VALUES: frozenset[TimeConfidence] = frozenset(get_args(TimeConfidence))

# Strongest first. A larger rank is weaker, so aggregate reduction is a max.
_SOURCE_RANK: dict[TemporalSource, int] = {source: rank for rank, source in enumerate(get_args(TemporalSource))}
_TIME_CONFIDENCE_BY_SOURCE: dict[TemporalSource, TimeConfidence] = {
    "provider_ts": "recorded",
    "hook_event_ts": "recorded",
    "sort_key": "estimated",
    "file_mtime": "estimated",
    "materialization_ts": "unknown",
    "fallback_date": "unknown",
}


def weakest_source(a: TemporalSource, b: TemporalSource) -> TemporalSource:
    """Return the weaker (less event-authoritative) temporal source."""

    return a if _SOURCE_RANK[a] >= _SOURCE_RANK[b] else b


def weakest_of(sources: Sequence[TemporalSource]) -> TemporalSource | None:
    """Reduce a sequence to its weakest source; return ``None`` when empty."""

    if not sources:
        return None
    result = sources[0]
    for source in sources[1:]:
        result = weakest_source(result, source)
    return result


def time_confidence_for_source(source: str | None) -> TimeConfidence:
    """Project source provenance to the public recorded/estimated/unknown axis."""

    if source is None or not is_valid_temporal_source(source):
        return "unknown"
    return _TIME_CONFIDENCE_BY_SOURCE[source]


def time_confidence_for_sources(sources: Sequence[TemporalSource]) -> TimeConfidence:
    """Project an aggregate through the weakest source in the input set."""

    return time_confidence_for_source(weakest_of(sources))


def is_valid_temporal_source(value: str) -> TypeGuard[TemporalSource]:
    """Return whether *value* belongs to the canonical temporal vocabulary."""

    return value in TEMPORAL_SOURCE_VALUES


__all__ = [
    "TEMPORAL_SOURCE_VALUES",
    "TIME_CONFIDENCE_VALUES",
    "TemporalSource",
    "TimeConfidence",
    "is_valid_temporal_source",
    "time_confidence_for_source",
    "time_confidence_for_sources",
    "weakest_of",
    "weakest_source",
]

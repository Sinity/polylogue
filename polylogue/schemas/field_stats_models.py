"""Typed field-statistics models and thresholds."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

ENUM_MAX_CARDINALITY = 50
ENUM_VALUE_CAP = 200
REF_MATCH_THRESHOLD = 0.7


@dataclass
class FieldStats:
    """Statistics collected for a single JSON path across all samples."""

    path: str
    observed_values: Counter[str] = field(default_factory=Counter)
    detected_formats: Counter[str] = field(default_factory=Counter)
    num_min: float | None = None
    num_max: float | None = None
    total_samples: int = 0
    present_count: int = 0
    array_lengths: list[int] = field(default_factory=list)
    is_multiline: int = 0
    value_count: int = 0
    value_conversation_ids: dict[str, set[str]] = field(default_factory=dict)
    string_lengths: list[int] = field(default_factory=list)
    newline_counts: list[int] = field(default_factory=list)
    numeric_values: list[float] = field(default_factory=list)
    distinct_value_count: int = 0
    values_per_conversation: dict[str, set[str]] = field(default_factory=dict)
    _ordered_samples: list[list[float]] = field(default_factory=list)
    co_occurring_fields: Counter[str] = field(default_factory=Counter)
    object_key_counts: list[int] = field(default_factory=list)
    max_depth_seen: int = 0

    @property
    def frequency(self) -> float:
        return self.present_count / self.total_samples if self.total_samples else 0.0

    @property
    def dominant_format(self) -> str | None:
        if not self.detected_formats or not self.value_count:
            return None
        fmt, count = self.detected_formats.most_common(1)[0]
        if count / self.value_count >= 0.8:
            return str(fmt) if fmt is not None else None
        return None

    @property
    def is_enum_like(self) -> bool:
        if not self.observed_values:
            return False
        return len(self.observed_values) <= ENUM_MAX_CARDINALITY

    @property
    def string_length_stats(self) -> dict[str, float] | None:
        if not self.string_lengths:
            return None
        n = len(self.string_lengths)
        avg = sum(self.string_lengths) / n
        if n > 1:
            variance = sum((x - avg) ** 2 for x in self.string_lengths) / (n - 1)
            stddev = math.sqrt(variance)
        else:
            stddev = 0.0
        return {
            "min": min(self.string_lengths),
            "max": max(self.string_lengths),
            "avg": avg,
            "stddev": stddev,
        }

    @property
    def newline_rate(self) -> float:
        return self.is_multiline / self.value_count if self.value_count else 0.0

    @property
    def monotonicity_score(self) -> float | None:
        if not self._ordered_samples:
            return None
        total_pairs = 0
        increasing_pairs = 0
        for seq in self._ordered_samples:
            for index in range(len(seq) - 1):
                total_pairs += 1
                if seq[index + 1] >= seq[index]:
                    increasing_pairs += 1
        return increasing_pairs / total_pairs if total_pairs else None

    @property
    def avg_array_length(self) -> float | None:
        return sum(self.array_lengths) / len(self.array_lengths) if self.array_lengths else None

    @property
    def avg_object_fanout(self) -> float | None:
        return sum(self.object_key_counts) / len(self.object_key_counts) if self.object_key_counts else None

    @property
    def approximate_entropy(self) -> float | None:
        if not self.observed_values or self.value_count == 0:
            return None
        total = sum(self.observed_values.values())
        entropy = 0.0
        for count in self.observed_values.values():
            if count > 0:
                probability = count / total
                entropy -= probability * math.log2(probability)
        return entropy


__all__ = [
    "ENUM_MAX_CARDINALITY",
    "ENUM_VALUE_CAP",
    "FieldStats",
    "REF_MATCH_THRESHOLD",
]

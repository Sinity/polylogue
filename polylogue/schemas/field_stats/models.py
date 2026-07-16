"""Typed field-statistics models and thresholds."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from polylogue.schemas.field_stats.distributions import CategoricalSketch, DistributionSketch

ENUM_MAX_CARDINALITY = 50
ENUM_VALUE_CAP = 200
REF_MATCH_THRESHOLD = 0.7
LEGACY_SAMPLE_CAP = 2_000
SESSION_EVIDENCE_CAP = 16


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
    value_session_ids: dict[str, set[str]] = field(default_factory=dict)
    string_lengths: list[int] = field(default_factory=list)
    newline_counts: list[int] = field(default_factory=list)
    numeric_values: list[float] = field(default_factory=list)
    distinct_value_count: int = 0
    values_per_session: dict[str, set[str]] = field(default_factory=dict)
    _ordered_samples: list[list[float]] = field(default_factory=list)
    co_occurring_fields: Counter[str] = field(default_factory=Counter)
    object_key_counts: list[int] = field(default_factory=list)
    max_depth_seen: int = 0
    ref_target: str | None = None
    documents_present: set[int] = field(default_factory=set)
    type_counts: Counter[str] = field(default_factory=Counter)
    null_count: int = 0
    document_encountered_count: int = 0
    document_non_null_count: int = 0
    string_length_distribution: DistributionSketch = field(default_factory=DistributionSketch)
    newline_distribution: DistributionSketch = field(default_factory=DistributionSketch)
    numeric_distribution: DistributionSketch = field(default_factory=DistributionSketch)
    array_length_distribution: DistributionSketch = field(default_factory=DistributionSketch)
    object_fanout_distribution: DistributionSketch = field(default_factory=DistributionSketch)
    categorical_distribution: CategoricalSketch = field(default_factory=CategoricalSketch)
    object_key_distribution: CategoricalSketch = field(default_factory=CategoricalSketch)
    co_occurrence_distribution: CategoricalSketch = field(default_factory=CategoricalSketch)
    boolean_counts: Counter[str] = field(default_factory=Counter)
    ordered_pair_count: int = 0
    ordered_increasing_pair_count: int = 0
    overflow_value_count: int = 0
    truncated_evidence: Counter[str] = field(default_factory=Counter)
    _last_encountered_document: int | None = field(default=None, repr=False)
    _last_non_null_document: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Backfill sketches for direct fixtures using legacy sample lists."""
        if not self.string_length_distribution.count:
            for value in self.string_lengths:
                self.string_length_distribution.observe(value)
        if not self.newline_distribution.count:
            for value in self.newline_counts:
                self.newline_distribution.observe(value)
        if not self.numeric_distribution.count:
            for numeric_value in self.numeric_values:
                self.numeric_distribution.observe(numeric_value)
        if not self.array_length_distribution.count:
            for value in self.array_lengths:
                self.array_length_distribution.observe(value)
        if not self.object_fanout_distribution.count:
            for value in self.object_key_counts:
                self.object_fanout_distribution.observe(value)
        if not self.categorical_distribution.count:
            for categorical_value, categorical_count in self.observed_values.items():
                self.categorical_distribution.observe(categorical_value, count=categorical_count)
        if self._ordered_samples and not self.ordered_pair_count:
            for sequence in self._ordered_samples:
                self.observe_ordered_sequence(sequence)

    def observe_document(self, sample_idx: int, *, non_null: bool) -> None:
        if self._last_encountered_document != sample_idx:
            self.document_encountered_count += 1
            self._last_encountered_document = sample_idx
        if non_null and self._last_non_null_document != sample_idx:
            self.document_non_null_count += 1
            self._last_non_null_document = sample_idx
            if len(self.documents_present) < LEGACY_SAMPLE_CAP:
                self.documents_present.add(sample_idx)
            else:
                self.truncated_evidence["document_ids"] += 1

    def observe_ordered_sequence(self, sequence: list[float]) -> None:
        for index in range(len(sequence) - 1):
            self.ordered_pair_count += 1
            if sequence[index + 1] >= sequence[index]:
                self.ordered_increasing_pair_count += 1

    @property
    def frequency(self) -> float:
        return self.present_count / self.total_samples if self.total_samples else 0.0

    @property
    def document_frequency(self) -> float:
        """Fraction of input documents where this path appears at least once.

        Unlike `frequency`, this denominator-stable metric works correctly for
        paths that recurse through arrays or dynamic-key dicts: each document
        contributes at most once regardless of how many array items or dynamic
        keys carry the path. Suitable for the `x-polylogue-frequency` schema
        annotation.
        """
        if not self.total_samples:
            return 0.0
        documents_seen = self.document_non_null_count or (
            len(self.documents_present) if self.documents_present else min(self.present_count, self.total_samples)
        )
        return documents_seen / self.total_samples

    @property
    def dominant_format(self) -> str | None:
        if not self.detected_formats or not self.value_count:
            return None
        fmt, count = self.detected_formats.most_common(1)[0]
        if count / self.value_count >= 0.8:
            return str(fmt)
        return None

    @property
    def is_enum_like(self) -> bool:
        if not self.observed_values:
            return False
        return len(self.observed_values) <= ENUM_MAX_CARDINALITY

    @property
    def string_length_stats(self) -> dict[str, float] | None:
        if not self.string_length_distribution.count:
            return None
        return {
            "min": self.string_length_distribution.minimum or 0.0,
            "max": self.string_length_distribution.maximum or 0.0,
            "avg": self.string_length_distribution.mean,
            "stddev": self.string_length_distribution.stddev,
        }

    @property
    def newline_rate(self) -> float:
        return self.is_multiline / self.value_count if self.value_count else 0.0

    @property
    def monotonicity_score(self) -> float | None:
        if self.ordered_pair_count:
            return self.ordered_increasing_pair_count / self.ordered_pair_count
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
        return self.array_length_distribution.mean if self.array_length_distribution.count else None

    @property
    def avg_object_fanout(self) -> float | None:
        return self.object_fanout_distribution.mean if self.object_fanout_distribution.count else None

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
    "LEGACY_SAMPLE_CAP",
    "REF_MATCH_THRESHOLD",
    "SESSION_EVIDENCE_CAP",
]

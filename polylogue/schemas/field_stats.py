"""FieldStats accumulation and format detection for schema inference.

Collects per-JSON-path statistics across a corpus of data samples.
These statistics drive the x-polylogue-* schema annotations produced by
_annotate_schema in schema_inference.py.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


# UUID pattern for detecting dynamic keys
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def is_dynamic_key(key: str) -> bool:
    """Check if a key looks like a dynamic identifier (UUID, hash, etc)."""
    if UUID_PATTERN.match(key):
        return True
    if re.match(r"^[0-9a-f]{24,}$", key, re.IGNORECASE):
        return True
    return bool(re.match(r"^(msg|node|conv|item|att)-[0-9a-f-]+$", key, re.IGNORECASE))


# Format detection patterns — ordered by specificity (most specific first)
_FORMAT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("uuid4", re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)),
    ("uuid", re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)),
    ("hex-id", re.compile(r"^[0-9a-f]{24,}$", re.I)),
    ("iso8601", re.compile(
        r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")),
    ("unix-epoch-str", re.compile(r"^\d{10}(\.\d+)?$")),
    ("url", re.compile(r"^https?://")),
    ("mime-type", re.compile(r"^[a-z]+/[a-z0-9][a-z0-9.+\-]*$", re.I)),
    ("base64", re.compile(r"^[A-Za-z0-9+/]{40,}={0,2}$")),
    ("email", re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")),
]

# Thresholds
_ENUM_MAX_CARDINALITY = 50  # max distinct values to count as enum-like
_ENUM_VALUE_CAP = 200  # max values to track per field (memory bound during collection)
_REF_MATCH_THRESHOLD = 0.7  # fraction of values that must match keys


@dataclass
class FieldStats:
    """Statistics collected for a single JSON path across all samples."""

    path: str
    observed_values: Counter = field(default_factory=Counter)
    detected_formats: Counter = field(default_factory=Counter)
    num_min: float | None = None
    num_max: float | None = None
    total_samples: int = 0
    present_count: int = 0
    array_lengths: list[int] = field(default_factory=list)
    is_multiline: int = 0  # count of values containing newlines
    value_count: int = 0  # total non-null values seen
    # Maps each observed string value → set of conversation IDs that contained it.
    # Populated only when conversation_ids are supplied to _collect_field_stats.
    # Used to enforce the cross-conversation privacy threshold in _annotate_schema.
    value_conversation_ids: dict[str, set[str]] = field(default_factory=dict)

    @property
    def frequency(self) -> float:
        """Fraction of samples where this field was present."""
        return self.present_count / self.total_samples if self.total_samples else 0.0

    @property
    def dominant_format(self) -> str | None:
        """Most common detected format, if it covers ≥80% of values."""
        if not self.detected_formats or not self.value_count:
            return None
        fmt, count = self.detected_formats.most_common(1)[0]
        if count / self.value_count >= 0.8:
            return fmt
        return None

    @property
    def is_enum_like(self) -> bool:
        """Whether this field has low-cardinality string values."""
        if not self.observed_values:
            return False
        return len(self.observed_values) <= _ENUM_MAX_CARDINALITY


def _detect_string_format(value: str) -> str | None:
    """Detect the format of a string value."""
    if not value or len(value) > 500:
        return None
    for fmt_name, pattern in _FORMAT_PATTERNS:
        if pattern.match(value):
            return fmt_name
    return None


def _detect_numeric_format(value: float | int) -> str | None:
    """Detect whether a numeric value is a Unix epoch timestamp."""
    if isinstance(value, bool):
        return None
    try:
        fval = float(value)
        if math.isnan(fval) or math.isinf(fval):
            return None
        # Unix epoch range: 2000-01-01 to 2040-01-01
        if 946684800.0 <= fval <= 2208988800.0:
            return "unix-epoch"
    except (TypeError, ValueError):
        pass
    return None


def _collect_field_stats(
    samples: list[dict[str, Any]],
    *,
    conversation_ids: list[str | None] | None = None,
    max_depth: int = 15,
) -> dict[str, FieldStats]:
    """Walk all samples and collect per-JSON-path statistics.

    Tracks: string value sets (for enum detection), format patterns,
    numeric ranges, field frequency, and array lengths.

    Args:
        samples: Raw data dicts to analyze
        conversation_ids: Optional parallel list of conversation IDs for each sample.
            When provided, each string value is annotated with the conversation(s)
            it appeared in, enabling the cross-conversation privacy threshold in
            _annotate_schema (min_conversation_count).  Length must equal len(samples).
        max_depth: Maximum nesting depth to traverse

    Returns:
        Mapping of JSON path → FieldStats
    """
    all_stats: dict[str, FieldStats] = {}
    # Track all dict key sets for reference detection
    dict_key_sets: dict[str, set[str]] = {}  # path → set of observed keys

    def _ensure_stats(path: str) -> FieldStats:
        if path not in all_stats:
            all_stats[path] = FieldStats(path=path)
        return all_stats[path]

    def _walk(value: Any, path: str, depth: int, sample_idx: int) -> None:
        if depth > max_depth:
            return

        stats = _ensure_stats(path)
        stats.total_samples = max(stats.total_samples, sample_idx + 1)

        if value is None:
            return

        stats.present_count += 1
        stats.value_count += 1

        if isinstance(value, dict):
            # Track keys for reference detection
            if path not in dict_key_sets:
                dict_key_sets[path] = set()
            dict_key_sets[path].update(value.keys())

            # Separate static vs dynamic keys
            static_keys = {}
            dynamic_values = []
            for k, v in value.items():
                if is_dynamic_key(k):
                    dynamic_values.append(v)
                else:
                    static_keys[k] = v

            # Walk static properties
            for k, v in static_keys.items():
                _walk(v, f"{path}.{k}", depth + 1, sample_idx)

            # Walk dynamic properties under wildcard path
            for v in dynamic_values:
                _walk(v, f"{path}.*", depth + 1, sample_idx)

        elif isinstance(value, list):
            stats.array_lengths.append(len(value))
            for _i, item in enumerate(value):
                # Use [*] for array items (not [0], [1] — we want aggregate stats)
                _walk(item, f"{path}[*]", depth + 1, sample_idx)

        elif isinstance(value, str):
            # Track observed values (capped for memory)
            if len(stats.observed_values) < _ENUM_VALUE_CAP:
                stats.observed_values[value] += 1
                # Track which conversation this value came from (for privacy threshold)
                if conversation_ids is not None:
                    conv_id = (
                        conversation_ids[sample_idx]
                        if sample_idx < len(conversation_ids)
                        else None
                    )
                    if conv_id is not None:
                        stats.value_conversation_ids.setdefault(value, set()).add(conv_id)

            # Detect format
            fmt = _detect_string_format(value)
            if fmt:
                stats.detected_formats[fmt] += 1

            # Track multiline content
            if "\n" in value:
                stats.is_multiline += 1

        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            fval = float(value)
            if not (math.isnan(fval) or math.isinf(fval)):
                if stats.num_min is None or fval < stats.num_min:
                    stats.num_min = fval
                if stats.num_max is None or fval > stats.num_max:
                    stats.num_max = fval

                # Detect numeric format (unix epoch)
                fmt = _detect_numeric_format(value)
                if fmt:
                    stats.detected_formats[fmt] += 1

    for idx, sample in enumerate(samples):
        _walk(sample, "$", 0, idx)

    # Fix total_samples for all stats (some paths only seen in subset)
    n = len(samples)
    for stats in all_stats.values():
        stats.total_samples = n

    # Reference detection: for each string field, check if its values
    # are mostly keys in some dict field
    for path, stats in all_stats.items():
        if stats.observed_values:
            observed = set(stats.observed_values.keys())
            if len(observed) > _ENUM_MAX_CARDINALITY:
                # High-cardinality string field — check for references
                for dict_path, keys in dict_key_sets.items():
                    if dict_path == path:
                        continue
                    if not keys:
                        continue
                    overlap = len(observed & keys)
                    if overlap / len(observed) >= _REF_MATCH_THRESHOLD:
                        stats._ref_target = dict_path  # type: ignore[attr-defined]

    return all_stats

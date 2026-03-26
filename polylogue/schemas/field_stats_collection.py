"""Field-statistics collection over raw JSON samples."""

from __future__ import annotations

import math
from typing import Any

from polylogue.schemas.field_stats_detection import (
    _detect_numeric_format,
    _detect_string_format,
    is_dynamic_key,
)
from polylogue.schemas.field_stats_models import (
    ENUM_MAX_CARDINALITY,
    ENUM_VALUE_CAP,
    REF_MATCH_THRESHOLD,
    FieldStats,
)


def _collect_field_stats(
    samples: list[dict[str, Any]],
    *,
    conversation_ids: list[str | None] | None = None,
    max_depth: int = 15,
) -> dict[str, FieldStats]:
    """Walk all samples and collect per-JSON-path statistics."""
    all_stats: dict[str, FieldStats] = {}
    dict_key_sets: dict[str, set[str]] = {}

    def _ensure_stats(path: str) -> FieldStats:
        if path not in all_stats:
            all_stats[path] = FieldStats(path=path)
        return all_stats[path]

    co_occurrence: dict[str, dict[int, set[str]]] = {}
    numeric_sample_cap = 500
    string_length_cap = 2000

    def _walk(value: Any, path: str, depth: int, sample_idx: int) -> None:
        if depth > max_depth:
            return

        stats = _ensure_stats(path)
        stats.total_samples = max(stats.total_samples, sample_idx + 1)
        stats.max_depth_seen = max(stats.max_depth_seen, depth)

        if value is None:
            return

        stats.present_count += 1
        stats.value_count += 1

        if isinstance(value, dict):
            if path not in dict_key_sets:
                dict_key_sets[path] = set()
            dict_key_sets[path].update(value.keys())
            stats.object_key_counts.append(len(value))

            static_keys: dict[str, Any] = {}
            dynamic_values: list[Any] = []
            for key, item in value.items():
                if is_dynamic_key(key):
                    dynamic_values.append(item)
                else:
                    static_keys[key] = item

            if path not in co_occurrence:
                co_occurrence[path] = {}
            co_occurrence[path][sample_idx] = set(static_keys.keys())

            for key, item in static_keys.items():
                _walk(item, f"{path}.{key}", depth + 1, sample_idx)
            for item in dynamic_values:
                _walk(item, f"{path}.*", depth + 1, sample_idx)
            return

        if isinstance(value, list):
            stats.array_lengths.append(len(value))
            numeric_seq: list[float] = []
            for item in value:
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    fval = float(item)
                    if not (math.isnan(fval) or math.isinf(fval)):
                        numeric_seq.append(fval)
                _walk(item, f"{path}[*]", depth + 1, sample_idx)
            if len(numeric_seq) >= 2:
                item_stats = _ensure_stats(f"{path}[*]")
                item_stats._ordered_samples.append(numeric_seq)
            return

        if isinstance(value, str):
            if len(stats.string_lengths) < string_length_cap:
                stats.string_lengths.append(len(value))
            if len(stats.observed_values) < ENUM_VALUE_CAP:
                if value not in stats.observed_values:
                    stats.distinct_value_count += 1
                stats.observed_values[value] += 1
                if conversation_ids is not None:
                    conv_id = conversation_ids[sample_idx] if sample_idx < len(conversation_ids) else None
                    if conv_id is not None:
                        stats.value_conversation_ids.setdefault(value, set()).add(conv_id)
                        stats.values_per_conversation.setdefault(conv_id, set()).add(value)
            else:
                stats.distinct_value_count += 1

            fmt = _detect_string_format(value)
            if fmt:
                stats.detected_formats[fmt] += 1

            newline_count = value.count("\n")
            if newline_count > 0:
                stats.is_multiline += 1
            stats.newline_counts.append(newline_count)
            return

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            fval = float(value)
            if not (math.isnan(fval) or math.isinf(fval)):
                if stats.num_min is None or fval < stats.num_min:
                    stats.num_min = fval
                if stats.num_max is None or fval > stats.num_max:
                    stats.num_max = fval
                if len(stats.numeric_values) < numeric_sample_cap:
                    stats.numeric_values.append(fval)
                fmt = _detect_numeric_format(value)
                if fmt:
                    stats.detected_formats[fmt] += 1

    for idx, sample in enumerate(samples):
        _walk(sample, "$", 0, idx)

    total_samples = len(samples)
    for stats in all_stats.values():
        stats.total_samples = total_samples

    for parent_path, samples_map in co_occurrence.items():
        for sibling_names in samples_map.values():
            for child_name in sibling_names:
                child_path = f"{parent_path}.{child_name}"
                if child_path not in all_stats:
                    continue
                child_stats = all_stats[child_path]
                for other_name in sibling_names:
                    if other_name != child_name:
                        child_stats.co_occurring_fields[other_name] += 1

    for path, stats in all_stats.items():
        if not stats.observed_values:
            continue
        observed = set(stats.observed_values.keys())
        if len(observed) <= ENUM_MAX_CARDINALITY:
            continue
        for dict_path, keys in dict_key_sets.items():
            if dict_path == path or not keys:
                continue
            overlap = len(observed & keys)
            if overlap / len(observed) >= REF_MATCH_THRESHOLD:
                stats._ref_target = dict_path  # type: ignore[attr-defined]

    return all_stats


__all__ = ["_collect_field_stats"]

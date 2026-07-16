"""Field-statistics collection over raw JSON samples."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TypeAlias

from polylogue.schemas.field_stats.detection import (
    _detect_numeric_format,
    _detect_string_format,
    is_dynamic_key,
)
from polylogue.schemas.field_stats.models import (
    ENUM_MAX_CARDINALITY,
    ENUM_VALUE_CAP,
    LEGACY_SAMPLE_CAP,
    REF_MATCH_THRESHOLD,
    SESSION_EVIDENCE_CAP,
    FieldStats,
)

SampleMapping: TypeAlias = Mapping[str, object]
JSONContainer: TypeAlias = dict[str, object]
JSONList: TypeAlias = list[object]
FieldStatsByPath: TypeAlias = dict[str, FieldStats]
DictKeySetsByPath: TypeAlias = dict[str, set[str]]

_DICT_KEY_EVIDENCE_CAP = 2_048
_CO_OCCURRENCE_FIELD_CAP = 256
_ORDERED_SEQUENCE_CAP = 256
_VALUES_PER_SESSION_CAP = 256


def _type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "other"


def _append_bounded(
    values: list[int] | list[float] | list[list[float]], value: object, stats: FieldStats, key: str
) -> None:
    if len(values) < LEGACY_SAMPLE_CAP:
        values.append(value)  # type: ignore[arg-type]
    else:
        stats.truncated_evidence[key] += 1


def _increment_bounded(counter: dict[str, int], key: str, stats: FieldStats, evidence_key: str) -> None:
    if key in counter or len(counter) < _CO_OCCURRENCE_FIELD_CAP:
        counter[key] = counter.get(key, 0) + 1
    else:
        stats.truncated_evidence[evidence_key] += 1


def _collect_field_stats(
    samples: Sequence[SampleMapping],
    *,
    session_ids: Sequence[str | None] | None = None,
    max_depth: int = 15,
) -> FieldStatsByPath:
    """Walk all samples and collect per-JSON-path statistics."""
    all_stats: FieldStatsByPath = {}
    dict_key_sets: DictKeySetsByPath = {}

    def _ensure_stats(path: str) -> FieldStats:
        if path not in all_stats:
            all_stats[path] = FieldStats(path=path)
        return all_stats[path]

    numeric_sample_cap = 500
    string_length_cap = 2000

    def _walk(value: object, path: str, depth: int, sample_idx: int) -> None:
        if depth > max_depth:
            return

        stats = _ensure_stats(path)
        stats.total_samples = max(stats.total_samples, sample_idx + 1)
        stats.max_depth_seen = max(stats.max_depth_seen, depth)
        stats.type_counts[_type_name(value)] += 1
        stats.observe_document(sample_idx, non_null=value is not None)

        if value is None:
            stats.null_count += 1
            return

        stats.present_count += 1
        stats.value_count += 1

        if isinstance(value, Mapping):
            if path not in dict_key_sets:
                dict_key_sets[path] = set()
            key_evidence = dict_key_sets[path]
            for key in value:
                stats.object_key_distribution.observe(str(key))
                if key in key_evidence or len(key_evidence) < _DICT_KEY_EVIDENCE_CAP:
                    key_evidence.add(key)
                else:
                    stats.truncated_evidence["dictionary_keys"] += 1
            _append_bounded(stats.object_key_counts, len(value), stats, "object_fanout_samples")
            stats.object_fanout_distribution.observe(len(value))

            static_keys: JSONContainer = {}
            dynamic_values: JSONList = []
            for key, item in value.items():
                if is_dynamic_key(key):
                    dynamic_values.append(item)
                else:
                    static_keys[str(key)] = item

            sibling_names = set(static_keys)
            for child_name in sibling_names:
                child_stats = _ensure_stats(f"{path}.{child_name}")
                for other_name in sibling_names:
                    if other_name != child_name:
                        child_stats.co_occurrence_distribution.observe(other_name)
                        _increment_bounded(
                            child_stats.co_occurring_fields,
                            other_name,
                            child_stats,
                            "co_occurring_fields",
                        )

            for key, item in static_keys.items():
                _walk(item, f"{path}.{key}", depth + 1, sample_idx)
            for item in dynamic_values:
                _walk(item, f"{path}.*", depth + 1, sample_idx)
            return

        if isinstance(value, list):
            _append_bounded(stats.array_lengths, len(value), stats, "array_length_samples")
            stats.array_length_distribution.observe(len(value))
            numeric_seq: list[float] = []
            for item in value:
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    fval = float(item)
                    if not (math.isnan(fval) or math.isinf(fval)):
                        numeric_seq.append(fval)
                _walk(item, f"{path}[*]", depth + 1, sample_idx)
            if len(numeric_seq) >= 2:
                item_stats = _ensure_stats(f"{path}[*]")
                item_stats.observe_ordered_sequence(numeric_seq)
                if len(item_stats._ordered_samples) < _ORDERED_SEQUENCE_CAP:
                    item_stats._ordered_samples.append(numeric_seq[:LEGACY_SAMPLE_CAP])
                else:
                    item_stats.truncated_evidence["ordered_sequences"] += 1
            return

        if isinstance(value, str):
            stats.categorical_distribution.observe(value)
            if len(stats.string_lengths) < string_length_cap:
                stats.string_lengths.append(len(value))
            else:
                stats.truncated_evidence["string_length_samples"] += 1
            stats.string_length_distribution.observe(len(value))
            if value in stats.observed_values or len(stats.observed_values) < ENUM_VALUE_CAP:
                if value not in stats.observed_values:
                    stats.distinct_value_count += 1
                stats.observed_values[value] += 1
                if session_ids is not None:
                    conv_id = session_ids[sample_idx] if sample_idx < len(session_ids) else None
                    if conv_id is not None:
                        sessions = stats.value_session_ids.setdefault(value, set())
                        if conv_id in sessions or len(sessions) < SESSION_EVIDENCE_CAP:
                            sessions.add(conv_id)
                        else:
                            stats.truncated_evidence["enum_session_ids"] += 1
                        if (
                            conv_id in stats.values_per_session
                            or len(stats.values_per_session) < _VALUES_PER_SESSION_CAP
                        ):
                            stats.values_per_session.setdefault(conv_id, set()).add(value)
                        else:
                            stats.truncated_evidence["sessions"] += 1
            else:
                stats.distinct_value_count += 1
                stats.overflow_value_count += 1

            fmt = _detect_string_format(value)
            if fmt:
                stats.detected_formats[fmt] += 1

            newline_count = value.count("\n")
            if newline_count > 0:
                stats.is_multiline += 1
            _append_bounded(stats.newline_counts, newline_count, stats, "newline_samples")
            stats.newline_distribution.observe(newline_count)
            return

        if isinstance(value, bool):
            stats.boolean_counts["true" if value else "false"] += 1
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
                else:
                    stats.truncated_evidence["numeric_samples"] += 1
                stats.numeric_distribution.observe(fval)
                fmt = _detect_numeric_format(value)
                if fmt:
                    stats.detected_formats[fmt] += 1

    for idx, sample in enumerate(samples):
        _walk(sample, "$", 0, idx)

    total_samples = len(samples)
    for stats in all_stats.values():
        stats.total_samples = total_samples

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
                stats.ref_target = dict_path

    return all_stats


__all__ = ["_collect_field_stats"]

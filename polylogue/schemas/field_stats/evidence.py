"""Versioned, compact field-statistics evidence for incremental inference."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import overload

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.schemas.field_stats.distributions import CategoricalSketch, DistributionSketch
from polylogue.schemas.field_stats.models import EQUALITY_EVIDENCE_CAP, SESSION_EVIDENCE_CAP, FieldStats

FIELD_EVIDENCE_VERSION = 1


def _counter_state(counter: Mapping[str, int]) -> list[JSONValue]:
    return [[key, value] for key, value in sorted(counter.items())]


def _counter_from_state(value: object) -> Counter[str]:
    if not isinstance(value, list):
        return Counter()
    return Counter(
        {
            item[0]: int(item[1])
            for item in value
            if isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], int)
            and not isinstance(item[1], bool)
        }
    )


def _distribution_state(stats: FieldStats) -> JSONDocument:
    return {
        "string_length": stats.string_length_distribution.to_state(),
        "newline": stats.newline_distribution.to_state(),
        "numeric": stats.numeric_distribution.to_state(),
        "array_length": stats.array_length_distribution.to_state(),
        "object_fanout": stats.object_fanout_distribution.to_state(),
        "categorical": stats.categorical_distribution.to_state(),
        "object_key": stats.object_key_distribution.to_state(),
        "co_occurrence": stats.co_occurrence_distribution.to_state(),
    }


def _state_int(value: JSONValue) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _state_string(value: JSONValue) -> str | None:
    return value if isinstance(value, str) else None


@overload
def _distribution_from_state(state: object, name: str, kind: type[DistributionSketch]) -> DistributionSketch: ...


@overload
def _distribution_from_state(state: object, name: str, kind: type[CategoricalSketch]) -> CategoricalSketch: ...


def _distribution_from_state(
    state: object, name: str, kind: type[DistributionSketch] | type[CategoricalSketch]
) -> DistributionSketch | CategoricalSketch:
    if not isinstance(state, dict):
        return kind()
    value = state.get(name)
    if not isinstance(value, dict):
        return kind()
    return kind.from_state(value)


def _string_tokens(values: Iterable[str]) -> list[JSONValue]:
    return list(values)


def serialize_field_stats(stats: FieldStats) -> JSONDocument:
    """Project one field's statistics without retaining source literals.

    Equality is retained as bounded SHA-256 tokens. Only the fixed structural
    role vocabulary remains readable, which is the explicit allowlist used by
    semantic-role inference and public enum annotation.
    """
    safe_sessions: list[JSONValue] = []
    for value in sorted(stats.safe_observed_values):
        value_digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        safe_sessions.append([value, _string_tokens(sorted(stats.equality_session_tokens.get(value_digest, set())))])
    return {
        "version": FIELD_EVIDENCE_VERSION,
        "path": stats.path,
        "counts": {
            "total_samples": stats.total_samples,
            "present": stats.present_count,
            "value": stats.value_count,
            "null": stats.null_count,
            "encountered_documents": stats.document_encountered_count,
            "non_null_documents": stats.document_non_null_count,
            "multiline": stats.is_multiline,
            "ordered_pairs": stats.ordered_pair_count,
            "ordered_increasing_pairs": stats.ordered_increasing_pair_count,
            "max_depth": stats.max_depth_seen,
            "slash_values": stats.slash_value_count,
        },
        "number_range": [stats.num_min, stats.num_max],
        "types": _counter_state(stats.type_counts),
        "formats": _counter_state(stats.detected_formats),
        "booleans": _counter_state(stats.boolean_counts),
        "co_occurring_fields": _counter_state(stats.co_occurring_fields),
        "truncated": _counter_state(stats.truncated_evidence),
        "equality_hashes": _counter_state(stats.equality_hash_counts),
        "equality_sessions": [
            [digest, _string_tokens(sorted(tokens))] for digest, tokens in sorted(stats.equality_session_tokens.items())
        ],
        "safe_values": _counter_state(stats.safe_observed_values),
        "safe_value_sessions": safe_sessions,
        "first_seen": stats.field_first_seen,
        "last_seen": stats.field_last_seen,
        "distributions": _distribution_state(stats),
    }


def deserialize_field_stats(state: JSONDocument) -> FieldStats:
    """Restore field statistics from :func:`serialize_field_stats`."""
    if state.get("version") != FIELD_EVIDENCE_VERSION:
        raise ValueError("unsupported field evidence version")
    path = state.get("path")
    if not isinstance(path, str):
        raise ValueError("field evidence path is missing")
    counts = state.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("field evidence counts are missing")
    distributions = state.get("distributions")
    safe_values = _counter_from_state(state.get("safe_values"))
    stats = FieldStats(
        path=path,
        observed_values=Counter(safe_values),
        safe_observed_values=safe_values,
        detected_formats=_counter_from_state(state.get("formats")),
        type_counts=_counter_from_state(state.get("types")),
        boolean_counts=_counter_from_state(state.get("booleans")),
        co_occurring_fields=_counter_from_state(state.get("co_occurring_fields")),
        truncated_evidence=_counter_from_state(state.get("truncated")),
        equality_hash_counts=_counter_from_state(state.get("equality_hashes")),
        total_samples=_state_int(counts.get("total_samples", 0)),
        present_count=_state_int(counts.get("present", 0)),
        value_count=_state_int(counts.get("value", 0)),
        null_count=_state_int(counts.get("null", 0)),
        document_encountered_count=_state_int(counts.get("encountered_documents", 0)),
        document_non_null_count=_state_int(counts.get("non_null_documents", 0)),
        is_multiline=_state_int(counts.get("multiline", 0)),
        ordered_pair_count=_state_int(counts.get("ordered_pairs", 0)),
        ordered_increasing_pair_count=_state_int(counts.get("ordered_increasing_pairs", 0)),
        max_depth_seen=_state_int(counts.get("max_depth", 0)),
        slash_value_count=_state_int(counts.get("slash_values", 0)),
        field_first_seen=_state_string(state.get("first_seen", None)),
        field_last_seen=_state_string(state.get("last_seen", None)),
        string_length_distribution=_distribution_from_state(distributions, "string_length", DistributionSketch),
        newline_distribution=_distribution_from_state(distributions, "newline", DistributionSketch),
        numeric_distribution=_distribution_from_state(distributions, "numeric", DistributionSketch),
        array_length_distribution=_distribution_from_state(distributions, "array_length", DistributionSketch),
        object_fanout_distribution=_distribution_from_state(distributions, "object_fanout", DistributionSketch),
        categorical_distribution=_distribution_from_state(distributions, "categorical", CategoricalSketch),
        object_key_distribution=_distribution_from_state(distributions, "object_key", CategoricalSketch),
        co_occurrence_distribution=_distribution_from_state(distributions, "co_occurrence", CategoricalSketch),
    )
    number_range = state.get("number_range")
    if isinstance(number_range, list) and len(number_range) == 2:
        stats.num_min = float(number_range[0]) if isinstance(number_range[0], (int, float)) else None
        stats.num_max = float(number_range[1]) if isinstance(number_range[1], (int, float)) else None
    sessions = state.get("equality_sessions")
    if isinstance(sessions, list):
        stats.equality_session_tokens = {
            digest: {token for token in tokens if isinstance(token, str)}
            for item in sessions
            if isinstance(item, list)
            and len(item) == 2
            and isinstance((digest := item[0]), str)
            and isinstance((tokens := item[1]), list)
        }
    safe_sessions = state.get("safe_value_sessions")
    if isinstance(safe_sessions, list):
        for item in safe_sessions:
            if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], list):
                value = item[0]
                digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
                stats.value_session_ids[value] = {token for token in item[1] if isinstance(token, str)}
                stats.equality_session_tokens.setdefault(digest, set()).update(stats.value_session_ids[value])
    return stats


def merge_field_stats(
    stats_by_source: Iterable[Mapping[str, FieldStats]], *, total_samples: int | None = None
) -> dict[str, FieldStats]:
    """Merge current source summaries in caller-supplied deterministic order."""
    merged: dict[str, FieldStats] = {}
    for source_stats in stats_by_source:
        for path, source in sorted(source_stats.items()):
            target = merged.setdefault(path, FieldStats(path=path))
            _merge_one(target, source)
            _bound_equality_evidence(target)
    for stats in merged.values():
        if total_samples is not None:
            stats.total_samples = total_samples
        stats.observed_values = Counter(stats.safe_observed_values)
    return merged


def _merge_one(target: FieldStats, source: FieldStats) -> None:
    target.total_samples += source.total_samples
    target.present_count += source.present_count
    target.value_count += source.value_count
    target.null_count += source.null_count
    target.document_encountered_count += source.document_encountered_count
    target.document_non_null_count += source.document_non_null_count
    target.is_multiline += source.is_multiline
    target.ordered_pair_count += source.ordered_pair_count
    target.ordered_increasing_pair_count += source.ordered_increasing_pair_count
    target.max_depth_seen = max(target.max_depth_seen, source.max_depth_seen)
    target.slash_value_count += source.slash_value_count
    target.num_min = (
        source.num_min
        if target.num_min is None
        else min(target.num_min, source.num_min)
        if source.num_min is not None
        else target.num_min
    )
    target.num_max = (
        source.num_max
        if target.num_max is None
        else max(target.num_max, source.num_max)
        if source.num_max is not None
        else target.num_max
    )
    for name in (
        "type_counts",
        "detected_formats",
        "boolean_counts",
        "co_occurring_fields",
        "truncated_evidence",
        "equality_hash_counts",
        "safe_observed_values",
    ):
        getattr(target, name).update(getattr(source, name))
    for name in (
        "string_length_distribution",
        "newline_distribution",
        "numeric_distribution",
        "array_length_distribution",
        "object_fanout_distribution",
        "categorical_distribution",
        "object_key_distribution",
        "co_occurrence_distribution",
    ):
        getattr(target, name).merge(getattr(source, name))
    for digest, tokens in source.equality_session_tokens.items():
        target.equality_session_tokens.setdefault(digest, set()).update(tokens)
    for value, tokens in source.value_session_ids.items():
        target.value_session_ids.setdefault(value, set()).update(tokens)
    candidates = [value for value in (target.field_first_seen, source.field_first_seen) if value]
    target.field_first_seen = min(candidates) if candidates else None
    candidates = [value for value in (target.field_last_seen, source.field_last_seen) if value]
    target.field_last_seen = max(candidates) if candidates else None


def _bound_equality_evidence(stats: FieldStats) -> None:
    if len(stats.equality_hash_counts) > EQUALITY_EVIDENCE_CAP:
        retained = set(sorted(stats.equality_hash_counts)[:EQUALITY_EVIDENCE_CAP])
        removed = len(stats.equality_hash_counts) - len(retained)
        stats.equality_hash_counts = Counter(
            {digest: count for digest, count in stats.equality_hash_counts.items() if digest in retained}
        )
        stats.equality_session_tokens = {
            digest: tokens for digest, tokens in stats.equality_session_tokens.items() if digest in retained
        }
        stats.truncated_evidence["equality_hashes"] += removed
    for tokens in stats.equality_session_tokens.values():
        if len(tokens) > SESSION_EVIDENCE_CAP:
            tokens.intersection_update(sorted(tokens)[:SESSION_EVIDENCE_CAP])
            stats.truncated_evidence["equality_sessions"] += 1
    for _value, tokens in stats.value_session_ids.items():
        if len(tokens) > SESSION_EVIDENCE_CAP:
            tokens.intersection_update(sorted(tokens)[:SESSION_EVIDENCE_CAP])
            stats.truncated_evidence["enum_session_ids"] += 1


__all__ = ["FIELD_EVIDENCE_VERSION", "deserialize_field_stats", "merge_field_stats", "serialize_field_stats"]

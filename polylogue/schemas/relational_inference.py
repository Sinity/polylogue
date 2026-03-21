"""Relational inference for schema fields.

Detects structural relationships between JSON paths based on statistical
evidence from FieldStats:
    - foreign-key references (value sets matching dict key sets)
    - temporal offsets (time deltas between timestamp fields)
    - mutually exclusive field groups (fields that never co-occur)
    - string-length distributions worth preserving in annotations
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from polylogue.schemas.field_stats import FieldStats


@dataclass(frozen=True, slots=True)
class ForeignKeyRelation:
    """A detected foreign-key-like reference between fields."""

    source_path: str
    target_path: str
    match_ratio: float  # fraction of source values found in target keys
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TimeDeltaRelation:
    """A detected temporal offset between two timestamp fields."""

    field_a: str
    field_b: str
    min_delta: float  # seconds
    max_delta: float
    avg_delta: float
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MutualExclusion:
    """A group of fields that never co-occur in the same object."""

    parent_path: str
    field_names: frozenset[str]
    sample_count: int  # number of samples observed
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StringLengthProfile:
    """String length distribution for a field worth preserving."""

    path: str
    min_length: int
    max_length: int
    avg_length: float
    stddev: float
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationalAnnotations:
    """All detected relational annotations."""

    foreign_keys: list[ForeignKeyRelation] = field(default_factory=list)
    time_deltas: list[TimeDeltaRelation] = field(default_factory=list)
    mutual_exclusions: list[MutualExclusion] = field(default_factory=list)
    string_lengths: list[StringLengthProfile] = field(default_factory=list)


def infer_relations(stats: dict[str, FieldStats]) -> RelationalAnnotations:
    """Detect all relational annotations from collected field statistics."""
    return RelationalAnnotations(
        foreign_keys=_detect_foreign_keys(stats),
        time_deltas=_detect_time_deltas(stats),
        mutual_exclusions=_detect_mutual_exclusions(stats),
        string_lengths=_detect_string_lengths(stats),
    )


# ---------------------------------------------------------------------------
# Foreign key detection
# ---------------------------------------------------------------------------

_FK_MATCH_THRESHOLD = 0.6


def _detect_foreign_keys(stats: dict[str, FieldStats]) -> list[ForeignKeyRelation]:
    """Detect fields whose values mostly match keys in some dict field."""
    results: list[ForeignKeyRelation] = []

    # Find fields with _ref_target already set by _collect_field_stats
    for path, fs in stats.items():
        ref_target = getattr(fs, "_ref_target", None)
        if ref_target:
            results.append(ForeignKeyRelation(
                source_path=path,
                target_path=ref_target,
                match_ratio=1.0,  # already passed threshold in field_stats
                evidence={"source": "field_stats_ref_detection"},
            ))

    # Also check for high-cardinality string fields that reference other paths
    # by looking for path segment overlap (e.g., $.mapping.*.parent → $.mapping.*)
    for path, fs in stats.items():
        if not fs.observed_values or len(fs.observed_values) <= 5:
            continue

        # Look for paths whose observed values match the observed values
        # of another ID-like field
        terminal = path.rsplit(".", 1)[-1].lower() if "." in path else path.lower()
        if terminal not in {"parent", "parentid", "parent_id", "parentuuid",
                           "parent_uuid", "ref", "reference", "source_id"}:
            continue

        # Already found via _ref_target? Skip.
        if getattr(fs, "_ref_target", None):
            continue

        # Check if values match any other field's values
        observed = set(fs.observed_values.keys())
        for other_path, other_fs in stats.items():
            if other_path == path:
                continue
            other_terminal = other_path.rsplit(".", 1)[-1].lower() if "." in other_path else other_path.lower()
            if other_terminal not in {"id", "uuid", "key", "node_id"}:
                continue
            if not other_fs.observed_values:
                continue
            other_values = set(other_fs.observed_values.keys())
            if not other_values:
                continue
            overlap = len(observed & other_values)
            ratio = overlap / len(observed) if observed else 0
            if ratio >= _FK_MATCH_THRESHOLD:
                results.append(ForeignKeyRelation(
                    source_path=path,
                    target_path=other_path,
                    match_ratio=ratio,
                    evidence={
                        "overlap_count": overlap,
                        "source_count": len(observed),
                        "target_count": len(other_values),
                    },
                ))

    return results


# ---------------------------------------------------------------------------
# Time delta detection
# ---------------------------------------------------------------------------

def _detect_time_deltas(stats: dict[str, FieldStats]) -> list[TimeDeltaRelation]:
    """Detect temporal offsets between timestamp-like fields in the same object."""
    results: list[TimeDeltaRelation] = []

    # Find pairs of timestamp fields at the same depth level
    ts_fields: list[tuple[str, FieldStats]] = []
    for path, fs in stats.items():
        fmt = fs.dominant_format
        if fmt in {"unix-epoch", "unix-epoch-str", "iso8601"}:
            ts_fields.append((path, fs))
        elif fs.num_min is not None and fs.num_max is not None:
            if 946684800.0 <= fs.num_min and fs.num_max <= 2208988800.0:
                ts_fields.append((path, fs))

    # Compare pairs of timestamp fields at the same depth
    for i, (path_a, fs_a) in enumerate(ts_fields):
        for path_b, fs_b in ts_fields[i + 1:]:
            # Must be at the same depth (siblings in the same object)
            parent_a = path_a.rsplit(".", 1)[0] if "." in path_a else "$"
            parent_b = path_b.rsplit(".", 1)[0] if "." in path_b else "$"
            if parent_a != parent_b:
                continue

            # Both must have numeric ranges
            if fs_a.num_min is None or fs_b.num_min is None:
                continue
            if fs_a.num_max is None or fs_b.num_max is None:
                continue

            # Compute approximate delta range
            # (these are approximate since we don't have paired values)
            min_delta = abs(fs_b.num_min - fs_a.num_max)
            max_delta = abs(fs_b.num_max - fs_a.num_min)
            avg_delta = (min_delta + max_delta) / 2

            if max_delta > 0:
                results.append(TimeDeltaRelation(
                    field_a=path_a,
                    field_b=path_b,
                    min_delta=min_delta,
                    max_delta=max_delta,
                    avg_delta=avg_delta,
                    evidence={
                        "field_a_range": [fs_a.num_min, fs_a.num_max],
                        "field_b_range": [fs_b.num_min, fs_b.num_max],
                    },
                ))

    return results


# ---------------------------------------------------------------------------
# Mutual exclusion detection
# ---------------------------------------------------------------------------

def _detect_mutual_exclusions(stats: dict[str, FieldStats]) -> list[MutualExclusion]:
    """Detect groups of fields that never co-occur in the same object."""
    results: list[MutualExclusion] = []

    # Group fields by parent path
    parent_children: dict[str, list[str]] = {}
    for path in stats:
        if "." not in path:
            continue
        parent, child = path.rsplit(".", 1)
        if "[*]" in child:
            continue
        parent_children.setdefault(parent, []).append(child)

    # For each parent, check co-occurrence patterns
    for parent, children in parent_children.items():
        if len(children) < 2:
            continue

        # Build co-occurrence matrix from the stats
        # Two fields are mutually exclusive if their co-occurrence count is 0
        exclusive_pairs: list[tuple[str, str]] = []
        for i, name_a in enumerate(children):
            path_a = f"{parent}.{name_a}"
            fs_a = stats.get(path_a)
            if not fs_a or fs_a.present_count == 0:
                continue
            for name_b in children[i + 1:]:
                path_b = f"{parent}.{name_b}"
                fs_b = stats.get(path_b)
                if not fs_b or fs_b.present_count == 0:
                    continue

                # Check if name_b ever appears in name_a's co-occurring fields
                co_count = fs_a.co_occurring_fields.get(name_b, 0)
                if co_count == 0:
                    exclusive_pairs.append((name_a, name_b))

        # Group exclusive pairs into cliques (simplified: just report pairs)
        for name_a, name_b in exclusive_pairs:
            path_a = f"{parent}.{name_a}"
            path_b = f"{parent}.{name_b}"
            fs_a = stats[path_a]
            fs_b = stats[path_b]
            results.append(MutualExclusion(
                parent_path=parent,
                field_names=frozenset({name_a, name_b}),
                sample_count=max(fs_a.total_samples, fs_b.total_samples),
                evidence={
                    f"{name_a}_present": fs_a.present_count,
                    f"{name_b}_present": fs_b.present_count,
                },
            ))

    return results


# ---------------------------------------------------------------------------
# String length distribution
# ---------------------------------------------------------------------------

def _detect_string_lengths(stats: dict[str, FieldStats]) -> list[StringLengthProfile]:
    """Extract interesting string length distributions."""
    results: list[StringLengthProfile] = []

    for path, fs in stats.items():
        sl = fs.string_length_stats
        if not sl:
            continue

        # Only include fields with meaningful variance or notable bounds
        avg = sl["avg"]
        stddev = sl["stddev"]

        # Skip trivially short fields (status codes, roles, booleans)
        if avg < 5 and stddev < 2:
            continue

        # Skip if too few samples
        if len(fs.string_lengths) < 3:
            continue

        results.append(StringLengthProfile(
            path=path,
            min_length=sl["min"],
            max_length=sl["max"],
            avg_length=avg,
            stddev=stddev,
            evidence={
                "sample_count": len(fs.string_lengths),
                "multiline_rate": round(fs.newline_rate, 3),
            },
        ))

    return results


__all__ = [
    "ForeignKeyRelation",
    "MutualExclusion",
    "RelationalAnnotations",
    "StringLengthProfile",
    "TimeDeltaRelation",
    "infer_relations",
]

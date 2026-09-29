"""Typed models for relational schema annotations."""

from __future__ import annotations

from dataclasses import dataclass, field

from polylogue.core.json import JSONDocument

RelationEvidence = JSONDocument


@dataclass(frozen=True, slots=True)
class ForeignKeyRelation:
    """A detected foreign-key-like reference between fields."""

    source_path: str
    target_path: str
    match_ratio: float
    evidence: RelationEvidence = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TimeDeltaRelation:
    """A detected temporal offset between two timestamp fields."""

    field_a: str
    field_b: str
    min_delta: float
    max_delta: float
    avg_delta: float
    evidence: RelationEvidence = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MutualExclusion:
    """A group of fields that never co-occur in the same object."""

    parent_path: str
    field_names: frozenset[str]
    sample_count: int
    evidence: RelationEvidence = field(default_factory=dict)


@dataclass
class RelationalAnnotations:
    """All detected relational annotations."""

    foreign_keys: list[ForeignKeyRelation] = field(default_factory=list)
    time_deltas: list[TimeDeltaRelation] = field(default_factory=list)
    mutual_exclusions: list[MutualExclusion] = field(default_factory=list)


__all__ = [
    "ForeignKeyRelation",
    "MutualExclusion",
    "RelationalAnnotations",
    "RelationEvidence",
    "TimeDeltaRelation",
]

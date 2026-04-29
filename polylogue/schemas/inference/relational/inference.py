"""Small public root for relational schema inference."""

from __future__ import annotations

from polylogue.schemas.field_stats import FieldStats
from polylogue.schemas.relational_inference_exclusions import detect_mutual_exclusions
from polylogue.schemas.relational_inference_foreign_keys import detect_foreign_keys
from polylogue.schemas.relational_inference_models import (
    ForeignKeyRelation,
    MutualExclusion,
    RelationalAnnotations,
    StringLengthProfile,
    TimeDeltaRelation,
)
from polylogue.schemas.relational_inference_strings import detect_string_lengths
from polylogue.schemas.relational_inference_time import detect_time_deltas


def infer_relations(stats: dict[str, FieldStats]) -> RelationalAnnotations:
    """Detect all relational annotations from collected field statistics."""
    return RelationalAnnotations(
        foreign_keys=detect_foreign_keys(stats),
        time_deltas=detect_time_deltas(stats),
        mutual_exclusions=detect_mutual_exclusions(stats),
        string_lengths=detect_string_lengths(stats),
    )


__all__ = [
    "ForeignKeyRelation",
    "MutualExclusion",
    "RelationalAnnotations",
    "StringLengthProfile",
    "TimeDeltaRelation",
    "infer_relations",
]

"""Relational constraint satisfaction for synthetic generation.

Uses ``x-polylogue-foreign-keys``, ``x-polylogue-time-deltas``,
``x-polylogue-mutually-exclusive``, and ``x-polylogue-string-lengths``
annotations on schemas to produce structurally coherent synthetic data.

These constraints are read from the schema root and applied during the
object-generation phase to enforce cross-field consistency.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from polylogue.schemas.synthetic.relation_solver_runtime import RelationConstraintSolverRuntimeMixin


@dataclass
class ForeignKeyGraph:
    """Reference graph built from x-polylogue-foreign-keys.

    Tracks generated ID values so foreign-key fields can reference
    previously generated IDs.
    """

    # source_path -> target_path
    references: dict[str, str] = field(default_factory=dict)
    # target_path -> list of generated values
    generated_ids: dict[str, list[str]] = field(default_factory=dict)

    def register_id(self, target_path: str, value: str) -> None:
        """Register a generated ID value for a target path."""
        self.generated_ids.setdefault(target_path, []).append(value)

    def resolve_reference(self, source_path: str, rng: random.Random) -> str | None:
        """Resolve a foreign-key reference to a previously generated ID."""
        target = self.references.get(source_path)
        if target is None:
            return None
        ids = self.generated_ids.get(target)
        if not ids:
            return None
        return rng.choice(ids)


@dataclass
class TimeDeltaConstraint:
    """Constraint between two timestamp fields."""

    field_a: str
    field_b: str
    min_delta: float
    max_delta: float
    avg_delta: float

    @property
    def stddev_approx(self) -> float:
        """Approximate stddev from the delta range."""
        return (self.max_delta - self.min_delta) / 4.0


@dataclass
class MutualExclusionGroup:
    """Group of fields that must not co-occur."""

    parent_path: str
    field_names: frozenset[str]


@dataclass
class StringLengthConstraint:
    """Length distribution constraint for a string field."""

    path: str
    min_length: int
    max_length: int
    avg_length: float
    stddev: float


class RelationConstraintSolver(RelationConstraintSolverRuntimeMixin):
    """Applies relational constraints from schema annotations during generation.

    Constructed once per corpus from the schema's root-level relational
    annotations.  Methods are called during object generation to enforce
    consistency.
    """

    def __init__(self, schema: dict[str, Any]) -> None:
        self.fk_graph = ForeignKeyGraph()
        self.time_deltas: list[TimeDeltaConstraint] = []
        self.mutual_exclusions: list[MutualExclusionGroup] = []
        self.string_lengths: dict[str, StringLengthConstraint] = {}
        self._time_delta_cls = TimeDeltaConstraint
        self._mutual_exclusion_cls = MutualExclusionGroup
        self._string_length_cls = StringLengthConstraint

        self._parse_foreign_keys(schema)
        self._parse_time_deltas(schema)
        self._parse_mutual_exclusions(schema)
        self._parse_string_lengths(schema)

    @property
    def has_constraints(self) -> bool:
        """Whether any relational constraints were parsed."""
        return bool(
            self.fk_graph.references
            or self.time_deltas
            or self.mutual_exclusions
            or self.string_lengths
        )



__all__ = [
    "ForeignKeyGraph",
    "MutualExclusionGroup",
    "RelationConstraintSolver",
    "StringLengthConstraint",
    "TimeDeltaConstraint",
]

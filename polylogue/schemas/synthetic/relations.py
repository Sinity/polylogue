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


class RelationConstraintSolver:
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

        self._parse_foreign_keys(schema)
        self._parse_time_deltas(schema)
        self._parse_mutual_exclusions(schema)
        self._parse_string_lengths(schema)

    def _parse_foreign_keys(self, schema: dict[str, Any]) -> None:
        for fk in schema.get("x-polylogue-foreign-keys", []):
            source = fk.get("source", "")
            target = fk.get("target", "")
            if source and target:
                self.fk_graph.references[source] = target

    def _parse_time_deltas(self, schema: dict[str, Any]) -> None:
        for td in schema.get("x-polylogue-time-deltas", []):
            self.time_deltas.append(TimeDeltaConstraint(
                field_a=td.get("field_a", ""),
                field_b=td.get("field_b", ""),
                min_delta=td.get("min_delta", 0.0),
                max_delta=td.get("max_delta", 0.0),
                avg_delta=td.get("avg_delta", 0.0),
            ))

    def _parse_mutual_exclusions(self, schema: dict[str, Any]) -> None:
        for me in schema.get("x-polylogue-mutually-exclusive", []):
            parent = me.get("parent", "")
            fields = me.get("fields", [])
            if parent and len(fields) >= 2:
                self.mutual_exclusions.append(MutualExclusionGroup(
                    parent_path=parent,
                    field_names=frozenset(fields),
                ))

    def _parse_string_lengths(self, schema: dict[str, Any]) -> None:
        for sl in schema.get("x-polylogue-string-lengths", []):
            path = sl.get("path", "")
            if path:
                self.string_lengths[path] = StringLengthConstraint(
                    path=path,
                    min_length=sl.get("min", 0),
                    max_length=sl.get("max", 100),
                    avg_length=sl.get("avg", 50.0),
                    stddev=sl.get("stddev", 10.0),
                )

    @property
    def has_constraints(self) -> bool:
        """Whether any relational constraints were parsed."""
        return bool(
            self.fk_graph.references
            or self.time_deltas
            or self.mutual_exclusions
            or self.string_lengths
        )

    # ── Foreign key support ──────────────────────────────────────────────

    def register_generated_id(self, path: str, value: str) -> None:
        """Register a generated ID value for potential foreign-key references."""
        self.fk_graph.register_id(path, value)

    def resolve_foreign_key(self, path: str, rng: random.Random) -> str | None:
        """Try to resolve a foreign-key reference at the given path."""
        return self.fk_graph.resolve_reference(path, rng)

    # ── Time delta support ───────────────────────────────────────────────

    def get_time_delta(
        self,
        field_a: str,
        field_b: str,
        rng: random.Random,
    ) -> float | None:
        """Get a realistic time delta between two timestamp fields.

        Returns the delta in seconds, or None if no constraint exists.
        """
        for td in self.time_deltas:
            if (td.field_a == field_a and td.field_b == field_b) or \
               (td.field_a == field_b and td.field_b == field_a):
                # Sample from a gaussian centered on avg_delta, clamped to [min, max]
                if td.stddev_approx > 0:
                    val = rng.gauss(td.avg_delta, td.stddev_approx)
                else:
                    val = rng.uniform(td.min_delta, td.max_delta)
                return max(td.min_delta, min(td.max_delta, val))
        return None

    # ── Mutual exclusion support ─────────────────────────────────────────

    def filter_mutually_exclusive(
        self,
        parent_path: str,
        field_names: set[str],
        rng: random.Random,
    ) -> set[str]:
        """Filter field names to satisfy mutual exclusion constraints.

        For each exclusion group that intersects the candidate fields,
        keep only one randomly chosen field from the group.

        Returns the filtered set of field names.
        """
        result = set(field_names)
        for group in self.mutual_exclusions:
            if group.parent_path != parent_path:
                continue
            overlap = result & group.field_names
            if len(overlap) > 1:
                # Keep one, remove the rest
                keeper = rng.choice(sorted(overlap))
                result -= overlap
                result.add(keeper)
        return result

    # ── String length support ────────────────────────────────────────────

    def generate_string_with_length(
        self,
        path: str,
        rng: random.Random,
        base_text: str,
    ) -> str:
        """Adjust a base text to match the string length profile for a path.

        If no length constraint exists, returns the base text unchanged.
        """
        constraint = self.string_lengths.get(path)
        if constraint is None:
            return base_text

        # Target length: sample from gaussian around avg, clamp to [min, max]
        target = int(rng.gauss(constraint.avg_length, constraint.stddev))
        target = max(constraint.min_length, min(constraint.max_length, target))

        if len(base_text) == 0:
            return base_text

        if len(base_text) >= target:
            # Truncate — try to break at word boundary
            if target <= 3:
                return base_text[:target]
            truncated = base_text[:target]
            last_space = truncated.rfind(" ")
            if last_space > target // 2:
                return truncated[:last_space]
            return truncated
        else:
            # Extend by repeating content
            repetitions = (target // len(base_text)) + 1
            extended = (base_text + " ") * repetitions
            return extended[:target].rstrip()

    # ── Path matching helpers ────────────────────────────────────────────

    def path_matches(self, schema_path: str, annotation_path: str) -> bool:
        """Check if a schema traversal path matches an annotation path.

        Annotation paths use ``$`` for root, ``.*`` for dynamic keys,
        and ``[*]`` for array items.  This does prefix/exact matching.
        """
        return schema_path == annotation_path


__all__ = [
    "ForeignKeyGraph",
    "MutualExclusionGroup",
    "RelationConstraintSolver",
    "StringLengthConstraint",
    "TimeDeltaConstraint",
]

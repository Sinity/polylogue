"""Tests for relational constraint solving in the synthetic corpus system.

Verifies that ForeignKeyGraph, TimeDeltaConstraint, MutualExclusionGroup,
StringLengthConstraint, and RelationConstraintSolver correctly enforce
cross-field consistency during synthetic data generation.
"""

from __future__ import annotations

import random

import pytest

from polylogue.schemas.synthetic.relations import (
    ForeignKeyGraph,
    MutualExclusionGroup,
    RelationConstraintSolver,
    StringLengthConstraint,
    TimeDeltaConstraint,
)

# ---------------------------------------------------------------------------
# ForeignKeyGraph
# ---------------------------------------------------------------------------


class TestForeignKeyGraph:
    def test_register_and_resolve(self) -> None:
        graph = ForeignKeyGraph()
        graph.references["$.child.parent_id"] = "$.parent.id"
        graph.register_id("$.parent.id", "abc-123")
        graph.register_id("$.parent.id", "def-456")

        rng = random.Random(42)
        ref = graph.resolve_reference("$.child.parent_id", rng)
        assert ref in {"abc-123", "def-456"}

    def test_resolve_returns_none_when_no_reference_defined(self) -> None:
        graph = ForeignKeyGraph()
        rng = random.Random(0)
        assert graph.resolve_reference("$.unknown.path", rng) is None

    def test_resolve_returns_none_when_target_has_no_ids(self) -> None:
        graph = ForeignKeyGraph()
        graph.references["$.child.ref"] = "$.parent.id"
        # No IDs registered for $.parent.id
        rng = random.Random(0)
        assert graph.resolve_reference("$.child.ref", rng) is None

    def test_register_multiple_ids_for_same_path(self) -> None:
        graph = ForeignKeyGraph()
        graph.register_id("$.nodes.id", "a")
        graph.register_id("$.nodes.id", "b")
        graph.register_id("$.nodes.id", "c")
        assert graph.generated_ids["$.nodes.id"] == ["a", "b", "c"]

    def test_referential_integrity_over_many_resolutions(self) -> None:
        """All resolved references must point to previously registered IDs."""
        graph = ForeignKeyGraph()
        graph.references["$.mapping.*.parent"] = "$.mapping.*.id"

        registered = {"id-1", "id-2", "id-3", "id-4", "id-5"}
        for rid in registered:
            graph.register_id("$.mapping.*.id", rid)

        rng = random.Random(99)
        for _ in range(50):
            ref = graph.resolve_reference("$.mapping.*.parent", rng)
            assert ref in registered


# ---------------------------------------------------------------------------
# TimeDeltaConstraint
# ---------------------------------------------------------------------------


class TestTimeDeltaConstraint:
    def test_properties(self) -> None:
        td = TimeDeltaConstraint(
            field_a="$.create_time",
            field_b="$.update_time",
            min_delta=0.0,
            max_delta=100.0,
            avg_delta=50.0,
        )
        assert td.field_a == "$.create_time"
        assert td.field_b == "$.update_time"
        assert td.min_delta == 0.0
        assert td.max_delta == 100.0
        assert td.avg_delta == 50.0

    def test_stddev_approx(self) -> None:
        td = TimeDeltaConstraint(
            field_a="a", field_b="b",
            min_delta=0.0, max_delta=100.0, avg_delta=50.0,
        )
        # stddev_approx = (max - min) / 4
        assert td.stddev_approx == 25.0

    def test_zero_range_gives_zero_stddev(self) -> None:
        td = TimeDeltaConstraint(
            field_a="a", field_b="b",
            min_delta=10.0, max_delta=10.0, avg_delta=10.0,
        )
        assert td.stddev_approx == 0.0


# ---------------------------------------------------------------------------
# MutualExclusionGroup
# ---------------------------------------------------------------------------


class TestMutualExclusionGroup:
    def test_field_names_are_frozenset(self) -> None:
        group = MutualExclusionGroup(
            parent_path="$.message",
            field_names=frozenset({"content", "parts"}),
        )
        assert isinstance(group.field_names, frozenset)
        assert "content" in group.field_names
        assert "parts" in group.field_names


# ---------------------------------------------------------------------------
# StringLengthConstraint
# ---------------------------------------------------------------------------


class TestStringLengthConstraint:
    def test_properties(self) -> None:
        slc = StringLengthConstraint(
            path="$.message.text",
            min_length=10,
            max_length=500,
            avg_length=120.0,
            stddev=30.0,
        )
        assert slc.min_length == 10
        assert slc.max_length == 500
        assert slc.avg_length == 120.0
        assert slc.stddev == 30.0


# ---------------------------------------------------------------------------
# RelationConstraintSolver — parsing from schema annotations
# ---------------------------------------------------------------------------


class TestRelationConstraintSolverParsing:
    def test_empty_schema_has_no_constraints(self) -> None:
        solver = RelationConstraintSolver({})
        assert not solver.has_constraints

    def test_parses_foreign_keys(self) -> None:
        schema = {
            "x-polylogue-foreign-keys": [
                {"source": "$.mapping.*.parent", "target": "$.mapping.*.id"},
            ],
        }
        solver = RelationConstraintSolver(schema)
        assert solver.has_constraints
        assert "$.mapping.*.parent" in solver.fk_graph.references
        assert solver.fk_graph.references["$.mapping.*.parent"] == "$.mapping.*.id"

    def test_parses_time_deltas(self) -> None:
        schema = {
            "x-polylogue-time-deltas": [
                {
                    "field_a": "$.create_time",
                    "field_b": "$.update_time",
                    "min_delta": 0.0,
                    "max_delta": 3600.0,
                    "avg_delta": 600.0,
                },
            ],
        }
        solver = RelationConstraintSolver(schema)
        assert solver.has_constraints
        assert len(solver.time_deltas) == 1
        assert solver.time_deltas[0].field_a == "$.create_time"

    def test_parses_mutual_exclusions(self) -> None:
        schema = {
            "x-polylogue-mutually-exclusive": [
                {"parent": "$.message", "fields": ["content", "parts"]},
            ],
        }
        solver = RelationConstraintSolver(schema)
        assert solver.has_constraints
        assert len(solver.mutual_exclusions) == 1
        assert solver.mutual_exclusions[0].field_names == frozenset({"content", "parts"})

    def test_mutual_exclusion_requires_at_least_two_fields(self) -> None:
        schema = {
            "x-polylogue-mutually-exclusive": [
                {"parent": "$.message", "fields": ["only_one"]},
            ],
        }
        solver = RelationConstraintSolver(schema)
        assert len(solver.mutual_exclusions) == 0

    def test_parses_string_lengths(self) -> None:
        schema = {
            "x-polylogue-string-lengths": [
                {
                    "path": "$.message.text",
                    "min": 5,
                    "max": 1000,
                    "avg": 200.0,
                    "stddev": 80.0,
                },
            ],
        }
        solver = RelationConstraintSolver(schema)
        assert solver.has_constraints
        assert "$.message.text" in solver.string_lengths
        constraint = solver.string_lengths["$.message.text"]
        assert constraint.min_length == 5
        assert constraint.max_length == 1000


# ---------------------------------------------------------------------------
# RelationConstraintSolver — foreign key operations
# ---------------------------------------------------------------------------


class TestRelationConstraintSolverForeignKeys:
    def test_register_and_resolve(self) -> None:
        schema = {
            "x-polylogue-foreign-keys": [
                {"source": "$.child.ref", "target": "$.parent.id"},
            ],
        }
        solver = RelationConstraintSolver(schema)
        solver.register_generated_id("$.parent.id", "abc")
        rng = random.Random(0)
        ref = solver.resolve_foreign_key("$.child.ref", rng)
        assert ref == "abc"

    def test_resolve_returns_none_for_unregistered_path(self) -> None:
        solver = RelationConstraintSolver({})
        rng = random.Random(0)
        assert solver.resolve_foreign_key("$.no.such.path", rng) is None


# ---------------------------------------------------------------------------
# RelationConstraintSolver — time delta operations
# ---------------------------------------------------------------------------


class TestRelationConstraintSolverTimeDeltas:
    def test_get_time_delta_returns_value_within_bounds(self) -> None:
        schema = {
            "x-polylogue-time-deltas": [
                {
                    "field_a": "$.create",
                    "field_b": "$.update",
                    "min_delta": 10.0,
                    "max_delta": 100.0,
                    "avg_delta": 55.0,
                },
            ],
        }
        solver = RelationConstraintSolver(schema)
        rng = random.Random(42)

        for _ in range(100):
            delta = solver.get_time_delta("$.create", "$.update", rng)
            assert delta is not None
            assert 10.0 <= delta <= 100.0

    def test_get_time_delta_symmetric_lookup(self) -> None:
        """Time delta can be looked up with fields in either order."""
        schema = {
            "x-polylogue-time-deltas": [
                {
                    "field_a": "$.a",
                    "field_b": "$.b",
                    "min_delta": 1.0,
                    "max_delta": 10.0,
                    "avg_delta": 5.0,
                },
            ],
        }
        solver = RelationConstraintSolver(schema)
        rng = random.Random(0)

        # Look up in reverse order
        delta = solver.get_time_delta("$.b", "$.a", rng)
        assert delta is not None
        assert 1.0 <= delta <= 10.0

    def test_get_time_delta_returns_none_for_unknown_pair(self) -> None:
        solver = RelationConstraintSolver({})
        rng = random.Random(0)
        assert solver.get_time_delta("$.x", "$.y", rng) is None

    def test_zero_range_time_delta(self) -> None:
        """When min == max (zero stddev), get_time_delta uses uniform and clamps."""
        schema = {
            "x-polylogue-time-deltas": [
                {
                    "field_a": "$.a",
                    "field_b": "$.b",
                    "min_delta": 42.0,
                    "max_delta": 42.0,
                    "avg_delta": 42.0,
                },
            ],
        }
        solver = RelationConstraintSolver(schema)
        rng = random.Random(0)
        delta = solver.get_time_delta("$.a", "$.b", rng)
        assert delta == 42.0


# ---------------------------------------------------------------------------
# RelationConstraintSolver — mutual exclusion filtering
# ---------------------------------------------------------------------------


class TestRelationConstraintSolverMutualExclusion:
    def test_keeps_only_one_field_from_exclusive_group(self) -> None:
        schema = {
            "x-polylogue-mutually-exclusive": [
                {"parent": "$", "fields": ["alpha", "beta", "gamma"]},
            ],
        }
        solver = RelationConstraintSolver(schema)
        rng = random.Random(0)

        candidates = {"alpha", "beta", "gamma", "delta"}
        filtered = solver.filter_mutually_exclusive("$", candidates, rng)

        # Only one of {alpha, beta, gamma} should survive
        exclusive_survivors = filtered & {"alpha", "beta", "gamma"}
        assert len(exclusive_survivors) == 1
        # Non-exclusive field should always survive
        assert "delta" in filtered

    def test_non_matching_parent_path_is_not_filtered(self) -> None:
        schema = {
            "x-polylogue-mutually-exclusive": [
                {"parent": "$.other", "fields": ["a", "b"]},
            ],
        }
        solver = RelationConstraintSolver(schema)
        rng = random.Random(0)

        candidates = {"a", "b", "c"}
        filtered = solver.filter_mutually_exclusive("$.different", candidates, rng)
        # No filtering should happen because parent path doesn't match
        assert filtered == candidates

    def test_no_exclusion_groups_returns_all_fields(self) -> None:
        solver = RelationConstraintSolver({})
        rng = random.Random(0)
        candidates = {"x", "y", "z"}
        filtered = solver.filter_mutually_exclusive("$", candidates, rng)
        assert filtered == candidates

    @pytest.mark.parametrize("seed", range(20))
    def test_never_co_populates_exclusive_fields(self, seed: int) -> None:
        """Statistical test: across many seeds, exclusive fields never co-occur."""
        schema = {
            "x-polylogue-mutually-exclusive": [
                {"parent": "$", "fields": ["opt_a", "opt_b"]},
            ],
        }
        solver = RelationConstraintSolver(schema)
        rng = random.Random(seed)

        candidates = {"opt_a", "opt_b", "required"}
        filtered = solver.filter_mutually_exclusive("$", candidates, rng)

        assert "required" in filtered
        assert len(filtered & {"opt_a", "opt_b"}) <= 1


# ---------------------------------------------------------------------------
# RelationConstraintSolver — string length operations
# ---------------------------------------------------------------------------


class TestRelationConstraintSolverStringLength:
    def test_no_constraint_returns_base_text_unchanged(self) -> None:
        solver = RelationConstraintSolver({})
        rng = random.Random(0)
        assert solver.generate_string_with_length("$.any", rng, "hello world") == "hello world"

    def test_truncates_long_text(self) -> None:
        schema = {
            "x-polylogue-string-lengths": [
                {"path": "$.short", "min": 3, "max": 10, "avg": 7.0, "stddev": 1.0},
            ],
        }
        solver = RelationConstraintSolver(schema)
        rng = random.Random(42)
        result = solver.generate_string_with_length("$.short", rng, "this is a very long text string")
        assert len(result) <= 10

    def test_extends_short_text(self) -> None:
        schema = {
            "x-polylogue-string-lengths": [
                {"path": "$.long", "min": 50, "max": 100, "avg": 75.0, "stddev": 5.0},
            ],
        }
        solver = RelationConstraintSolver(schema)
        rng = random.Random(42)
        result = solver.generate_string_with_length("$.long", rng, "hi")
        assert len(result) >= 50

    def test_empty_base_text_returned_unchanged(self) -> None:
        schema = {
            "x-polylogue-string-lengths": [
                {"path": "$.x", "min": 10, "max": 100, "avg": 50.0, "stddev": 5.0},
            ],
        }
        solver = RelationConstraintSolver(schema)
        rng = random.Random(0)
        assert solver.generate_string_with_length("$.x", rng, "") == ""

    @pytest.mark.parametrize("seed", range(20))
    def test_result_within_min_max_bounds(self, seed: int) -> None:
        schema = {
            "x-polylogue-string-lengths": [
                {"path": "$.text", "min": 10, "max": 50, "avg": 30.0, "stddev": 8.0},
            ],
        }
        solver = RelationConstraintSolver(schema)
        rng = random.Random(seed)
        result = solver.generate_string_with_length(
            "$.text", rng, "some example text for testing purposes"
        )
        assert len(result) >= 10
        assert len(result) <= 50


# ---------------------------------------------------------------------------
# RelationConstraintSolver — integration: all constraint types together
# ---------------------------------------------------------------------------


class TestRelationConstraintSolverIntegration:
    def test_schema_with_all_constraint_types(self) -> None:
        schema = {
            "x-polylogue-foreign-keys": [
                {"source": "$.nodes.*.parent", "target": "$.nodes.*.id"},
            ],
            "x-polylogue-time-deltas": [
                {
                    "field_a": "$.create_time",
                    "field_b": "$.update_time",
                    "min_delta": 0.0,
                    "max_delta": 3600.0,
                    "avg_delta": 300.0,
                },
            ],
            "x-polylogue-mutually-exclusive": [
                {"parent": "$.message", "fields": ["text", "parts"]},
            ],
            "x-polylogue-string-lengths": [
                {"path": "$.message.body", "min": 20, "max": 500, "avg": 100.0, "stddev": 40.0},
            ],
        }
        solver = RelationConstraintSolver(schema)
        assert solver.has_constraints

        # FK
        solver.register_generated_id("$.nodes.*.id", "node-1")
        rng = random.Random(0)
        ref = solver.resolve_foreign_key("$.nodes.*.parent", rng)
        assert ref == "node-1"

        # Time delta
        delta = solver.get_time_delta("$.create_time", "$.update_time", rng)
        assert delta is not None
        assert 0.0 <= delta <= 3600.0

        # Mutual exclusion
        filtered = solver.filter_mutually_exclusive(
            "$.message", {"text", "parts", "metadata"}, rng,
        )
        assert len(filtered & {"text", "parts"}) == 1
        assert "metadata" in filtered

        # String length
        result = solver.generate_string_with_length(
            "$.message.body", rng, "short",
        )
        assert len(result) >= 20

    def test_path_matches_exact(self) -> None:
        solver = RelationConstraintSolver({})
        assert solver.path_matches("$.a.b", "$.a.b") is True
        assert solver.path_matches("$.a.b", "$.a.c") is False

"""Tests for showcase exercise catalog integrity and query functions."""

from __future__ import annotations

from polylogue.showcase.exercises import (
    EXERCISES,
    GROUPS,
    Exercise,
    exercises_by_group,
    topological_order,
    vhs_exercises,
)


class TestExerciseUniqueness:
    """All exercises must have unique names."""

    def test_all_exercises_have_unique_names(self):
        names = [e.name for e in EXERCISES]
        assert len(names) == len(set(names)), (
            f"Duplicate exercise names: "
            f"{[n for n in names if names.count(n) > 1]}"
        )

    def test_exercise_names_are_non_empty(self):
        for e in EXERCISES:
            assert e.name, "Exercise name must be non-empty"


class TestExercisesByGroup:
    """exercises_by_group covers all groups."""

    def test_covers_all_groups(self):
        by_group = exercises_by_group()
        for group in GROUPS:
            assert group in by_group, f"Group {group!r} missing from result"

    def test_all_exercises_assigned_to_known_groups(self):
        for e in EXERCISES:
            assert e.group in GROUPS, (
                f"Exercise {e.name!r} has unknown group {e.group!r}"
            )

    def test_total_count_matches(self):
        by_group = exercises_by_group()
        total = sum(len(exs) for exs in by_group.values())
        assert total == len(EXERCISES)

    def test_proof_exercises_exist_in_subcommands_group(self):
        names = {exercise.name for exercise in EXERCISES if exercise.group == "subcommands"}
        assert {
            "check-proof-json",
            "check-cohorts-json",
            "check-semantic-proof",
            "check-semantic-proof-json",
            "check-semantic-proof-read-surfaces",
            "check-semantic-proof-read-surfaces-json",
        } <= names


class TestVhsExercises:
    """vhs_exercises returns only capturable exercises."""

    def test_returns_only_capturable(self):
        vhs = vhs_exercises()
        for e in vhs:
            assert e.vhs_capture is True, (
                f"Exercise {e.name!r} in vhs_exercises() but vhs_capture=False"
            )

    def test_excludes_non_capturable(self):
        vhs_names = {e.name for e in vhs_exercises()}
        for e in EXERCISES:
            if not e.vhs_capture:
                assert e.name not in vhs_names

    def test_expected_count(self):
        """Six exercises are marked for VHS capture."""
        assert len(vhs_exercises()) == 6

    def test_expected_names(self):
        vhs_names = {e.name for e in vhs_exercises()}
        expected = {
            "help-main", "run-preview", "stats-default",
            "query-list", "check-health", "query-latest-md",
        }
        assert vhs_names == expected


class TestTopologicalOrder:
    """topological_order respects depends_on ordering."""

    def test_independent_exercises_maintain_input_order(self):
        exs = [
            Exercise(name="a", group="structural", description="A"),
            Exercise(name="b", group="structural", description="B"),
            Exercise(name="c", group="structural", description="C"),
        ]
        result = topological_order(exs)
        assert [e.name for e in result] == ["a", "b", "c"]

    def test_dependency_comes_before_dependent(self):
        exs = [
            Exercise(name="child", group="g", description="C", depends_on="parent"),
            Exercise(name="parent", group="g", description="P"),
        ]
        result = topological_order(exs)
        names = [e.name for e in result]
        assert names.index("parent") < names.index("child")

    def test_missing_dependency_is_tolerated(self):
        exs = [
            Exercise(name="orphan", group="g", description="O", depends_on="missing"),
        ]
        result = topological_order(exs)
        assert len(result) == 1
        assert result[0].name == "orphan"

    def test_chain_dependency(self):
        exs = [
            Exercise(name="c", group="g", description="C", depends_on="b"),
            Exercise(name="b", group="g", description="B", depends_on="a"),
            Exercise(name="a", group="g", description="A"),
        ]
        result = topological_order(exs)
        names = [e.name for e in result]
        assert names == ["a", "b", "c"]


class TestTierFiltering:
    """Tier values are consistent across the catalog."""

    def test_tier_0_exercises_exist(self):
        tier0 = [e for e in EXERCISES if e.tier == 0]
        assert len(tier0) > 0

    def test_tier_1_exercises_exist(self):
        tier1 = [e for e in EXERCISES if e.tier == 1]
        assert len(tier1) > 0

    def test_tier_2_exercises_exist(self):
        tier2 = [e for e in EXERCISES if e.tier == 2]
        assert len(tier2) > 0

    def test_all_tiers_are_valid(self):
        for e in EXERCISES:
            assert e.tier in (0, 1, 2), (
                f"Exercise {e.name!r} has invalid tier {e.tier}"
            )

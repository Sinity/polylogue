"""Tests for showcase exercise catalog integrity and query functions."""

from __future__ import annotations

from polylogue.showcase.exercises import (
    EXERCISES,
    GROUPS,
    QA_EXTRA_EXERCISES,
    QA_EXTRA_SCENARIOS,
    Exercise,
    exercises_by_group,
    topological_order,
    vhs_exercises,
)
from polylogue.showcase.generators import (
    command_help_exercise_names,
    generate_qa_extra_scenarios,
    inventory_command_paths,
    json_contract_exercise_names,
)


class TestExerciseUniqueness:
    """All exercises must have unique names."""

    def test_all_exercises_have_unique_names(self) -> None:
        names = [e.name for e in EXERCISES]
        assert len(names) == len(set(names)), f"Duplicate exercise names: {[n for n in names if names.count(n) > 1]}"

    def test_exercise_names_are_non_empty(self) -> None:
        for e in EXERCISES:
            assert e.name, "Exercise name must be non-empty"


class TestExercisesByGroup:
    """exercises_by_group covers all groups."""

    def test_covers_all_groups(self) -> None:
        by_group = exercises_by_group()
        for group in GROUPS:
            assert group in by_group, f"Group {group!r} missing from result"

    def test_all_exercises_assigned_to_known_groups(self) -> None:
        for e in EXERCISES:
            assert e.group in GROUPS, f"Exercise {e.name!r} has unknown group {e.group!r}"

    def test_total_count_matches(self) -> None:
        by_group = exercises_by_group()
        total = sum(len(exs) for exs in by_group.values())
        assert total == len(EXERCISES)

    def test_proof_exercises_exist_in_subcommands_group(self) -> None:
        names = {exercise.name for exercise in EXERCISES if exercise.group == "subcommands"}
        assert {
            "doctor-proof-json",
            "doctor-cohorts-json",
        } <= names

    def test_all_command_paths_have_generated_help_exercises(self) -> None:
        expected = command_help_exercise_names()
        observed = {
            exercise.name
            for exercise in EXERCISES
            if exercise.group == "structural" and exercise.name.startswith("help-") and exercise.name != "help-main"
        }
        assert observed == expected

    def test_inventory_includes_nested_command_paths(self) -> None:
        observed = {command_path.display_name for command_path in inventory_command_paths()}
        assert {
            "insights analytics",
            "run render",
            "schema explain",
        } <= observed

    def test_all_json_contract_commands_have_generated_exercises(self) -> None:
        expected = json_contract_exercise_names()
        observed = {
            exercise.name: exercise
            for exercise in EXERCISES
            if exercise.group == "subcommands" and exercise.name.startswith("json-")
        }
        assert set(observed) == expected
        for exercise in observed.values():
            assert exercise.assertion.stdout_is_valid_json is True
            assert exercise.output_ext == ".json"

    def test_json_contract_exercises_use_curated_runnable_args(self) -> None:
        observed = {
            exercise.name: exercise
            for exercise in EXERCISES
            if exercise.group == "subcommands" and exercise.name.startswith("json-")
        }
        assert observed["json-doctor-action-event-preview"].args == [
            "doctor",
            "--format",
            "json",
            "--repair",
            "--preview",
            "--target",
            "action_event_read_model",
        ]
        assert observed["json-doctor-session-insights-preview"].args == [
            "doctor",
            "--format",
            "json",
            "--repair",
            "--preview",
            "--target",
            "session_insights",
        ]
        assert observed["json-run-embed"].args == ["run", "embed", "--stats", "--format", "json"]
        assert "json-schema-compare" not in observed
        assert "json-schema-generate" not in observed
        assert "json-schema-promote" not in observed

    def test_runtime_aligned_json_contract_exercise_preserves_declared_targets(self) -> None:
        observed = {
            exercise.name: exercise
            for exercise in EXERCISES
            if exercise.group == "subcommands" and exercise.name.startswith("json-")
        }

        action_preview = observed["json-doctor-action-event-preview"]

        assert action_preview.path_targets == ("action-event-repair-loop",)
        assert action_preview.artifact_targets == (
            "action_event_rows",
            "action_event_fts",
            "action_event_readiness",
        )
        assert action_preview.operation_targets == (
            "cli.json-contract",
            "project-action-event-readiness",
        )
        assert action_preview.tags == (
            "generated",
            "json-contract",
            "maintenance",
            "action-events",
        )

        session_preview = observed["json-doctor-session-insights-preview"]
        assert session_preview.path_targets == ("session-insight-repair-loop",)
        assert session_preview.artifact_targets == (
            "session_insight_rows",
            "session_insight_fts",
            "session_insight_readiness",
        )
        assert session_preview.operation_targets == (
            "cli.json-contract",
            "project-session-insight-readiness",
        )
        assert session_preview.tags == (
            "generated",
            "json-contract",
            "maintenance",
            "session-insights",
        )

        profiles = observed["json-insights-profiles"]
        assert profiles.path_targets == ("session-profile-query-loop",)
        assert profiles.artifact_targets == (
            "session_profile_rows",
            "session_profile_merged_fts",
            "session_profile_results",
        )
        assert profiles.operation_targets == (
            "cli.json-contract",
            "query-session-profiles",
        )
        assert profiles.tags == (
            "generated",
            "json-contract",
            "insights",
            "session-profiles",
        )

        threads = observed["json-insights-threads"]
        assert threads.path_targets == ("work-thread-query-loop",)
        assert threads.artifact_targets == ("work_thread_rows", "work_thread_fts", "work_thread_results")
        assert threads.operation_targets == (
            "cli.json-contract",
            "query-work-threads",
        )
        assert threads.tags == (
            "generated",
            "json-contract",
            "insights",
            "threads",
        )

    def test_static_reparse_preview_exercise_preserves_declared_targets(self) -> None:
        observed = {exercise.name: exercise for exercise in EXERCISES}

        reparse_preview = observed["run-preview-reparse"]

        assert reparse_preview.args == [
            "run",
            "--preview",
            "--reparse",
            "--source",
            "inbox",
            "parse",
        ]
        assert reparse_preview.origin == "authored.showcase-catalog"
        assert reparse_preview.path_targets == ("raw-reparse-loop",)
        assert reparse_preview.artifact_targets == (
            "raw_validation_state",
            "validation_backlog",
            "parse_backlog",
            "parse_quarantine",
        )
        assert reparse_preview.operation_targets == (
            "plan-validation-backlog",
            "plan-parse-backlog",
        )
        assert reparse_preview.tags == (
            "pipeline",
            "reparse",
            "maintenance",
        )

    def test_exported_qa_extra_roots_match_generated_family(self) -> None:
        expected_names = [scenario.name for scenario in generate_qa_extra_scenarios()]

        assert [scenario.name for scenario in QA_EXTRA_SCENARIOS] == expected_names
        assert [exercise.name for exercise in QA_EXTRA_EXERCISES] == expected_names


class TestVhsExercises:
    """vhs_exercises returns only capturable exercises."""

    def test_returns_only_capturable(self) -> None:
        vhs = vhs_exercises()
        for e in vhs:
            assert e.vhs_capture is True, f"Exercise {e.name!r} in vhs_exercises() but vhs_capture=False"

    def test_excludes_non_capturable(self) -> None:
        vhs_names = {e.name for e in vhs_exercises()}
        for e in EXERCISES:
            if not e.vhs_capture:
                assert e.name not in vhs_names

    def test_expected_count(self) -> None:
        """Six exercises are marked for VHS capture."""
        assert len(vhs_exercises()) == 6

    def test_expected_names(self) -> None:
        vhs_names = {e.name for e in vhs_exercises()}
        expected = {
            "help-main",
            "run-preview",
            "stats-default",
            "query-list",
            "doctor-readiness",
            "query-latest-md",
        }
        assert vhs_names == expected


class TestTopologicalOrder:
    """topological_order respects depends_on ordering."""

    def test_independent_exercises_maintain_input_order(self) -> None:
        exs = [
            Exercise(name="a", group="structural", description="A"),
            Exercise(name="b", group="structural", description="B"),
            Exercise(name="c", group="structural", description="C"),
        ]
        result = topological_order(exs)
        assert [e.name for e in result] == ["a", "b", "c"]

    def test_dependency_comes_before_dependent(self) -> None:
        exs = [
            Exercise(name="child", group="g", description="C", depends_on="parent"),
            Exercise(name="parent", group="g", description="P"),
        ]
        result = topological_order(exs)
        names = [e.name for e in result]
        assert names.index("parent") < names.index("child")

    def test_missing_dependency_is_tolerated(self) -> None:
        exs = [
            Exercise(name="orphan", group="g", description="O", depends_on="missing"),
        ]
        result = topological_order(exs)
        assert len(result) == 1
        assert result[0].name == "orphan"

    def test_chain_dependency(self) -> None:
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

    def test_tier_0_exercises_exist(self) -> None:
        tier0 = [e for e in EXERCISES if e.tier == 0]
        assert len(tier0) > 0

    def test_tier_1_exercises_exist(self) -> None:
        tier1 = [e for e in EXERCISES if e.tier == 1]
        assert len(tier1) > 0

    def test_tier_2_exercises_exist(self) -> None:
        tier2 = [e for e in EXERCISES if e.tier == 2]
        assert len(tier2) > 0

    def test_all_tiers_are_valid(self) -> None:
        for e in EXERCISES:
            assert e.tier in (0, 1, 2), f"Exercise {e.name!r} has invalid tier {e.tier}"

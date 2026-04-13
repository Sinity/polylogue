"""Tests for showcase scenario compilation helpers."""

from __future__ import annotations

from polylogue.showcase.exercise_models import Validation
from polylogue.showcase.scenario_models import ExerciseScenario, compile_exercise_scenarios


def test_exercise_scenario_compiles_to_exercise() -> None:
    scenario = ExerciseScenario(
        scenario_id="json-doctor",
        group="subcommands",
        description="doctor JSON contract",
        args=("doctor", "--json"),
        validation=Validation(stdout_is_valid_json=True),
        needs_data=False,
        output_ext=".json",
        artifact_class="json",
        origin="generated.json-contract",
        artifact_targets=("doctor_runtime",),
        operation_targets=("cli.json-contract",),
        tags=("generated", "json-contract"),
    )

    exercise = scenario.compile()

    assert exercise.name == "json-doctor"
    assert exercise.args == ["doctor", "--json"]
    assert exercise.validation.stdout_is_valid_json is True
    assert exercise.output_ext == ".json"
    assert exercise.artifact_class == "json"


def test_compile_exercise_scenarios_preserves_order() -> None:
    scenarios = (
        ExerciseScenario(scenario_id="a", group="structural", description="A"),
        ExerciseScenario(scenario_id="b", group="structural", description="B"),
    )

    exercises = compile_exercise_scenarios(scenarios)

    assert [exercise.name for exercise in exercises] == ["a", "b"]

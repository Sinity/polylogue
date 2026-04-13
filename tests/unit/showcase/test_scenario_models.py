"""Tests for showcase scenario compilation helpers."""

from __future__ import annotations

from polylogue.scenarios import ScenarioProjectionSourceKind, polylogue_execution
from polylogue.showcase.exercise_models import Exercise, Validation


def test_exercise_scenario_compiles_to_exercise() -> None:
    scenario = Exercise(
        name="json-doctor",
        group="subcommands",
        description="doctor JSON contract",
        execution=polylogue_execution("doctor", "--json"),
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
    assert exercise.origin == "generated.json-contract"
    assert exercise.artifact_targets == ("doctor_runtime",)
    assert exercise.operation_targets == ("cli.json-contract",)
    assert exercise.tags == ("generated", "json-contract")


def test_exercise_scenario_compiles_its_own_projection_entry() -> None:
    scenario = Exercise(
        name="json-doctor",
        group="subcommands",
        description="doctor JSON contract",
        origin="generated.json-contract",
        artifact_targets=("doctor_runtime",),
        operation_targets=("cli.json-contract",),
        tags=("generated", "json-contract"),
    )

    projection = scenario.to_projection_entry()

    assert projection.source_kind is ScenarioProjectionSourceKind.EXERCISE
    assert projection.name == "json-doctor"
    assert projection.description == "doctor JSON contract"
    assert projection.origin == "generated.json-contract"
    assert projection.artifact_targets == ("doctor_runtime",)
    assert projection.operation_targets == ("cli.json-contract",)
    assert projection.tags == ("generated", "json-contract")
    assert projection.source_payload["execution"]["kind"] == "polylogue"

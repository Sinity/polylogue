from __future__ import annotations

from dataclasses import dataclass

from polylogue.scenarios import CorpusSpec, ExecutableScenario, ScenarioProjectionSourceKind, pytest_execution


@dataclass(frozen=True, kw_only=True)
class _ExecutableFixture(ExecutableScenario):
    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.VALIDATION_LANE


def test_executable_scenario_exposes_pytest_targets() -> None:
    scenario = _ExecutableFixture(
        name="machine-contract",
        description="machine contract lane",
        execution=pytest_execution("tests/unit/cli/test_machine_contract.py"),
        origin="authored.validation-lane",
        operation_targets=("cli.json-contract",),
        tags=("contract", "json"),
    )

    projection = scenario.to_projection_entry()

    assert scenario.tests == ("tests/unit/cli/test_machine_contract.py",)
    assert projection.source_kind is ScenarioProjectionSourceKind.VALIDATION_LANE
    assert projection.name == "machine-contract"
    assert projection.operation_targets == ("cli.json-contract",)
    assert projection.tags == ("contract", "json")
    assert projection.source_payload["execution"]["kind"] == "pytest"
    assert projection.source_payload["execution"]["argv"] == ["tests/unit/cli/test_machine_contract.py"]


def test_executable_scenario_projection_payload_preserves_corpus_specs() -> None:
    scenario = _ExecutableFixture(
        name="synthetic-lane",
        description="synthetic contract lane",
        execution=pytest_execution("tests/unit/cli/test_machine_contract.py"),
        corpus_specs=(CorpusSpec.for_provider("chatgpt", count=2),),
        origin="authored.validation-lane",
    )

    projection = scenario.to_projection_entry()

    assert projection.source_payload["corpus_specs"][0]["provider"] == "chatgpt"

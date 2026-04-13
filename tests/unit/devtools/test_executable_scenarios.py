from __future__ import annotations

from dataclasses import dataclass

from devtools.executable_scenarios import ExecutableScenario
from devtools.execution_specs import pytest_execution
from polylogue.scenarios import ScenarioProjectionSourceKind


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

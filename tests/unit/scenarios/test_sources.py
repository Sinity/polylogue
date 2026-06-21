from __future__ import annotations

from dataclasses import dataclass

from polylogue.scenarios import NamedScenarioSource, ScenarioProjectionSourceKind


@dataclass(frozen=True, kw_only=True)
class _NamedFixture(NamedScenarioSource):
    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.VALIDATION_LANE


def test_named_scenario_source_projects_name_and_description() -> None:
    source = _NamedFixture(
        name="contract-help",
        description="contract help text",
        origin="generated.test",
        artifact_targets=("archive_readiness",),
        tags=("contract", "help"),
    )

    projection = source.to_projection_entry()

    assert projection.source_kind is ScenarioProjectionSourceKind.VALIDATION_LANE
    assert projection.name == "contract-help"
    assert projection.description == "contract help text"
    assert projection.origin == "generated.test"
    assert projection.artifact_targets == ("archive_readiness",)
    assert projection.tags == ("contract", "help")

from __future__ import annotations

from dataclasses import dataclass

from polylogue.scenarios import NamedScenarioSource, ScenarioProjectionSourceKind


@dataclass(frozen=True, kw_only=True)
class _NamedFixture(NamedScenarioSource):
    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.EXERCISE


def test_named_scenario_source_projects_name_and_description() -> None:
    source = _NamedFixture(
        name="exercise-help",
        description="exercise help text",
        origin="generated.test",
        artifact_targets=("archive_health",),
        tags=("exercise", "help"),
    )

    projection = source.to_projection_entry()

    assert projection.source_kind is ScenarioProjectionSourceKind.EXERCISE
    assert projection.name == "exercise-help"
    assert projection.description == "exercise help text"
    assert projection.origin == "generated.test"
    assert projection.artifact_targets == ("archive_health",)
    assert projection.tags == ("exercise", "help")

"""Scenario models that compile into concrete showcase exercises."""

from __future__ import annotations

from dataclasses import dataclass, field

from polylogue.scenarios import ExecutableScenario, ScenarioProjectionSourceKind, polylogue_execution
from polylogue.showcase.exercise_models import Exercise, Validation


@dataclass(frozen=True, kw_only=True)
class ExerciseScenario(ExecutableScenario):
    """Authored scenario metadata for one CLI-backed showcase proof."""

    group: str
    validation: Validation = field(default_factory=Validation)
    needs_data: bool = False
    writes: bool = False
    depends_on: str | None = None
    output_ext: str = ".txt"
    tier: int = 1
    env: str = "any"
    timeout_s: float = 120.0
    vhs_capture: bool = False
    artifact_class: str = "text"
    capture_steps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.execution is None:
            object.__setattr__(self, "execution", polylogue_execution())

    @property
    def args(self) -> tuple[str, ...]:
        if self.execution is None:
            return ()
        return self.execution.polylogue_args

    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.EXERCISE

    def compile(self) -> Exercise:
        """Lower the scenario into the concrete showcase exercise artifact."""
        return Exercise(
            name=self.name,
            group=self.group,
            description=self.description,
            execution=self.execution or polylogue_execution(),
            validation=self.validation,
            needs_data=self.needs_data,
            writes=self.writes,
            depends_on=self.depends_on,
            output_ext=self.output_ext,
            tier=self.tier,
            env=self.env,
            timeout_s=self.timeout_s,
            vhs_capture=self.vhs_capture,
            artifact_class=self.artifact_class,
            capture_steps=self.capture_steps,
            origin=self.origin,
            path_targets=self.path_targets,
            artifact_targets=self.artifact_targets,
            operation_targets=self.operation_targets,
            tags=self.tags,
        )


def compile_exercise_scenarios(scenarios: tuple[ExerciseScenario, ...]) -> tuple[Exercise, ...]:
    """Compile authored scenarios into concrete showcase exercises."""
    return tuple(scenario.compile() for scenario in scenarios)


__all__ = ["ExerciseScenario", "compile_exercise_scenarios"]

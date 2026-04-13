"""Scenario models that compile into concrete showcase exercises."""

from __future__ import annotations

from dataclasses import dataclass, field

from polylogue.showcase.exercise_models import Exercise, Validation


@dataclass(frozen=True)
class ExerciseScenario:
    """Authored scenario metadata for one CLI-backed showcase proof."""

    scenario_id: str
    group: str
    description: str
    args: tuple[str, ...] = ()
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
    origin: str = "authored"
    artifact_targets: tuple[str, ...] = ()
    operation_targets: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def compile(self) -> Exercise:
        """Lower the scenario into the concrete showcase exercise artifact."""
        return Exercise(
            name=self.scenario_id,
            group=self.group,
            description=self.description,
            args=list(self.args),
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
        )


def compile_exercise_scenarios(scenarios: tuple[ExerciseScenario, ...]) -> tuple[Exercise, ...]:
    """Compile authored scenarios into concrete showcase exercises."""
    return tuple(scenario.compile() for scenario in scenarios)


__all__ = ["ExerciseScenario", "compile_exercise_scenarios"]

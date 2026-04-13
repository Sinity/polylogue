"""Typed showcase exercise models."""

from __future__ import annotations

from dataclasses import dataclass, field

from polylogue.scenarios import (
    AssertionSpec,
    ExecutableScenario,
    ExecutionKind,
    ExecutionSpec,
    ScenarioProjectionSourceKind,
    polylogue_execution,
)


@dataclass(frozen=True, kw_only=True)
class Exercise(ExecutableScenario):
    """A single showcase exercise — one CLI invocation with validation."""

    group: str  # structural | sources | pipeline | query-read | query-write | subcommands | advanced
    execution: ExecutionSpec = field(default_factory=polylogue_execution)
    assertion: AssertionSpec = field(default_factory=AssertionSpec)
    needs_data: bool = False  # Requires populated database
    writes: bool = False  # Mutates state — skip in --live mode
    depends_on: str | None = None  # Exercise that must complete first
    output_ext: str = ".txt"  # .txt / .json / .md / .csv / .html / .org
    tier: int = 1  # Complexity tier: 0=fast/structural, 1=basic, 2=advanced
    env: str = "any"  # "any" | "seeded" | "live"
    timeout_s: float = 120.0  # Per-exercise timeout in seconds
    vhs_capture: bool = False  # Whether this exercise should be captured as VHS recording
    artifact_class: str = "text"  # "text" | "json" | "visual" | "bundle"
    capture_steps: tuple[str, ...] = ()  # Optional VHS interaction steps for complex scenarios

    @property
    def args(self) -> list[str]:
        return list(self.execution.polylogue_args)

    def __post_init__(self) -> None:
        if self.execution is None:
            object.__setattr__(self, "execution", polylogue_execution())
        if self.execution.kind is not ExecutionKind.POLYLOGUE:
            raise ValueError("showcase exercises require polylogue execution")
        super().__post_init__()

    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.EXERCISE

    def compile(self) -> Exercise:
        return self

    @property
    def invoke_args(self) -> list[str]:
        return list(self.execution.polylogue_invoke_args)

    @property
    def display_command(self) -> list[str]:
        command = self.execution.display_command
        return list(command) if command is not None else ["polylogue"]

    @property
    def display_command_text(self) -> str:
        return " ".join(self.display_command)

    @property
    def args_text(self) -> str:
        return " ".join(self.execution.polylogue_args) if self.execution.polylogue_args else "(default stats)"


__all__ = ["AssertionSpec", "Exercise"]

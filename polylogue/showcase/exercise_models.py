"""Typed showcase exercise models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from polylogue.scenarios import ExecutionKind, ExecutionSpec, ScenarioMetadata, polylogue_execution

if TYPE_CHECKING:
    from polylogue.scenarios import CorpusSpec


@dataclass(frozen=True)
class Validation:
    """Expected outcome for an exercise."""

    exit_code: int | None = 0  # None = delegated to custom validator
    stdout_contains: tuple[str, ...] = ()
    stdout_not_contains: tuple[str, ...] = ()
    stdout_is_valid_json: bool = False
    stdout_min_lines: int | None = None
    custom: Callable[[str, int], str | None] | None = None  # (output, exit_code) -> error|None


@dataclass(frozen=True)
class Exercise(ScenarioMetadata):
    """A single showcase exercise — one CLI invocation with validation."""

    name: str  # Unique ID e.g. "query.list-json"
    group: str  # structural | sources | pipeline | query-read | query-write | subcommands | advanced
    description: str  # Human-readable, used in cookbook headings
    execution: ExecutionSpec = field(default_factory=polylogue_execution)
    corpus_specs: tuple[CorpusSpec, ...] = ()
    validation: Validation = field(default_factory=Validation)
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
        if self.execution.kind is not ExecutionKind.POLYLOGUE:
            raise ValueError("showcase exercises require polylogue execution")

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


__all__ = ["Exercise", "Validation"]

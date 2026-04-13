"""Typed showcase exercise models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


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
class Exercise:
    """A single showcase exercise — one CLI invocation with validation."""

    name: str  # Unique ID e.g. "query.list-json"
    group: str  # structural | sources | pipeline | query-read | query-write | subcommands | advanced
    description: str  # Human-readable, used in cookbook headings
    args: list[str] = field(default_factory=list)  # CLI args (without 'polylogue' prefix)
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
    origin: str = "authored"
    artifact_targets: tuple[str, ...] = ()
    operation_targets: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


__all__ = ["Exercise", "Validation"]

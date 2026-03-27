"""Multi-dimensional exercise classification for showcase exercises.

Replaces the flat ``group`` field with a structured taxonomy where exercises
are described by what they test.  The tier (complexity level) is derived
from dimensions rather than manually assigned.
"""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.types import ExerciseIOMode


@dataclass(frozen=True)
class ExerciseDimensions:
    """Multi-dimensional classification for a showcase exercise."""

    capability: str  # "query" | "ingest" | "schema" | "format" | "filter" | "mutation" | "meta"
    scope: str  # "structural" | "single-provider" | "cross-provider" | "full-archive"
    io_mode: ExerciseIOMode
    output_format: str  # "text" | "json" | "markdown" | "html" | "csv" | "yaml" | "org" | "obsidian" | "mixed"
    complexity: str  # "smoke" | "basic" | "combinatorial" | "stress"

    @property
    def derived_tier(self) -> int:
        """Derive tier from complexity dimension."""
        return {
            "smoke": 0,
            "basic": 1,
            "combinatorial": 2,
            "stress": 2,
        }.get(self.complexity, 1)

    @property
    def vhs_eligible(self) -> bool:
        """Whether this exercise is eligible for automatic VHS recording."""
        return self.scope == "structural" and self.complexity == "smoke"


# Convenience constructors for common patterns
def structural_smoke(
    capability: str = "meta",
    output_format: str = "text",
) -> ExerciseDimensions:
    """Quick structural/smoke exercise (tier 0)."""
    return ExerciseDimensions(
        capability=capability,
        scope="structural",
        io_mode=ExerciseIOMode.READ,
        output_format=output_format,
        complexity="smoke",
    )


def query_read(
    output_format: str = "text",
    complexity: str = "basic",
    scope: str = "full-archive",
) -> ExerciseDimensions:
    """Read-only query exercise."""
    return ExerciseDimensions(
        capability="query",
        scope=scope,
        io_mode=ExerciseIOMode.READ,
        output_format=output_format,
        complexity=complexity,
    )


def query_write(
    output_format: str = "text",
    complexity: str = "basic",
) -> ExerciseDimensions:
    """Write/mutation query exercise."""
    return ExerciseDimensions(
        capability="mutation",
        scope="full-archive",
        io_mode=ExerciseIOMode.WRITE,
        output_format=output_format,
        complexity=complexity,
    )


def schema_exercise(
    complexity: str = "basic",
    io_mode: ExerciseIOMode = ExerciseIOMode.READ,
) -> ExerciseDimensions:
    """Schema-related exercise."""
    return ExerciseDimensions(
        capability="schema",
        scope="single-provider",
        io_mode=ExerciseIOMode.from_string(io_mode),
        output_format="json",
        complexity=complexity,
    )


__all__ = [
    "ExerciseDimensions",
    "query_read",
    "query_write",
    "schema_exercise",
    "structural_smoke",
]

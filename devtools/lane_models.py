"""Shared control-plane lane metadata."""

from __future__ import annotations

from dataclasses import dataclass

from devtools.execution_specs import ExecutionSpec
from polylogue.scenarios import ScenarioMetadata


@dataclass(frozen=True)
class LaneEntry(ScenarioMetadata):
    """One named control-plane lane."""

    name: str
    description: str
    timeout_s: int
    category: str
    execution: ExecutionSpec

    @property
    def is_composite(self) -> bool:
        return self.execution.is_composite

    @property
    def command(self) -> list[str] | None:
        command = self.execution.command
        return list(command) if command is not None else None

    @property
    def sub_lanes(self) -> tuple[str, ...]:
        return self.execution.members


__all__ = ["LaneEntry"]

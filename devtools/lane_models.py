"""Shared control-plane lane metadata."""

from __future__ import annotations

from dataclasses import dataclass

from devtools.execution_specs import ExecutionSpec
from polylogue.scenarios import ScenarioMetadata, ScenarioProjectionSource, ScenarioProjectionSourceKind


@dataclass(frozen=True)
class LaneEntry(ScenarioProjectionSource, ScenarioMetadata):
    """One named control-plane lane."""

    name: str
    description: str
    timeout_s: int
    category: str
    execution: ExecutionSpec

    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.VALIDATION_LANE

    @property
    def projection_name(self) -> str:
        return self.name

    @property
    def projection_description(self) -> str:
        return self.description

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

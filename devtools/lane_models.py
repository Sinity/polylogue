"""Shared control-plane lane metadata."""

from __future__ import annotations

from dataclasses import dataclass

from devtools.executable_scenarios import ExecutableScenario
from polylogue.scenarios import ScenarioProjectionSourceKind


@dataclass(frozen=True, kw_only=True)
class LaneEntry(ExecutableScenario):
    """One named control-plane lane."""

    timeout_s: int
    category: str

    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.VALIDATION_LANE

    @property
    def sub_lanes(self) -> tuple[str, ...]:
        return self.members


__all__ = ["LaneEntry"]

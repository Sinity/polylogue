"""Shared control-plane lane metadata."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.scenarios import ExecutableScenario, ScenarioProjectionSourceKind


@dataclass(frozen=True, kw_only=True)
class LaneEntry(ExecutableScenario):
    """One named control-plane lane."""

    timeout_s: int
    category: str
    family: str | None = None

    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.VALIDATION_LANE

    @property
    def sub_lanes(self) -> tuple[str, ...]:
        return self.members

    def projection_source_payload(self) -> dict[str, object]:
        payload = super().projection_source_payload()
        payload["timeout_s"] = self.timeout_s
        payload["category"] = self.category
        if self.family is not None:
            payload["family"] = self.family
        return payload


__all__ = ["LaneEntry"]

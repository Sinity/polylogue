"""Shared authored scenario-source models."""

from __future__ import annotations

from dataclasses import dataclass

from .metadata import ScenarioMetadata
from .projections import ScenarioProjectionSource


@dataclass(frozen=True, kw_only=True)
class NamedScenarioSource(ScenarioProjectionSource, ScenarioMetadata):
    """One authored scenario source with a stable name and description."""

    name: str
    description: str

    @property
    def projection_name(self) -> str:
        return self.name

    @property
    def projection_description(self) -> str:
        return self.description


__all__ = ["NamedScenarioSource"]

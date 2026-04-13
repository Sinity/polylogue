"""Shared scenario-projection metadata for control-plane surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum

from .metadata import ScenarioMetadata


class ScenarioProjectionSourceKind(str, Enum):
    EXERCISE = "exercise"
    VALIDATION_LANE = "validation-lane"
    MUTATION_CAMPAIGN = "mutation-campaign"
    BENCHMARK_CAMPAIGN = "benchmark-campaign"
    SYNTHETIC_BENCHMARK = "synthetic-benchmark"
    INFERRED_CORPUS = "inferred-corpus"


@dataclass(frozen=True)
class ScenarioProjectionEntry(ScenarioMetadata):
    source_kind: ScenarioProjectionSourceKind
    name: str
    description: str
    source_payload: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_object(
        cls,
        *,
        source_kind: ScenarioProjectionSourceKind,
        name: str,
        description: str,
        obj: object,
        source_payload: Mapping[str, object] | None = None,
    ) -> ScenarioProjectionEntry:
        metadata = ScenarioMetadata.from_object(obj)
        return cls(
            source_kind=source_kind,
            name=name,
            description=description,
            origin=metadata.origin,
            path_targets=metadata.path_targets,
            artifact_targets=metadata.artifact_targets,
            operation_targets=metadata.operation_targets,
            tags=metadata.tags,
            source_payload=dict(source_payload or {}),
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["source_kind"] = self.source_kind.value
        return data


__all__ = [
    "ScenarioProjectionEntry",
    "ScenarioProjectionSourceKind",
]

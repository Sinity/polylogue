"""Shared runtime scenario-coverage computation for control-plane surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from devtools.quality_registry import build_quality_registry
from polylogue.artifact_graph import build_artifact_graph
from polylogue.scenarios import ScenarioMetadata
from polylogue.showcase.exercises import EXERCISES


@dataclass(frozen=True, slots=True)
class ScenarioCoverageRef:
    source: str
    name: str
    origin: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "name": self.name,
            "origin": self.origin,
        }


@dataclass(frozen=True, slots=True)
class RuntimeScenarioCoverage:
    artifacts: dict[str, tuple[ScenarioCoverageRef, ...]]
    operations: dict[str, tuple[ScenarioCoverageRef, ...]]
    uncovered_artifacts: tuple[str, ...]
    uncovered_operations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": {
                name: [ref.to_dict() for ref in refs]
                for name, refs in self.artifacts.items()
            },
            "operations": {
                name: [ref.to_dict() for ref in refs]
                for name, refs in self.operations.items()
            },
            "uncovered_artifacts": list(self.uncovered_artifacts),
            "uncovered_operations": list(self.uncovered_operations),
        }


def build_runtime_scenario_coverage() -> RuntimeScenarioCoverage:
    registry = build_quality_registry()
    scenario_like_items = [
        ("exercise", exercise.name, ScenarioMetadata.from_object(exercise))
        for exercise in EXERCISES
    ]
    scenario_like_items.extend(
        ("benchmark-campaign", campaign.name, ScenarioMetadata.from_object(campaign))
        for campaign in registry.benchmark_campaigns
    )
    scenario_like_items.extend(
        ("synthetic-benchmark", campaign.name, ScenarioMetadata.from_object(campaign))
        for campaign in registry.synthetic_benchmark_campaigns
    )

    graph = build_artifact_graph()
    artifact_refs: dict[str, list[ScenarioCoverageRef]] = {name: [] for name in graph.by_name()}
    operation_refs: dict[str, list[ScenarioCoverageRef]] = {operation.name: [] for operation in graph.operations}

    for source_kind, name, metadata in scenario_like_items:
        ref = ScenarioCoverageRef(source=source_kind, name=name, origin=metadata.origin)
        for artifact in metadata.resolve_runtime_artifacts():
            artifact_refs[artifact.name].append(ref)
        for operation in metadata.resolve_runtime_operations():
            operation_refs[operation.name].append(ref)

    covered_artifacts = {
        name: tuple(refs)
        for name, refs in artifact_refs.items()
        if refs
    }
    covered_operations = {
        name: tuple(refs)
        for name, refs in operation_refs.items()
        if refs
    }

    return RuntimeScenarioCoverage(
        artifacts=covered_artifacts,
        operations=covered_operations,
        uncovered_artifacts=tuple(sorted(name for name, refs in artifact_refs.items() if not refs)),
        uncovered_operations=tuple(sorted(name for name, refs in operation_refs.items() if not refs)),
    )


__all__ = [
    "RuntimeScenarioCoverage",
    "ScenarioCoverageRef",
    "build_runtime_scenario_coverage",
]

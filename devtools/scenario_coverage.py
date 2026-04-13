"""Shared runtime scenario-coverage computation for control-plane surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from devtools.quality_registry import QualityRegistry, build_quality_registry
from polylogue.artifact_graph import build_artifact_graph


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


def build_runtime_scenario_coverage(*, registry: QualityRegistry | None = None) -> RuntimeScenarioCoverage:
    quality_registry = registry or build_quality_registry()

    graph = build_artifact_graph()
    artifact_refs: dict[str, list[ScenarioCoverageRef]] = {name: [] for name in graph.by_name()}
    operation_refs: dict[str, list[ScenarioCoverageRef]] = {operation.name: [] for operation in graph.operations}

    for projection in quality_registry.scenario_projections:
        ref = ScenarioCoverageRef(source=projection.source_kind, name=projection.name, origin=projection.origin)
        for artifact in projection.resolve_runtime_artifacts():
            artifact_refs[artifact.name].append(ref)
        for operation in projection.resolve_runtime_operations():
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

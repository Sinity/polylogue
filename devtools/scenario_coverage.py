"""Shared runtime scenario-coverage computation for control-plane surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from devtools.scenario_projection_catalog import build_scenario_projection_entries
from polylogue.artifact_graph import build_artifact_graph
from polylogue.operations import build_declared_operation_catalog
from polylogue.scenarios import ScenarioProjectionEntry


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
class RuntimePathCoverage:
    name: str
    refs: tuple[ScenarioCoverageRef, ...]
    uncovered_artifacts: tuple[str, ...]
    uncovered_operations: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.uncovered_artifacts and not self.uncovered_operations

    def to_dict(self) -> dict[str, Any]:
        return {
            "refs": [ref.to_dict() for ref in self.refs],
            "uncovered_artifacts": list(self.uncovered_artifacts),
            "uncovered_operations": list(self.uncovered_operations),
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class RuntimeScenarioCoverage:
    artifacts: dict[str, tuple[ScenarioCoverageRef, ...]]
    operations: dict[str, tuple[ScenarioCoverageRef, ...]]
    maintenance_targets: dict[str, tuple[ScenarioCoverageRef, ...]]
    declared_operations: dict[str, tuple[ScenarioCoverageRef, ...]]
    paths: dict[str, RuntimePathCoverage]
    uncovered_artifacts: tuple[str, ...]
    uncovered_operations: tuple[str, ...]
    uncovered_maintenance_targets: tuple[str, ...]
    uncovered_declared_operations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": {name: [ref.to_dict() for ref in refs] for name, refs in self.artifacts.items()},
            "operations": {name: [ref.to_dict() for ref in refs] for name, refs in self.operations.items()},
            "maintenance_targets": {
                name: [ref.to_dict() for ref in refs] for name, refs in self.maintenance_targets.items()
            },
            "declared_operations": {
                name: [ref.to_dict() for ref in refs] for name, refs in self.declared_operations.items()
            },
            "paths": {name: path.to_dict() for name, path in self.paths.items()},
            "uncovered_artifacts": list(self.uncovered_artifacts),
            "uncovered_operations": list(self.uncovered_operations),
            "uncovered_maintenance_targets": list(self.uncovered_maintenance_targets),
            "uncovered_declared_operations": list(self.uncovered_declared_operations),
        }


def build_runtime_scenario_coverage(
    *,
    projections: tuple[ScenarioProjectionEntry, ...] | None = None,
) -> RuntimeScenarioCoverage:
    scenario_projections = projections or build_scenario_projection_entries()
    graph = build_artifact_graph()
    declared_operation_catalog = build_declared_operation_catalog()
    artifact_refs: dict[str, list[ScenarioCoverageRef]] = {name: [] for name in graph.by_name()}
    operation_refs: dict[str, list[ScenarioCoverageRef]] = {operation.name: [] for operation in graph.operations}
    maintenance_target_refs: dict[str, list[ScenarioCoverageRef]] = {
        target.name: [] for target in graph.maintenance_targets
    }
    declared_operation_refs: dict[str, list[ScenarioCoverageRef]] = {
        operation.name: [] for operation in declared_operation_catalog.specs
    }

    for projection in scenario_projections:
        ref = ScenarioCoverageRef(source=projection.source_kind.value, name=projection.name, origin=projection.origin)
        for artifact in projection.resolve_runtime_artifacts():
            artifact_refs[artifact.name].append(ref)
        for operation in projection.resolve_runtime_operations():
            operation_refs[operation.name].append(ref)
        maintenance_targets = {
            *projection.resolve_runtime_maintenance_targets(),
            *graph.maintenance_targets_for_operation_names(projection.runtime_operation_targets()),
        }
        for target in maintenance_targets:
            maintenance_target_refs[target.name].append(ref)
        for operation in projection.resolve_declared_operations():
            declared_operation_refs[operation.name].append(ref)

    covered_artifacts = {name: tuple(refs) for name, refs in artifact_refs.items() if refs}
    covered_operations = {name: tuple(refs) for name, refs in operation_refs.items() if refs}
    covered_maintenance_targets = {name: tuple(refs) for name, refs in maintenance_target_refs.items() if refs}
    covered_declared_operations = {name: tuple(refs) for name, refs in declared_operation_refs.items() if refs}
    path_coverage: dict[str, RuntimePathCoverage] = {}
    for path in graph.paths:
        relevant_operations = tuple(operation.name for operation in graph.operations_for_path(path))
        refs = {
            *(ref for node_name in path.nodes for ref in artifact_refs[node_name]),
            *(ref for operation_name in relevant_operations for ref in operation_refs[operation_name]),
        }
        path_coverage[path.name] = RuntimePathCoverage(
            name=path.name,
            refs=tuple(sorted(refs, key=lambda ref: (ref.source, ref.name, ref.origin))),
            uncovered_artifacts=tuple(sorted(node_name for node_name in path.nodes if not artifact_refs[node_name])),
            uncovered_operations=tuple(
                sorted(operation_name for operation_name in relevant_operations if not operation_refs[operation_name])
            ),
        )

    return RuntimeScenarioCoverage(
        artifacts=covered_artifacts,
        operations=covered_operations,
        maintenance_targets=covered_maintenance_targets,
        declared_operations=covered_declared_operations,
        paths=path_coverage,
        uncovered_artifacts=tuple(sorted(name for name, refs in artifact_refs.items() if not refs)),
        uncovered_operations=tuple(sorted(name for name, refs in operation_refs.items() if not refs)),
        uncovered_maintenance_targets=tuple(sorted(name for name, refs in maintenance_target_refs.items() if not refs)),
        uncovered_declared_operations=tuple(sorted(name for name, refs in declared_operation_refs.items() if not refs)),
    )


__all__ = [
    "RuntimeScenarioCoverage",
    "RuntimePathCoverage",
    "ScenarioCoverageRef",
    "build_runtime_scenario_coverage",
]

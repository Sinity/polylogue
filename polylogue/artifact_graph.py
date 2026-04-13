"""Explicit artifact/dependency map for Polylogue runtime semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from polylogue.artifacts import (
    ArtifactLayer,
    ArtifactNode,
    ArtifactPath,
    build_runtime_artifact_nodes,
    build_runtime_artifact_paths,
)
from polylogue.maintenance_targets import MaintenanceTargetSpec, build_maintenance_target_catalog
from polylogue.operations import OperationSpec, build_runtime_operation_catalog


@dataclass(frozen=True, slots=True)
class ArtifactGraph:
    """Named artifact nodes plus curated high-value paths through them."""

    nodes: tuple[ArtifactNode, ...]
    paths: tuple[ArtifactPath, ...]
    operations: tuple[OperationSpec, ...]
    maintenance_targets: tuple[MaintenanceTargetSpec, ...]

    def by_name(self) -> dict[str, ArtifactNode]:
        return {node.name: node for node in self.nodes}

    def operation_by_name(self) -> dict[str, OperationSpec]:
        return {operation.name: operation for operation in self.operations}

    def path_by_name(self) -> dict[str, ArtifactPath]:
        return {path.name: path for path in self.paths}

    def maintenance_target_by_name(self) -> dict[str, MaintenanceTargetSpec]:
        return {target.name: target for target in self.maintenance_targets}

    def artifact_names(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes)

    def operation_names(self) -> tuple[str, ...]:
        return tuple(operation.name for operation in self.operations)

    def path_names(self) -> tuple[str, ...]:
        return tuple(path.name for path in self.paths)

    def maintenance_target_names(self) -> tuple[str, ...]:
        return tuple(target.name for target in self.maintenance_targets)

    def resolve_artifacts(self, names: tuple[str, ...]) -> tuple[ArtifactNode, ...]:
        by_name = self.by_name()
        return tuple(by_name[name] for name in names if name in by_name)

    def resolve_operations(self, names: tuple[str, ...]) -> tuple[OperationSpec, ...]:
        by_name = self.operation_by_name()
        return tuple(by_name[name] for name in names if name in by_name)

    def resolve_paths(self, names: tuple[str, ...]) -> tuple[ArtifactPath, ...]:
        by_name = self.path_by_name()
        return tuple(by_name[name] for name in names if name in by_name)

    def resolve_maintenance_targets(self, names: tuple[str, ...]) -> tuple[MaintenanceTargetSpec, ...]:
        by_name = self.maintenance_target_by_name()
        return tuple(by_name[name] for name in names if name in by_name)

    def operations_for_path(self, path: ArtifactPath | str) -> tuple[OperationSpec, ...]:
        path_name = path if isinstance(path, str) else path.name
        return tuple(
            operation
            for operation in self.operations
            if path_name in operation.path_targets
        )

    def artifacts_for_maintenance_target(self, target: MaintenanceTargetSpec | str) -> tuple[ArtifactNode, ...]:
        target_name = target if isinstance(target, str) else target.name
        return tuple(node for node in self.nodes if target_name in node.repair_targets)

    def maintenance_targets_for_operation_names(self, names: tuple[str, ...]) -> tuple[MaintenanceTargetSpec, ...]:
        return build_maintenance_target_catalog().maintenance_targets_for_operation_names(names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "paths": [path.to_dict() for path in self.paths],
            "operations": [operation.to_dict() for operation in self.operations],
            "maintenance_targets": [target.to_dict() for target in self.maintenance_targets],
        }


def build_artifact_graph() -> ArtifactGraph:
    nodes = build_runtime_artifact_nodes()
    paths = build_runtime_artifact_paths()
    operations = build_runtime_operation_catalog().specs
    maintenance_targets = build_maintenance_target_catalog().specs
    maintenance_target_names = {target.name for target in maintenance_targets}
    invalid_refs = {
        node.name: tuple(target for target in node.repair_targets if target not in maintenance_target_names)
        for node in nodes
        if any(target not in maintenance_target_names for target in node.repair_targets)
    }
    if invalid_refs:
        raise ValueError(f"Artifact graph declared unknown maintenance targets: {invalid_refs}")
    return ArtifactGraph(nodes=nodes, paths=paths, operations=operations, maintenance_targets=maintenance_targets)


__all__ = [
    "ArtifactGraph",
    "ArtifactLayer",
    "ArtifactNode",
    "ArtifactPath",
    "build_artifact_graph",
]

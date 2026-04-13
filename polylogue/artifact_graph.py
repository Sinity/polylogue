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
from polylogue.operations import OperationSpec, build_runtime_operation_catalog


@dataclass(frozen=True, slots=True)
class ArtifactGraph:
    """Named artifact nodes plus curated high-value paths through them."""

    nodes: tuple[ArtifactNode, ...]
    paths: tuple[ArtifactPath, ...]
    operations: tuple[OperationSpec, ...]

    def by_name(self) -> dict[str, ArtifactNode]:
        return {node.name: node for node in self.nodes}

    def operation_by_name(self) -> dict[str, OperationSpec]:
        return {operation.name: operation for operation in self.operations}

    def path_by_name(self) -> dict[str, ArtifactPath]:
        return {path.name: path for path in self.paths}

    def artifact_names(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes)

    def operation_names(self) -> tuple[str, ...]:
        return tuple(operation.name for operation in self.operations)

    def path_names(self) -> tuple[str, ...]:
        return tuple(path.name for path in self.paths)

    def resolve_artifacts(self, names: tuple[str, ...]) -> tuple[ArtifactNode, ...]:
        by_name = self.by_name()
        return tuple(by_name[name] for name in names if name in by_name)

    def resolve_operations(self, names: tuple[str, ...]) -> tuple[OperationSpec, ...]:
        by_name = self.operation_by_name()
        return tuple(by_name[name] for name in names if name in by_name)

    def resolve_paths(self, names: tuple[str, ...]) -> tuple[ArtifactPath, ...]:
        by_name = self.path_by_name()
        return tuple(by_name[name] for name in names if name in by_name)

    def operations_for_path(self, path: ArtifactPath | str) -> tuple[OperationSpec, ...]:
        path_name = path if isinstance(path, str) else path.name
        selected_path = self.path_by_name()[path_name]
        path_nodes = set(selected_path.nodes)
        return tuple(
            operation
            for operation in self.operations
            if set(operation.consumes).union(operation.produces).intersection(path_nodes)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "paths": [path.to_dict() for path in self.paths],
            "operations": [operation.to_dict() for operation in self.operations],
        }


def build_artifact_graph() -> ArtifactGraph:
    nodes = build_runtime_artifact_nodes()
    paths = build_runtime_artifact_paths()
    operations = build_runtime_operation_catalog().specs
    return ArtifactGraph(nodes=nodes, paths=paths, operations=operations)


__all__ = [
    "ArtifactGraph",
    "ArtifactLayer",
    "ArtifactNode",
    "ArtifactPath",
    "build_artifact_graph",
]

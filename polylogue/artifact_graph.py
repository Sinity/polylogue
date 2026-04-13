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
from polylogue.operations import OperationSpec, build_runtime_operation_specs


@dataclass(frozen=True, slots=True)
class ArtifactGraph:
    """Named artifact nodes plus curated high-value paths through them."""

    nodes: tuple[ArtifactNode, ...]
    paths: tuple[ArtifactPath, ...]
    operations: tuple[OperationSpec, ...]

    def by_name(self) -> dict[str, ArtifactNode]:
        return {node.name: node for node in self.nodes}

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "paths": [path.to_dict() for path in self.paths],
            "operations": [operation.to_dict() for operation in self.operations],
        }


def build_artifact_graph() -> ArtifactGraph:
    nodes = build_runtime_artifact_nodes()
    paths = build_runtime_artifact_paths()
    operations = build_runtime_operation_specs()
    return ArtifactGraph(nodes=nodes, paths=paths, operations=operations)


__all__ = [
    "ArtifactGraph",
    "ArtifactLayer",
    "ArtifactNode",
    "ArtifactPath",
    "build_artifact_graph",
]

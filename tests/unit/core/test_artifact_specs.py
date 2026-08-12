from __future__ import annotations

from polylogue.artifacts import build_runtime_artifact_nodes, build_runtime_artifact_paths
from polylogue.artifacts.graph import build_artifact_graph


def test_runtime_artifact_specs_connect_operation_targets_to_declared_paths() -> None:
    graph = build_artifact_graph()

    for operation in graph.operations:
        for path_name in operation.path_targets:
            path = graph.path_by_name()[path_name]
            assert {*operation.consumes, *operation.produces}.issubset(path.nodes)


def test_runtime_artifact_paths_reference_only_declared_nodes() -> None:
    nodes = build_runtime_artifact_nodes()
    paths = build_runtime_artifact_paths()
    node_names = {node.name for node in nodes}

    for path in paths:
        assert path.nodes
        assert set(path.nodes).issubset(node_names)

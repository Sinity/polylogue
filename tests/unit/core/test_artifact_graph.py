from __future__ import annotations

from polylogue.artifact_graph import ArtifactLayer, build_artifact_graph
from polylogue.artifacts import build_runtime_artifact_nodes, build_runtime_artifact_paths
from polylogue.operations import OperationKind, build_runtime_operation_catalog


def test_artifact_graph_contains_the_current_runtime_paths() -> None:
    graph = build_artifact_graph()
    nodes = graph.by_name()
    operations = {operation.name: operation for operation in graph.operations}

    assert set(nodes) >= {
        "raw_validation_state",
        "validation_backlog",
        "parse_backlog",
        "parse_quarantine",
        "message_source_rows",
        "message_fts",
        "tool_use_source_blocks",
        "action_event_rows",
        "action_event_fts",
        "action_event_health",
        "session_product_source_conversations",
        "session_product_rows",
        "session_product_fts",
        "session_product_health",
        "conversation_query_results",
        "archive_health",
    }
    assert nodes["raw_validation_state"].layer is ArtifactLayer.DURABLE
    assert nodes["message_fts"].layer is ArtifactLayer.INDEX
    assert nodes["message_fts"].depends_on == ("message_source_rows",)
    assert nodes["action_event_fts"].layer is ArtifactLayer.INDEX
    assert nodes["action_event_fts"].depends_on == ("action_event_rows",)
    assert nodes["action_event_health"].depends_on == ("action_event_rows", "action_event_fts")
    assert nodes["session_product_fts"].layer is ArtifactLayer.INDEX
    assert nodes["session_product_health"].depends_on == ("session_product_rows", "session_product_fts")
    assert nodes["parse_quarantine"].depends_on == ("raw_validation_state",)
    assert nodes["conversation_query_results"].depends_on == ("message_fts",)
    assert nodes["archive_health"].depends_on == ("message_fts", "action_event_health", "session_product_health")
    assert operations["plan-validation-backlog"].produces == ("validation_backlog",)
    assert operations["plan-parse-backlog"].produces == ("parse_backlog", "parse_quarantine")
    assert operations["index-message-fts"].produces == ("message_fts",)
    assert operations["materialize-action-events"].produces == ("action_event_rows", "action_event_fts")
    assert operations["query-conversations"].produces == ("conversation_query_results",)
    assert operations["project-action-event-health"].consumes == ("action_event_rows", "action_event_fts")
    assert operations["materialize-session-products"].produces == ("session_product_rows", "session_product_fts")
    assert operations["project-session-product-health"].consumes == ("session_product_rows", "session_product_fts")
    assert operations["project-archive-health"].consumes == (
        "message_fts",
        "action_event_health",
        "session_product_health",
    )
    assert operations["plan-validation-backlog"].kind is OperationKind.PLANNING


def test_artifact_graph_operations_come_from_runtime_operation_catalog() -> None:
    graph = build_artifact_graph()
    authored = build_runtime_operation_catalog()

    assert tuple(operation.name for operation in graph.operations) == authored.names()
    assert graph.operations == authored.specs


def test_artifact_graph_paths_reference_only_declared_nodes() -> None:
    graph = build_artifact_graph()
    node_names = set(graph.by_name())

    assert {path.name for path in graph.paths} == {
        "raw-reparse-loop",
        "message-fts-health-loop",
        "conversation-query-loop",
        "action-event-repair-loop",
        "session-product-repair-loop",
    }
    for path in graph.paths:
        assert path.nodes
        assert set(path.nodes).issubset(node_names)


def test_artifact_graph_nodes_and_paths_come_from_runtime_artifact_specs() -> None:
    graph = build_artifact_graph()

    assert graph.nodes == build_runtime_artifact_nodes()
    assert graph.paths == build_runtime_artifact_paths()


def test_artifact_graph_serializes_layers_as_strings() -> None:
    payload = build_artifact_graph().to_dict()

    assert any(node["layer"] == "durable" for node in payload["nodes"])
    assert any(path["name"] == "raw-reparse-loop" for path in payload["paths"])
    assert any(operation["name"] == "plan-validation-backlog" for operation in payload["operations"])
    assert any(operation["kind"] == "planning" for operation in payload["operations"])


def test_artifact_graph_operations_reference_only_declared_nodes() -> None:
    graph = build_artifact_graph()
    node_names = set(graph.by_name())
    path_names = set(graph.path_names())

    for operation in graph.operations:
        assert set(operation.consumes).issubset(node_names)
        assert set(operation.produces).issubset(node_names)
        assert set(operation.path_targets).issubset(path_names)


def test_artifact_graph_resolves_runtime_targets() -> None:
    graph = build_artifact_graph()

    assert "action_event_rows" in graph.artifact_names()
    assert "materialize-session-products" in graph.operation_names()
    assert tuple(artifact.name for artifact in graph.resolve_artifacts(("action_event_rows", "missing"))) == (
        "action_event_rows",
    )
    assert tuple(
        operation.name for operation in graph.resolve_operations(("project-action-event-health", "missing"))
    ) == ("project-action-event-health",)


def test_artifact_graph_lists_operations_for_each_runtime_path() -> None:
    graph = build_artifact_graph()

    assert tuple(operation.name for operation in graph.operations_for_path("raw-reparse-loop")) == (
        "plan-validation-backlog",
        "plan-parse-backlog",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("message-fts-health-loop")) == (
        "index-message-fts",
        "project-archive-health",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("conversation-query-loop")) == (
        "index-message-fts",
        "query-conversations",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("action-event-repair-loop")) == (
        "materialize-action-events",
        "project-action-event-health",
    )
    assert tuple(operation.name for operation in graph.operations_for_path("session-product-repair-loop")) == (
        "materialize-session-products",
        "project-session-product-health",
    )

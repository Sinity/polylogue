from __future__ import annotations

from polylogue.artifact_graph import ArtifactLayer, build_artifact_graph


def test_artifact_graph_contains_the_two_proven_vertical_paths() -> None:
    graph = build_artifact_graph()
    nodes = graph.by_name()
    operations = {operation.name: operation for operation in graph.operations}

    assert set(nodes) >= {
        "raw_validation_state",
        "validation_backlog",
        "parse_backlog",
        "parse_quarantine",
        "tool_use_source_blocks",
        "action_event_rows",
        "action_event_fts",
        "action_event_health",
    }
    assert nodes["raw_validation_state"].layer is ArtifactLayer.DURABLE
    assert nodes["action_event_fts"].layer is ArtifactLayer.INDEX
    assert nodes["action_event_fts"].depends_on == ("action_event_rows",)
    assert nodes["action_event_health"].depends_on == ("action_event_rows", "action_event_fts")
    assert nodes["parse_quarantine"].depends_on == ("raw_validation_state",)
    assert operations["plan-validation-backlog"].produces == ("validation_backlog",)
    assert operations["plan-parse-backlog"].produces == ("parse_backlog", "parse_quarantine")
    assert operations["project-action-event-health"].consumes == ("action_event_rows", "action_event_fts")


def test_artifact_graph_paths_reference_only_declared_nodes() -> None:
    graph = build_artifact_graph()
    node_names = set(graph.by_name())

    assert {path.name for path in graph.paths} == {
        "raw-reparse-loop",
        "action-event-repair-loop",
    }
    for path in graph.paths:
        assert path.nodes
        assert set(path.nodes).issubset(node_names)


def test_artifact_graph_serializes_layers_as_strings() -> None:
    payload = build_artifact_graph().to_dict()

    assert any(node["layer"] == "durable" for node in payload["nodes"])
    assert any(path["name"] == "raw-reparse-loop" for path in payload["paths"])
    assert any(operation["name"] == "plan-validation-backlog" for operation in payload["operations"])

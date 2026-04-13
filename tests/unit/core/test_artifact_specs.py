from __future__ import annotations

from polylogue.artifacts import build_runtime_artifact_nodes, build_runtime_artifact_paths


def test_runtime_artifact_specs_expose_the_curated_vertical_paths() -> None:
    nodes = build_runtime_artifact_nodes()
    paths = build_runtime_artifact_paths()

    assert {node.name for node in nodes} >= {
        "raw_validation_state",
        "validation_backlog",
        "parse_backlog",
        "parse_quarantine",
        "tool_use_source_blocks",
        "action_event_rows",
        "action_event_fts",
        "action_event_health",
    }
    assert {path.name for path in paths} == {
        "raw-reparse-loop",
        "action-event-repair-loop",
    }


def test_runtime_artifact_paths_reference_only_declared_nodes() -> None:
    nodes = build_runtime_artifact_nodes()
    paths = build_runtime_artifact_paths()
    node_names = {node.name for node in nodes}

    for path in paths:
        assert path.nodes
        assert set(path.nodes).issubset(node_names)

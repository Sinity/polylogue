from __future__ import annotations

import json

from devtools import artifact_graph
from polylogue.artifacts.graph import build_artifact_graph


def test_render_artifact_graph_text_covers_the_runtime_graph() -> None:
    graph = build_artifact_graph()
    rendered = artifact_graph.render_artifact_graph(as_json=False)

    assert rendered.count("Artifact Paths:") == 1
    assert rendered.count("Artifact Operations:") == 1
    assert rendered.count("Maintenance Targets:") == 1
    for path in graph.paths:
        assert f"- {path.name}: {path.description}" in rendered
    for operation in graph.operations:
        assert f"- {operation.name} [{operation.kind.value}]: {operation.description}" in rendered
    for target in graph.maintenance_targets:
        assert f"- {target.name} [{target.mode.value}/{target.category.value}]: {target.description}" in rendered


def test_render_artifact_graph_json_is_machine_readable() -> None:
    payload = json.loads(artifact_graph.render_artifact_graph(as_json=True))

    assert payload == build_artifact_graph().to_dict()

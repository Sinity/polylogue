"""Render the runtime artifact graph for architectural inspection."""

from __future__ import annotations

import argparse
import json
import sys

from polylogue.artifact_graph import build_artifact_graph


def render_artifact_graph(*, as_json: bool) -> str:
    graph = build_artifact_graph()
    if as_json:
        return json.dumps(graph.to_dict(), indent=2)

    lines: list[str] = ["Artifact Paths:"]
    for path in graph.paths:
        lines.append(f"- {path.name}: {path.description}")
        for node_name in path.nodes:
            node = graph.by_name()[node_name]
            dependencies = f" <- {', '.join(node.depends_on)}" if node.depends_on else ""
            lines.append(f"  - {node.name} [{node.layer.value}]{dependencies}")
    lines.append("")
    lines.append("Artifact Operations:")
    for operation in graph.operations:
        consumes = ", ".join(operation.consumes) if operation.consumes else "—"
        produces = ", ".join(operation.produces) if operation.produces else "—"
        lines.append(f"- {operation.name} [{operation.kind.value}]: {operation.description}")
        lines.append(f"  - consumes: {consumes}")
        lines.append(f"  - produces: {produces}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit the artifact graph as JSON.")
    args = parser.parse_args(argv)
    sys.stdout.write(render_artifact_graph(as_json=args.json))
    sys.stdout.write("\n")
    return 0


__all__ = ["main", "render_artifact_graph"]

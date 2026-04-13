"""Render the runtime artifact graph for architectural inspection."""

from __future__ import annotations

import argparse
import json
import sys

from polylogue.artifact_graph import build_artifact_graph
from polylogue.scenarios import ScenarioMetadata
from polylogue.showcase.exercises import EXERCISES


def _runtime_scenario_coverage() -> dict[str, dict[str, list[dict[str, str]]]]:
    from devtools.quality_registry import build_quality_registry

    registry = build_quality_registry()
    scenario_like_items = [
        ("exercise", exercise.name, ScenarioMetadata.from_object(exercise))
        for exercise in EXERCISES
    ]
    scenario_like_items.extend(
        ("benchmark-campaign", campaign.name, ScenarioMetadata.from_object(campaign))
        for campaign in registry.benchmark_campaigns
    )
    scenario_like_items.extend(
        ("synthetic-benchmark", campaign.name, ScenarioMetadata.from_object(campaign))
        for campaign in registry.synthetic_benchmark_campaigns
    )

    graph = build_artifact_graph()
    artifact_refs: dict[str, list[dict[str, str]]] = {name: [] for name in graph.by_name()}
    operation_refs: dict[str, list[dict[str, str]]] = {operation.name: [] for operation in graph.operations}

    for source_kind, name, metadata in scenario_like_items:
        for artifact_name in metadata.runtime_artifact_targets():
            artifact_refs[artifact_name].append({"source": source_kind, "name": name, "origin": metadata.origin})
        for operation_name in metadata.runtime_operation_targets():
            operation_refs[operation_name].append({"source": source_kind, "name": name, "origin": metadata.origin})

    covered_artifacts = {name: refs for name, refs in artifact_refs.items() if refs}
    covered_operations = {name: refs for name, refs in operation_refs.items() if refs}

    return {
        "artifacts": covered_artifacts,
        "operations": covered_operations,
        "uncovered_artifacts": sorted(name for name, refs in artifact_refs.items() if not refs),
        "uncovered_operations": sorted(name for name, refs in operation_refs.items() if not refs),
    }


def render_artifact_graph(*, as_json: bool) -> str:
    graph = build_artifact_graph()
    coverage = _runtime_scenario_coverage()
    if as_json:
        payload = graph.to_dict()
        payload["scenario_coverage"] = coverage
        return json.dumps(payload, indent=2)

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
    lines.append("")
    lines.append("Runtime Scenario Coverage:")
    if not coverage["artifacts"] and not coverage["operations"]:
        lines.append("- none")
        return "\n".join(lines)
    for artifact_name, refs in sorted(coverage["artifacts"].items()):
        rendered_refs = ", ".join(f"{ref['source']}:{ref['name']}" for ref in refs)
        lines.append(f"- artifact {artifact_name}: {rendered_refs}")
    for operation_name, refs in sorted(coverage["operations"].items()):
        rendered_refs = ", ".join(f"{ref['source']}:{ref['name']}" for ref in refs)
        lines.append(f"- operation {operation_name}: {rendered_refs}")
    if coverage["uncovered_artifacts"]:
        lines.append("- uncovered artifacts: " + ", ".join(coverage["uncovered_artifacts"]))
    if coverage["uncovered_operations"]:
        lines.append("- uncovered operations: " + ", ".join(coverage["uncovered_operations"]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit the artifact graph as JSON.")
    args = parser.parse_args(argv)
    sys.stdout.write(render_artifact_graph(as_json=args.json))
    sys.stdout.write("\n")
    return 0


__all__ = ["main", "render_artifact_graph"]

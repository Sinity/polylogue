"""Render the runtime artifact graph for architectural inspection."""

from __future__ import annotations

import argparse
import json
import sys

from devtools.scenario_coverage import build_runtime_scenario_coverage
from polylogue.artifact_graph import build_artifact_graph


def render_artifact_graph(*, as_json: bool) -> str:
    graph = build_artifact_graph()
    coverage = build_runtime_scenario_coverage()
    if as_json:
        payload = graph.to_dict()
        payload["scenario_coverage"] = coverage.to_dict()
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
    lines.append("Maintenance Targets:")
    for target in graph.maintenance_targets:
        artifacts = ", ".join(node.name for node in graph.artifacts_for_maintenance_target(target)) or "—"
        operations = (
            ", ".join(
                operation
                for operation in (target.doctor_readiness_operation, target.doctor_repair_operation)
                if operation
            )
            or "—"
        )
        lines.append(f"- {target.name} [{target.mode.value}/{target.category.value}]: {target.description}")
        lines.append(f"  - artifacts: {artifacts}")
        lines.append(f"  - operations: {operations}")
    lines.append("")
    lines.append("Runtime Path Coverage:")
    for path_name, path_coverage in sorted(coverage.paths.items()):
        status = "complete" if path_coverage.complete else "partial"
        rendered_refs = ", ".join(f"{ref.source}:{ref.name}" for ref in path_coverage.refs) or "—"
        lines.append(f"- {path_name} [{status}]: {rendered_refs}")
        if path_coverage.uncovered_artifacts:
            lines.append(f"  - uncovered artifacts: {', '.join(path_coverage.uncovered_artifacts)}")
        if path_coverage.uncovered_operations:
            lines.append(f"  - uncovered operations: {', '.join(path_coverage.uncovered_operations)}")
    lines.append("")
    lines.append("Runtime Scenario Coverage:")
    if not coverage.artifacts and not coverage.operations:
        lines.append("- none")
        return "\n".join(lines)
    for artifact_name, refs in sorted(coverage.artifacts.items()):
        rendered_refs = ", ".join(f"{ref.source}:{ref.name}" for ref in refs)
        lines.append(f"- artifact {artifact_name}: {rendered_refs}")
    for operation_name, refs in sorted(coverage.operations.items()):
        rendered_refs = ", ".join(f"{ref.source}:{ref.name}" for ref in refs)
        lines.append(f"- operation {operation_name}: {rendered_refs}")
    for target_name, refs in sorted(coverage.maintenance_targets.items()):
        rendered_refs = ", ".join(f"{ref.source}:{ref.name}" for ref in refs)
        lines.append(f"- maintenance {target_name}: {rendered_refs}")
    if coverage.uncovered_artifacts:
        lines.append("- uncovered artifacts: " + ", ".join(coverage.uncovered_artifacts))
    if coverage.uncovered_operations:
        lines.append("- uncovered operations: " + ", ".join(coverage.uncovered_operations))
    if coverage.uncovered_maintenance_targets:
        lines.append("- uncovered maintenance targets: " + ", ".join(coverage.uncovered_maintenance_targets))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit the artifact graph as JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with exit 1 if any artifacts, operations, or maintenance targets are uncovered.",
    )
    args = parser.parse_args(argv)
    sys.stdout.write(render_artifact_graph(as_json=args.json))
    sys.stdout.write("\n")
    if args.strict:
        coverage = build_runtime_scenario_coverage()
        if coverage.uncovered_artifacts or coverage.uncovered_operations or coverage.uncovered_maintenance_targets:
            return 1
    return 0


__all__ = ["main", "render_artifact_graph"]

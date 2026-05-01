"""Render authored scenario-bearing verification projections."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable

from devtools.scenario_projection_catalog import build_scenario_projection_entries
from polylogue.scenarios import ScenarioProjectionEntry


def _select_projections(
    projections: Iterable[ScenarioProjectionEntry],
    *,
    source_kinds: tuple[str, ...],
    path_target: str | None,
    artifact_target: str | None,
    operation_target: str | None,
    tag: str | None,
) -> tuple[ScenarioProjectionEntry, ...]:
    selected: list[ScenarioProjectionEntry] = []
    for entry in projections:
        if source_kinds and entry.source_kind.value not in source_kinds:
            continue
        if path_target and path_target not in entry.runtime_path_targets():
            continue
        if artifact_target and artifact_target not in entry.artifact_targets:
            continue
        if operation_target and operation_target not in entry.operation_targets:
            continue
        if tag and tag not in entry.tags:
            continue
        selected.append(entry)
    return tuple(selected)


def render_scenario_projections(
    *,
    as_json: bool,
    source_kinds: tuple[str, ...] = (),
    path_target: str | None = None,
    artifact_target: str | None = None,
    operation_target: str | None = None,
    tag: str | None = None,
) -> str:
    projections = _select_projections(
        build_scenario_projection_entries(),
        source_kinds=source_kinds,
        path_target=path_target,
        artifact_target=artifact_target,
        operation_target=operation_target,
        tag=tag,
    )
    if as_json:
        return json.dumps([entry.to_dict() for entry in projections], indent=2)

    lines = [f"Scenario Projections ({len(projections)}):"]
    if not projections:
        lines.append("- none")
        return "\n".join(lines)
    for entry in projections:
        lines.append(f"- {entry.source_kind.value}:{entry.name} [{entry.origin}]")
        lines.append(f"  - description: {entry.description}")
        lines.append(
            f"  - path targets: {', '.join(entry.runtime_path_targets()) if entry.runtime_path_targets() else '—'}"
        )
        lines.append(f"  - artifact targets: {', '.join(entry.artifact_targets) if entry.artifact_targets else '—'}")
        lines.append(f"  - operation targets: {', '.join(entry.operation_targets) if entry.operation_targets else '—'}")
        lines.append(f"  - tags: {', '.join(entry.tags) if entry.tags else '—'}")
        if entry.docs_role or entry.caption or entry.demonstrates:
            lines.append(f"  - docs role: {entry.docs_role or '—'}")
            lines.append(f"  - caption: {entry.caption or '—'}")
            lines.append(f"  - demonstrates: {', '.join(entry.demonstrates) if entry.demonstrates else '—'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit scenario projections as JSON.")
    parser.add_argument(
        "--source-kind",
        dest="source_kinds",
        action="append",
        default=[],
        help="Restrict to a specific projection source kind (repeatable).",
    )
    parser.add_argument(
        "--path-target", default=None, help="Restrict to projections covering this runtime path target."
    )
    parser.add_argument(
        "--artifact-target", default=None, help="Restrict to projections covering this artifact target."
    )
    parser.add_argument(
        "--operation-target",
        default=None,
        help="Restrict to projections covering this operation target.",
    )
    parser.add_argument("--tag", default=None, help="Restrict to projections carrying this tag.")
    args = parser.parse_args(argv)
    sys.stdout.write(
        render_scenario_projections(
            as_json=args.json,
            source_kinds=tuple(args.source_kinds),
            path_target=args.path_target,
            artifact_target=args.artifact_target,
            operation_target=args.operation_target,
            tag=args.tag,
        )
    )
    sys.stdout.write("\n")
    return 0


__all__ = ["main", "render_scenario_projections"]

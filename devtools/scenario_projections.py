"""Render authored scenario-bearing verification projections."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from devtools.quality_registry import build_quality_registry


def render_scenario_projections(*, as_json: bool) -> str:
    projections = build_quality_registry().scenario_projections
    if as_json:
        return json.dumps([asdict(entry) for entry in projections], indent=2)

    lines = ["Scenario Projections:"]
    if not projections:
        lines.append("- none")
        return "\n".join(lines)
    for entry in projections:
        lines.append(f"- {entry.source_kind}:{entry.name} [{entry.origin}]")
        lines.append(f"  - description: {entry.description}")
        lines.append(f"  - artifact targets: {', '.join(entry.artifact_targets) if entry.artifact_targets else '—'}")
        lines.append(f"  - operation targets: {', '.join(entry.operation_targets) if entry.operation_targets else '—'}")
        lines.append(f"  - tags: {', '.join(entry.tags) if entry.tags else '—'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit scenario projections as JSON.")
    args = parser.parse_args(argv)
    sys.stdout.write(render_scenario_projections(as_json=args.json))
    sys.stdout.write("\n")
    return 0


__all__ = ["main", "render_scenario_projections"]

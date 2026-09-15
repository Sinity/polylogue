"""Render docs/api-parity.md from the declared semantic-operation matrix.

The table is derived, never transcribed: MCP rows come from
``polylogue.mcp.declarations.registry`` through ``polylogue.api.parity``, and
each cell is either a live binding or a recorded intentional absence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from devtools.command_catalog import control_plane_command
from devtools.render_support import write_if_changed
from polylogue.api.parity import (
    EXCLUSIONS,
    ROOT_QUERY_MODE_TOKEN,
    SemanticOperation,
    is_async_operation,
    public_facade_callables,
    semantic_operations,
)

OUTPUT = "docs/api-parity.md"
_RENDER_COMMAND = control_plane_command("render api-parity")


def _cell(operation: SemanticOperation, surface: str) -> str:
    binding = operation.binding(surface)
    if binding is None:
        return "**missing row**"
    if not binding.bound:
        return f"_absent_: {binding.absence_reason}"
    if surface == "cli":
        if binding.target == ROOT_QUERY_MODE_TOKEN:
            return f"`polylogue {binding.target} …` (root query mode)"
        return f"`polylogue {binding.target}`"
    if surface == "mcp":
        return f"`{binding.target}`"
    name = binding.target.rsplit(".", 1)[-1]
    prefix = "await " if is_async_operation(name) else ""
    return f"`{prefix}archive.{name}()`"


def build_document() -> str:
    operations = semantic_operations()
    lines = [
        "[← Back to README](../README.md)",
        "",
        "# CLI / MCP / Python operation parity",
        "",
        f"<!-- GENERATED FILE: edit polylogue/api/parity.py, then run `{_RENDER_COMMAND}`. -->",
        "",
        "Parity is governed by stable operation identities, not by name-shaped",
        "reflection. Each row below is one semantic operation; each cell is either a",
        "live binding that must resolve or an intentional absence with its reason.",
        "MCP rows are derived from the MCP declaration registry, so a new tool adds a",
        "row here automatically.",
        "",
        f"Drift check: `{control_plane_command('verify api-parity --check')}` (also run by",
        f"`{control_plane_command('verify --quick')}` as `gate api-parity`).",
        "",
        "## Operations",
        "",
        "| Operation | Summary | CLI | MCP | Python |",
        "| --- | --- | --- | --- | --- |",
    ]
    for operation in operations:
        summary = operation.summary.replace("|", "\\|").strip()
        lines.append(
            f"| `{operation.operation_id}` | {summary} | {_cell(operation, 'cli')} | "
            f"{_cell(operation, 'mcp')} | {_cell(operation, 'python')} |"
        )
    lines.extend(
        [
            "",
            "## Classification of the public Python facade",
            "",
            f"Every public callable on `polylogue.api.Polylogue` ({len(public_facade_callables())} at",
            "render time) is either bound by an operation above or listed here as an",
            "explicit exclusion. An unclassified callable fails the parity gate.",
            "",
            "| Category | Reason | Callables |",
            "| --- | --- | --- |",
        ]
    )
    by_category: dict[str, tuple[str, list[str]]] = {}
    for excluded in EXCLUSIONS:
        reason, members = by_category.setdefault(excluded.category, (excluded.reason, []))
        members.append(excluded.name)
    for category in sorted(by_category):
        reason, members = by_category[category]
        rendered = ", ".join(f"`{name}`" for name in sorted(members))
        lines.append(f"| `{category}` | {reason} | {rendered} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the CLI/MCP/Python operation parity matrix.")
    parser.add_argument("--output", default=OUTPUT, help=f"Target markdown file (default: {OUTPUT})")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the committed matrix is out of sync with the declarations.",
    )
    args = parser.parse_args(argv)

    output_path = Path(args.output).expanduser()
    rendered = build_document()
    if args.check:
        try:
            current = output_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            current = None
        if current != rendered:
            print(f"render api-parity: out of sync: {output_path}", file=sys.stderr)
            print(f"render api-parity: run: {_RENDER_COMMAND}", file=sys.stderr)
            return 1
        print(f"render api-parity: sync OK: {output_path}")
        return 0

    write_if_changed(output_path, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

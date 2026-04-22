"""Render the generated command catalog inside docs/devtools.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from devtools.command_catalog import (
    CommandSpec,
    control_plane_command,
    featured_command_specs,
    grouped_command_specs,
    verification_lab_command_specs,
)
from devtools.render_support import write_if_changed

MARKER = "devtools-command-catalog"


def _render_table(category: str, commands: list[CommandSpec]) -> list[str]:
    lines = [
        f"### {category.title()}",
        "",
        "| Command | Description |",
        "| --- | --- |",
    ]
    for spec in commands:
        lines.append(f"| `{spec.invocation}` | {spec.description} |")
    lines.append("")
    return lines


def _render_featured_commands(commands: tuple[CommandSpec, ...]) -> list[str]:
    lines = [
        "## Core Loop",
        "",
        "These are the commands worth remembering during normal repo work:",
        "",
    ]
    for spec in commands:
        lines.append(f"- `{spec.invocation}`: {spec.use_when or spec.description}")
        if spec.examples:
            lines.append(f"  Common forms: {', '.join(f'`{example}`' for example in spec.examples)}.")
    lines.append("")
    return lines


def _render_verification_lab_surface(commands: tuple[CommandSpec, ...]) -> list[str]:
    if not commands:
        return []
    lines = [
        "## Verification Lab Surface",
        "",
        "The proof-lab operator surface intentionally lives in `devtools` for now. These commands operate on",
        "repo proof obligations and evidence records, not end-user archive workflows.",
        "",
        "| Command | Role |",
        "| --- | --- |",
    ]
    for spec in commands:
        lines.append(f"| `{spec.invocation}` | {spec.use_when or spec.description} |")
    lines.append("")
    return lines


def build_command_catalog() -> str:
    groups = grouped_command_specs()
    featured = featured_command_specs()
    verification_lab = verification_lab_command_specs()
    lines = [
        "<!-- BEGIN GENERATED: devtools-command-catalog -->",
        "## Command Catalog",
        "",
        "Use these discovery commands before scripting or dispatching subcommands:",
        "",
        "```bash",
        control_plane_command("--help"),
        control_plane_command("--list-commands"),
        control_plane_command("--list-commands", "--json"),
        control_plane_command("status"),
        control_plane_command("status", "--json"),
        "```",
        "",
    ]
    lines.extend(_render_verification_lab_surface(verification_lab))
    if featured:
        lines.extend(_render_featured_commands(featured))
    for category, commands in groups.items():
        lines.extend(_render_table(category, commands))
    lines.append("<!-- END GENERATED: devtools-command-catalog -->")
    return "\n".join(lines)


def replace_marked_section(text: str, replacement: str) -> str:
    begin = f"<!-- BEGIN GENERATED: {MARKER} -->"
    end = f"<!-- END GENERATED: {MARKER} -->"
    start = text.find(begin)
    finish = text.find(end)
    if start == -1 or finish == -1 or finish < start:
        raise ValueError(f"marker block not found: {MARKER}")
    finish += len(end)
    return text[:start] + replacement + text[finish:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the generated command catalog inside docs/devtools.md.")
    parser.add_argument("--output", default="docs/devtools.md", help="Target markdown file (default: docs/devtools.md)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the target file is out of sync with the rendered command catalog.",
    )
    args = parser.parse_args(argv)

    output_path = Path(args.output).expanduser()
    try:
        rendered = replace_marked_section(output_path.read_text(encoding="utf-8"), build_command_catalog())
    except ValueError as exc:
        print(f"render-devtools-reference: {exc}", file=sys.stderr)
        return 1

    if args.check:
        current = output_path.read_text(encoding="utf-8")
        if current != rendered:
            print(f"render-devtools-reference: out of sync: {output_path}", file=sys.stderr)
            print(
                f"render-devtools-reference: run: {control_plane_command('render-devtools-reference')}",
                file=sys.stderr,
            )
            return 1
        print(f"render-devtools-reference: sync OK: {output_path}")
        return 0

    write_if_changed(output_path, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

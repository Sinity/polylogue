"""Render the generated command catalog inside docs/devtools.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from devtools.command_catalog import CommandSpec, grouped_command_specs

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


def build_command_catalog() -> str:
    groups = grouped_command_specs()
    lines = [
        "<!-- BEGIN GENERATED: devtools-command-catalog -->",
        "## Command Catalog",
        "",
        "Use these discovery commands before calling control-plane subcommands programmatically:",
        "",
        "```bash",
        "python -m devtools --help",
        "python -m devtools --list-commands",
        "python -m devtools --list-commands --json",
        "python -m devtools status",
        "python -m devtools status --json",
        "```",
        "",
    ]
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


def write_if_changed(output_path: Path, content: str) -> None:
    try:
        current = output_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        current = None
    if current == content:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(output_path)


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
            print("render-devtools-reference: run: python -m devtools render-devtools-reference", file=sys.stderr)
            return 1
        print(f"render-devtools-reference: sync OK: {output_path}")
        return 0

    write_if_changed(output_path, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

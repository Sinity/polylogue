"""Unified command surface for repository-maintenance tools."""

from __future__ import annotations

import argparse
import json
import sys

from devtools.command_catalog import COMMAND_SPECS, COMMANDS, grouped_command_specs


def _print_command_inventory(*, as_json: bool) -> None:
    if as_json:
        payload = {
            "commands": [spec.to_dict() for spec in COMMAND_SPECS],
            "categories": [
                {
                    "name": category,
                    "commands": [spec.name for spec in specs],
                }
                for category, specs in grouped_command_specs().items()
            ],
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    print("Commands:")
    for category, specs in grouped_command_specs().items():
        print(f"\n  {category}:")
        for spec in specs:
            print(f"    {spec.name:<25} {spec.description}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Polylogue developer tools.")
    parser.add_argument(
        "--list-commands",
        action="store_true",
        help="List available commands instead of running one.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON for `--list-commands` or command-specific JSON surfaces.",
    )
    parser.add_argument("command", nargs="?", help="Subcommand to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the subcommand")
    parsed = parser.parse_args(argv)

    if parsed.json and not parsed.list_commands and parsed.command is None:
        parser.error("--json requires --list-commands or a command that supports JSON output")

    if parsed.list_commands:
        if parsed.command is not None:
            parser.error("--list-commands does not accept a command argument")
        _print_command_inventory(as_json=parsed.json)
        return 0

    if not parsed.command:
        parser.print_help()
        print()
        _print_command_inventory(as_json=False)
        return 0

    spec = COMMANDS.get(parsed.command)
    if spec is None:
        parser.error(f"unknown command: {parsed.command}")
    command_args = parsed.args
    if parsed.json:
        command_args = ["--json", *command_args]
    return spec.resolve_main()(command_args)


if __name__ == "__main__":
    raise SystemExit(main())

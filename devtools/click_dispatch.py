"""Click-based CLI dispatch for devtools commands.

Generates Click commands from the CommandSpec catalog and preserves:
- Path syntax: ``devtools <group> <command> <args>``
- ``--json`` flag forwarding to subcommands (root or local)
- ``--list-commands --json`` machine-output contract
- Generated docs rendering
"""

from __future__ import annotations

import json as json_mod
import sys
from typing import Any

import click

from devtools.command_catalog import (
    COMMAND_SPECS,
    CommandSpec,
    grouped_command_specs,
    verification_lab_command_specs,
)

GROUP_HELP: dict[str, str] = {
    "bench": "Run benchmark, mutation, SLO, and resource-budget commands.",
    "lab": "Run verification-lab evidence, schema, probe, and policy commands.",
    "provider": "Inspect provider/importer package readiness.",
    "render": "Render and check generated repository surfaces.",
    "release": "Build, smoke, and validate release/distribution readiness.",
    "verify": "Run the local verification baseline or focused checks. Use --inner-help for baseline flags.",
    "workspace": "Inspect and maintain local agent workspace state.",
}


def _print_inventory(*, json: bool) -> None:
    verification_lab = verification_lab_command_specs()
    if json:
        payload = {
            "commands": [spec.to_dict() for spec in COMMAND_SPECS],
            "surfaces": {
                "verification_lab": [spec.name for spec in verification_lab],
            },
            "categories": [
                {
                    "name": category,
                    "commands": [spec.name for spec in specs],
                }
                for category, specs in grouped_command_specs().items()
            ],
        }
        json_mod.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    click.echo("Commands:")
    if verification_lab:
        click.echo("\n  verification lab surface:")
        for spec in verification_lab:
            click.echo(f"    {spec.name:<25} {spec.description}")
    for category, specs in grouped_command_specs().items():
        click.echo(f"\n  {category}:")
        for spec in specs:
            click.echo(f"    {spec.name:<25} {spec.description}")


class _PreservedEpilogCommand(click.Command):
    """Click command that emits the epilog verbatim, preserving newlines."""

    def format_epilog(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        if not self.epilog:
            return
        formatter.write("\n")
        for line in self.epilog.splitlines():
            formatter.write(line + "\n")


class _DefaultCommandGroup(click.Group):
    """Group that falls back to a default command when no subcommand matches."""

    def __init__(
        self,
        name: str,
        *,
        default_command: click.Command,
        help: str | None = None,
        context_settings: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, help=help, context_settings=context_settings)
        self.no_args_is_help = False
        default_command.hidden = True
        self.default_command = default_command
        self.add_command(default_command, "_default")

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if not args:
            args = ["_default"]
        return super().parse_args(ctx, args)

    def resolve_command(
        self,
        ctx: click.Context,
        args: list[str],
    ) -> tuple[str | None, click.Command | None, list[str]]:
        if not args or args[0] not in self.commands:
            return self.default_command.name, self.default_command, args
        return super().resolve_command(ctx, args)


def _build_epilog(spec: CommandSpec) -> str | None:
    """Render ``use_when`` and ``examples`` from a CommandSpec into a help epilog.

    Returns ``None`` when neither field has content so Click omits the
    section entirely.
    """
    sections: list[str] = []
    if spec.use_when:
        sections.append(f"Use when:\n  {spec.use_when}")
    if spec.examples:
        example_lines = "\n".join(f"  {line}" for line in spec.examples)
        sections.append(f"Examples:\n{example_lines}")
    if not sections:
        return None
    return "\n\n".join(sections)


def _make_command(spec: CommandSpec) -> click.Command:
    """Create a Click command from a CommandSpec.

    Args after the command name are forwarded as-is to the spec's
    resolve_main() entrypoint.  The ``--json`` flag is accepted both at
    the root group level (propagated via ctx.obj) and locally on each
    command.  ``--inner-help`` proxies ``--help`` to the wrapped
    argparse entrypoint so callers can discover sub-flags the catalog
    does not declare.
    """
    from devtools.command_catalog import COMMANDS

    def callback(args: tuple[str, ...], json_flag: bool = False, inner_help: bool = False) -> None:
        ctx = click.get_current_context()
        root_json = ctx.obj.get("json", False) if ctx.obj else False
        argv = list(args)
        if inner_help:
            argv = [*argv, "--help"]
        if json_flag or root_json:
            argv = [*argv, "--json"]
        # Resolve at call time so monkeypatching COMMANDS works (used in tests)
        cmd_spec = COMMANDS.get(spec.name, spec)
        exit_code = cmd_spec.resolve_main()(argv)
        ctx.exit(exit_code)

    params: list[click.Parameter] = [
        click.Argument(
            ["args"],
            nargs=-1,
            required=False,
        ),
        click.Option(
            ["--json", "json_flag"],
            is_flag=True,
            help="Emit machine-readable JSON for this command.",
            expose_value=True,
        ),
        click.Option(
            ["--inner-help", "inner_help"],
            is_flag=True,
            help="Forward --help to the wrapped command for its native flag list.",
            expose_value=True,
        ),
    ]

    cmd = _PreservedEpilogCommand(
        name=spec.command_path[-1],
        help=spec.description,
        epilog=_build_epilog(spec),
        callback=callback,
        params=params,
    )
    # Subcommands that use argparse internally need unknown options forwarded
    # as-is rather than rejected by Click's option parser.  This allows
    # modules like devtools/task_history.py with their own sub-subcommands and
    # --flags to work transparently.
    cmd.allow_extra_args = True
    cmd.ignore_unknown_options = True
    return cmd


def _ensure_group(
    parent: click.Group,
    name: str,
    *,
    default_command: click.Command | None = None,
) -> click.Group:
    existing = parent.commands.get(name)
    if existing is not None:
        if not isinstance(existing, click.Group):
            raise ValueError(f"cannot register devtools group {name!r}: command already exists")
        return existing
    if default_command is None:
        group = click.Group(name=name, help=GROUP_HELP.get(name))
    else:
        group = _DefaultCommandGroup(
            name=name,
            help=GROUP_HELP.get(name),
            default_command=default_command,
            context_settings={
                "ignore_unknown_options": True,
                "allow_extra_args": True,
            },
        )
    parent.add_command(group)
    return group


def _make_cli() -> click.Group:
    """Build the root Click group with all CommandSpec commands registered."""

    @click.group(name="devtools", invoke_without_command=True)
    @click.option(
        "--json", is_flag=True, help="Emit machine-readable JSON for --list-commands or command-specific JSON surfaces."
    )
    @click.option("--list-commands", is_flag=True, help="List available commands instead of running one.")
    @click.pass_context
    def cli(ctx: click.Context, json: bool, list_commands: bool) -> None:
        """Polylogue developer tools."""
        ctx.ensure_object(dict)
        ctx.obj["json"] = json

        if list_commands:
            _print_inventory(json=json)
            ctx.exit(0)

        if ctx.invoked_subcommand is None:
            click.echo(cli.get_help(ctx))
            ctx.exit(0)

    nested_group_roots = {
        spec.command_path[0]
        for spec in COMMAND_SPECS
        if len(spec.command_path) > 1 and any(other.command_path == spec.command_path[:1] for other in COMMAND_SPECS)
    }

    for spec in COMMAND_SPECS:
        cmd = _make_command(spec)
        parent = cli
        if len(spec.command_path) == 1 and spec.command_path[0] in nested_group_roots:
            _ensure_group(parent, spec.command_path[0], default_command=cmd)
            continue
        for group_name in spec.command_path[:-1]:
            parent = _ensure_group(parent, group_name)
        parent.add_command(cmd)

    return cli


cli = _make_cli()


def _dispatch(argv: list[str]) -> int:
    """Run the Click CLI and translate ``SystemExit`` to an int return code."""
    try:
        cli(args=argv, prog_name="devtools", standalone_mode=True)
        return 0
    except SystemExit as e:
        code = e.code
        return code if isinstance(code, int) else 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for programmatic use of the Click-based devtools CLI.

    Converts argv to Click invocation and returns the exit code.  Every
    invocation appends a JSONL record to ``.agent/task-history/tasks.jsonl`` so
    agent task history is self-populating (see ``devtools workspace tasks``).  Set
    ``POLYLOGUE_TASK_HISTORY_DISABLE=1`` to opt out (also suppressed during a
    ``devtools workspace tasks replay`` to avoid double-logging the outer wrapper).
    """
    import os
    import time

    from devtools import task_history as task_history_mod

    args_list = list(argv or [])
    if not args_list or args_list[0].startswith("-"):
        # Bare invocation or root option only — skip auto-log.
        return _dispatch(args_list)

    command_name = args_list[0]
    inner_args = args_list[1:]
    for spec in sorted(COMMAND_SPECS, key=lambda item: len(item.command_path), reverse=True):
        path = spec.command_path
        if tuple(args_list[: len(path)]) == path:
            command_name = " ".join(path)
            inner_args = args_list[len(path) :]
            break

    if task_history_mod.auto_log_disabled():
        return _dispatch(args_list)

    started = time.perf_counter()
    exit_code = 0
    try:
        exit_code = _dispatch(args_list)
        return exit_code
    finally:
        duration_ms = (time.perf_counter() - started) * 1000.0
        task_history_mod.record_invocation(
            command=command_name,
            args=inner_args,
            duration_ms=duration_ms,
            exit_code=exit_code,
            cwd=os.getcwd(),
        )

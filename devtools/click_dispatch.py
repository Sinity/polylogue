"""Click-based CLI dispatch for devtools commands.

Generates Click commands from the CommandSpec catalog and preserves:
- Flat CLI syntax: ``devtools <command> <args>``
- ``--json`` flag forwarding to subcommands (root or local)
- ``--list-commands --json`` machine-output contract
- Generated docs rendering
"""

from __future__ import annotations

import json as json_mod
import sys

import click

from devtools.command_catalog import (
    COMMAND_SPECS,
    CommandSpec,
    grouped_command_specs,
    verification_lab_command_specs,
)


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


def _make_command(spec: CommandSpec) -> click.Command:
    """Create a Click command from a CommandSpec.

    Args after the command name are forwarded as-is to the spec's
    resolve_main() entrypoint.  The ``--json`` flag is accepted both at
    the root group level (propagated via ctx.obj) and locally on each
    command.
    """
    from devtools.command_catalog import COMMANDS

    def callback(args: tuple[str, ...], json_flag: bool = False) -> None:
        ctx = click.get_current_context()
        root_json = ctx.obj.get("json", False) if ctx.obj else False
        argv = list(args)
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
    ]

    cmd = click.Command(
        name=spec.name,
        help=spec.description,
        callback=callback,
        params=params,
    )
    # Subcommands that use argparse internally need unknown options forwarded
    # as-is rather than rejected by Click's option parser.  This allows
    # modules like devtools/xtask.py with their own sub-subcommands and
    # --flags to work transparently.
    cmd.allow_extra_args = True
    cmd.ignore_unknown_options = True
    return cmd


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

    for spec in COMMAND_SPECS:
        cmd = _make_command(spec)
        cli.add_command(cmd)

    return cli


cli = _make_cli()


def main(argv: list[str] | None = None) -> int:
    """Entry point for programmatic use of the Click-based devtools CLI.

    Converts argv to Click invocation and returns the exit code.
    """
    try:
        cli(args=argv or [], standalone_mode=True)
        return 0
    except SystemExit as e:
        code = e.code
        return code if isinstance(code, int) else 0

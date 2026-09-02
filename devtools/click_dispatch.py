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
from pathlib import Path

import click

from devtools import system_exit
from devtools.checkout_guard import (
    CheckoutImportMismatchError,
    assert_polylogue_matches_checkout,
)
from devtools.command_catalog import (
    COMMAND_SPECS,
    CommandSpec,
    grouped_command_specs,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

GROUP_HELP: dict[str, str] = {
    "archive": "Run archive-facing evidence and generation actuators.",
    "bench": "Run benchmark, SLO, and resource-budget profiles.",
    "cache": "Maintain the shared fixture caches.",
    "schema": "Inspect, generate, and commit provider schema packages.",
    "smoke": "Probe deployed binaries, routes, and live service shapes.",
}


def _print_inventory(*, json: bool) -> None:
    if json:
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
        json_mod.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    click.echo("Commands:")
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


def _declared_flag_dests(spec: CommandSpec) -> list[tuple[str, str]]:
    """Click destinations for the flags a spec surfaces in its own help."""
    return [(flag, flag.lstrip("-").replace("-", "_")) for flag, _help in spec.flags]


def _make_command(spec: CommandSpec) -> click.Command:
    """Create a Click command from a CommandSpec.

    Args after the command name are forwarded as-is to the spec's
    resolve_main() entrypoint.  The ``--json`` flag is accepted both at
    the root group level (propagated via ctx.obj) and locally on each
    command that declares ``json_flag``.
    """
    from devtools.command_catalog import COMMANDS

    def callback(args: tuple[str, ...], json_flag: bool = False, **declared: bool) -> None:
        ctx = click.get_current_context()
        root_json = ctx.obj.get("json", False) if ctx.obj else False
        argv = [*args, *(flag for flag, dest in _declared_flag_dests(spec) if declared.get(dest))]
        if spec.json_flag and (json_flag or root_json):
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
    ]
    for flag, dest in _declared_flag_dests(spec):
        params.append(click.Option([flag, dest], is_flag=True, help=dict(spec.flags)[flag], expose_value=True))
    if spec.json_flag:
        params.append(
            click.Option(
                ["--json", "json_flag"],
                is_flag=True,
                help="Emit machine-readable JSON for this command.",
                expose_value=True,
            )
        )

    cmd = _PreservedEpilogCommand(
        name=spec.command_path[-1],
        help=spec.description,
        epilog=_build_epilog(spec),
        callback=callback,
        params=params,
    )
    # Subcommands use argparse internally, so unknown options must be forwarded
    # as-is rather than rejected by Click's option parser.
    cmd.allow_extra_args = True
    cmd.ignore_unknown_options = True
    return cmd


def _ensure_group(parent: click.Group, name: str) -> click.Group:
    existing = parent.commands.get(name)
    if existing is not None:
        if not isinstance(existing, click.Group):
            raise ValueError(f"cannot register devtools group {name!r}: command already exists")
        return existing
    group = click.Group(name=name, help=GROUP_HELP.get(name))
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

    for spec in COMMAND_SPECS:
        cmd = _make_command(spec)
        parent = cli
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
        translation = system_exit.translate_system_exit(e)
        if translation.message is not None:
            print(translation.message, file=sys.stderr)
        return translation.code


def main(argv: list[str] | None = None) -> int:
    """Entry point for programmatic use of the Click-based devtools CLI."""

    command_argv = list(argv or [])
    try:
        assert_polylogue_matches_checkout(_REPO_ROOT, context="devtools")
    except CheckoutImportMismatchError as exc:
        sys.stderr.write(f"{exc}\n")
        return 125

    return _dispatch(command_argv)

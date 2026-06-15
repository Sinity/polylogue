"""Generated-style coverage for public CLI action contracts (#1816)."""

from __future__ import annotations

import click

from polylogue.cli.action_contracts import (
    ACTION_CONTRACT_BY_PATH,
    ACTION_CONTRACTS,
    PUBLIC_ACTION_FLOOR,
    VIRTUAL_ACTION_PATHS,
    CliActionContract,
)
from polylogue.cli.click_app import cli
from polylogue.cli.command_inventory import CommandPath, iter_command_paths
from polylogue.cli.query_group import _split_query_mode_args


def _command_paths() -> dict[tuple[str, ...], click.Command]:
    return {entry.path: entry.command for entry in iter_command_paths(cli)}


def _command_for_contract(entry: CliActionContract, commands: dict[tuple[str, ...], click.Command]) -> click.Command:
    if entry.path == ("find",):
        return cli
    return commands[entry.path]


def _choice_values(command: click.Command) -> set[str]:
    values: set[str] = set()
    ctx = click.Context(command)
    for param in command.get_params(ctx):
        if not isinstance(param, click.Option):
            continue
        if not {"--format", "-f", "--output-format", "--json"}.intersection(param.opts):
            continue
        if isinstance(param.type, click.Choice):
            values.update(str(choice) for choice in param.type.choices)
        if "--json" in param.opts:
            values.add("json")
    return values


def test_every_public_floor_action_has_exactly_one_contract() -> None:
    """Every v0 public floor action must declare one executable contract."""
    declared = set(ACTION_CONTRACT_BY_PATH)
    expected = set(PUBLIC_ACTION_FLOOR)
    assert declared == expected, (
        "The #1816 public CLI action floor and ACTION_CONTRACTS drifted.\n"
        f"Missing contracts: {sorted(expected - declared)}\n"
        f"Unexpected contracts: {sorted(declared - expected)}"
    )
    assert len(ACTION_CONTRACTS) == len(declared), "ACTION_CONTRACTS contains duplicate path entries"


def test_contract_paths_resolve_to_click_or_virtual_counterpart() -> None:
    """Every contract path must map to a live Click path or declared virtual grammar action."""
    commands = _command_paths()
    stale = sorted(
        path for path in ACTION_CONTRACT_BY_PATH if path not in commands and path not in VIRTUAL_ACTION_PATHS
    )
    assert not stale, (
        f"These CLI action contracts do not correspond to a live Click command or declared virtual action: {stale}"
    )


def test_virtual_find_counterpart_is_query_parser_keyword() -> None:
    """`find` is public grammar, so prove the parser still recognizes it."""
    click_args, query_terms, has_subcommand, explicit_query = _split_query_mode_args(cli, ["find", "needle"])
    assert click_args == []
    assert query_terms == ("needle",)
    assert not has_subcommand
    assert explicit_query


def test_floor_click_paths_without_contract_are_reported() -> None:
    """The non-virtual floor paths currently present in Click must be contracted."""
    commands = _command_paths()
    missing = sorted(path for path in PUBLIC_ACTION_FLOOR if path in commands and path not in ACTION_CONTRACT_BY_PATH)
    assert not missing, f"Public floor Click paths lack CliActionContract entries: {missing}"


def test_declared_machine_formats_are_supported_by_click_options() -> None:
    """Contracts may not claim JSON/NDJSON unless the live command exposes it."""
    commands = _command_paths()
    unsupported: list[str] = []
    for entry in ACTION_CONTRACTS:
        command = _command_for_contract(entry, commands)
        choices = _choice_values(command)
        if "json" in entry.formats and "json" not in choices:
            unsupported.append(f"{CommandPath(entry.path, command).display_name}: json")
        if "ndjson" in entry.formats and "ndjson" not in choices:
            unsupported.append(f"{CommandPath(entry.path, command).display_name}: ndjson")
    assert not unsupported, f"Contracts declare unsupported machine formats: {unsupported}"


def test_default_format_is_declared_for_every_contract() -> None:
    """The default format must be one of the contract's declared formats."""
    invalid = [entry.path for entry in ACTION_CONTRACTS if entry.default_format not in entry.formats]
    assert not invalid, f"Contracts have default_format outside formats: {invalid}"

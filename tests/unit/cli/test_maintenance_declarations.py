"""The maintenance CLI group is derived from its typed declaration family.

Anti-vacuity for this module: add a command to the Click group without a
declaration, delete a declaration whose command is still registered, rename or
delete a declared command function, drop a nested group's ``nested_group``
flag, or change a declared short help, and one of these tests goes red. The
examples are run through the production Click tree, not a double.
"""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path

import click
from click.testing import CliRunner

from polylogue.cli.click_app import cli as root_cli
from polylogue.cli.click_command_registration import _LazyCommand, _NestedLazyGroup
from polylogue.cli.commands.maintenance import maintenance_group
from polylogue.declarations import DeclarationRegistry, HandlerBinding
from polylogue.declarations.diagnostics import diagnose_registry
from polylogue.maintenance.declarations import (
    MAINTENANCE_COMMAND_DECLARATIONS,
    MAINTENANCE_KERNEL_REGISTRY,
    declaration_for_command,
)

ROOT = Path(__file__).resolve().parents[3]


def test_registered_commands_are_exactly_the_declared_family() -> None:
    """The Click group holds no command the declaration family does not own."""

    assert set(maintenance_group.commands) == {declaration.cli_name for declaration in MAINTENANCE_COMMAND_DECLARATIONS}
    assert len(MAINTENANCE_KERNEL_REGISTRY) == len(MAINTENANCE_COMMAND_DECLARATIONS)


def test_declared_short_help_and_command_type_reach_the_live_group() -> None:
    """Registration metadata is read from the declaration, not a second table."""

    for declaration in MAINTENANCE_COMMAND_DECLARATIONS:
        command = maintenance_group.commands[declaration.cli_name]
        assert command.short_help == declaration.short_help
        expected = _NestedLazyGroup if declaration.nested_group else _LazyCommand
        assert isinstance(command, expected), declaration.cli_name


def test_declared_handler_is_a_real_click_command_in_its_owning_module() -> None:
    """Resolve each lazily-bound handler eagerly, the way dispatch will.

    Anti-vacuity: rename a maintenance command function and this fails here
    instead of at the first real invocation of that subcommand.
    """

    for declaration in MAINTENANCE_COMMAND_DECLARATIONS:
        module = importlib.import_module(declaration.module)
        handler = getattr(module, declaration.attribute)
        assert isinstance(handler, click.Command), f"{declaration.cli_name} -> {declaration.attribute}"
        if declaration.nested_group:
            assert isinstance(handler, click.Group), declaration.cli_name


def test_declared_examples_cross_the_real_cli_adapter() -> None:
    """o21.3 AC4: every declared example resolves through the production tree."""

    runner = CliRunner()
    for declaration in MAINTENANCE_COMMAND_DECLARATIONS:
        for example in declaration.kernel.examples:
            declared_argv = dict(example.arguments)["argv"]
            assert isinstance(declared_argv, tuple)
            argv = [str(item) for item in declared_argv]
            assert argv[: len(declaration.invocation)] == list(declaration.invocation)
            result = runner.invoke(root_cli, argv)
            assert result.exit_code == 0, f"{argv}: {result.output}\n{result.exception!r}"


def test_live_registry_resolves_every_declared_binding() -> None:
    assert diagnose_registry(MAINTENANCE_KERNEL_REGISTRY, root=ROOT) == ()


def test_a_deleted_command_function_is_an_actionable_diagnostic() -> None:
    """A declared handler with no producer fails completeness with a repair."""

    broken = replace(
        MAINTENANCE_COMMAND_DECLARATIONS[0].kernel,
        handlers=(
            HandlerBinding(
                surface="cli",
                owner_path="polylogue/cli/commands/maintenance/_archive_plan.py",
                symbol="archive_plan_command_deleted",
                binding_key="ops maintenance archive-plan",
            ),
        ),
    )
    registry = DeclarationRegistry()
    registry.register(broken)
    diagnostics = diagnose_registry(registry, root=ROOT)
    assert [item.code for item in diagnostics] == ["unresolved_handler_symbol"]
    assert "archive_plan_command_deleted" in diagnostics[0].message
    assert diagnostics[0].repair_command


def test_unknown_command_names_its_exact_repair() -> None:
    try:
        declaration_for_command("not-a-command")
    except KeyError as exc:
        assert "polylogue/maintenance/declarations.py" in str(exc)
        assert "devtools test" in str(exc)
    else:  # pragma: no cover - the lookup must refuse
        raise AssertionError("unknown maintenance command resolved")

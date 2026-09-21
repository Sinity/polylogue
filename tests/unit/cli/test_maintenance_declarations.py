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
from unittest.mock import patch

import click
from click.testing import CliRunner

from polylogue.cli.click_app import cli as root_cli
from polylogue.cli.click_command_registration import _LazyCommand, _NestedLazyGroup
from polylogue.cli.commands.maintenance import maintenance_group
from polylogue.declarations import DeclarationRegistry, HandlerBinding
from polylogue.declarations.diagnostics import diagnose_registry
from polylogue.maintenance.declarations import (
    CAMPAIGN_ACTUATOR_RETIREMENTS,
    MAINTENANCE_COMMAND_DECLARATIONS,
    MAINTENANCE_KERNEL_REGISTRY,
    declaration_for_command,
    retirement_diagnostics,
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


def test_campaign_actuator_retirements_name_a_live_actuator_and_its_commands() -> None:
    """Every declared retirement points at something that is actually there.

    The declaration is the head-side signal polylogue-6kur AC6 asks for: bead
    state is external to a feature branch and cannot gate a check, so the
    actuator, the campaign work it serves, and the condition that retires it
    are recorded in the tree instead.

    Anti-vacuity: rename either actuator class, or point a record's
    ``owner_path`` at a different file than its dotted module, and this goes
    red.
    """

    assert CAMPAIGN_ACTUATOR_RETIREMENTS
    for record in CAMPAIGN_ACTUATOR_RETIREMENTS:
        module_path, separator, symbol = record.actuator.partition(":")
        assert separator, record.actuator
        assert record.owner_path.removesuffix(".py").replace("/", ".") == module_path
        assert f"class {symbol}(" in (ROOT / record.owner_path).read_text(encoding="utf-8")
        assert record.commands and record.serves and record.retires_when
        for command in record.commands:
            assert command.startswith("ops maintenance ")
            declaration_for_command(command.removeprefix("ops maintenance "))


def test_retirement_check_refuses_in_both_directions() -> None:
    """The check is a deletion *successor*, not a standing deletion demand.

    A check that only ever demanded deletion could not express "this actuator
    is still owed"; a check that only ever demanded presence could not end it.
    Both refusals are exercised here against the same tree, with the recorded
    condition as the only difference.

    Anti-vacuity: drop either branch of ``retirement_diagnostics`` and the
    corresponding assertion goes red; the third assertion is the control that
    keeps a blanket refusal from passing the other two.
    """

    live = retirement_diagnostics(root=ROOT)
    assert live == (), [item.message for item in live]

    met = [replace(record, condition_met=True) for record in CAMPAIGN_ACTUATOR_RETIREMENTS]
    with patch("polylogue.maintenance.declarations.CAMPAIGN_ACTUATOR_RETIREMENTS", tuple(met)):
        overdue = retirement_diagnostics(root=ROOT)
    assert {item.code for item in overdue} == {"actuator-retirement-overdue"}
    assert len(overdue) == len(CAMPAIGN_ACTUATOR_RETIREMENTS)
    for record, diagnostic in zip(CAMPAIGN_ACTUATOR_RETIREMENTS, overdue, strict=True):
        assert record.actuator in diagnostic.message
        assert record.retires_when in diagnostic.message

    vanished = [
        replace(record, actuator=f"{record.owner_path.removesuffix('.py').replace('/', '.')}:NoSuchActuator")
        for record in CAMPAIGN_ACTUATOR_RETIREMENTS
    ]
    with patch("polylogue.maintenance.declarations.CAMPAIGN_ACTUATOR_RETIREMENTS", tuple(vanished)):
        premature = retirement_diagnostics(root=ROOT)
    assert {item.code for item in premature} == {"actuator-retirement-premature"}


def test_declaration_bindings_gate_owns_the_retirement_check() -> None:
    """The gate reads the declaration; no new gate name was invented.

    ``devtools gate declaration-bindings`` already resolves this family's
    declared bindings against the live checkout, and its ``domain`` hook is the
    declared seam for bindings the shared kernel cannot resolve.

    Anti-vacuity: unwire ``_maintenance_domain_diagnostics`` from ``REGISTRIES``
    and this goes red while ``retirement_diagnostics`` still passes its own
    tests -- a declaration nothing reads.
    """

    from devtools.verify_declaration_bindings import REGISTRIES

    domain = REGISTRIES["maintenance"].domain
    assert domain is not None
    met = [replace(record, condition_met=True) for record in CAMPAIGN_ACTUATOR_RETIREMENTS]
    with patch("polylogue.maintenance.declarations.CAMPAIGN_ACTUATOR_RETIREMENTS", tuple(met)):
        assert {item.code for item in domain()} == {"actuator-retirement-overdue"}

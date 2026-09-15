"""Executable query declarations own the capability route's identity.

Anti-vacuity for this module: rename an ``ArchiveStore.query_*`` executor,
point a field's ``spec_attr``/``plan_attr`` at an attribute that no longer
exists, add a query field or terminal unit whose declaration does not reach
``explain(subject="capability")``, or serve a capability row whose
``declaration_id`` no declaration owns, and one of these tests goes red. Unit
examples are parsed by the production parser, not compared to another
generated string.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

from polylogue.archive.query.capability_catalog import capability_detail_page
from polylogue.archive.query.declarations import (
    QUERY_CAPABILITY_DECLARATIONS,
    QUERY_KERNEL_REGISTRY,
    capability_declaration_rows,
    query_binding_diagnostics,
)
from polylogue.archive.query.fields import QUERY_FIELD_DESCRIPTORS, mcp_query_field_names
from polylogue.archive.query.metadata import query_unit_descriptors
from polylogue.declarations import DeclarationRegistry, HandlerBinding
from polylogue.declarations.diagnostics import diagnose_registry

ROOT = Path(__file__).resolve().parents[4]


def test_declaration_identity_has_exactly_one_owner() -> None:
    """The served rows are the declarations; the route re-derives nothing."""

    served = capability_declaration_rows()
    assert [row["declaration_id"] for row in served] == [
        declaration.kernel.declaration_id for declaration in QUERY_CAPABILITY_DECLARATIONS
    ]
    assert len(QUERY_KERNEL_REGISTRY) == len(served)


def test_every_live_field_and_terminal_unit_is_declared() -> None:
    """o21.3 AC5: the family is derived from the live descriptors.

    Anti-vacuity: add a ``QueryFieldDescriptor`` or a terminal
    ``QueryUnitDescriptor`` and it must appear here with no second edit;
    a hand-maintained projection would omit it.
    """

    declared = {declaration.kernel.declaration_id for declaration in QUERY_CAPABILITY_DECLARATIONS}
    expected = {f"query.field.{field.name}" for field in QUERY_FIELD_DESCRIPTORS}
    expected |= {f"query.unit.{descriptor.unit}" for descriptor in query_unit_descriptors(terminal_supported=True)}
    assert declared == expected


def test_capability_detail_page_serves_only_declared_rows() -> None:
    """The bounded MCP detail route is a view over the declaration family."""

    page = capability_detail_page(limit=25)
    assert page["total"] == len(QUERY_CAPABILITY_DECLARATIONS)
    owned = {declaration.kernel.declaration_id for declaration in QUERY_CAPABILITY_DECLARATIONS}
    items = page["items"]
    assert isinstance(items, list)
    assert {str(cast(dict[str, object], item)["declaration_id"]) for item in items} <= owned


def test_declared_unit_examples_parse_through_the_production_parser() -> None:
    """o21.3 AC4: unit examples are executable, not documentation prose."""

    from polylogue.archive.query.expression import parse_unit_source_expression

    for declaration in QUERY_CAPABILITY_DECLARATIONS:
        if declaration.kind != "unit":
            continue
        expression = str(dict(declaration.kernel.examples[0].arguments)["expression"])
        parsed = parse_unit_source_expression(expression)
        assert parsed is not None, expression


def test_declared_field_examples_name_live_boundary_spellings() -> None:
    """A declared MCP parameter must exist in the live MCP query vocabulary."""

    boundary = mcp_query_field_names()
    seen = 0
    for declaration in QUERY_CAPABILITY_DECLARATIONS:
        if declaration.kind != "field":
            continue
        for example in declaration.kernel.examples:
            arguments = dict(example.arguments)
            if "mcp_parameter" not in arguments:
                continue
            seen += 1
            assert str(arguments["mcp_parameter"]) in boundary, declaration.kernel.declaration_id
    assert seen > 0, "no field declared an MCP parameter example; the check would be vacuous"


def test_live_domain_bindings_resolve() -> None:
    """Every declared spec/plan attribute and unit executor exists today."""

    assert query_binding_diagnostics() == ()


def test_live_registry_resolves_every_declared_binding() -> None:
    assert diagnose_registry(QUERY_KERNEL_REGISTRY, root=ROOT) == ()


def test_a_renamed_unit_executor_is_an_actionable_diagnostic() -> None:
    """o21.3 AC3: a unit whose SQL executor no longer exists fails with a repair.

    The production route reaches the executor by ``getattr(archive,
    descriptor.sql_query_method)``; without this the rename surfaces as a
    runtime ``AttributeError`` under a real query instead of a gate finding.

    Anti-vacuity: this mutates only the declared symbol, so a gate that stopped
    resolving handler symbols would report nothing.
    """

    message_unit = next(
        declaration
        for declaration in QUERY_CAPABILITY_DECLARATIONS
        if declaration.kernel.declaration_id == "query.unit.message"
    )
    assert any(handler.symbol == "query_messages" for handler in message_unit.kernel.handlers)
    broken = replace(
        message_unit.kernel,
        handlers=(
            HandlerBinding(
                surface="archive-sql-executor",
                owner_path="polylogue/storage/sqlite/archive_tiers/archive.py",
                symbol="query_messages_renamed_away",
                binding_key="ArchiveStore.query_messages_renamed_away",
            ),
        ),
    )
    registry = DeclarationRegistry()
    registry.register(broken)
    diagnostics = diagnose_registry(registry, root=ROOT)
    assert [item.code for item in diagnostics] == ["unresolved_handler_symbol"]
    assert "query_messages_renamed_away" in diagnostics[0].message
    assert diagnostics[0].repair_command

"""Actionable registration diagnostics over the shared declaration kernel.

Anti-vacuity: each test names the mutation that turns it red -- a handler
symbol that no longer exists, an owner path that was moved or deleted, an
output with no target, or a live registry whose declared handlers stop
resolving.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from polylogue.declarations import (
    CompatibilityKey,
    CompletenessEdge,
    DeclarationRegistry,
    DeclarationSpec,
    ExampleSpec,
    HandlerBinding,
    OutputSpec,
)
from polylogue.declarations.diagnostics import diagnose_registry, format_diagnostic, resolve_declaration

ROOT = Path(__file__).resolve().parents[3]

_DECLARATION = DeclarationSpec(
    declaration_id="synthetic.probe",
    family_id="synthetic",
    public_name="probe",
    owner_path="polylogue/declarations/diagnostics.py",
    compatibility=CompatibilityKey("synthetic", "executable", "read", "single-object", "rebuildable"),
    producer="polylogue.declarations.diagnostics.diagnose_registry",
    role_gate="library",
    schema_ref="polylogue.declarations.diagnostics:diagnose_registry",
    discovery_text="Synthetic probe declaration.",
    repair_command="devtools gate declaration-bindings",
    handlers=(
        HandlerBinding(
            surface="library",
            owner_path="polylogue/declarations/diagnostics.py",
            symbol="diagnose_registry",
            binding_key="polylogue.declarations.diagnostics:diagnose_registry",
        ),
    ),
    outputs=(OutputSpec(name="report", kind="python", schema_ref="Diagnostic", target_path="stdout"),),
    examples=(ExampleSpec(name="minimal", summary="Diagnose one registry."),),
    completeness_edges=(
        CompletenessEdge(
            producer="synthetic.probe",
            consumer="devtools/verify_declaration_bindings.py",
            kind="gate",
            owner_path="devtools/verify_declaration_bindings.py",
        ),
    ),
)


def _registry(declaration: DeclarationSpec) -> DeclarationRegistry:
    registry = DeclarationRegistry()
    registry.register(declaration)
    return registry


def test_complete_declaration_reports_nothing() -> None:
    assert diagnose_registry(_registry(_DECLARATION), root=ROOT) == ()


def test_unresolved_handler_symbol_is_reported() -> None:
    broken = replace(
        _DECLARATION,
        handlers=(replace(_DECLARATION.handlers[0], symbol="symbol_that_vanished"),),
    )
    codes = {item.code for item in diagnose_registry(_registry(broken), root=ROOT)}
    assert "unresolved_handler_symbol" in codes


def test_unresolved_owner_path_is_reported() -> None:
    broken = replace(_DECLARATION, owner_path="polylogue/declarations/file_that_moved.py")
    diagnostics = diagnose_registry(_registry(broken), root=ROOT)
    assert "unresolved_owner_path" in {item.code for item in diagnostics}


def test_missing_output_target_is_reported() -> None:
    broken = replace(
        _DECLARATION,
        outputs=(replace(_DECLARATION.outputs[0], target_path=""),),
    )
    assert "missing_output_target" in {item.code for item in diagnose_registry(_registry(broken), root=ROOT)}


def test_every_diagnostic_names_a_path_and_a_repair_command() -> None:
    broken = replace(_DECLARATION, owner_path="polylogue/declarations/file_that_moved.py")
    for diagnostic in diagnose_registry(_registry(broken), root=ROOT):
        line = format_diagnostic(diagnostic)
        assert diagnostic.owner_path in line and diagnostic.repair_command in line


def test_resolution_reports_every_declared_binding() -> None:
    resolutions = resolve_declaration(_DECLARATION, root=ROOT)
    assert {item.kind for item in resolutions} == {"owner-path", "handler-owner-path", "handler-symbol"}
    assert all(item.resolved for item in resolutions)


def test_live_registries_resolve() -> None:
    """The production MCP declarations resolve against the checkout.

    Anti-vacuity: rename a registered MCP handler function without updating its
    declaration and this goes red.
    """

    from devtools.verify_declaration_bindings import run as run_bindings

    report = run_bindings(root=ROOT)
    assert report == dict.fromkeys(report, ())

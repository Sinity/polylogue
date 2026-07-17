"""Actionable completeness diagnostics for declaration registries."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.declarations.models import DeclarationSpec
from polylogue.declarations.registry import DeclarationRegistry


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One source-locatable declaration repair."""

    code: str
    message: str
    declaration_id: str
    owner_path: str
    repair_command: str


def _diagnostic(declaration: DeclarationSpec, code: str, missing: str) -> Diagnostic:
    return Diagnostic(
        code=code,
        message=f"{declaration.declaration_id}: missing {missing}",
        declaration_id=declaration.declaration_id,
        owner_path=declaration.owner_path,
        repair_command=declaration.repair_command,
    )


def validate_declaration(declaration: DeclarationSpec) -> tuple[Diagnostic, ...]:
    """Return every missing shared edge without coercing the declaration."""

    diagnostics: list[Diagnostic] = []
    if not declaration.producer:
        diagnostics.append(_diagnostic(declaration, "missing_producer", "producer"))
    if not declaration.handlers:
        diagnostics.append(_diagnostic(declaration, "missing_handler", "handler binding"))
    if not declaration.role_gate:
        diagnostics.append(_diagnostic(declaration, "missing_role_gate", "role gate"))
    if not declaration.schema_ref:
        diagnostics.append(_diagnostic(declaration, "missing_schema", "schema reference"))
    if not declaration.examples:
        diagnostics.append(_diagnostic(declaration, "missing_example", "example"))
    if not declaration.outputs:
        diagnostics.append(_diagnostic(declaration, "missing_output", "generated/runtime output"))
    if not declaration.completeness_edges:
        diagnostics.append(_diagnostic(declaration, "missing_consumer_edge", "consumer completeness edge"))
    if not declaration.discovery_text:
        diagnostics.append(_diagnostic(declaration, "missing_discovery", "discovery text"))
    return tuple(diagnostics)


def validate_registry(registry: DeclarationRegistry) -> tuple[Diagnostic, ...]:
    """Return deterministic diagnostics for every declaration in the registry."""

    diagnostics = [diagnostic for item in registry.declarations() for diagnostic in validate_declaration(item)]
    return tuple(sorted(diagnostics, key=lambda item: (item.declaration_id, item.code)))


__all__ = ["Diagnostic", "validate_declaration", "validate_registry"]

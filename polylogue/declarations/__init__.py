"""Typed declaration kernel shared by domain registries."""

from polylogue.declarations.models import (
    CompatibilityKey,
    CompletenessEdge,
    DeclarationSpec,
    ExampleSpec,
    FamilySpec,
    HandlerBinding,
    JSONScalar,
    JSONValue,
    OutputSpec,
)
from polylogue.declarations.registry import DeclarationConflictError, DeclarationRegistry
from polylogue.declarations.validation import Diagnostic, validate_declaration, validate_registry

__all__ = [
    "CompatibilityKey",
    "CompletenessEdge",
    "DeclarationConflictError",
    "DeclarationRegistry",
    "DeclarationSpec",
    "Diagnostic",
    "ExampleSpec",
    "FamilySpec",
    "HandlerBinding",
    "JSONScalar",
    "JSONValue",
    "OutputSpec",
    "validate_declaration",
    "validate_registry",
]

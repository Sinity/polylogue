"""Typed declaration/derivation kernel shared by domain registries."""

from polylogue.declarations.derive import (
    DeclarationDeriver,
    DerivationInput,
    derivation_inputs,
    normalized_derivation_bytes,
)
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
    "DeclarationDeriver",
    "DeclarationRegistry",
    "DeclarationSpec",
    "DerivationInput",
    "Diagnostic",
    "ExampleSpec",
    "FamilySpec",
    "HandlerBinding",
    "JSONScalar",
    "JSONValue",
    "OutputSpec",
    "derivation_inputs",
    "normalized_derivation_bytes",
    "validate_declaration",
    "validate_registry",
]

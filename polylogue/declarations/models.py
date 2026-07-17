"""Storage-free declaration authoring primitives.

The shared kernel deliberately knows nothing about MCP, origins, queries,
storage, maintenance, or any other domain registry.  Domain packages wrap
these immutable records with their own semantic fields and dispatch rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = (
    JSONScalar
    | list["JSONValue"]
    | dict[str, "JSONValue"]
    | tuple["JSONValue", ...]
    | tuple[tuple[str, "JSONValue"], ...]
)


@dataclass(frozen=True, slots=True)
class CompatibilityKey:
    """Dimensions that must match before declarations share a family."""

    identity: str
    lifecycle: str
    authority: str
    access_result_shape: str
    durability: str

    def differences(self, other: CompatibilityKey) -> tuple[tuple[str, str, str], ...]:
        """Return every incompatible dimension in stable field order."""

        fields = ("identity", "lifecycle", "authority", "access_result_shape", "durability")
        return tuple(
            (field, str(getattr(self, field)), str(getattr(other, field)))
            for field in fields
            if getattr(self, field) != getattr(other, field)
        )


@dataclass(frozen=True, slots=True)
class HandlerBinding:
    """One executable handler expected to realize a declaration."""

    surface: str
    owner_path: str
    symbol: str
    binding_key: str


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """One generated or runtime output owned by a declaration."""

    name: str
    kind: str
    schema_ref: str
    target_path: str


@dataclass(frozen=True, slots=True)
class ExampleSpec:
    """One deterministic example invocation or discovery example."""

    name: str
    summary: str
    arguments: tuple[tuple[str, JSONValue], ...] = ()


@dataclass(frozen=True, slots=True)
class CompletenessEdge:
    """A producer/consumer edge whose absence makes a declaration incomplete."""

    producer: str
    consumer: str
    kind: str
    owner_path: str


@dataclass(frozen=True, slots=True)
class DeclarationSpec:
    """Shared declaration contract consumed by domain-specific registries."""

    declaration_id: str
    family_id: str
    public_name: str
    owner_path: str
    compatibility: CompatibilityKey
    producer: str
    role_gate: str
    schema_ref: str
    discovery_text: str
    repair_command: str
    handlers: tuple[HandlerBinding, ...]
    outputs: tuple[OutputSpec, ...]
    examples: tuple[ExampleSpec, ...]
    completeness_edges: tuple[CompletenessEdge, ...]


@dataclass(frozen=True, slots=True)
class FamilySpec:
    """Normalized family projection built by :class:`DeclarationRegistry`."""

    family_id: str
    compatibility: CompatibilityKey
    declaration_ids: tuple[str, ...]
    owner_paths: tuple[str, ...]


__all__ = [
    "CompatibilityKey",
    "CompletenessEdge",
    "DeclarationSpec",
    "ExampleSpec",
    "FamilySpec",
    "HandlerBinding",
    "JSONScalar",
    "JSONValue",
    "OutputSpec",
]

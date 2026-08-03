"""Stable derivation inputs for declaration-owned generated artifacts.

Each declaration carries every metadata field a domain surface needs, but a
renderer for names, contracts, schema/doc fragments, examples, discovery
text, or completeness projections should not have to know the full
:class:`~polylogue.declarations.models.DeclarationSpec` shape -- it should
receive a narrow, typed, source-provenanced slice. This module derives those
slices deterministically (registration order never affects output) and
exposes one :class:`~typing.Protocol` per artifact kind so a domain renderer
can declare exactly which slice it consumes. Nothing here renders a file;
that remains a domain-surface responsibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Protocol

from polylogue.declarations.models import CompletenessEdge, DeclarationSpec, ExampleSpec, HandlerBinding, OutputSpec
from polylogue.declarations.registry import DeclarationRegistry


def _deterministic_bytes(payload: list[dict[str, object]]) -> bytes:
    """Serialize a list of already-normalized projections deterministically."""

    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


@dataclass(frozen=True, slots=True)
class DerivationInput:
    """Normalized full-declaration projection handed to a generic renderer."""

    declaration_id: str
    public_name: str
    owner_path: str
    normalized: dict[str, object]


@dataclass(frozen=True, slots=True)
class NameInput:
    """Stable name/identity projection: family and public-name derivation."""

    declaration_id: str
    family_id: str
    public_name: str
    owner_path: str


@dataclass(frozen=True, slots=True)
class ContractInput:
    """Stable contract projection: producer, role gate, and handler bindings."""

    declaration_id: str
    producer: str
    role_gate: str
    handlers: tuple[HandlerBinding, ...]
    owner_path: str


@dataclass(frozen=True, slots=True)
class SchemaDocInput:
    """Stable schema/doc-fragment projection: schema ref and generated outputs."""

    declaration_id: str
    schema_ref: str
    outputs: tuple[OutputSpec, ...]
    owner_path: str


@dataclass(frozen=True, slots=True)
class ExampleInput:
    """Stable example projection: deterministic invocation/discovery examples."""

    declaration_id: str
    examples: tuple[ExampleSpec, ...]
    owner_path: str


@dataclass(frozen=True, slots=True)
class DiscoveryInput:
    """Stable discovery-text projection."""

    declaration_id: str
    public_name: str
    discovery_text: str
    owner_path: str


@dataclass(frozen=True, slots=True)
class CompletenessInput:
    """Stable completeness projection: producer/consumer edges owned here."""

    declaration_id: str
    completeness_edges: tuple[CompletenessEdge, ...]
    owner_path: str


class DeclarationDeriver(Protocol):
    """Protocol implemented by a full-declaration domain-specific derivation."""

    def derive(self, declarations: tuple[DerivationInput, ...]) -> bytes: ...


class NameDeriver(Protocol):
    """Protocol implemented by a name/identity domain-specific derivation."""

    def derive(self, names: tuple[NameInput, ...]) -> bytes: ...


class ContractDeriver(Protocol):
    """Protocol implemented by a contract domain-specific derivation."""

    def derive(self, contracts: tuple[ContractInput, ...]) -> bytes: ...


class SchemaDocDeriver(Protocol):
    """Protocol implemented by a schema/doc-fragment domain-specific derivation."""

    def derive(self, schema_docs: tuple[SchemaDocInput, ...]) -> bytes: ...


class ExampleDeriver(Protocol):
    """Protocol implemented by an example domain-specific derivation."""

    def derive(self, examples: tuple[ExampleInput, ...]) -> bytes: ...


class DiscoveryDeriver(Protocol):
    """Protocol implemented by a discovery-text domain-specific derivation."""

    def derive(self, discovery: tuple[DiscoveryInput, ...]) -> bytes: ...


class CompletenessDeriver(Protocol):
    """Protocol implemented by a completeness-projection domain-specific derivation."""

    def derive(self, completeness: tuple[CompletenessInput, ...]) -> bytes: ...


def _normalized(declaration: DeclarationSpec) -> dict[str, object]:
    return asdict(declaration)


def derivation_inputs(registry: DeclarationRegistry) -> tuple[DerivationInput, ...]:
    """Return stable typed full-declaration inputs, independent of registration order."""

    return tuple(
        DerivationInput(
            declaration_id=declaration.declaration_id,
            public_name=declaration.public_name,
            owner_path=declaration.owner_path,
            normalized=_normalized(declaration),
        )
        for declaration in registry.declarations()
    )


def name_inputs(registry: DeclarationRegistry) -> tuple[NameInput, ...]:
    """Return stable typed name/identity inputs plus source provenance."""

    return tuple(
        NameInput(
            declaration_id=declaration.declaration_id,
            family_id=declaration.family_id,
            public_name=declaration.public_name,
            owner_path=declaration.owner_path,
        )
        for declaration in registry.declarations()
    )


def contract_inputs(registry: DeclarationRegistry) -> tuple[ContractInput, ...]:
    """Return stable typed contract inputs plus source provenance."""

    return tuple(
        ContractInput(
            declaration_id=declaration.declaration_id,
            producer=declaration.producer,
            role_gate=declaration.role_gate,
            handlers=declaration.handlers,
            owner_path=declaration.owner_path,
        )
        for declaration in registry.declarations()
    )


def schema_doc_inputs(registry: DeclarationRegistry) -> tuple[SchemaDocInput, ...]:
    """Return stable typed schema/doc-fragment inputs plus source provenance."""

    return tuple(
        SchemaDocInput(
            declaration_id=declaration.declaration_id,
            schema_ref=declaration.schema_ref,
            outputs=declaration.outputs,
            owner_path=declaration.owner_path,
        )
        for declaration in registry.declarations()
    )


def example_inputs(registry: DeclarationRegistry) -> tuple[ExampleInput, ...]:
    """Return stable typed example inputs plus source provenance."""

    return tuple(
        ExampleInput(
            declaration_id=declaration.declaration_id,
            examples=declaration.examples,
            owner_path=declaration.owner_path,
        )
        for declaration in registry.declarations()
    )


def discovery_inputs(registry: DeclarationRegistry) -> tuple[DiscoveryInput, ...]:
    """Return stable typed discovery-text inputs plus source provenance."""

    return tuple(
        DiscoveryInput(
            declaration_id=declaration.declaration_id,
            public_name=declaration.public_name,
            discovery_text=declaration.discovery_text,
            owner_path=declaration.owner_path,
        )
        for declaration in registry.declarations()
    )


def completeness_inputs(registry: DeclarationRegistry) -> tuple[CompletenessInput, ...]:
    """Return stable typed completeness-projection inputs plus source provenance."""

    return tuple(
        CompletenessInput(
            declaration_id=declaration.declaration_id,
            completeness_edges=declaration.completeness_edges,
            owner_path=declaration.owner_path,
        )
        for declaration in registry.declarations()
    )


def normalized_derivation_bytes(registry: DeclarationRegistry) -> bytes:
    """Serialize full-declaration derivation inputs deterministically for drift checks."""

    payload = [item.normalized for item in derivation_inputs(registry)]
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def normalized_name_bytes(registry: DeclarationRegistry) -> bytes:
    """Serialize name/identity inputs deterministically for drift checks."""

    return _deterministic_bytes([asdict(item) for item in name_inputs(registry)])


def normalized_contract_bytes(registry: DeclarationRegistry) -> bytes:
    """Serialize contract inputs deterministically for drift checks."""

    return _deterministic_bytes([asdict(item) for item in contract_inputs(registry)])


def normalized_schema_doc_bytes(registry: DeclarationRegistry) -> bytes:
    """Serialize schema/doc-fragment inputs deterministically for drift checks."""

    return _deterministic_bytes([asdict(item) for item in schema_doc_inputs(registry)])


def normalized_example_bytes(registry: DeclarationRegistry) -> bytes:
    """Serialize example inputs deterministically for drift checks."""

    return _deterministic_bytes([asdict(item) for item in example_inputs(registry)])


def normalized_discovery_bytes(registry: DeclarationRegistry) -> bytes:
    """Serialize discovery-text inputs deterministically for drift checks."""

    return _deterministic_bytes([asdict(item) for item in discovery_inputs(registry)])


def normalized_completeness_bytes(registry: DeclarationRegistry) -> bytes:
    """Serialize completeness-projection inputs deterministically for drift checks."""

    return _deterministic_bytes([asdict(item) for item in completeness_inputs(registry)])


__all__ = [
    "CompletenessDeriver",
    "CompletenessInput",
    "ContractDeriver",
    "ContractInput",
    "DeclarationDeriver",
    "DerivationInput",
    "DiscoveryDeriver",
    "DiscoveryInput",
    "ExampleDeriver",
    "ExampleInput",
    "NameDeriver",
    "NameInput",
    "SchemaDocDeriver",
    "SchemaDocInput",
    "completeness_inputs",
    "contract_inputs",
    "derivation_inputs",
    "discovery_inputs",
    "example_inputs",
    "name_inputs",
    "normalized_completeness_bytes",
    "normalized_contract_bytes",
    "normalized_derivation_bytes",
    "normalized_discovery_bytes",
    "normalized_example_bytes",
    "normalized_name_bytes",
    "normalized_schema_doc_bytes",
    "schema_doc_inputs",
]

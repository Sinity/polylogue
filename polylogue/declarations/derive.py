"""Stable derivation inputs for declaration-owned generated artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Protocol

from polylogue.declarations.models import DeclarationSpec
from polylogue.declarations.registry import DeclarationRegistry


@dataclass(frozen=True, slots=True)
class DerivationInput:
    """Normalized declaration input handed to a domain renderer."""

    declaration_id: str
    public_name: str
    owner_path: str
    normalized: dict[str, object]


class DeclarationDeriver(Protocol):
    """Protocol implemented by domain-specific artifact derivations."""

    def derive(self, declarations: tuple[DerivationInput, ...]) -> bytes: ...


def _normalized(declaration: DeclarationSpec) -> dict[str, object]:
    return asdict(declaration)


def derivation_inputs(registry: DeclarationRegistry) -> tuple[DerivationInput, ...]:
    """Return stable typed inputs for names, contracts, docs, and completeness."""

    return tuple(
        DerivationInput(
            declaration_id=declaration.declaration_id,
            public_name=declaration.public_name,
            owner_path=declaration.owner_path,
            normalized=_normalized(declaration),
        )
        for declaration in registry.declarations()
    )


def normalized_derivation_bytes(registry: DeclarationRegistry) -> bytes:
    """Serialize derivation inputs deterministically for drift checks."""

    payload = [item.normalized for item in derivation_inputs(registry)]
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


__all__ = ["DeclarationDeriver", "DerivationInput", "derivation_inputs", "normalized_derivation_bytes"]

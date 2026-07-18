"""Storage-neutral identity values for rebuildable derivations.

A :class:`DerivationKey` says *what result would be current*.  It deliberately
excludes attempt generation, scheduler/producer identity, eligibility and
privacy policy, lifecycle state, cost, and result-integrity hashes.  Domains
such as embeddings and FTS own their own ledgers and transitions while sharing
this value shape.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

DerivationAtom: TypeAlias = str | int | float | bool | bytes | None

_KEY_DOMAIN = b"polylogue.derivation-key.v1\x00"
_SUBJECT_DOMAIN = b"polylogue.derivation-subject.v1\x00"
_IDENTITY_DOMAIN = b"polylogue.derivation-identity.v1\x00"


def _digest(domain: bytes, payload: bytes) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(domain)
    hasher.update(payload)
    return hasher.digest()


def _canonical_atom(value: DerivationAtom) -> object:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("derivation identity floats must be finite")
    return value


@dataclass(frozen=True, slots=True)
class DerivationSubject:
    """Logical subject reference plus the grain of the derived output."""

    reference: str
    grain: str

    def __post_init__(self) -> None:
        if not self.reference:
            raise ValueError("derivation subject reference must not be empty")
        if not self.grain:
            raise ValueError("derivation subject grain must not be empty")

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            {"reference": self.reference, "grain": self.grain},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def digest(self) -> bytes:
        return _digest(_SUBJECT_DOMAIN, self.canonical_bytes())


@dataclass(frozen=True, slots=True)
class DerivationIdentity:
    """One namespaced, canonically ordered set of computational identity fields."""

    namespace: str
    fields: tuple[tuple[str, DerivationAtom], ...]

    def __post_init__(self) -> None:
        if not self.namespace:
            raise ValueError("derivation identity namespace must not be empty")
        names = tuple(name for name, _ in self.fields)
        if any(not name for name in names):
            raise ValueError("derivation identity field names must not be empty")
        if len(names) != len(set(names)):
            raise ValueError("derivation identity field names must be unique")
        if names != tuple(sorted(names)):
            raise ValueError("derivation identity fields must be sorted by name")
        for _, value in self.fields:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("derivation identity floats must be finite")

    @classmethod
    def from_mapping(cls, namespace: str, fields: Mapping[str, DerivationAtom]) -> DerivationIdentity:
        """Freeze a mapping into canonical key order."""

        return cls(namespace=namespace, fields=tuple(sorted(fields.items())))

    def canonical_bytes(self) -> bytes:
        payload = {
            "namespace": self.namespace,
            "fields": [[name, _canonical_atom(value)] for name, value in self.fields],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def digest(self) -> bytes:
        return _digest(_IDENTITY_DOMAIN, self.canonical_bytes())

    def field(self, name: str) -> DerivationAtom:
        for field_name, value in self.fields:
            if field_name == name:
                return value
        raise KeyError(name)


@runtime_checkable
class DerivationKeyLike(Protocol):
    """Structural protocol reusable without adopting another domain's ledger."""

    subject: DerivationSubject
    source_identity: DerivationIdentity
    recipe_identity: DerivationIdentity
    output_contract: DerivationIdentity

    @abstractmethod
    def digest(self) -> bytes:
        """Return the canonical SHA-256 derivation-key digest."""


def compose_derivation_key_digest(
    *,
    subject_digest: bytes,
    source_identity_digest: bytes,
    recipe_identity_digest: bytes,
    output_contract_digest: bytes,
) -> bytes:
    """Compose a key digest from the four independently typed components."""

    components = (
        subject_digest,
        source_identity_digest,
        recipe_identity_digest,
        output_contract_digest,
    )
    if any(len(component) != hashlib.sha256().digest_size for component in components):
        raise ValueError("derivation key component digests must be SHA-256 values")
    return _digest(_KEY_DOMAIN, b"".join(components))


@dataclass(frozen=True, slots=True)
class DerivationKey:
    """Exact source + recipe + output identity for one logical subject/grain."""

    subject: DerivationSubject
    source_identity: DerivationIdentity
    recipe_identity: DerivationIdentity
    output_contract: DerivationIdentity

    def digest(self) -> bytes:
        return compose_derivation_key_digest(
            subject_digest=self.subject.digest(),
            source_identity_digest=self.source_identity.digest(),
            recipe_identity_digest=self.recipe_identity.digest(),
            output_contract_digest=self.output_contract.digest(),
        )

    def canonical_bytes(self) -> bytes:
        payload = {
            "subject": json.loads(self.subject.canonical_bytes()),
            "source_identity": json.loads(self.source_identity.canonical_bytes()),
            "recipe_identity": json.loads(self.recipe_identity.canonical_bytes()),
            "output_contract": json.loads(self.output_contract.canonical_bytes()),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


__all__ = [
    "DerivationAtom",
    "DerivationIdentity",
    "DerivationKey",
    "DerivationKeyLike",
    "DerivationSubject",
    "compose_derivation_key_digest",
]

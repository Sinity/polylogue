"""Scope contracts for terminal query actions.

The request and result fingerprints deliberately accept different evidence
types.  This prevents a result from certifying itself by reusing the lowered
request object.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


class ScopeMismatchError(RuntimeError):
    """The executor applied a different scope from the lowered request."""


PredicateLowerer = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class SurfaceSpec:
    """Declaration of the scope contract consumed by one terminal action."""

    action: str
    unit: str
    supports_predicate_pushdown: bool
    predicate_lowerer: PredicateLowerer | None = None

    def __post_init__(self) -> None:
        if self.supports_predicate_pushdown and self.predicate_lowerer is None:
            raise ValueError(f"{self.action} declares predicate pushdown without a lowerer")


def _fingerprint(evidence: Mapping[str, Any]) -> str:
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "scope:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def request_scope_fingerprint(lowered_predicate: Mapping[str, Any]) -> str:
    """Fingerprint the immutable predicate produced by request lowering."""

    return _fingerprint({"kind": "lowered-predicate", "predicate": dict(lowered_predicate)})


def result_scope_fingerprint(applied_scope: Mapping[str, Any]) -> str:
    """Fingerprint executor evidence read from the applied query scope."""

    return _fingerprint({"kind": "applied-scope", "scope": dict(applied_scope)})


def assert_scope_match(requested: str, applied: str) -> None:
    if requested != applied:
        raise ScopeMismatchError(f"query scope changed during execution: requested={requested} applied={applied}")


def validate_surface_specs(specs: tuple[SurfaceSpec, ...]) -> None:
    """Validate the complete action declaration set at registration time."""

    actions = [spec.action for spec in specs]
    if len(actions) != len(set(actions)):
        raise ValueError("duplicate terminal action declaration")
    for spec in specs:
        if spec.supports_predicate_pushdown and spec.predicate_lowerer is None:
            raise ValueError(f"{spec.action} declares predicate pushdown without a lowerer")


__all__ = [
    "ScopeMismatchError",
    "SurfaceSpec",
    "assert_scope_match",
    "request_scope_fingerprint",
    "result_scope_fingerprint",
    "validate_surface_specs",
]

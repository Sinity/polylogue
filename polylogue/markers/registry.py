"""Declare-once marker kind registry and lowering completeness checks."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from polylogue.core.enums import AssertionKind
from polylogue.markers.models import MarkerKindSpec


class MarkerRegistry:
    def __init__(self, specs: Iterable[MarkerKindSpec] = ()) -> None:
        self._specs: dict[str, MarkerKindSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: MarkerKindSpec) -> MarkerKindSpec:
        if not spec.kind or spec.kind != spec.kind.lower():
            raise ValueError("marker kind must be a non-empty lowercase token")
        if spec.kind in self._specs:
            raise ValueError(f"duplicate marker declaration: {spec.kind}")
        if spec.lowering_target is None:
            raise ValueError(
                f"marker {spec.kind!r} has no lowering owner; add lowering_target to polylogue/markers/registry.py"
            )
        self._specs[spec.kind] = spec
        return spec

    def get(self, kind: str) -> MarkerKindSpec | None:
        return self._specs.get(kind)

    def __contains__(self, kind: str) -> bool:
        return kind in self._specs

    def __iter__(self) -> Iterator[MarkerKindSpec]:
        yield from (self._specs[key] for key in sorted(self._specs))

    def validate(self) -> None:
        missing = [spec.kind for spec in self if spec.lowering_target is None]
        if missing:
            raise ValueError(f"unregistered/ownerless marker lowerings: {', '.join(missing)}")


def marker_spec(registry: MarkerRegistry, kind: str) -> MarkerKindSpec | None:
    """Resolve a kind through the declaration registry."""
    return registry.get(kind)


# The syntax is intentionally small.  All representative kinds use the same
# parser path; only this declaration table selects their typed owner.
MARKER_REGISTRY = MarkerRegistry(
    (
        MarkerKindSpec("goal", "text", AssertionKind.NOTE, "session goal"),
        MarkerKindSpec("assertion", "text", AssertionKind.NOTE, "assertion candidate"),
        MarkerKindSpec("decision", "text", AssertionKind.DECISION, "decision candidate"),
        MarkerKindSpec("event", "text", AssertionKind.RUN_STATE, "event candidate"),
        MarkerKindSpec("finding", "text", AssertionKind.FINDING, "finding candidate"),
        MarkerKindSpec("handoff", "text", AssertionKind.HANDOFF, "handoff candidate"),
        MarkerKindSpec("policy", "text", AssertionKind.DECISION, "policy candidate"),
        MarkerKindSpec("note", "text", AssertionKind.NOTE, "note candidate"),
    )
)
MARKER_REGISTRY.validate()
marker_registry = MARKER_REGISTRY

__all__ = ["MARKER_REGISTRY", "MarkerRegistry", "marker_registry"]

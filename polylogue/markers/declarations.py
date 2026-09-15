"""Shared-kernel projection of the marker authoring family.

:mod:`polylogue.markers.registry` already declares each authoring kind once,
but that declaration was invisible to the shared kernel: nothing resolved a
marker's lowering owner, its production handlers, or its authoring examples
until a rebuild hit them.  This module projects every registered
:class:`~polylogue.markers.models.MarkerKindSpec` into a
:class:`~polylogue.declarations.DeclarationSpec` so ``devtools gate
declaration-bindings`` resolves the family against the live checkout.

The projection deliberately lives *beside* the registry rather than inside it.
``polylogue/markers/registry.py`` and ``polylogue/markers/models.py`` are both
inside the derived-schema identity closure -- the session materializer reaches
them through ``polylogue.markers.scan_block`` -- so declaring the kernel there
would move the archive's schema identity for a developer-experience surface
and force a full reconvergence.  Nothing in the closure imports this module.

The declaration table is *derived*, never transcribed: adding a
``MarkerKindSpec`` to ``MARKER_REGISTRY`` adds its kernel record here with no
second edit, and removing one removes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from polylogue.declarations import (
    CompatibilityKey,
    CompletenessEdge,
    DeclarationRegistry,
    DeclarationSpec,
    ExampleSpec,
    HandlerBinding,
    OutputSpec,
    validate_registry,
)
from polylogue.markers.models import MarkerKindSpec
from polylogue.markers.registry import MARKER_REGISTRY, MarkerRegistry

REPAIR_COMMAND: Final = "devtools test tests/unit/markers tests/unit/declarations"

#: The single production route that turns a parsed marker into a durable row.
MARKER_PRODUCER: Final = "polylogue.markers.lowering.lower_markers"

#: The derived-enrichment consumer that scans blocks during materialization.
MARKER_CONSUMER: Final = "polylogue.storage.derived.session.rebuild.scan_block"

_LINE_SIGIL_TEMPLATE: Final = "::{kind}: {body}"
_INLINE_SIGIL_TEMPLATE: Final = "[[{kind}: {body}]]"


def line_marker_example(kind: str, body: str) -> str:
    """Render one line-anchored marker in the production grammar."""

    return _LINE_SIGIL_TEMPLATE.format(kind=kind, body=body)


def inline_marker_example(kind: str, body: str) -> str:
    """Render one inline marker in the production grammar."""

    return _INLINE_SIGIL_TEMPLATE.format(kind=kind, body=body)


@dataclass(frozen=True, slots=True)
class MarkerDeclaration:
    """One marker kind and its shared-kernel record.

    The domain spec keeps its own semantics (payload shape, lowering owner,
    authority); the kernel carries only what the shared completeness graph
    needs.  Neither is derived from the other's storage.
    """

    kernel: DeclarationSpec
    spec: MarkerKindSpec

    @property
    def kind(self) -> str:
        return self.spec.kind


def _compatibility(spec: MarkerKindSpec) -> CompatibilityKey:
    return CompatibilityKey(
        identity="marker-kind",
        lifecycle="declared",
        authority=spec.authority,
        access_result_shape="assertion-candidate",
        durability="durable-user-assertion",
    )


def _kernel(spec: MarkerKindSpec) -> DeclarationSpec:
    lowering_target = spec.lowering_target
    if lowering_target is None:  # pragma: no cover - MarkerRegistry.register refuses this
        raise ValueError(f"marker {spec.kind!r} has no lowering owner")
    return DeclarationSpec(
        declaration_id=f"marker.kind.{spec.kind}",
        family_id="marker.kind",
        public_name=f"marker:{spec.kind}",
        owner_path="polylogue/markers/registry.py",
        compatibility=_compatibility(spec),
        producer=MARKER_PRODUCER,
        role_gate="assertion.author_kind:agent",
        schema_ref=f"polylogue.core.enums.AssertionKind.{lowering_target.name}",
        discovery_text=spec.description,
        repair_command=REPAIR_COMMAND,
        handlers=(
            HandlerBinding(
                surface="marker-parser",
                owner_path="polylogue/markers/parser.py",
                symbol="parse_markers",
                binding_key=f"marker:{spec.kind}",
            ),
            HandlerBinding(
                surface="marker-lowering",
                owner_path="polylogue/markers/lowering.py",
                symbol="lower_markers",
                binding_key=f"assertion-kind:{lowering_target.value}",
            ),
            HandlerBinding(
                surface="user-write",
                owner_path="polylogue/storage/sqlite/archive_tiers/user_write.py",
                symbol="upsert_assertion",
                binding_key=f"assertions:{lowering_target.value}",
            ),
        ),
        outputs=(
            OutputSpec(
                name="assertion",
                kind="user-assertion-row",
                schema_ref=f"assertions.kind={lowering_target.value}",
                target_path="user.db:assertions",
            ),
        ),
        # Both forms of the live grammar, rendered from the declared kind.
        # ``tests/unit/markers/test_marker_declarations.py`` parses every one
        # of these through the production parser and asserts it lowers to the
        # declared assertion kind, so an example can never drift into prose
        # the parser would reject.
        examples=(
            ExampleSpec(
                name="line",
                summary=f"Line-anchored {spec.kind} marker ({spec.description}).",
                arguments=(("text", line_marker_example(spec.kind, spec.description)),),
            ),
            ExampleSpec(
                name="inline",
                summary=f"Inline {spec.kind} marker ({spec.description}).",
                arguments=(("text", inline_marker_example(spec.kind, spec.description)),),
            ),
        ),
        completeness_edges=(
            CompletenessEdge(
                producer=MARKER_PRODUCER,
                consumer=MARKER_CONSUMER,
                kind="derived-block-enrichment",
                owner_path="polylogue/storage/derived/session/rebuild.py",
            ),
        ),
    )


def marker_declarations(registry: MarkerRegistry = MARKER_REGISTRY) -> tuple[MarkerDeclaration, ...]:
    """Project every registered marker kind, in the registry's stable order."""

    return tuple(MarkerDeclaration(kernel=_kernel(spec), spec=spec) for spec in registry)


def build_marker_kernel_registry(registry: MarkerRegistry = MARKER_REGISTRY) -> DeclarationRegistry:
    """Register every projected marker kernel into a fresh kernel registry."""

    kernel_registry = DeclarationRegistry()
    for declaration in marker_declarations(registry):
        kernel_registry.register(declaration.kernel)
    return kernel_registry


MARKER_DECLARATIONS: Final[tuple[MarkerDeclaration, ...]] = marker_declarations()
MARKER_DECLARATION_BY_KIND: Final[dict[str, MarkerDeclaration]] = {
    declaration.kind: declaration for declaration in MARKER_DECLARATIONS
}
MARKER_KERNEL_REGISTRY: Final[DeclarationRegistry] = build_marker_kernel_registry()

_DIAGNOSTICS = validate_registry(MARKER_KERNEL_REGISTRY)
if _DIAGNOSTICS:  # pragma: no cover - a structurally incomplete projection must not import
    raise RuntimeError("incomplete marker declaration registry: " + "; ".join(item.message for item in _DIAGNOSTICS))


def declaration_for_marker(kind: str) -> MarkerDeclaration:
    """Resolve one marker declaration or name the exact repair."""

    try:
        return MARKER_DECLARATION_BY_KIND[kind]
    except KeyError as exc:
        raise KeyError(
            f"marker kind {kind!r} has no declaration; add its MarkerKindSpec to MARKER_REGISTRY in "
            f"polylogue/markers/registry.py and run {REPAIR_COMMAND}"
        ) from exc


__all__ = [
    "MARKER_CONSUMER",
    "MARKER_DECLARATIONS",
    "MARKER_DECLARATION_BY_KIND",
    "MARKER_KERNEL_REGISTRY",
    "MARKER_PRODUCER",
    "REPAIR_COMMAND",
    "MarkerDeclaration",
    "build_marker_kernel_registry",
    "declaration_for_marker",
    "inline_marker_example",
    "line_marker_example",
    "marker_declarations",
]

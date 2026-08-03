"""Synthetic domain proof for the shared DeclarationSpec kernel.

Simulates a fake domain family the way ``polylogue/mcp/declarations`` (the
real t46.8.1 pilot) does: two compatible declarations that legitimately share
a family, one declaration that differs on each of the five compatibility
dimensions in isolation (so the diagnostic can be checked names exactly that
dimension and no other), deterministic per-artifact-kind derivation across
shuffled registration order, and typed-deriver usage for every artifact kind
the kernel promises (names, contracts, schema/doc fragments, examples,
discovery text, completeness).
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from polylogue.declarations import (
    CompatibilityKey,
    CompletenessEdge,
    DeclarationConflictError,
    DeclarationRegistry,
    DeclarationSpec,
    ExampleSpec,
    HandlerBinding,
    OutputSpec,
    normalized_completeness_bytes,
    normalized_contract_bytes,
    normalized_derivation_bytes,
    normalized_discovery_bytes,
    normalized_example_bytes,
    normalized_name_bytes,
    normalized_schema_doc_bytes,
)

_BASE_COMPATIBILITY = CompatibilityKey(
    identity="synthetic-record",
    lifecycle="executable",
    authority="read",
    access_result_shape="exhaustive-page",
    durability="rebuildable",
)

_DIMENSION_OVERRIDES: dict[str, str] = {
    "identity": "synthetic-record-v2",
    "lifecycle": "reserved",
    "authority": "write",
    "access_result_shape": "single-object",
    "durability": "durable",
}


def _fake_domain_declaration(
    name: str,
    *,
    family: str,
    compatibility: CompatibilityKey = _BASE_COMPATIBILITY,
) -> DeclarationSpec:
    """Build one declaration as the fake ``widget`` domain family would."""

    return DeclarationSpec(
        declaration_id=f"widget.{name}",
        family_id=family,
        public_name=f"widget-{name}",
        owner_path=f"tests/fixtures/widgets/{name}.py",
        compatibility=compatibility,
        producer=f"widget.operations.{name}",
        role_gate="widget.role:read",
        schema_ref=f"widget.schema.{name}",
        discovery_text=f"Discover the {name} widget.",
        repair_command=f"devtools render widget-contract --name {name}",
        handlers=(HandlerBinding("widget-cli", f"tests/fixtures/widgets/{name}.py", name, f"widget:{name}"),),
        outputs=(OutputSpec("docs", "markdown-fragment", f"widget.schema.{name}", f"docs/widgets/{name}.md"),),
        examples=(ExampleSpec("minimal", f"Minimal {name} widget invocation.", (("name", name),)),),
        completeness_edges=(
            CompletenessEdge(f"widget.{name}", "widget.consumer.cli", "coverage", "tests/fixtures/widgets/cli.py"),
        ),
    )


def test_two_compatible_widget_declarations_share_one_family() -> None:
    """Production dependency: DeclarationRegistry family unification for a real domain shape.

    Anti-vacuity mutation: a bug that assigns each declaration a distinct
    family (rather than unifying on ``family_id`` + matching compatibility)
    would make ``families()`` report two single-member families instead of
    one two-member family.
    """

    registry = DeclarationRegistry()
    registry.register(_fake_domain_declaration("gauge", family="widget"))
    registry.register(_fake_domain_declaration("dial", family="widget"))

    families = registry.families()
    assert len(families) == 1
    assert families[0].family_id == "widget"
    assert families[0].declaration_ids == ("widget.dial", "widget.gauge")


@pytest.mark.parametrize("dimension", sorted(_DIMENSION_OVERRIDES))
def test_family_rejects_declaration_differing_on_one_dimension(dimension: str) -> None:
    """Production dependency: five-dimension compatibility-key enforcement, isolated per axis.

    Anti-vacuity mutation: a compatibility check that only compares a subset
    of the five dimensions (e.g. drops ``durability``) would let this
    single-axis-differing declaration join the family silently, or the
    reported diagnostic would omit the varied dimension name.
    """

    registry = DeclarationRegistry()
    registry.register(_fake_domain_declaration("gauge", family="widget"))

    overrides = {dimension: _DIMENSION_OVERRIDES[dimension]}
    incompatible_key = replace(_BASE_COMPATIBILITY, **overrides)
    incompatible = _fake_domain_declaration("rogue", family="widget", compatibility=incompatible_key)

    with pytest.raises(DeclarationConflictError) as caught:
        registry.register(incompatible)

    differing_dimensions = [field for field, _, _ in caught.value.differences]
    assert differing_dimensions == [dimension]
    left, right = caught.value.differences[0][1], caught.value.differences[0][2]
    assert left == str(getattr(_BASE_COMPATIBILITY, dimension))
    assert right == str(overrides[dimension])
    # The registry must never coerce the incompatible declaration in: the
    # family stays exactly as it was before the failed registration attempt.
    assert registry.declarations() == (_fake_domain_declaration("gauge", family="widget"),)


def test_widget_family_derivation_is_order_independent_across_every_artifact_kind() -> None:
    """Production dependency: per-artifact-kind deterministic derivation (o21.1 AC 3).

    Anti-vacuity mutation: any typed projection function (name/contract/
    schema-doc/example/discovery/completeness) that leaks dict/set iteration
    order, or omits sorting by declaration id, would make one of these
    byte comparisons fail under a shuffled registration order.
    """

    declarations = [_fake_domain_declaration(name, family=name) for name in ("gauge", "dial", "lever", "switch")]

    forward = DeclarationRegistry()
    for declaration in declarations:
        forward.register(declaration)

    shuffled = list(declarations)
    random.Random(20260721).shuffle(shuffled)
    reordered = DeclarationRegistry()
    for declaration in shuffled:
        reordered.register(declaration)

    assert normalized_derivation_bytes(forward) == normalized_derivation_bytes(reordered)
    assert normalized_name_bytes(forward) == normalized_name_bytes(reordered)
    assert normalized_contract_bytes(forward) == normalized_contract_bytes(reordered)
    assert normalized_schema_doc_bytes(forward) == normalized_schema_doc_bytes(reordered)
    assert normalized_example_bytes(forward) == normalized_example_bytes(reordered)
    assert normalized_discovery_bytes(forward) == normalized_discovery_bytes(reordered)
    assert normalized_completeness_bytes(forward) == normalized_completeness_bytes(reordered)

    # And the registry's own public ordering is also independent of insertion order.
    assert tuple(item.declaration_id for item in forward.declarations()) == tuple(
        item.declaration_id for item in reordered.declarations()
    )

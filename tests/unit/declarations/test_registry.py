"""Contracts for the storage-free shared declaration kernel."""

from __future__ import annotations

import ast
from pathlib import Path

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
    normalized_derivation_bytes,
    validate_declaration,
)


def _declaration(
    name: str,
    *,
    family: str = "family",
    compatibility: CompatibilityKey | None = None,
) -> DeclarationSpec:
    return DeclarationSpec(
        declaration_id=f"test.{name}",
        family_id=family,
        public_name=name,
        owner_path=f"tests/{name}.py",
        compatibility=compatibility
        or CompatibilityKey(
            identity="stable-id",
            lifecycle="retained",
            authority="read",
            access_result_shape="single-object",
            durability="derived",
        ),
        producer="test.operation",
        role_gate="test.role:read",
        schema_ref="test.schema",
        discovery_text=f"Discover {name}.",
        repair_command="devtools render test-contract",
        handlers=(HandlerBinding("test", f"tests/{name}.py", name, f"test:{name}"),),
        outputs=(OutputSpec("runtime", "single-object", "test.schema", f"test://{name}"),),
        examples=(ExampleSpec("minimal", "Minimal example."),),
        completeness_edges=(CompletenessEdge(f"test.{name}", "test.consumer", "coverage", "tests/test_consumer.py"),),
    )


def test_registry_derivation_is_independent_of_registration_order() -> None:
    """Production dependency: DeclarationRegistry stable ordering and derivation normalization.

    Anti-vacuity mutation: returning insertion order from ``declarations()`` makes
    the normalized byte comparison fail.
    """

    forward = DeclarationRegistry()
    reverse = DeclarationRegistry()
    declarations = (_declaration("alpha", family="alpha"), _declaration("beta", family="beta"))
    for declaration in declarations:
        forward.register(declaration)
    for declaration in reversed(declarations):
        reverse.register(declaration)

    assert normalized_derivation_bytes(forward) == normalized_derivation_bytes(reverse)
    assert tuple(item.declaration_id for item in forward.declarations()) == ("test.alpha", "test.beta")


def test_family_rejects_every_incompatible_semantic_dimension() -> None:
    """Production dependency: compatibility-key enforcement in DeclarationRegistry.

    Anti-vacuity mutation: dropping any field from ``CompatibilityKey.differences``
    removes that field from the asserted diagnostic.
    """

    registry = DeclarationRegistry()
    registry.register(_declaration("first"))
    conflicting = CompatibilityKey(
        identity="other-id",
        lifecycle="deprecated",
        authority="write",
        access_result_shape="page",
        durability="durable",
    )

    with pytest.raises(DeclarationConflictError) as caught:
        registry.register(_declaration("second", compatibility=conflicting))

    assert [field for field, _, _ in caught.value.differences] == [
        "identity",
        "lifecycle",
        "authority",
        "access_result_shape",
        "durability",
    ]
    assert caught.value.owner_path == "tests/second.py"


def test_validation_reports_source_and_exact_repair_command() -> None:
    """Production dependency: declaration completeness diagnostics.

    Anti-vacuity mutation: changing the diagnostic owner or repair command makes
    the source-locatable assertions fail.
    """

    complete = _declaration("complete")
    incomplete = DeclarationSpec(
        declaration_id=complete.declaration_id,
        family_id=complete.family_id,
        public_name=complete.public_name,
        owner_path=complete.owner_path,
        compatibility=complete.compatibility,
        producer="",
        role_gate="",
        schema_ref="",
        discovery_text="",
        repair_command=complete.repair_command,
        handlers=(),
        outputs=(),
        examples=(),
        completeness_edges=(),
    )

    diagnostics = validate_declaration(incomplete)

    assert {item.code for item in diagnostics} == {
        "missing_consumer_edge",
        "missing_discovery",
        "missing_example",
        "missing_handler",
        "missing_output",
        "missing_producer",
        "missing_role_gate",
        "missing_schema",
    }
    assert {item.owner_path for item in diagnostics} == {"tests/complete.py"}
    assert {item.repair_command for item in diagnostics} == {"devtools render test-contract"}


def test_shared_kernel_has_no_domain_or_runtime_imports() -> None:
    """Production dependency: the shared declaration package import boundary.

    Anti-vacuity mutation: importing MCP, storage, operations, or configuration
    into the kernel adds a forbidden import collected by this AST check.
    """

    forbidden = (
        "polylogue.api",
        "polylogue.archive",
        "polylogue.config",
        "polylogue.insights",
        "polylogue.mcp",
        "polylogue.operations",
        "polylogue.storage",
    )
    imported: list[tuple[Path, str]] = []
    for path in sorted(Path("polylogue/declarations").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend((path, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.append((path, node.module))

    violations = [(path, module) for path, module in imported if module.startswith(forbidden)]
    assert violations == []

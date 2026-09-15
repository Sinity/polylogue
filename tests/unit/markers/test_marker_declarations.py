"""The marker family's shared-kernel declarations bind to the production route.

Anti-vacuity for this module: renaming ``lower_markers``/``parse_markers``,
moving ``upsert_assertion``, changing the marker sigil, or pointing a marker's
``lowering_target`` at a different ``AssertionKind`` each turns one of these
tests red. A declaration that merely agrees with another declaration proves
nothing, so every example here is parsed by the production parser and lowered
through the production assertion writer into a real ``user.db``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.declarations import DeclarationRegistry, HandlerBinding
from polylogue.declarations.diagnostics import diagnose_registry
from polylogue.markers import MARKER_REGISTRY, MarkerKindSpec, MarkerRegistry, candidates_for_block, lower_markers
from polylogue.markers.declarations import (
    MARKER_DECLARATIONS,
    MARKER_KERNEL_REGISTRY,
    MarkerDeclaration,
    build_marker_kernel_registry,
    declaration_for_marker,
)
from polylogue.markers.parser import parse_markers

ROOT = Path(__file__).resolve().parents[3]


def _example_text(declaration: MarkerDeclaration, name: str) -> str:
    for example in declaration.kernel.examples:
        if example.name == name:
            return str(dict(example.arguments)["text"])
    raise AssertionError(f"{declaration.kind} declares no {name!r} example")


def test_every_registered_marker_kind_has_exactly_one_declaration() -> None:
    """The kernel table is derived from MARKER_REGISTRY, never transcribed."""

    assert {spec.kind for spec in MARKER_REGISTRY} == {declaration.kind for declaration in MARKER_DECLARATIONS}
    assert len(MARKER_KERNEL_REGISTRY) == len(MARKER_DECLARATIONS)


def test_adding_a_kind_requires_only_the_typed_family_module() -> None:
    """o21.3 AC5: a new member needs its family declaration and nothing else.

    Anti-vacuity: if the projection were a hand-maintained second table, this
    registry would not contain ``synthetic-probe`` and the assertion fails.
    """

    registry = MarkerRegistry(
        (
            MarkerKindSpec("goal", "text", AssertionKind.NOTE, "session goal"),
            MarkerKindSpec("synthetic-probe", "text", AssertionKind.LESSON, "synthetic probe candidate"),
        )
    )
    kernel_registry = build_marker_kernel_registry(registry)
    assert {item.declaration_id for item in kernel_registry.declarations()} == {
        "marker.kind.goal",
        "marker.kind.synthetic-probe",
    }
    probe = kernel_registry.get("marker.kind.synthetic-probe")
    assert probe.schema_ref == "polylogue.core.enums.AssertionKind.LESSON"


def test_declared_examples_parse_through_the_production_parser() -> None:
    """Every declared example is real authoring syntax, not documentation prose."""

    for declaration in MARKER_DECLARATIONS:
        for example in declaration.kernel.examples:
            text = str(dict(example.arguments)["text"])
            matches = parse_markers(text)
            assert len(matches) == 1, f"{declaration.kind}/{example.name}: {text!r} -> {matches!r}"
            assert matches[0].kind == declaration.kind
            assert not matches[0].malformed


def test_declared_lowering_target_is_the_kind_the_production_route_emits() -> None:
    """The declared schema_ref names the AssertionKind the real scan produces."""

    for declaration in MARKER_DECLARATIONS:
        text = _example_text(declaration, "line")
        candidate = candidates_for_block("message-1", "block-2", text)[0]
        assertion_kind = candidate.assertion_kind
        assert assertion_kind is not None
        assert assertion_kind is declaration.spec.lowering_target
        assert declaration.kernel.schema_ref.endswith(f".{assertion_kind.name}")


def test_declared_output_target_is_the_row_the_production_writer_persists(tmp_path: Path) -> None:
    """The declared output crosses the real user.db writer, not a double.

    Anti-vacuity: change ``lower_markers`` to stop writing ``assertions``, or
    point a declaration's output at another table, and this fails.
    """

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    user_db = tmp_path / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    conn = sqlite3.connect(user_db)
    try:
        for declaration in MARKER_DECLARATIONS:
            output = declaration.kernel.outputs[0]
            assert output.target_path == "user.db:assertions"
            text = _example_text(declaration, "line")
            candidates = candidates_for_block("message-1", f"block-{declaration.kind}", text)
            ids = lower_markers(conn, candidates, now_ms=123)
            assert len(ids) == 1
            row = conn.execute(
                "SELECT kind, author_kind, status FROM assertions WHERE assertion_id = ?", ids
            ).fetchone()
            assert row is not None, declaration.kind
            lowering_target = declaration.spec.lowering_target
            assert lowering_target is not None
            assert row[0] == lowering_target.value
            assert output.schema_ref == f"assertions.kind={row[0]}"
            assert row[1] == "agent"
            assert row[2] == AssertionStatus.CANDIDATE.value
    finally:
        conn.close()


def test_live_registry_resolves_every_declared_binding() -> None:
    assert diagnose_registry(MARKER_KERNEL_REGISTRY, root=ROOT) == ()


def test_a_renamed_production_handler_is_an_actionable_diagnostic() -> None:
    """o21.3 AC3: a consumer reference with no producer fails with a repair.

    Anti-vacuity: this mutates only the declared symbol, so if the gate stopped
    resolving handler symbols the diagnostic would disappear.
    """

    broken = replace(
        MARKER_DECLARATIONS[0].kernel,
        handlers=(
            HandlerBinding(
                surface="marker-lowering",
                owner_path="polylogue/markers/lowering.py",
                symbol="lower_markers_renamed_away",
                binding_key="assertion-kind:note",
            ),
        ),
    )
    registry = DeclarationRegistry()
    registry.register(broken)
    diagnostics = diagnose_registry(registry, root=ROOT)
    assert [item.code for item in diagnostics] == ["unresolved_handler_symbol"]
    assert diagnostics[0].repair_command
    assert "lower_markers_renamed_away" in diagnostics[0].message


def test_unknown_kind_names_its_exact_repair() -> None:
    try:
        declaration_for_marker("not-a-marker")
    except KeyError as exc:
        assert "polylogue/markers/registry.py" in str(exc)
        assert "devtools test" in str(exc)
    else:  # pragma: no cover - the lookup must refuse
        raise AssertionError("unknown marker kind resolved")

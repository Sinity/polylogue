"""Production plan contracts for derived index fast-forwards."""

from __future__ import annotations

import json

import pytest

from devtools import verify_schema_upgrade_lane as schema_policy
from polylogue.storage.sqlite import lifecycle
from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.lifecycle import (
    DerivedDeltaClass,
    FastForwardOperation,
    FastForwardOperationKind,
    IndexDeltaDeclaration,
    IndexFastForwardPlan,
    index_delta_declaration_report,
    index_fast_forward_plan,
)

_DEPLOYED_FAST_FORWARD_TARGET = 35


def test_v32_to_current_plan_declares_ma2_as_an_index_only_delta() -> None:
    """Exercise the production declaration used by clone fast-forward selection."""
    plan = index_fast_forward_plan(32, _DEPLOYED_FAST_FORWARD_TARGET)

    assert plan is not None
    ma2 = next(declaration for declaration in plan.declarations if declaration.version == 34)
    assert DerivedDeltaClass.INDEX_ONLY in ma2.classes
    assert ("index", "idx_web_constructs_message") in plan.canonical_objects
    assert any(operation.kind is FastForwardOperationKind.CREATE_INDEX for operation in ma2.operations)


def test_semantic_delta_routes_a_plan_away_from_sql_fast_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parser-dependent delta cannot be mistaken for a clone-only SQL repair."""
    declaration = IndexDeltaDeclaration(
        version=36,
        classes=(DerivedDeltaClass.SEMANTIC_REPARSE,),
        operations=(
            FastForwardOperation(
                name="v36-parser-shape",
                kind=FastForwardOperationKind.REPLACE_TABLE,
                objects=(("table", "sessions"),),
            ),
        ),
    )
    plan = IndexFastForwardPlan(source_version=35, target_version=36, declarations=(declaration,))
    monkeypatch.setattr(lifecycle, "INDEX_DELTA_DECLARATIONS", (declaration,))

    assert plan.requires_semantic_reparse is True
    assert plan.eligible_for_sql_fast_forward is False
    assert lifecycle.index_fast_forward_plan(35, 36) is None


def test_current_index_schema_has_a_complete_delta_declaration() -> None:
    """Exercise the exact declaration report consumed by the schema policy lint."""
    report = index_delta_declaration_report(INDEX_SCHEMA_VERSION)

    assert report["ok"] is True
    assert report["missing_versions"] == ()


def test_plan_orders_declarations_before_validating_contiguity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registry order cannot silently downgrade a complete clone-safe plan."""
    monkeypatch.setattr(lifecycle, "INDEX_DELTA_DECLARATIONS", tuple(reversed(lifecycle.INDEX_DELTA_DECLARATIONS)))

    plan = lifecycle.index_fast_forward_plan(32, _DEPLOYED_FAST_FORWARD_TARGET)

    assert plan is not None
    assert tuple(declaration.version for declaration in plan.declarations) == (33, 34, 35)


def test_nonsemantic_delta_without_operations_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A declared class cannot advance an index version without executable SQL."""
    empty_declaration = IndexDeltaDeclaration(
        version=37,
        classes=(DerivedDeltaClass.INDEX_ONLY,),
    )
    monkeypatch.setattr(
        lifecycle,
        "INDEX_DELTA_DECLARATIONS",
        (*lifecycle.INDEX_DELTA_DECLARATIONS, empty_declaration),
    )

    report = lifecycle.index_delta_declaration_report(37)

    assert report["ok"] is False
    assert report["invalid_versions"] == (37,)
    assert lifecycle.index_fast_forward_plan(32, 37) is None


def test_delta_without_a_declared_class_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operation cannot create a clone route without a delta classification."""
    unclassified_declaration = IndexDeltaDeclaration(
        version=37,
        classes=(),
        operations=(
            FastForwardOperation(
                name="v36-unclassified-index",
                kind=FastForwardOperationKind.CREATE_INDEX,
                objects=(("index", "idx_web_constructs_message"),),
            ),
        ),
    )
    monkeypatch.setattr(
        lifecycle,
        "INDEX_DELTA_DECLARATIONS",
        (*lifecycle.INDEX_DELTA_DECLARATIONS, unclassified_declaration),
    )

    assert lifecycle.index_delta_declaration_report(37)["invalid_versions"] == (37,)
    assert lifecycle.index_fast_forward_plan(32, 37) is None


def test_schema_policy_rejects_an_index_bump_without_a_delta_declaration(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real lab command must fail before an undeclared bump reaches CI."""
    monkeypatch.setattr(schema_policy, "INDEX_SCHEMA_VERSION", INDEX_SCHEMA_VERSION + 1)

    exit_code = schema_policy.main(["--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["index_delta_declarations"]["missing_versions"] == [INDEX_SCHEMA_VERSION + 1]

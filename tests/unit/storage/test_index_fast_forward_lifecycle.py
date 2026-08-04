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

_DEPLOYED_FAST_FORWARD_TARGET = 36


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


def test_v36_origin_check_is_a_clone_safe_constraint_copy_forward() -> None:
    """The v36 Origin widening does not require raw parser semantics."""
    plan = index_fast_forward_plan(35, 36)

    assert plan is not None
    declaration = plan.declarations[0]
    assert declaration.classes == (DerivedDeltaClass.CONSTRAINT_ONLY,)
    assert declaration.operations[0].objects == (("table", "sessions"), ("table", "session_links"))


def test_v37_cache_removal_is_a_clone_safe_declared_delta() -> None:
    plan = index_fast_forward_plan(36, 37)

    assert plan is not None
    declaration = plan.declarations[0]
    assert declaration.classes == (DerivedDeltaClass.CACHE_REMOVAL,)
    assert declaration.operations[0].kind is FastForwardOperationKind.DROP_TABLE
    assert declaration.operations[0].objects == (
        ("table", "session_runs"),
        ("table", "session_observed_events"),
        ("table", "session_context_snapshots"),
    )


def test_v61_pricing_column_drop_is_a_clone_safe_constraint_copy_forward() -> None:
    """polylogue-resk: session_model_usage's pricing-column drop is CONSTRAINT_ONLY."""
    plan = index_fast_forward_plan(60, 61)

    assert plan is not None
    declaration = plan.declarations[0]
    assert declaration.classes == (DerivedDeltaClass.CONSTRAINT_ONLY,)
    assert declaration.operations[0].kind is FastForwardOperationKind.REPLACE_TABLE
    assert declaration.operations[0].objects == (("table", "session_model_usage"),)


def test_v64_fingerprint_stamp_delta_requires_semantic_reparse() -> None:
    declaration = next(d for d in lifecycle.INDEX_DELTA_DECLARATIONS if d.version == 64)

    assert declaration.classes == (DerivedDeltaClass.SEMANTIC_REPARSE,)
    assert lifecycle.index_fast_forward_plan(63, 64) is None


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
    assert tuple(declaration.version for declaration in plan.declarations) == (33, 34, 35, 36)


def test_nonsemantic_delta_without_operations_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A declared class cannot advance an index version without executable SQL."""
    empty_declaration = IndexDeltaDeclaration(
        version=37,
        classes=(DerivedDeltaClass.INDEX_ONLY,),
    )
    # Isolate to declarations at or below 36 (the versions this plan actually
    # needs) rather than the live module tuple: the live tuple keeps growing
    # past 37 as later schema versions get declared, and every one of those
    # would otherwise leak into invalid_versions via the
    # `declaration.version > current_version` check below.
    base_declarations = tuple(
        declaration for declaration in lifecycle.INDEX_DELTA_DECLARATIONS if declaration.version <= 36
    )
    monkeypatch.setattr(
        lifecycle,
        "INDEX_DELTA_DECLARATIONS",
        (*base_declarations, empty_declaration),
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
    # Same isolation rationale as test_nonsemantic_delta_without_operations_is_rejected above.
    base_declarations = tuple(
        declaration for declaration in lifecycle.INDEX_DELTA_DECLARATIONS if declaration.version <= 36
    )
    monkeypatch.setattr(
        lifecycle,
        "INDEX_DELTA_DECLARATIONS",
        (*base_declarations, unclassified_declaration),
    )

    assert lifecycle.index_delta_declaration_report(37)["invalid_versions"] == (37,)
    assert lifecycle.index_fast_forward_plan(32, 37) is None


def test_schema_policy_rejects_an_index_bump_without_a_delta_declaration(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real lab command must fail before an undeclared bump reaches CI."""
    # Isolate to declarations at or below the current (pre-bump) schema
    # version: index_delta_declaration_report's missing_versions accumulates
    # over the whole expected range, so any future currently-undeclared gap
    # elsewhere in the live tuple would otherwise leak into this assertion
    # alongside the version this test deliberately leaves undeclared.
    complete_declarations = tuple(
        declaration for declaration in lifecycle.INDEX_DELTA_DECLARATIONS if declaration.version <= INDEX_SCHEMA_VERSION
    )
    monkeypatch.setattr(lifecycle, "INDEX_DELTA_DECLARATIONS", complete_declarations)
    monkeypatch.setattr(schema_policy, "INDEX_SCHEMA_VERSION", INDEX_SCHEMA_VERSION + 1)

    exit_code = schema_policy.main(["--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["index_delta_declarations"]["missing_versions"] == [INDEX_SCHEMA_VERSION + 1]


def test_targeted_reprocess_scope_requires_a_dimension() -> None:
    """An empty scope would silently mean 'every session' -- refuse it."""
    with pytest.raises(ValueError, match="origin and/or session_ids"):
        lifecycle.TargetedReprocessScope()


def test_targeted_reprocess_scope_builds_the_declared_predicate() -> None:
    origin_only = lifecycle.TargetedReprocessScope(origin="codex-session")
    assert origin_only.matching_predicate_sql() == ("origin = ?", ("codex-session",))

    sessions_only = lifecycle.TargetedReprocessScope(session_ids=("a:1", "a:2"))
    assert sessions_only.matching_predicate_sql() == ("session_id IN (?, ?)", ("a:1", "a:2"))

    both = lifecycle.TargetedReprocessScope(origin="codex-session", session_ids=("a:1",))
    assert both.matching_predicate_sql() == ("origin = ? AND session_id IN (?)", ("codex-session", "a:1"))


def test_declaration_requires_scope_and_class_together() -> None:
    """The class/scope pairing is enforced at construction, not just by the report."""
    with pytest.raises(ValueError, match="must be declared together"):
        IndexDeltaDeclaration(
            version=54,
            classes=(DerivedDeltaClass.SHAPE_FORWARD_TARGETED_REPROCESS,),
            operations=(
                FastForwardOperation(
                    name="synthetic",
                    kind=FastForwardOperationKind.REPLACE_TABLE,
                    objects=(("table", "sessions"),),
                ),
            ),
            # reprocess_scope deliberately omitted.
        )
    with pytest.raises(ValueError, match="must be declared together"):
        IndexDeltaDeclaration(
            version=54,
            classes=(DerivedDeltaClass.CONSTRAINT_ONLY,),
            operations=(
                FastForwardOperation(
                    name="synthetic",
                    kind=FastForwardOperationKind.REPLACE_TABLE,
                    objects=(("table", "sessions"),),
                ),
            ),
            reprocess_scope=lifecycle.TargetedReprocessScope(origin="codex-session"),
        )


def test_plan_aggregates_pending_reprocess_scopes_across_declarations() -> None:
    """A plan crossing a SHAPE_FORWARD_TARGETED_REPROCESS delta surfaces its scope as data."""
    declaration = IndexDeltaDeclaration(
        version=54,
        classes=(DerivedDeltaClass.SHAPE_FORWARD_TARGETED_REPROCESS,),
        operations=(
            FastForwardOperation(
                name="synthetic",
                kind=FastForwardOperationKind.REPLACE_TABLE,
                objects=(("table", "sessions"),),
            ),
        ),
        reprocess_scope=lifecycle.TargetedReprocessScope(origin="codex-session"),
    )
    plan = lifecycle.IndexFastForwardPlan(source_version=53, target_version=54, declarations=(declaration,))

    assert plan.eligible_for_sql_fast_forward is True
    assert plan.requires_semantic_reparse is False
    assert plan.pending_reprocess_scopes == ((54, lifecycle.TargetedReprocessScope(origin="codex-session")),)


def test_v44_is_declared_shape_forward_targeted_reprocess_scoped_to_codex() -> None:
    """polylogue-9rw0.1 AC#4: v44 is re-declared under the new class.

    ANTI-VACUITY: reverting v44's declared ``classes`` back to
    ``(SEMANTIC_REPARSE,)`` makes ``eligible_for_sql_fast_forward`` False and
    this assertion fail -- the whole point of the new class is that v44 now
    fast-forwards instead of forcing a full rebuild to backfill two nullable
    columns on one origin's sessions.
    """
    v44 = next(d for d in lifecycle.INDEX_DELTA_DECLARATIONS if d.version == 44)

    assert v44.classes == (DerivedDeltaClass.SHAPE_FORWARD_TARGETED_REPROCESS,)
    assert v44.reprocess_scope == lifecycle.TargetedReprocessScope(origin="codex-session")

    plan = lifecycle.index_fast_forward_plan(43, 44)
    assert plan is not None
    assert plan.eligible_for_sql_fast_forward is True
    assert plan.pending_reprocess_scopes == ((44, lifecycle.TargetedReprocessScope(origin="codex-session")),)

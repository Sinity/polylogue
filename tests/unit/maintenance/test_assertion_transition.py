from __future__ import annotations

import json
import sqlite3

import pytest

from polylogue.maintenance.assertion_transition import (
    IdentityMigrationMap,
    ObjectRefDisposition,
    ObjectRefReconciliationError,
    SourceIdentityClaims,
    TransitionBinding,
    apply_assertion_transition,
    enumerate_assertion_object_refs,
    reconcile_object_refs,
)


def _binding() -> TransitionBinding:
    return TransitionBinding("pre", "candidate", "source", "package", (("user", 11),))


def test_exact_reconciliation_distinguishes_preserved_restored_missing_and_orphaned() -> None:
    restored = "session:chatgpt:restored"
    missing = "session:chatgpt:missing"
    orphan = "session:chatgpt:orphan"
    plan = reconcile_object_refs(
        (restored, missing, orphan),
        candidate_refs=(restored,),
        predecessor_refs=(),
        source_claims=SourceIdentityClaims.from_refs((restored, missing)),
        binding=_binding(),
    )
    assert [row.disposition for row in plan.rows] == [
        ObjectRefDisposition.EXPECTED_RESTORED,
        ObjectRefDisposition.BLOCKING_MISSING,
        ObjectRefDisposition.ORPHANED,
    ]


def test_old_index_presence_is_preserved_and_empty_plan_is_explicit() -> None:
    ref = "session:codex:one"
    plan = reconcile_object_refs(
        (ref,),
        candidate_refs=(ref,),
        predecessor_refs=(ref,),
        source_claims=SourceIdentityClaims.from_refs((ref,)),
        binding=_binding(),
    )
    assert plan.is_empty
    assert plan.rows[0].disposition is ObjectRefDisposition.PRESERVED


def test_map_is_typed_total_and_has_inverse() -> None:
    old, new = "session:old:1", "session:new:1"
    mapping = IdentityMigrationMap("parser-v2", ((old, new),))
    plan = reconcile_object_refs(
        (old,),
        candidate_refs=(new,),
        source_claims=SourceIdentityClaims.from_refs(()),
        migration_map=mapping,
        binding=_binding(),
    )
    assert plan.forward == ((old, new),)
    assert plan.inverse == ((new, old),)
    assert plan.rows[0].disposition is ObjectRefDisposition.EXPLICITLY_MIGRATED


def test_conflicting_map_fails_closed_and_unmapped_refs_remain_orphaned() -> None:
    with pytest.raises(ObjectRefReconciliationError):
        IdentityMigrationMap("parser", (("session:a", "session:b"), ("session:a", "session:c")))
    mapping = IdentityMigrationMap("parser", (("session:a", "session:b"),))
    plan = reconcile_object_refs(
        ("session:a", "session:c"),
        candidate_refs=("session:b",),
        source_claims=SourceIdentityClaims.from_refs(()),
        migration_map=mapping,
        binding=_binding(),
    )
    assert plan.rows[1].disposition is ObjectRefDisposition.ORPHANED


def test_enumeration_and_apply_update_all_durable_assertion_reference_columns() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE assertions (assertion_id TEXT PRIMARY KEY, scope_ref TEXT, target_ref TEXT NOT NULL, evidence_refs_json TEXT, supersedes_json TEXT)"
    )
    old, new = "session:old:1", "session:new:1"
    conn.execute("INSERT INTO assertions VALUES (?, ?, ?, ?, ?)", ("a", old, old, json.dumps([old]), json.dumps([old])))
    assert enumerate_assertion_object_refs(conn) == (old,)
    plan = reconcile_object_refs(
        (old,),
        candidate_refs=(new,),
        source_claims=SourceIdentityClaims.from_refs(()),
        migration_map=IdentityMigrationMap("parser", ((old, new),)),
        binding=_binding(),
    )
    apply_assertion_transition(conn, plan, binding=_binding(), verified_backup=True)
    row = conn.execute("SELECT scope_ref, target_ref, evidence_refs_json, supersedes_json FROM assertions").fetchone()
    assert row == (new, new, json.dumps([new]), json.dumps([new]))


def test_apply_requires_matching_binding_backup_and_rolls_back_on_audit_failure() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE assertions (assertion_id TEXT PRIMARY KEY, scope_ref TEXT, target_ref TEXT NOT NULL, evidence_refs_json TEXT, supersedes_json TEXT)"
    )
    old, new = "session:old:1", "session:new:1"
    conn.execute("INSERT INTO assertions VALUES (?, ?, ?, ?, ?)", ("a", None, old, "[]", "[]"))
    plan = reconcile_object_refs(
        (old,),
        candidate_refs=(new,),
        source_claims=SourceIdentityClaims.from_refs(()),
        migration_map=IdentityMigrationMap("parser", ((old, new),)),
        binding=_binding(),
    )
    with pytest.raises(ObjectRefReconciliationError, match="backup"):
        apply_assertion_transition(conn, plan, binding=_binding(), verified_backup=False)
    with pytest.raises(RuntimeError):
        apply_assertion_transition(
            conn,
            plan,
            binding=_binding(),
            verified_backup=True,
            append_audit=lambda _: (_ for _ in ()).throw(RuntimeError("audit")),
        )
    assert conn.execute("SELECT target_ref FROM assertions").fetchone() == (old,)


def test_apply_refuses_a_stale_binding_and_leaves_assertions_untouched() -> None:
    """The binding pins the plan to one candidate and durable-tier version set.

    Applying a plan built against a different candidate would rewrite
    irreplaceable user assertions toward identities that candidate never
    contained, so staleness must refuse before any UPDATE runs.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE assertions (assertion_id TEXT PRIMARY KEY, scope_ref TEXT, target_ref TEXT NOT NULL, evidence_refs_json TEXT, supersedes_json TEXT)"
    )
    old, new = "session:old:1", "session:new:1"
    conn.execute("INSERT INTO assertions VALUES (?, ?, ?, ?, ?)", ("a", None, old, "[]", "[]"))
    plan = reconcile_object_refs(
        (old,),
        candidate_refs=(new,),
        source_claims=SourceIdentityClaims.from_refs(()),
        migration_map=IdentityMigrationMap("parser", ((old, new),)),
        binding=_binding(),
    )
    superseded = TransitionBinding("pre", "a-later-candidate", "source", "package", (("user", 11),))

    with pytest.raises(ObjectRefReconciliationError, match="stale"):
        apply_assertion_transition(conn, plan, binding=superseded, verified_backup=True)

    assert conn.execute("SELECT target_ref FROM assertions").fetchone() == (old,)

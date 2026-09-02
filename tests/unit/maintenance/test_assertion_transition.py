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
    enumerate_durable_reference_inventory,
    reconcile_object_refs,
)
from polylogue.storage.sqlite.archive_tiers.audit import AUDIT_DDL
from polylogue.storage.sqlite.archive_tiers.user import USER_DDL


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


def test_assertion_transition_classifies_evidence_without_parsing_external_locators() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE assertions (
            assertion_id TEXT PRIMARY KEY, scope_ref TEXT, target_ref TEXT NOT NULL,
            evidence_refs_json TEXT, supersedes_json TEXT
        );
        """
    )
    old = ("message:session:old-0", "message:session:old-1")
    new = ("message:session:new-0", "message:session:new-1")
    evidence = "codex-session:demo::m::0"
    conn.execute(
        "INSERT INTO assertions VALUES (?, ?, ?, ?, ?)",
        ("a", old[0], old[0], json.dumps([old[1], evidence, "/tmp/receipt.json"]), "[]"),
    )
    refs = tuple(item.value for item in enumerate_durable_reference_inventory(conn))
    assert set(refs) == {*old, evidence}
    plan = reconcile_object_refs(
        refs,
        candidate_refs=(*new, evidence),
        source_claims=SourceIdentityClaims.from_refs(()),
        migration_map=IdentityMigrationMap("producer-v2", tuple(zip(old, new, strict=True))),
        binding=_binding(),
    )
    apply_assertion_transition(conn, plan, binding=_binding(), verified_backup=True)
    assert conn.execute("SELECT evidence_refs_json FROM assertions").fetchone()[0] == json.dumps(
        [new[1], evidence, "/tmp/receipt.json"]
    )


def test_missing_candidate_endpoint_blocks_before_mutation() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE assertions (assertion_id TEXT PRIMARY KEY, scope_ref TEXT, target_ref TEXT NOT NULL, evidence_refs_json TEXT, supersedes_json TEXT)"
    )
    conn.execute("INSERT INTO assertions VALUES (?, ?, ?, ?, ?)", ("a", None, "message:old", "[]", "[]"))
    binding = _binding()
    plan = reconcile_object_refs(
        ("message:old",),
        candidate_refs=(),
        source_claims=SourceIdentityClaims.from_refs(()),
        migration_map=IdentityMigrationMap("producer", (("message:old", "message:new"),)),
        binding=binding,
    )

    with pytest.raises(ObjectRefReconciliationError, match="blocking missing"):
        apply_assertion_transition(conn, plan, binding=binding, verified_backup=True)
    assert conn.execute("SELECT target_ref FROM assertions").fetchone() == ("message:old",)


def test_position_only_reconstruction_is_blocked_without_a_producer_map() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE assertions (assertion_id TEXT PRIMARY KEY, scope_ref TEXT, target_ref TEXT NOT NULL, evidence_refs_json TEXT, supersedes_json TEXT)"
    )
    conn.execute("INSERT INTO assertions VALUES (?, ?, ?, ?, ?)", ("a", None, "message:old", "[]", "[]"))
    binding = _binding()
    plan = reconcile_object_refs(
        ("message:old",),
        candidate_refs=("message:new",),
        source_claims=SourceIdentityClaims.from_refs(("message:old",)),
        binding=binding,
    )

    assert plan.rows[0].disposition is ObjectRefDisposition.BLOCKING_MISSING
    with pytest.raises(ObjectRefReconciliationError, match="blocking missing"):
        apply_assertion_transition(conn, plan, binding=binding, verified_backup=True)
    assert conn.execute("SELECT target_ref FROM assertions").fetchone() == ("message:old",)


def test_omitted_relation_in_a_complete_schema_blocks_apply() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(USER_DDL)
    conn.execute("DROP TABLE result_set_members")
    binding = _binding()
    plan = reconcile_object_refs(
        ("message:old",),
        candidate_refs=("message:new",),
        source_claims=SourceIdentityClaims.from_refs(()),
        migration_map=IdentityMigrationMap("producer", (("message:old", "message:new"),)),
        binding=binding,
    )

    with pytest.raises(ObjectRefReconciliationError, match="result_set_members"):
        apply_assertion_transition(conn, plan, binding=binding, verified_backup=True)


def test_result_set_members_are_sealed_against_identity_rewrites() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE assertions (
            assertion_id TEXT PRIMARY KEY, scope_ref TEXT, target_ref TEXT NOT NULL,
            evidence_refs_json TEXT, supersedes_json TEXT
        );
        CREATE TABLE result_set_members (
            result_set_id TEXT NOT NULL, rank INTEGER NOT NULL, member_ref TEXT NOT NULL,
            PRIMARY KEY (result_set_id, rank)
        );
        INSERT INTO assertions VALUES ('a', NULL, 'message:old', '[]', '[]');
        INSERT INTO result_set_members VALUES ('finding', 0, 'message:old');
        """
    )
    binding = _binding()
    plan = reconcile_object_refs(
        ("message:old",),
        candidate_refs=("message:new",),
        source_claims=SourceIdentityClaims.from_refs(()),
        migration_map=IdentityMigrationMap("producer", (("message:old", "message:new"),)),
        binding=binding,
    )

    with pytest.raises(ObjectRefReconciliationError, match="sealed durable relation: user.result_set_members"):
        apply_assertion_transition(conn, plan, binding=binding, verified_backup=True)
    assert conn.execute("SELECT member_ref FROM result_set_members").fetchone() == ("message:old",)


def test_inventory_includes_evidence_refs_and_rejects_undeclared_reference_columns() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE assertions (
            assertion_id TEXT PRIMARY KEY, scope_ref TEXT, target_ref TEXT NOT NULL,
            author_ref TEXT, evidence_refs_json TEXT, supersedes_json TEXT
        );
        CREATE TABLE unexpected_relation (target_ref TEXT NOT NULL);
        """
    )
    evidence = "codex-session:demo::message-old::0"
    conn.execute(
        "INSERT INTO assertions VALUES (?, ?, ?, ?, ?, ?)",
        ("a", "message:session:old", "message:session:old", "user:local", json.dumps([evidence, "/tmp/receipt"]), "[]"),
    )
    with pytest.raises(ObjectRefReconciliationError, match="lacks a descriptor"):
        enumerate_durable_reference_inventory(conn)

    conn.execute("DROP TABLE unexpected_relation")
    inventory = enumerate_durable_reference_inventory(conn)
    assert {item.value for item in inventory} == {"message:session:old", "user:local", evidence}


def test_fresh_durable_ddl_is_covered_by_the_reference_catalog() -> None:
    user = sqlite3.connect(":memory:")
    audit = sqlite3.connect(":memory:")
    user.executescript(USER_DDL)
    audit.executescript(AUDIT_DDL)
    assert enumerate_durable_reference_inventory(user, audit) == ()


def test_evidence_reference_can_use_an_exact_migration_map() -> None:
    old = "codex-session:demo::message-old::0"
    new = "codex-session:demo::message-new::0"
    plan = reconcile_object_refs(
        (old,),
        candidate_refs=(new,),
        source_claims=SourceIdentityClaims.from_refs(()),
        migration_map=IdentityMigrationMap("producer", ((old, new),)),
        binding=_binding(),
    )
    assert plan.rows[0].disposition is ObjectRefDisposition.EXPLICITLY_MIGRATED
    assert plan.forward == ((old, new),)


def test_native_evidence_identity_can_contain_slashes() -> None:
    old = "browser-capture:c/with spaces::message-id"
    new = "browser-capture:c/with spaces::replacement-id"
    plan = reconcile_object_refs(
        (old,),
        candidate_refs=(new,),
        source_claims=SourceIdentityClaims.from_refs(()),
        migration_map=IdentityMigrationMap("producer", ((old, new),)),
        binding=_binding(),
    )
    assert plan.forward == ((old, new),)


def test_plan_must_classify_every_durable_public_reference() -> None:
    user = sqlite3.connect(":memory:")
    user.executescript(
        """
        CREATE TABLE assertions (
            assertion_id TEXT PRIMARY KEY, scope_ref TEXT, target_ref TEXT NOT NULL,
            evidence_refs_json TEXT, supersedes_json TEXT
        );
        CREATE TABLE context_deliveries (
            snapshot_ref TEXT PRIMARY KEY, evidence_refs_json TEXT NOT NULL
        );
        INSERT INTO assertions VALUES ('a', NULL, 'message:old', '[]', '[]');
        INSERT INTO context_deliveries VALUES
            ('context-snapshot:s1', '["message:unclassified", "/tmp/receipt"]');
        """
    )
    old, new = "message:old", "message:new"
    plan = reconcile_object_refs(
        (old,),
        candidate_refs=(new,),
        source_claims=SourceIdentityClaims.from_refs(()),
        migration_map=IdentityMigrationMap("producer", ((old, new),)),
        binding=_binding(),
    )
    with pytest.raises(ObjectRefReconciliationError, match="does not classify durable user references"):
        apply_assertion_transition(user, plan, binding=_binding(), verified_backup=True)
    assert user.execute("SELECT target_ref FROM assertions").fetchone() == (old,)


@pytest.mark.parametrize(
    ("tier", "ddl", "select"),
    [
        (
            "user",
            """
            CREATE TABLE annotation_batches (batch_id TEXT PRIMARY KEY, target_ref TEXT NOT NULL);
            INSERT INTO annotation_batches VALUES ('batch', 'message:old');
            """,
            "SELECT target_ref FROM annotation_batches",
        ),
        (
            "user",
            """
            CREATE TABLE context_deliveries (snapshot_ref TEXT PRIMARY KEY, evidence_refs_json TEXT NOT NULL);
            INSERT INTO context_deliveries VALUES ('context-snapshot:old', '[\"message:old\"]');
            """,
            "SELECT evidence_refs_json FROM context_deliveries",
        ),
        (
            "audit",
            """
            CREATE TABLE operation_targets (operation_id TEXT NOT NULL, ordinal INTEGER NOT NULL, target_ref TEXT NOT NULL,
                PRIMARY KEY (operation_id, ordinal));
            INSERT INTO operation_targets VALUES ('operation', 0, 'message:old');
            """,
            "SELECT target_ref FROM operation_targets",
        ),
    ],
)
def test_sealed_durable_history_is_never_rewritten(tier: str, ddl: str, select: str) -> None:
    user = sqlite3.connect(":memory:")
    audit = sqlite3.connect(":memory:")
    connection = user if tier == "user" else audit
    connection.executescript(ddl)
    old, new = "message:old", "message:new"
    refs = (old, "context-snapshot:old") if tier == "user" and "context_deliveries" in ddl else (old,)
    plan = reconcile_object_refs(
        refs,
        candidate_refs=tuple("context-snapshot:new" if ref == "context-snapshot:old" else new for ref in refs),
        source_claims=SourceIdentityClaims.from_refs(()),
        migration_map=IdentityMigrationMap(
            "producer",
            tuple((ref, "context-snapshot:new" if ref == "context-snapshot:old" else new) for ref in refs),
        ),
        binding=_binding(),
    )
    with pytest.raises(ObjectRefReconciliationError, match=f"sealed durable relation: {tier}."):
        apply_assertion_transition(user, plan, binding=_binding(), verified_backup=True, audit_conn=audit)
    assert connection.execute(select).fetchone()[0] != new

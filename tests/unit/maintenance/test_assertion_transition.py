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
    _insert_result_set_successor,
    _result_set_successor_id,
    apply_assertion_transition,
    enumerate_assertion_object_refs,
    enumerate_durable_reference_inventory,
    reconcile_object_refs,
)
from polylogue.storage.sqlite.archive_tiers.audit import AUDIT_DDL
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_DDL
from polylogue.storage.sqlite.holdout_cohorts import mark_holdout
from polylogue.storage.sqlite.query_objects import (
    EvaluationReceipt,
    get_result_set,
    membership_merkle_root,
    put_evaluation_receipt,
    put_query,
    put_result_set,
    put_retained_query_run,
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


def test_result_set_member_transition_clones_the_manifest_and_repoints_owners() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.USER)
    query = put_query(
        conn,
        {"field": "origin", "value": "claude-code"},
        grain="message",
        lane="dialogue",
        rank_policy="fixture",
        created_at_ms=1,
    )
    old = tuple(f"message:claude-code:legacy:{index}" for index in range(20))
    new = tuple(f"message:claude-code:current:n:{index}" for index in range(20))
    put_result_set(
        conn,
        result_set_id="finding",
        query_hash=query.query_hash,
        grain="message",
        corpus_epoch="before",
        member_refs=old,
        exactness="exact",
        persistence_class="finding",
        created_at_ms=2,
    )
    put_retained_query_run(
        conn, run_id="qr_finding", query_hash=query.query_hash, result_set_id="finding", retained_at_ms=3
    )
    put_evaluation_receipt(
        conn,
        query_hash=query.query_hash,
        result_set_id="finding",
        receipt=EvaluationReceipt("receipt", "source", "user", "index", "build"),
        created_at_ms=4,
    )
    conn.executemany(
        """
        INSERT INTO assertions (
            assertion_id, scope_ref, target_ref, kind, evidence_refs_json,
            created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, 'note', ?, 6, 6)
        """,
        (
            (
                f"assertion-{index}",
                "result-set:finding" if index == 0 else None,
                old[index % len(old)],
                json.dumps([old[(index + 1) % len(old)], "codex-session:demo::m::0", "/tmp/receipt"]),
            )
            for index in range(107)
        ),
    )
    binding = _binding()
    refs = tuple(item.value for item in enumerate_durable_reference_inventory(conn))
    plan = reconcile_object_refs(
        refs,
        candidate_refs=tuple(new) + tuple(ref for ref in refs if ref not in old),
        source_claims=SourceIdentityClaims.from_refs(()),
        migration_map=IdentityMigrationMap("producer", tuple(zip(old, new, strict=True))),
        binding=binding,
    )

    apply_assertion_transition(conn, plan, binding=binding, verified_backup=True)

    successor = conn.execute("SELECT result_set_id FROM result_sets WHERE result_set_id != 'finding'").fetchone()[0]
    assert successor != "finding"
    # Execution records name the manifest they actually ran against; that
    # manifest is retained, so they are not repointed.
    assert (
        conn.execute("SELECT result_set_id FROM retained_query_runs WHERE run_id = 'qr_finding'").fetchone()[0]
        == "finding"
    )
    assert (
        tuple(
            row[0]
            for row in conn.execute(
                "SELECT member_ref FROM result_set_members WHERE result_set_id = ? ORDER BY rank", (successor,)
            )
        )
        == new
    )
    assert (
        tuple(
            row[0]
            for row in conn.execute(
                "SELECT member_ref FROM result_set_members WHERE result_set_id = 'finding' ORDER BY rank"
            )
        )
        == old
    )
    manifest = get_result_set(conn, successor)
    assert manifest is not None
    assert manifest.member_count == len(new)
    assert manifest.membership_merkle_root == membership_merkle_root(new)
    assert tuple(
        conn.execute("SELECT result_set_id FROM query_evaluation_receipts WHERE receipt_id = 'receipt'").fetchone()
    ) == ("finding",)
    assert tuple(conn.execute("SELECT scope_ref FROM assertions WHERE assertion_id = 'assertion-0'").fetchone()) == (
        f"result-set:{successor}",
    )
    assert tuple(
        conn.execute(
            "SELECT COUNT(*) FROM assertions WHERE target_ref LIKE 'message:claude-code:current:n:%'"
        ).fetchone()
    ) == (107,)
    evidence = json.loads(
        conn.execute("SELECT evidence_refs_json FROM assertions WHERE assertion_id = 'assertion-0'").fetchone()[0]
    )
    assert evidence[-2:] == ["codex-session:demo::m::0", "/tmp/receipt"]


def test_corrupt_result_set_manifest_blocks_and_rolls_back() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.USER)
    query = put_query(
        conn,
        {"field": "origin", "value": "claude-code"},
        grain="message",
        lane="dialogue",
        rank_policy="fixture",
        created_at_ms=1,
    )
    old, new = "message:legacy", "message:current:n:1"
    put_result_set(
        conn,
        result_set_id="finding",
        query_hash=query.query_hash,
        grain="message",
        corpus_epoch="before",
        member_refs=(old,),
        exactness="exact",
        persistence_class="finding",
        created_at_ms=2,
    )
    conn.execute(
        "INSERT INTO assertions (assertion_id, target_ref, kind, created_at_ms, updated_at_ms) VALUES ('a', ?, 'note', 3, 3)",
        (old,),
    )
    conn.execute("UPDATE result_sets SET membership_merkle_root = ? WHERE result_set_id = 'finding'", ("0" * 64,))
    refs = tuple(item.value for item in enumerate_durable_reference_inventory(conn))
    plan = reconcile_object_refs(
        refs,
        candidate_refs=(new,),
        source_claims=SourceIdentityClaims.from_refs(()),
        migration_map=IdentityMigrationMap("producer", ((old, new),)),
        binding=_binding(),
    )

    with pytest.raises(ObjectRefReconciliationError, match="membership root"):
        apply_assertion_transition(conn, plan, binding=_binding(), verified_backup=True)
    assert tuple(conn.execute("SELECT target_ref FROM assertions WHERE assertion_id = 'a'").fetchone()) == (old,)
    assert tuple(conn.execute("SELECT COUNT(*) FROM result_sets").fetchone()) == (1,)


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
    file_ref = "file:artifact"
    conn.execute(
        "INSERT INTO assertions VALUES (?, ?, ?, ?, ?, ?)",
        (
            "a",
            "message:session:old",
            "message:session:old",
            "user:local",
            json.dumps([evidence, file_ref, "file:///tmp/receipt"]),
            "[]",
        ),
    )
    with pytest.raises(ObjectRefReconciliationError, match="lacks a descriptor"):
        enumerate_durable_reference_inventory(conn)

    conn.execute("DROP TABLE unexpected_relation")
    inventory = enumerate_durable_reference_inventory(conn)
    assert {item.value for item in inventory} == {"message:session:old", "user:local", evidence, file_ref}


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


def _whole_inventory_plan(conn: sqlite3.Connection, old: str, new: str, audit: sqlite3.Connection | None = None):  # type: ignore[no-untyped-def]
    """Plan that migrates ``old``→``new`` and preserves every other ref."""
    refs = tuple(item.value for item in enumerate_durable_reference_inventory(conn, audit))
    return reconcile_object_refs(
        refs,
        candidate_refs=tuple(new if ref == old else ref for ref in refs),
        source_claims=SourceIdentityClaims.from_refs(()),
        migration_map=IdentityMigrationMap("producer", ((old, new),)),
        binding=_binding(),
    )


def _member_transition_fixture(conn: sqlite3.Connection) -> tuple[object, tuple[str, ...], tuple[str, ...]]:
    """Seed one promoted result set whose members all need a new identity."""
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.USER)
    query = put_query(
        conn,
        {"field": "origin", "value": "claude-code"},
        grain="message",
        lane="dialogue",
        rank_policy="fixture",
        created_at_ms=1,
    )
    old = tuple(f"message:claude-code:legacy:{index}" for index in range(4))
    new = tuple(f"message:claude-code:current:n:{index}" for index in range(4))
    put_result_set(
        conn,
        result_set_id="finding",
        query_hash=query.query_hash,
        grain="message",
        corpus_epoch="before",
        member_refs=old,
        exactness="exact",
        persistence_class="finding",
        created_at_ms=2,
    )
    return query, old, new


def _member_transition_plan(conn: sqlite3.Connection, old: tuple[str, ...], new: tuple[str, ...]):  # type: ignore[no-untyped-def]
    refs = tuple(item.value for item in enumerate_durable_reference_inventory(conn))
    return reconcile_object_refs(
        refs,
        candidate_refs=tuple(new) + tuple(ref for ref in refs if ref not in old),
        source_claims=SourceIdentityClaims.from_refs(()),
        migration_map=IdentityMigrationMap("producer", tuple(zip(old, new, strict=True))),
        binding=_binding(),
    )


def test_retained_execution_records_survive_a_result_set_transition() -> None:
    """A replayed execution is still the same execution after a transition.

    Anti-vacuity: putting ``retained_query_runs`` back into
    ``_repoint_result_set_owners``'s ``owner_columns`` makes this red --
    ``put_retained_query_run`` compares ``result_set_id`` as exact-retry
    identity and would raise "conflicts with a different execution".
    """
    conn = sqlite3.connect(":memory:")
    query, old, new = _member_transition_fixture(conn)
    put_retained_query_run(
        conn, run_id="qr_finding", query_hash=query.query_hash, result_set_id="finding", retained_at_ms=3
    )
    plan = _member_transition_plan(conn, old, new)

    apply_assertion_transition(conn, plan, binding=_binding(), verified_backup=True)

    retry = put_retained_query_run(
        conn, run_id="qr_finding", query_hash=query.query_hash, result_set_id="finding", retained_at_ms=3
    )
    assert retry.result_set_id == "finding"


def test_result_set_successor_creation_is_retry_safe() -> None:
    """Creating the same deterministic successor twice converges.

    The successor id is a hash of the manifest it derives from, so a retried
    application recomputes exactly this id and this content.

    Anti-vacuity: restoring the plain ``INSERT INTO result_sets`` /
    ``INSERT INTO result_set_members`` makes this red with
    ``sqlite3.IntegrityError``.
    """
    conn = sqlite3.connect(":memory:")
    _query, old, new = _member_transition_fixture(conn)
    row = tuple(conn.execute("SELECT * FROM result_sets WHERE result_set_id = 'finding'").fetchone())
    successor_id = _result_set_successor_id("finding", old, new)

    _insert_result_set_successor(conn, successor_id, row, new)
    _insert_result_set_successor(conn, successor_id, row, new)

    assert conn.execute("SELECT COUNT(*) FROM result_sets WHERE result_set_id = ?", (successor_id,)).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM result_set_members WHERE result_set_id = ?", (successor_id,)).fetchone()[
        0
    ] == len(new)


def test_a_holdout_result_set_is_never_given_a_second_identity() -> None:
    """A holdout cohort keeps exactly one manifest id.

    Contamination is recorded per ``result_set_id`` and is permanent once
    true, so a second id would let an undeclared access under one id read as
    clean under the other.

    Anti-vacuity: restoring ``_clone_holdout_policy``'s successor INSERT
    makes this red -- the transition would succeed and
    ``has_holdout_contamination`` would answer independently for each id.
    """
    conn = sqlite3.connect(":memory:")
    _query, old, new = _member_transition_fixture(conn)
    mark_holdout(
        conn,
        result_set_id="finding",
        frame="evaluation",
        selection_definition={"kind": "fixture"},
        intended_confirmation_use="confirm",
        authority="user:local",
        created_epoch="before",
        created_at_ms=3,
    )
    plan = _member_transition_plan(conn, old, new)

    with pytest.raises(ObjectRefReconciliationError, match="holdout policy"):
        apply_assertion_transition(conn, plan, binding=_binding(), verified_backup=True)
    assert conn.execute("SELECT COUNT(*) FROM result_sets").fetchone()[0] == 1


def test_catalog_completeness_follows_the_tier_schema_version() -> None:
    """A durable file at an earlier migration is valid, not incomplete.

    Anti-vacuity: restoring the ``query_unit_frame_state`` marker check makes
    this red -- a user tier at version 10 carries that table but not
    ``query_excision_ledger``, and every transition against it would be
    refused as an incomplete catalog.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(USER_DDL)
    conn.execute("DROP TABLE query_excision_ledger")
    conn.execute("PRAGMA user_version = 10")
    old, new = "message:old", "message:new"
    conn.execute(
        "INSERT INTO assertions (assertion_id, target_ref, kind, created_at_ms, updated_at_ms) "
        "VALUES ('a', ?, 'note', 3, 3)",
        (old,),
    )
    plan = _whole_inventory_plan(conn, old, new)

    apply_assertion_transition(conn, plan, binding=_binding(), verified_backup=True)
    assert conn.execute("SELECT target_ref FROM assertions WHERE assertion_id = 'a'").fetchone()[0] == new


def test_neither_durable_tier_commits_before_the_caller_commits() -> None:
    """Releasing the savepoints leaves both tiers uncommitted.

    Anti-vacuity: removing the ``BEGIN`` that encloses each connection makes
    this red -- the outermost savepoint release would commit the user tier by
    itself, so a crash before the audit release would strand a durable
    transition with no audit record.
    """
    user = sqlite3.connect(":memory:")
    audit = sqlite3.connect(":memory:")
    user.executescript(USER_DDL)
    audit.executescript(AUDIT_DDL)
    old, new = "message:old", "message:new"
    user.execute(
        "INSERT INTO assertions (assertion_id, target_ref, kind, created_at_ms, updated_at_ms) "
        "VALUES ('a', ?, 'note', 3, 3)",
        (old,),
    )
    user.commit()
    plan = _whole_inventory_plan(user, old, new, audit)

    apply_assertion_transition(user, plan, binding=_binding(), verified_backup=True, audit_conn=audit)

    assert user.in_transaction
    assert audit.in_transaction
    user.rollback()
    assert user.execute("SELECT target_ref FROM assertions WHERE assertion_id = 'a'").fetchone() == (old,)

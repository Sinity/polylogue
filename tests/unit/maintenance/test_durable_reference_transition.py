"""The rebuild's durable-reference rebind step, end to end.

Anti-vacuity for the whole file: every assertion here fails if the transition
stops being computed. Drop the migration map and the message dispositions become
``orphaned``; drop the index-independent branch and content-addressed refs become
``orphaned``; drop the source claims and a missing session stops blocking.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from polylogue.maintenance.assertion_transition import (
    ObjectRefDisposition,
    ObjectRefReconciliationError,
    SourceIdentityClaims,
)
from polylogue.maintenance.durable_reference_transition import (
    DurableReferenceTransition,
    apply_durable_reference_transition,
    governed_target,
    index_identity,
    message_identity_migration_map,
    plan_durable_reference_transition,
    resolve_candidate_references,
    source_session_claims,
)
from polylogue.storage.sqlite.archive_tiers.audit import AUDIT_DDL
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
from polylogue.storage.sqlite.archive_tiers.user import USER_DDL
from polylogue.storage.sqlite.query_objects import (
    EvaluationReceipt,
    put_evaluation_receipt,
    put_query,
    put_result_set,
)

SESSION = "chatgpt-export:s1"
NATIVE_OLD = f"message:{SESSION}:m-1"
POSITIONAL_OLD = f"message:{SESSION}:0.0"
NATIVE_NEW = f"message:{SESSION}:n:m-1"
POSITIONAL_NEW = f"message:{SESSION}:p:0.0"
NATIVE_EVIDENCE_OLD = f"{SESSION}::{SESSION}:m-1"
POSITIONAL_EVIDENCE_OLD = f"{SESSION}::{SESSION}:0.0"
NATIVE_EVIDENCE_NEW = f"{SESSION}::{SESSION}:n:m-1"
POSITIONAL_EVIDENCE_NEW = f"{SESSION}::{SESSION}:p:0.0"

#: The message identity the live archive's index generation was built with.
_PREDECESSOR_DDL = """
CREATE TABLE sessions (
    session_id TEXT GENERATED ALWAYS AS (origin || ':' || native_id) STORED UNIQUE,
    origin TEXT NOT NULL,
    native_id TEXT NOT NULL,
    PRIMARY KEY(origin, native_id)
) STRICT;
CREATE TABLE messages (
    message_id TEXT GENERATED ALWAYS AS (
        session_id || ':' || COALESCE(native_id, position || '.' || variant_index)
    ) STORED UNIQUE,
    session_id TEXT NOT NULL,
    native_id TEXT,
    position INTEGER NOT NULL,
    variant_index INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(session_id, position, variant_index)
) STRICT;
CREATE TABLE blocks (
    block_id TEXT GENERATED ALWAYS AS (message_id || ':' || position) STORED UNIQUE,
    message_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY(message_id, position)
) STRICT;
"""


def _predecessor_index() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(_PREDECESSOR_DDL)
    conn.execute("INSERT INTO sessions (origin, native_id) VALUES ('chatgpt-export', 's1')")
    conn.executemany(
        "INSERT INTO messages (session_id, native_id, position) VALUES (?, ?, ?)",
        ((SESSION, None, 0), (SESSION, "m-1", 1)),
    )
    conn.execute("INSERT INTO blocks (message_id, position) VALUES (?, 0)", (f"{SESSION}:m-1",))
    conn.commit()
    return conn


def _candidate_index(*, native_id_shadowing_a_position: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    conn.execute(
        "INSERT INTO sessions (origin, native_id, content_hash) VALUES ('chatgpt-export', 's1', ?)", (b"\x00" * 32,)
    )
    rows = [(SESSION, None, 0, "positional"), (SESSION, "m-1", 1, "native")]
    if native_id_shadowing_a_position:
        rows.append((SESSION, "0.0", 2, "native"))
    conn.executemany(
        "INSERT INTO messages (session_id, native_id, position, identity_source, role, content_hash)"
        " VALUES (?, ?, ?, ?, 'user', ?)",
        ((session, native, position, source, b"\x00" * 32) for session, native, position, source in rows),
    )
    conn.execute(
        "INSERT INTO blocks (message_id, session_id, position, block_type) VALUES (?, ?, 0, 'text')",
        (f"{SESSION}:n:m-1", SESSION),
    )
    conn.commit()
    return conn


def _durable_tiers(refs: tuple[str, ...]) -> tuple[sqlite3.Connection, sqlite3.Connection]:
    user = sqlite3.connect(":memory:")
    user.executescript(USER_DDL)
    audit = sqlite3.connect(":memory:")
    audit.executescript(AUDIT_DDL)
    query = put_query(
        user,
        {"field": "origin", "value": "chatgpt-export"},
        grain="message",
        lane="dialogue",
        rank_policy="fixture",
        created_at_ms=1,
    )
    put_result_set(
        user,
        result_set_id="finding",
        query_hash=query.query_hash,
        grain="message",
        corpus_epoch="before",
        member_refs=refs,
        exactness="exact",
        persistence_class="finding",
        created_at_ms=2,
    )
    put_evaluation_receipt(
        user,
        query_hash=query.query_hash,
        result_set_id="finding",
        # The live archive stores a package build id here: it parses as an
        # evidence ref and addresses no index row.
        receipt=EvaluationReceipt("receipt", "source-gen", "user-gen", "index-gen", "polylogue:0.3.0"),
        created_at_ms=3,
    )
    user.commit()
    audit.commit()
    return user, audit


def _plan(
    user: sqlite3.Connection,
    audit: sqlite3.Connection,
    candidate: sqlite3.Connection,
    predecessor: sqlite3.Connection,
    *,
    claims: SourceIdentityClaims = SourceIdentityClaims.from_refs(()),
) -> DurableReferenceTransition:
    return plan_durable_reference_transition(
        user_conn=user,
        audit_conn=audit,
        candidate_index_conn=candidate,
        predecessor_index_conn=predecessor,
        source_claims=claims,
        package_version="test",
        producer="test-producer",
        candidate_index_path="candidate",
        predecessor_index_path="predecessor",
    )


def test_message_refs_are_explicitly_migrated_across_the_n_p_identity_change() -> None:
    user, audit = _durable_tiers((NATIVE_OLD, POSITIONAL_OLD))
    predecessor, candidate = _predecessor_index(), _candidate_index()

    transition = _plan(user, audit, candidate, predecessor)

    census = transition.receipt(applied=False)
    assert census["resolution"]["predecessor_resolved_cells"] == 3
    assert census["resolution"]["candidate_direct_resolved_cells"] == 1
    assert census["resolution"]["candidate_resolved_cells"] == 3
    assert census["durable_rows"]["after"] is None

    assert dict(transition.plan.forward) == {NATIVE_OLD: NATIVE_NEW, POSITIONAL_OLD: POSITIONAL_NEW}
    assert transition.dispositions[ObjectRefDisposition.EXPLICITLY_MIGRATED.value] == 2
    assert transition.dispositions[ObjectRefDisposition.ORPHANED.value] == 0
    assert transition.binding.migration_map_digest is not None

    apply_durable_reference_transition(user_conn=user, audit_conn=audit, transition=transition, verified_backup=True)

    applied = transition.receipt(applied=True)
    assert (
        applied["durable_rows"]["after"]["user.result_sets"] == census["durable_rows"]["before"]["user.result_sets"] + 1
    )
    assert (
        applied["durable_rows"]["after"]["user.result_set_members"]
        == census["durable_rows"]["before"]["user.result_set_members"] + 2
    )

    # The original manifest is immutable, so the old ids survive verbatim and
    # the transition lands as a successor manifest carrying the new ones.
    assert tuple(
        row[0]
        for row in user.execute(
            "SELECT member_ref FROM result_set_members WHERE result_set_id = 'finding' ORDER BY rank"
        )
    ) == (NATIVE_OLD, POSITIONAL_OLD)
    successor = user.execute("SELECT result_set_id FROM result_sets WHERE result_set_id != 'finding'").fetchone()[0]
    assert tuple(
        row[0]
        for row in user.execute(
            "SELECT member_ref FROM result_set_members WHERE result_set_id = ? ORDER BY rank", (successor,)
        )
    ) == (NATIVE_NEW, POSITIONAL_NEW)


def test_without_a_map_the_same_references_are_recorded_as_orphaned() -> None:
    """The disposition the rebuild takes silently today, made explicit."""
    user, audit = _durable_tiers((NATIVE_OLD, POSITIONAL_OLD))
    predecessor = _predecessor_index()
    empty_candidate = sqlite3.connect(":memory:")
    empty_candidate.executescript(INDEX_DDL)

    transition = _plan(user, audit, empty_candidate, predecessor)

    assert transition.plan.forward == ()
    assert transition.dispositions[ObjectRefDisposition.ORPHANED.value] == 2
    assert [row.disposition for row in transition.plan.rows if row.source.startswith("message:")] == [
        ObjectRefDisposition.ORPHANED,
        ObjectRefDisposition.ORPHANED,
    ]


def test_index_independent_reference_kinds_are_never_reported_as_lost() -> None:
    user, audit = _durable_tiers((NATIVE_OLD,))
    user.execute(
        "INSERT INTO assertions (assertion_id, target_ref, kind, evidence_refs_json, created_at_ms, updated_at_ms)"
        " VALUES ('a', 'insight:raw-authority-frontier@v1', 'note', ?, 1, 1)",
        (json.dumps(["user:local", "analysis:claim-vs-evidence", f"session:{SESSION}"]),),
    )
    predecessor, candidate = _predecessor_index(), _candidate_index()

    transition = _plan(user, audit, candidate, predecessor)

    dispositions = {row.source: row.disposition for row in transition.plan.rows}
    assert dispositions["insight:raw-authority-frontier@v1"] is ObjectRefDisposition.PRESERVED
    assert dispositions["user:local"] is ObjectRefDisposition.PRESERVED
    assert dispositions["analysis:claim-vs-evidence"] is ObjectRefDisposition.PRESERVED
    assert dispositions[f"session:{SESSION}"] is ObjectRefDisposition.PRESERVED


def test_a_source_claimed_session_missing_from_the_candidate_blocks_the_transition() -> None:
    missing = "session:chatgpt-export:lost"
    user, audit = _durable_tiers((NATIVE_OLD,))
    user.execute(
        "INSERT INTO assertions (assertion_id, target_ref, kind, evidence_refs_json, created_at_ms, updated_at_ms)"
        " VALUES ('a', ?, 'note', '[]', 1, 1)",
        (missing,),
    )
    predecessor, candidate = _predecessor_index(), _candidate_index()

    orphaned = _plan(user, audit, candidate, predecessor)
    assert {row.source: row.disposition for row in orphaned.plan.rows}[missing] is ObjectRefDisposition.ORPHANED

    claimed = _plan(user, audit, candidate, predecessor, claims=SourceIdentityClaims.from_refs((missing,)))
    assert {row.source: row.disposition for row in claimed.plan.rows}[missing] is ObjectRefDisposition.BLOCKING_MISSING
    with pytest.raises(ObjectRefReconciliationError, match="blocking missing"):
        apply_durable_reference_transition(user_conn=user, audit_conn=audit, transition=claimed, verified_backup=True)


def test_source_session_claims_come_from_acquired_raw_rows() -> None:
    source = sqlite3.connect(":memory:")
    source.execute("CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY, origin TEXT, native_id TEXT)")
    source.executemany(
        "INSERT INTO raw_sessions VALUES (?, ?, ?)",
        (("r1", "chatgpt-export", "s1"), ("r2", "claude-code-session", None)),
    )
    assert source_session_claims(source).refs == frozenset({"session:chatgpt-export:s1"})

    empty = sqlite3.connect(":memory:")
    with pytest.raises(ObjectRefReconciliationError, match="raw_sessions"):
        source_session_claims(empty)


def test_two_candidate_successors_for_one_old_id_refuse_rather_than_guess() -> None:
    candidate = _candidate_index(native_id_shadowing_a_position=True)
    with pytest.raises(ObjectRefReconciliationError, match="two candidate successors"):
        message_identity_migration_map(candidate, (POSITIONAL_OLD,), producer="test")


def test_evidence_message_refs_are_migrated_with_native_and_positional_ids() -> None:
    """Compact evidence refs retain their session and follow both namespaces."""
    user, audit = _durable_tiers((NATIVE_OLD, POSITIONAL_OLD))
    # The fixture has no assertion row by default; add one through the schema
    # directly so this test exercises the declared durable field inventory.
    user.execute(
        "INSERT INTO assertions (assertion_id, target_ref, kind, evidence_refs_json, created_at_ms, updated_at_ms) "
        "VALUES ('evidence', 'user:local', 'note', ?, 1, 1)",
        (json.dumps([NATIVE_EVIDENCE_OLD, POSITIONAL_EVIDENCE_OLD]),),
    )
    predecessor, candidate = _predecessor_index(), _candidate_index()

    transition = _plan(user, audit, candidate, predecessor)

    assert dict(transition.plan.forward)[NATIVE_EVIDENCE_OLD] == NATIVE_EVIDENCE_NEW
    assert dict(transition.plan.forward)[POSITIONAL_EVIDENCE_OLD] == POSITIONAL_EVIDENCE_NEW

    apply_durable_reference_transition(user_conn=user, audit_conn=audit, transition=transition, verified_backup=True)

    evidence = json.loads(
        user.execute("SELECT evidence_refs_json FROM assertions WHERE assertion_id = 'evidence'").fetchone()[0]
    )
    assert evidence == [NATIVE_EVIDENCE_NEW, POSITIONAL_EVIDENCE_NEW]


def test_an_index_governed_kind_without_an_exact_probe_refuses() -> None:
    candidate = _candidate_index()
    with pytest.raises(ObjectRefReconciliationError, match="no exact candidate probe"):
        resolve_candidate_references(candidate, ("action:some-tool-use-block:0",))
    assert governed_target("insight:x@v1") is None
    assert governed_target(f"block:{SESSION}:n:m-1:0") == ("blocks", "block_id", f"{SESSION}:n:m-1:0")


def test_the_binding_moves_with_the_candidate_index_identity() -> None:
    predecessor, candidate = _predecessor_index(), _candidate_index()
    assert index_identity(predecessor) != index_identity(candidate)
    user, audit = _durable_tiers((NATIVE_OLD,))
    transition = _plan(user, audit, candidate, predecessor)
    assert transition.binding.candidate_digest == index_identity(candidate)
    assert transition.binding.predecessor_digest == index_identity(predecessor)


def test_the_receipt_records_every_disposition_and_the_rewrites() -> None:
    user, audit = _durable_tiers((NATIVE_OLD, POSITIONAL_OLD))
    predecessor, candidate = _predecessor_index(), _candidate_index()
    receipt = _plan(user, audit, candidate, predecessor).receipt(applied=False)
    assert receipt["schema"] == "polylogue.durable-reference-transition.v1"
    assert receipt["applied"] is False
    assert len(receipt["references"]) == receipt["classified_references"]
    assert sorted(pair[0] for pair in receipt["forward"]) == sorted((NATIVE_OLD, POSITIONAL_OLD))
    assert receipt["dispositions"][ObjectRefDisposition.EXPLICITLY_MIGRATED.value] == 2


def test_production_index_ddl_still_uses_the_identity_this_route_maps_onto() -> None:
    """The map's `n:`/`p:` probe is only correct while the schema mints those ids."""
    assert "CASE WHEN native_id IS NULL THEN 'p:' || position || '.' || variant_index ELSE 'n:' || native_id END" in (
        INDEX_DDL
    )
    assert "COALESCE(native_id, position || '.' || variant_index)" not in INDEX_DDL


def test_an_opaque_build_ref_is_not_reported_as_a_lost_session() -> None:
    """`runtime_build_ref` holds `polylogue:0.3.0`; no Origin names that archive."""
    user, audit = _durable_tiers((NATIVE_OLD,))
    predecessor, candidate = _predecessor_index(), _candidate_index()

    transition = _plan(user, audit, candidate, predecessor)

    dispositions = {row.source: row.disposition for row in transition.plan.rows}
    assert dispositions["polylogue:0.3.0"] is ObjectRefDisposition.PRESERVED
    assert governed_target("polylogue:0.3.0") is None
    assert governed_target("codex-session:019f49d8") == ("sessions", "session_id", "codex-session:019f49d8")


def _archive_root(tmp_path: Path) -> tuple[Path, Path]:
    """A durable file set plus a candidate index generation, on disk.

    Every fixture connection commits before it is copied out: `backup` blocks
    on a source still inside a write transaction.
    """
    root = tmp_path / "archive"
    root.mkdir()
    user, audit = _durable_tiers((NATIVE_OLD, POSITIONAL_OLD))
    for connection, name in ((user, "user.db"), (audit, "audit.db")):
        target = sqlite3.connect(root / name)
        connection.backup(target)
        target.close()
    source = sqlite3.connect(root / "source.db")
    source.execute("CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY, origin TEXT, native_id TEXT)")
    source.execute("INSERT INTO raw_sessions VALUES ('r1', 'chatgpt-export', 's1')")
    source.commit()
    source.close()
    for connection, path in ((_predecessor_index(), root / "index.db"), (_candidate_index(), root / "candidate.db")):
        target = sqlite3.connect(path)
        connection.backup(target)
        target.close()
    return root, root / "candidate.db"


def test_the_maintenance_verb_plans_the_transition_against_a_real_archive(tmp_path: Path) -> None:
    """h8shq: the transition has a production caller, not just a library."""
    root, candidate = _archive_root(tmp_path)
    plan_path = tmp_path / "plan.json"

    result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "durable-reference-transition",
            "plan",
            "--archive-root",
            str(root),
            "--candidate-index",
            str(candidate),
            "--output",
            str(plan_path),
            "--output-format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    document = json.loads(plan_path.read_text(encoding="utf-8"))
    assert document["schema"] == "polylogue.durable-reference-transition-plan.v1"
    transition = document["transition"]
    assert transition["schema"] == "polylogue.durable-reference-transition.v1"
    assert transition["dispositions"][ObjectRefDisposition.EXPLICITLY_MIGRATED.value] == 2
    assert sorted(pair[1] for pair in transition["forward"]) == sorted((NATIVE_NEW, POSITIONAL_NEW))


def test_apply_refuses_a_plan_it_was_not_given_the_digest_for(tmp_path: Path) -> None:
    root, candidate = _archive_root(tmp_path)
    plan_path = tmp_path / "plan.json"
    runner = CliRunner()
    assert (
        runner.invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "durable-reference-transition",
                "plan",
                "--archive-root",
                str(root),
                "--candidate-index",
                str(candidate),
                "--output",
                str(plan_path),
            ],
        ).exit_code
        == 0
    )

    result = runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "durable-reference-transition",
            "apply",
            "--archive-root",
            str(root),
            "--plan",
            str(plan_path),
            "--authorize",
            "0" * 64,
            "--backup-manifest",
            str(plan_path),
        ],
    )

    assert result.exit_code != 0
    assert "authorization does not match" in result.output

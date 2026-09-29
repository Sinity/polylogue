"""Laws for what a query excision removes, scrubs, and refuses."""

from __future__ import annotations

import sqlite3

import pytest

from polylogue.security.query_excision import apply_query_excision, plan_query_excision
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.holdout_cohorts import mark_holdout, record_holdout_access
from polylogue.storage.sqlite.query_objects import (
    QueryObject,
    ResultSetManifest,
    get_query,
    get_result_set,
    put_query,
    put_query_name,
    put_result_set,
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.USER)
    return conn


def _query_with_relation(
    conn: sqlite3.Connection, *, result_set_id: str = "relation-only"
) -> tuple[QueryObject, ResultSetManifest]:
    query = put_query(
        conn,
        {"field": "origin", "value": "codex"},
        grain="session",
        lane="dialogue",
        rank_policy="mixed",
        created_at_ms=1,
    )
    relation = put_result_set(
        conn,
        result_set_id=result_set_id,
        query_hash=query.query_hash,
        grain="session",
        corpus_epoch="e1",
        member_refs=("session:one",),
        exactness="exact",
        persistence_class="pinned",
        created_at_ms=2,
    )
    return query, relation


def _insert_assertion(conn: sqlite3.Connection, assertion_id: str, target_ref: str) -> None:
    conn.execute(
        """
        INSERT INTO assertions (
            assertion_id, scope_ref, target_ref, key, kind, value_json, body_text,
            staleness_json, supersedes_json, evidence_refs_json, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, 'note', ?, ?, ?, ?, ?, 1, 1)
        """,
        (
            assertion_id,
            "repo:/home/operator/secret-project",
            target_ref,
            "api-key=sk-secret",
            '{"claim":"secret"}',
            "secret body",
            '{"observed":"secret"}',
            '["assertion:secret-predecessor"]',
            '["blob:sha256:secret"]',
        ),
    )


def test_relation_excision_preserves_owning_query_assertions() -> None:
    """Excising one snapshot must not destroy notes on its surviving query.

    Anti-vacuity: scanning the owning query's refs as well puts
    `note-on-query` into `finding_report_refs`, so the apply marks it deleted
    and clears its content -- irreplaceable user-tier state.
    """
    conn = _conn()
    query, relation = _query_with_relation(conn)
    put_query_name(conn, name="codex", query_hash=query.query_hash, watch=False, updated_at_ms=2)
    _insert_assertion(conn, "note-on-query", f"query:{query.query_hash}")
    _insert_assertion(conn, "note-on-relation", f"result-set:{relation.result_set_id}")

    plan = plan_query_excision(conn, f"result-set:{relation.result_set_id}")
    assert plan.finding_report_refs == ("note-on-relation",)
    # A relation-only plan reports only what it removes.
    assert plan.names == ()
    assert plan.edge_refs == ()

    apply_query_excision(conn, plan, reason="remove relation", actor="user:test", now_ms=3)

    assert get_query(conn, query.query_hash) is not None
    assert get_result_set(conn, relation.result_set_id) is None
    surviving = conn.execute("SELECT status, body_text FROM assertions WHERE assertion_id = 'note-on-query'").fetchone()
    assert tuple(surviving) == ("active", "secret body")


def test_query_excision_still_reaches_its_own_relations() -> None:
    """The opposite direction: a `query:` target keeps its full reach."""
    conn = _conn()
    query, relation = _query_with_relation(conn)
    put_query_name(conn, name="codex", query_hash=query.query_hash, watch=False, updated_at_ms=2)
    _insert_assertion(conn, "note-on-query", f"query:{query.query_hash}")
    _insert_assertion(conn, "note-on-relation", f"result-set:{relation.result_set_id}")

    plan = plan_query_excision(conn, query.ref)
    assert set(plan.finding_report_refs) == {"note-on-query", "note-on-relation"}
    assert plan.names == ("codex",)

    apply_query_excision(conn, plan, reason="secret", actor="user:test", now_ms=3)
    assert get_query(conn, query.query_hash) is None
    assert get_result_set(conn, relation.result_set_id) is None


def test_excision_scrubs_every_content_bearing_assertion_field() -> None:
    """A scrubbed assertion keeps no readable operator content.

    Anti-vacuity: clearing only `value_json` and `body_text` leaves the secret
    in `key`, the path in `scope_ref`, and operator JSON in `staleness_json`,
    `supersedes_json`, `evidence_refs_json`, and caller-controlled authorship.
    """
    conn = _conn()
    query, _relation = _query_with_relation(conn)
    _insert_assertion(conn, "note-on-query", f"query:{query.query_hash}")

    plan = plan_query_excision(conn, query.ref)
    apply_query_excision(conn, plan, reason="secret", actor="user:test", now_ms=9)

    row = conn.execute(
        """
        SELECT status, key, value_json, body_text, scope_ref, staleness_json,
               supersedes_json, evidence_refs_json, context_policy_json, visibility, author_ref, author_kind
        FROM assertions WHERE assertion_id = 'note-on-query'
        """
    ).fetchone()
    (
        status,
        key,
        value_json,
        body_text,
        scope_ref,
        staleness,
        supersedes,
        evidence,
        policy,
        visibility,
        author,
        author_kind,
    ) = row
    assert status == "deleted"
    assert (key, value_json, body_text, scope_ref, staleness) == (None, None, None, None, None)
    assert (supersedes, evidence) == ("[]", "[]")
    assert policy == '{"inject":false}'
    assert visibility == "private"
    # The operation ledger retains accountability; caller-controlled author
    # identity must not survive in the scrubbed assertion itself.
    assert (author, author_kind) == ("system:excision", "system")


def test_accessed_holdout_is_held_instead_of_failing_mid_transaction() -> None:
    """An accessed holdout refuses before any row is mutated.

    Anti-vacuity: without the plan-time classification, the apply runs the
    ledger insert and the assertion scrub and then raises
    `sqlite3.IntegrityError` on the `ON DELETE RESTRICT` policy delete.
    """
    conn = _conn()
    _query, relation = _query_with_relation(conn, result_set_id="holdout-relation")
    mark_holdout(
        conn,
        result_set_id=relation.result_set_id,
        frame="frame",
        selection_definition={"field": "origin"},
        intended_confirmation_use="evaluation",
        authority="user:test",
        created_epoch="e1",
        created_at_ms=2,
    )
    record_holdout_access(
        conn,
        receipt_id="holdout-read-1",
        result_set_id=relation.result_set_id,
        accessor_ref="user:test",
        declared_confirmation=True,
        accessed_at_ms=3,
    )
    _insert_assertion(conn, "note-on-relation", f"result-set:{relation.result_set_id}")

    plan = plan_query_excision(conn, f"result-set:{relation.result_set_id}")
    assert plan.held_refs == ("holdout-access-receipt:holdout-read-1",)

    receipt = apply_query_excision(conn, plan, reason="privacy", actor="user:test", now_ms=4)

    assert receipt.status == "held"
    assert receipt.held_refs == ("holdout-access-receipt:holdout-read-1",)
    assert get_result_set(conn, relation.result_set_id) is not None
    assert conn.execute("SELECT COUNT(*) FROM query_excision_ledger").fetchone()[0] == 0
    assert conn.execute("SELECT status FROM assertions WHERE assertion_id = 'note-on-relation'").fetchone()[0] == (
        "active"
    )


def test_unaccessed_holdout_is_still_excisable() -> None:
    """The opposite direction: a holdout nobody read is not blocked."""
    conn = _conn()
    _query, relation = _query_with_relation(conn, result_set_id="clean-holdout")
    mark_holdout(
        conn,
        result_set_id=relation.result_set_id,
        frame="frame",
        selection_definition={"field": "origin"},
        intended_confirmation_use="evaluation",
        authority="user:test",
        created_epoch="e1",
        created_at_ms=2,
    )

    plan = plan_query_excision(conn, f"result-set:{relation.result_set_id}")
    assert plan.held_refs == ()
    receipt = apply_query_excision(conn, plan, reason="privacy", actor="user:test", now_ms=4)

    assert receipt.status == "applied"
    assert get_result_set(conn, relation.result_set_id) is None


def test_graph_drift_between_plan_and_apply_is_refused() -> None:
    """A citing assertion added after planning holds the apply.

    Anti-vacuity: trusting the stale plan returns `applied` while
    `late-finding`'s content stays readable behind a completed excision.
    """
    conn = _conn()
    query, _relation = _query_with_relation(conn)
    plan = plan_query_excision(conn, query.ref)
    _insert_assertion(conn, "late-finding", f"query:{query.query_hash}")

    receipt = apply_query_excision(conn, plan, reason="secret", actor="user:test", now_ms=5)

    assert receipt.status == "held"
    assert receipt.held_refs == ("late-finding",)
    assert get_query(conn, query.query_hash) is not None
    assert conn.execute("SELECT body_text FROM assertions WHERE assertion_id = 'late-finding'").fetchone()[0] == (
        "secret body"
    )


def test_an_unchanged_graph_still_applies() -> None:
    """The opposite direction: re-resolution is not a blanket refusal."""
    conn = _conn()
    query, relation = _query_with_relation(conn)
    plan = plan_query_excision(conn, query.ref)

    receipt = apply_query_excision(conn, plan, reason="secret", actor="user:test", now_ms=5)

    assert receipt.status == "applied"
    assert get_query(conn, query.query_hash) is None
    assert get_result_set(conn, relation.result_set_id) is None


def test_relation_excision_disables_only_watches_whose_baseline_was_removed() -> None:
    """A watch whose baseline snapshot is excised stops being evaluated.

    Anti-vacuity: delete only the baseline row and both watched names keep
    ``watch=1``, so the standing-query stage recreates the tombstoned result
    set on every convergence pass.
    """
    from polylogue.storage.sqlite.query_objects import put_watched_query_baseline

    conn = _conn()
    query, _relation = _query_with_relation(conn)
    watched = put_result_set(
        conn,
        result_set_id="watch-baseline",
        query_hash=query.query_hash,
        grain="session",
        corpus_epoch="e1",
        member_refs=("session:one",),
        exactness="exact",
        persistence_class="watch",
        created_at_ms=2,
    )
    put_query_name(conn, name="watch-a", query_hash=query.query_hash, watch=True, updated_at_ms=2)
    put_query_name(conn, name="watch-b", query_hash=query.query_hash, watch=True, updated_at_ms=2)
    put_watched_query_baseline(conn, query_hash=query.query_hash, result_set_id=watched.result_set_id, updated_at_ms=2)
    other = put_query(
        conn,
        {"field": "origin", "value": "chatgpt-export"},
        grain="session",
        lane="dialogue",
        rank_policy="mixed",
        created_at_ms=2,
    )
    put_query_name(conn, name="unrelated", query_hash=other.query_hash, watch=True, updated_at_ms=2)
    receipt = apply_query_excision(
        conn,
        plan_query_excision(conn, "result-set:watch-baseline"),
        reason="remove baseline",
        actor="user:test",
        now_ms=3,
    )
    assert receipt.status == "applied"
    assert dict(conn.execute("SELECT name, watch FROM query_names")) == {"watch-a": 0, "watch-b": 0, "unrelated": 1}
    assert get_query(conn, query.query_hash) is not None
    assert conn.execute("SELECT COUNT(*) FROM watched_query_baselines").fetchone()[0] == 0


@pytest.mark.parametrize("target_kind", ["query", "result-set"])
@pytest.mark.parametrize(("actor", "now_ms"), [(" ", 3), ("user:test", -1)])
def test_a_refused_tombstone_aborts_before_any_deletion(target_kind: str, actor: str, now_ms: int) -> None:
    """A ledger row the schema refuses stops the excision before content goes.

    Anti-vacuity: with ``INSERT OR IGNORE`` the CHECK failure inserts nothing,
    the query and result set are deleted anyway, and an applied receipt comes
    back with no tombstone to stop their recreation.
    """
    conn = _conn()
    query, relation = _query_with_relation(conn)
    _insert_assertion(conn, "keep-note", query.ref)
    target = query.ref if target_kind == "query" else f"result-set:{relation.result_set_id}"
    with pytest.raises(sqlite3.IntegrityError):
        apply_query_excision(conn, plan_query_excision(conn, target), reason="privacy", actor=actor, now_ms=now_ms)
    assert get_query(conn, query.query_hash) is not None
    assert get_result_set(conn, relation.result_set_id) is not None
    assert conn.execute("SELECT COUNT(*) FROM query_excision_ledger").fetchone()[0] == 0
    assert conn.execute("SELECT status FROM assertions WHERE assertion_id='keep-note'").fetchone()[0] == "active"

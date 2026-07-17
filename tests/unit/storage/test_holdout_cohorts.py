from __future__ import annotations

import sqlite3

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.holdout_cohorts import (
    HoldoutAccessError,
    get_holdout_policy,
    has_holdout_contamination,
    is_holdout,
    list_holdout_access_receipts,
    mark_holdout,
    record_holdout_access,
    require_non_holdout_access,
)
from polylogue.storage.sqlite.query_objects import get_result_set, put_query, put_result_set


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.USER)
    return conn


def _seeded_result_set(conn: sqlite3.Connection, *, result_set_id: str = "rs-holdout") -> str:
    query = put_query(
        conn,
        {"field": "origin", "value": "codex-session"},
        grain="session",
        lane="dialogue",
        rank_policy="mixed",
        created_at_ms=1,
    )
    put_result_set(
        conn,
        result_set_id=result_set_id,
        query_hash=query.query_hash,
        grain="session",
        corpus_epoch="epoch-1",
        member_refs=("session:a", "session:b"),
        exactness="exact",
        persistence_class="cohort",
        created_at_ms=2,
    )
    return result_set_id


def test_not_marked_result_set_is_not_a_holdout_and_reads_are_unrestricted() -> None:
    conn = _conn()
    result_set_id = _seeded_result_set(conn)

    assert is_holdout(conn, result_set_id) is False
    require_non_holdout_access(conn, result_set_id, declared_confirmation=False)  # no raise


def test_mark_holdout_requires_an_existing_result_set() -> None:
    conn = _conn()

    with pytest.raises(KeyError):
        mark_holdout(
            conn,
            result_set_id="does-not-exist",
            frame="frame-a",
            selection_definition={"kind": "seeded"},
            intended_confirmation_use="precision@k evaluation",
            authority="operator",
            created_epoch="epoch-1",
            created_at_ms=10,
        )


def test_exploratory_query_cannot_read_a_seeded_holdout() -> None:
    conn = _conn()
    result_set_id = _seeded_result_set(conn)
    mark_holdout(
        conn,
        result_set_id=result_set_id,
        frame="frame-a",
        selection_definition={"kind": "seeded"},
        intended_confirmation_use="precision@k evaluation",
        authority="operator",
        created_epoch="epoch-1",
        created_at_ms=10,
    )

    assert is_holdout(conn, result_set_id) is True
    with pytest.raises(HoldoutAccessError, match="exploratory queries cannot read"):
        require_non_holdout_access(conn, result_set_id, declared_confirmation=False)


def test_declared_confirmation_run_can_access_with_a_visible_receipt() -> None:
    conn = _conn()
    result_set_id = _seeded_result_set(conn)
    mark_holdout(
        conn,
        result_set_id=result_set_id,
        frame="frame-a",
        selection_definition={"kind": "seeded"},
        intended_confirmation_use="precision@k evaluation",
        authority="operator",
        created_epoch="epoch-1",
        created_at_ms=10,
    )

    require_non_holdout_access(conn, result_set_id, declared_confirmation=True)  # no raise
    receipt = record_holdout_access(
        conn,
        receipt_id="receipt-1",
        result_set_id=result_set_id,
        accessor_ref="agent:confirmation-run",
        declared_confirmation=True,
        accessed_at_ms=20,
    )

    assert receipt.contamination is False
    receipts = list_holdout_access_receipts(conn, result_set_id)
    assert len(receipts) == 1
    assert receipts[0].receipt_id == "receipt-1"
    assert has_holdout_contamination(conn, result_set_id) is False


def test_undeclared_access_marks_permanent_contamination() -> None:
    conn = _conn()
    result_set_id = _seeded_result_set(conn)
    mark_holdout(
        conn,
        result_set_id=result_set_id,
        frame="frame-a",
        selection_definition={"kind": "seeded"},
        intended_confirmation_use="precision@k evaluation",
        authority="operator",
        created_epoch="epoch-1",
        created_at_ms=10,
    )

    receipt = record_holdout_access(
        conn,
        receipt_id="receipt-accident",
        result_set_id=result_set_id,
        accessor_ref="agent:exploratory-slip",
        declared_confirmation=False,
        accessed_at_ms=30,
        reason="unauthorized exploratory read",
    )

    assert receipt.contamination is True
    assert has_holdout_contamination(conn, result_set_id) is True

    # A later declared confirmation access does not retroactively clear it.
    record_holdout_access(
        conn,
        receipt_id="receipt-confirmation",
        result_set_id=result_set_id,
        accessor_ref="agent:confirmation-run",
        declared_confirmation=True,
        accessed_at_ms=40,
    )
    assert has_holdout_contamination(conn, result_set_id) is True
    assert len(list_holdout_access_receipts(conn, result_set_id)) == 2


def test_record_access_requires_the_result_set_to_be_a_holdout() -> None:
    conn = _conn()
    result_set_id = _seeded_result_set(conn)

    with pytest.raises(KeyError):
        record_holdout_access(
            conn,
            receipt_id="receipt-1",
            result_set_id=result_set_id,
            accessor_ref="agent:x",
            declared_confirmation=True,
            accessed_at_ms=1,
        )


def test_remarking_with_identical_policy_is_idempotent() -> None:
    conn = _conn()
    result_set_id = _seeded_result_set(conn)
    kwargs: dict[str, object] = {
        "result_set_id": result_set_id,
        "frame": "frame-a",
        "selection_definition": {"kind": "seeded"},
        "intended_confirmation_use": "precision@k evaluation",
        "authority": "operator",
        "created_epoch": "epoch-1",
        "created_at_ms": 10,
    }

    first = mark_holdout(conn, **kwargs)  # type: ignore[arg-type]
    second = mark_holdout(conn, **kwargs)  # type: ignore[arg-type]

    assert first == second


def test_remarking_with_a_different_policy_raises() -> None:
    conn = _conn()
    result_set_id = _seeded_result_set(conn)
    mark_holdout(
        conn,
        result_set_id=result_set_id,
        frame="frame-a",
        selection_definition={"kind": "seeded"},
        intended_confirmation_use="precision@k evaluation",
        authority="operator",
        created_epoch="epoch-1",
        created_at_ms=10,
    )

    with pytest.raises(ValueError, match="different declared policy"):
        mark_holdout(
            conn,
            result_set_id=result_set_id,
            frame="frame-b",
            selection_definition={"kind": "seeded"},
            intended_confirmation_use="precision@k evaluation",
            authority="operator",
            created_epoch="epoch-1",
            created_at_ms=10,
        )


def test_deleting_a_holdout_marked_result_set_is_blocked_by_the_durable_fk() -> None:
    """rxdo.9.4 AC: 'reset/excision preserve the declared durability
    semantics.' No excision mechanism exists for result_sets yet (see
    migration 009's ON DELETE RESTRICT), so the durable floor this PR ships
    is DB-level: a holdout-marked result set cannot be deleted out from
    under its policy row. A future excision mechanism must route through an
    explicit unmark-then-delete step rather than a raw DELETE, or it will
    hit this same IntegrityError."""

    conn = _conn()
    result_set_id = _seeded_result_set(conn)
    mark_holdout(
        conn,
        result_set_id=result_set_id,
        frame="frame-a",
        selection_definition={"kind": "seeded"},
        intended_confirmation_use="precision@k evaluation",
        authority="operator",
        created_epoch="epoch-1",
        created_at_ms=10,
    )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM result_sets WHERE result_set_id = ?", (result_set_id,))

    # Contrast: a non-holdout result set has no such protection -- the
    # restriction is specific to the holdout policy, not a blanket
    # disallow-all-deletes on result_sets.
    other_id = _seeded_result_set(conn, result_set_id="rs-no-holdout")
    conn.execute("DELETE FROM result_sets WHERE result_set_id = ?", (other_id,))
    assert get_result_set(conn, other_id) is None


def test_cohort_and_holdout_relation_identities_remain_distinct_while_sharing_the_manifest() -> None:
    """rxdo.9.4 design: holdout is a policy layered on the existing
    result_sets manifest, not a second relation type -- the cohort keeps its
    own persistence_class identity even after being marked as a holdout."""

    conn = _conn()
    result_set_id = _seeded_result_set(conn, result_set_id="rs-cohort-holdout")

    mark_holdout(
        conn,
        result_set_id=result_set_id,
        frame="frame-a",
        selection_definition={"kind": "seeded"},
        intended_confirmation_use="precision@k evaluation",
        authority="operator",
        created_epoch="epoch-1",
        created_at_ms=10,
    )

    row = conn.execute("SELECT persistence_class FROM result_sets WHERE result_set_id = ?", (result_set_id,)).fetchone()
    assert row is not None
    assert row[0] == "cohort"
    assert get_holdout_policy(conn, result_set_id) is not None

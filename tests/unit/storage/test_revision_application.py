from __future__ import annotations

import sqlite3
from dataclasses import replace
from typing import Any

import pytest

from polylogue.archive.revision_replay import ApplicationDecision
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
from polylogue.storage.sqlite.archive_tiers.revision_application import (
    FullSnapshotFoldAuthorization,
    RevisionApplicationReceipt,
    assert_session_fts_exact_sync,
    record_revision_application_sync,
)


def _receipt(
    *, generation: int = 1, revision: str = "revision-1", frontier: int | None = None
) -> RevisionApplicationReceipt:
    return RevisionApplicationReceipt(
        raw_id=f"raw-{generation}",
        session_id="codex-session:session",
        logical_source_key="codex:session",
        source_revision=revision,
        acquisition_generation=generation,
        decision=ApplicationDecision.SELECTED_BASELINE,
        accepted_raw_id=f"raw-{generation}",
        accepted_source_revision=revision,
        accepted_content_hash=bytes([generation]) * 32,
        accepted_frontier_kind="byte",
        accepted_frontier=frontier if frontier is not None else generation * 100,
    )


def _application_rows(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
    return conn.execute(
        """
        SELECT decision_id, raw_id, session_id, logical_source_key, source_revision,
               acquisition_generation, decision, accepted_raw_id,
               accepted_source_revision, accepted_content_hash,
               accepted_frontier_kind, accepted_frontier, baseline_raw_id,
               predecessor_raw_id, append_end_offset, detail, decided_at_ms
        FROM raw_revision_applications ORDER BY decision_id
        """
    ).fetchall()


def _head_rows(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
    return conn.execute(
        """
        SELECT logical_source_key, session_id, accepted_raw_id,
               accepted_source_revision, accepted_content_hash,
               accepted_frontier_kind, accepted_frontier,
               acquisition_generation, append_end_offset, decided_at_ms
        FROM raw_revision_heads ORDER BY logical_source_key
        """
    ).fetchall()


def test_application_receipt_is_idempotent_and_rejects_older_head() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    receipt = _receipt()
    record_revision_application_sync(conn, receipt, decided_at_ms=10)
    record_revision_application_sync(conn, receipt, decided_at_ms=20)
    assert conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone()[0] == 1
    assert conn.execute(
        "SELECT accepted_source_revision FROM raw_revision_heads WHERE logical_source_key = 'codex:session'"
    ).fetchone() == ("revision-1",)
    assert conn.execute(
        "SELECT accepted_frontier_kind, accepted_frontier FROM raw_revision_applications"
    ).fetchone() == (receipt.accepted_frontier_kind, receipt.accepted_frontier)

    record_revision_application_sync(conn, _receipt(generation=2, revision="revision-2"), decided_at_ms=30)
    with pytest.raises(RuntimeError, match="older accepted frontier"):
        record_revision_application_sync(conn, _receipt(generation=1, revision="revision-old"), decided_at_ms=40)


def test_equivalent_accepted_raw_reuses_semantic_receipt_identity() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    receipt = replace(
        _receipt(),
        decision=ApplicationDecision.SUPERSEDED,
        accepted_raw_id="representative-b",
        accepted_source_revision="semantic-revision",
    )
    record_revision_application_sync(conn, receipt, decided_at_ms=10)

    equivalent_representative = replace(receipt, accepted_raw_id="representative-a")
    assert equivalent_representative.decision_id != receipt.decision_id
    record_revision_application_sync(conn, equivalent_representative, decided_at_ms=20)

    assert conn.execute(
        "SELECT accepted_raw_id, accepted_source_revision, accepted_content_hash FROM raw_revision_applications"
    ).fetchall() == [("representative-b", "semantic-revision", receipt.accepted_content_hash)]


def test_session_fts_proof_detects_missing_row_and_trigger_mutation() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    conn.execute(
        "INSERT INTO sessions(native_id, origin, content_hash) VALUES ('session', 'codex-session', ?)",
        (b"s" * 32,),
    )
    conn.execute(
        """
        INSERT INTO messages(session_id, position, role, material_origin, content_hash)
        VALUES ('codex-session:session', 0, 'user', 'human_authored', ?)
        """,
        (b"m" * 32,),
    )
    conn.execute(
        """
        INSERT INTO blocks(message_id, session_id, position, block_type, text)
        VALUES ('codex-session:session:0.0', 'codex-session:session', 0, 'text', 'proof')
        """
    )
    assert_session_fts_exact_sync(conn, "codex-session:session")

    conn.execute("DELETE FROM messages_fts")
    with pytest.raises(RuntimeError, match="FTS proof failed"):
        assert_session_fts_exact_sync(conn, "codex-session:session")
    conn.execute("DROP TRIGGER messages_fts_ai")
    with pytest.raises(RuntimeError, match="canonical message FTS triggers"):
        assert_session_fts_exact_sync(conn, "codex-session:session")


@pytest.mark.parametrize(
    "changes",
    [
        {"accepted_frontier": 99},
        {"accepted_frontier_kind": "semantic"},
    ],
)
def test_equal_frontier_cas_rejects_conflicting_frontier_state(changes: dict[str, Any]) -> None:
    """(e) a genuinely older or incomparable frontier is still rejected regardless of raw identity."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    receipt = _receipt()
    record_revision_application_sync(conn, receipt, decided_at_ms=10)
    original_application = conn.execute(
        "SELECT accepted_frontier_kind, accepted_frontier FROM raw_revision_applications"
    ).fetchone()
    original_head = conn.execute(
        "SELECT accepted_raw_id, accepted_frontier_kind, accepted_frontier FROM raw_revision_heads"
    ).fetchone()

    with pytest.raises(RuntimeError, match="conflicting|rejected"):
        record_revision_application_sync(conn, replace(receipt, **changes), decided_at_ms=20)

    assert (
        conn.execute("SELECT accepted_frontier_kind, accepted_frontier FROM raw_revision_applications").fetchone()
        == original_application
        == ("byte", 100)
    )
    assert (
        conn.execute(
            "SELECT accepted_raw_id, accepted_frontier_kind, accepted_frontier FROM raw_revision_heads"
        ).fetchone()
        == original_head
        == (receipt.accepted_raw_id, "byte", 100)
    )


def test_decision_id_binds_accepted_frontier_evidence() -> None:
    receipt = _receipt()

    for changed in (
        replace(receipt, accepted_content_hash=b"x" * 32),
        replace(receipt, accepted_source_revision="revision-2"),
        replace(receipt, accepted_frontier_kind="semantic"),
        replace(receipt, accepted_frontier=200),
        replace(receipt, acquisition_generation=2),
        replace(receipt, append_end_offset=101),
        replace(receipt, raw_id="raw-other"),
        replace(receipt, session_id="codex-session:other"),
        replace(receipt, decision=ApplicationDecision.APPLIED_APPEND),
    ):
        assert receipt.decision_id != changed.decision_id


@pytest.mark.parametrize(
    "changes",
    [
        {"accepted_frontier": None},
        {"accepted_raw_id": None},
    ],
)
def test_application_receipt_rejects_uncoupled_accepted_identity_and_frontier(changes: dict[str, Any]) -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)

    with pytest.raises(ValueError, match="present or all absent|present or absent together"):
        record_revision_application_sync(conn, replace(_receipt(), **changes), decided_at_ms=10)

    assert conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone() == (0,)
    assert conn.execute("SELECT COUNT(*) FROM raw_revision_heads").fetchone() == (0,)


def test_persisted_higher_frontier_twin_under_same_id_cannot_rewrite_receipt_or_head() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    receipt = _receipt()
    record_revision_application_sync(conn, receipt, decided_at_ms=10)
    higher_frontier = 200
    conn.execute(
        "UPDATE raw_revision_applications SET accepted_frontier = ? WHERE decision_id = ?",
        (higher_frontier, receipt.decision_id),
    )
    conn.commit()

    with pytest.raises(RuntimeError, match="immutable evidence"):
        record_revision_application_sync(conn, receipt, decided_at_ms=20)

    assert conn.execute(
        "SELECT accepted_frontier_kind, accepted_frontier FROM raw_revision_applications"
    ).fetchone() == ("byte", higher_frontier)
    assert conn.execute(
        "SELECT accepted_raw_id, accepted_frontier_kind, accepted_frontier FROM raw_revision_heads"
    ).fetchone() == (receipt.accepted_raw_id, "byte", receipt.accepted_frontier)


@pytest.mark.parametrize(
    "changes",
    [
        {"accepted_content_hash": b"x" * 32},
        {"session_id": "codex-session:renamed"},
    ],
)
def test_equal_frontier_cas_allows_same_raw_supersede(changes: dict[str, Any]) -> None:
    """(a)+(b) same accepted_raw_id, equal frontier, differing derived semantics.

    A parser fix (differing content hash) or an identity-law fix (differing
    session_id) re-derives from the exact same accepted raw evidence -- the
    blob is content-addressed and unchanged, only the derivation changed. This
    must supersede the existing head rather than raise: blocking it would make
    every parser/identity improvement poison already-committed archives.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    receipt = _receipt()
    record_revision_application_sync(conn, receipt, decided_at_ms=10)

    reapplied = replace(receipt, **changes)
    assert reapplied.decision_id != receipt.decision_id
    record_revision_application_sync(conn, reapplied, decided_at_ms=20)

    assert conn.execute("SELECT COUNT(*) FROM raw_revision_applications").fetchone() == (2,)

    row = conn.execute(
        """
        SELECT session_id, accepted_raw_id, accepted_content_hash
        FROM raw_revision_heads WHERE logical_source_key = 'codex:session'
        """
    ).fetchone()
    assert row == (reapplied.session_id, reapplied.accepted_raw_id, reapplied.accepted_content_hash)


def test_equal_frontier_cas_rejects_cross_raw_conflict_without_fold_authorization() -> None:
    """(c) a different accepted_raw_id at the same frontier is a genuine conflict and still rejects.

    The error message must name both the existing and incoming accepted_raw_id
    so a rebuild failure is diagnosable without re-running under a debugger.
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    receipt = _receipt()
    record_revision_application_sync(conn, receipt, decided_at_ms=10)
    before_applications = _application_rows(conn)
    before_heads = _head_rows(conn)

    conflicting = replace(
        receipt,
        raw_id="raw-conflict",
        accepted_raw_id="raw-conflict",
        accepted_content_hash=b"z" * 32,
    )
    with pytest.raises(RuntimeError, match="conflicting accepted head") as excinfo:
        record_revision_application_sync(conn, conflicting, decided_at_ms=20)
    message = str(excinfo.value)
    assert receipt.accepted_raw_id is not None
    assert receipt.accepted_raw_id in message
    assert "raw-conflict" in message
    assert _application_rows(conn) == before_applications
    assert _head_rows(conn) == before_heads


def test_equal_frontier_cas_idempotent_same_everything_reapplication() -> None:
    """(d) re-applying the exact same receipt at an equal frontier is a clean no-op."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    receipt = _receipt()
    record_revision_application_sync(conn, receipt, decided_at_ms=10)
    before_applications = _application_rows(conn)
    before_heads = _head_rows(conn)
    record_revision_application_sync(conn, receipt, decided_at_ms=20)

    assert _application_rows(conn) == before_applications
    assert _head_rows(conn) == before_heads


def test_larger_full_snapshot_supersedes_deeper_append_generation() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    record_revision_application_sync(conn, _receipt(generation=0, revision="full-0", frontier=100), decided_at_ms=10)
    record_revision_application_sync(conn, _receipt(generation=2, revision="append-2", frontier=180), decided_at_ms=20)
    later_full = _receipt(generation=1, revision="full-1", frontier=220)
    record_revision_application_sync(conn, later_full, decided_at_ms=30)
    assert conn.execute("SELECT accepted_source_revision, accepted_frontier FROM raw_revision_heads").fetchone() == (
        "full-1",
        220,
    )


def test_equal_frontier_full_can_replace_equivalent_append_representation() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    full_zero = _receipt(generation=0, revision="full-0", frontier=100)
    record_revision_application_sync(conn, full_zero, decided_at_ms=10)
    append_one = replace(
        _receipt(generation=1, revision="append-1", frontier=140),
        decision=ApplicationDecision.APPLIED_APPEND,
        append_end_offset=140,
    )
    record_revision_application_sync(conn, append_one, decided_at_ms=15)
    append_head = replace(
        _receipt(generation=2, revision="append-2", frontier=180),
        decision=ApplicationDecision.APPLIED_APPEND,
        append_end_offset=180,
    )
    record_revision_application_sync(conn, append_head, decided_at_ms=20)
    full_head = replace(
        _receipt(generation=1, revision="full-1", frontier=180),
        accepted_content_hash=append_head.accepted_content_hash,
    )
    record_revision_application_sync(conn, full_head, decided_at_ms=30)
    assert conn.execute("SELECT accepted_source_revision, append_end_offset FROM raw_revision_heads").fetchone() == (
        "full-1",
        None,
    )


def test_equal_frontier_fold_authorization_is_bound_to_one_exact_head() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    append_head = replace(
        _receipt(generation=2, revision="append-2", frontier=180),
        decision=ApplicationDecision.APPLIED_APPEND,
        append_end_offset=180,
    )
    record_revision_application_sync(conn, append_head, decided_at_ms=10)
    authorization = FullSnapshotFoldAuthorization(
        logical_source_key="codex:session",
        session_id="codex-session:session",
        accepted_append_raw_id="raw-2",
        accepted_append_source_revision="append-2",
        accepted_append_content_hash=bytes([2]) * 32,
        frontier=180,
        full_raw_id="full",
        full_source_revision="full-revision",
    )
    full = replace(
        _receipt(generation=3, revision="full-revision", frontier=180),
        raw_id="full",
        accepted_raw_id="full",
        accepted_source_revision="full-revision",
        accepted_content_hash=b"different-normalized-hash-123456"[:32],
        append_end_offset=None,
        fold_authorization=authorization,
    )
    record_revision_application_sync(conn, full, decided_at_ms=20)
    assert conn.execute("SELECT accepted_raw_id FROM raw_revision_heads").fetchone() == ("full",)
    with pytest.raises(RuntimeError, match="conflicting accepted head"):
        record_revision_application_sync(
            conn,
            replace(
                full,
                raw_id="other",
                accepted_raw_id="other",
                accepted_content_hash=b"other-normalized-content-hash-123"[:32],
            ),
            decided_at_ms=30,
        )

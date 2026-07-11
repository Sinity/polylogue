from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from polylogue.archive.revision_replay import ApplicationDecision
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
from polylogue.storage.sqlite.archive_tiers.revision_application import (
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

    record_revision_application_sync(conn, _receipt(generation=2, revision="revision-2"), decided_at_ms=30)
    with pytest.raises(RuntimeError, match="older accepted frontier"):
        record_revision_application_sync(conn, _receipt(generation=1, revision="revision-old"), decided_at_ms=40)


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
        {"session_id": "codex-session:other"},
        {"accepted_content_hash": b"x" * 32},
        {"accepted_frontier": 99},
        {"accepted_frontier_kind": "semantic"},
    ],
)
def test_equal_frontier_cas_rejects_conflicting_semantic_state(changes: dict[str, object]) -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    receipt = _receipt()
    record_revision_application_sync(conn, receipt, decided_at_ms=10)

    with pytest.raises(RuntimeError, match="conflicting|rejected"):
        record_revision_application_sync(conn, replace(receipt, **changes), decided_at_ms=20)


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
    append_head = replace(
        _receipt(generation=2, revision="append-2", frontier=180),
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

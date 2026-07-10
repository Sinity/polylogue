from __future__ import annotations

import sqlite3

import pytest

from polylogue.archive.revision_replay import ApplicationDecision
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
from polylogue.storage.sqlite.archive_tiers.revision_application import (
    RevisionApplicationReceipt,
    assert_session_fts_exact_sync,
    record_revision_application_sync,
)


def _receipt(*, generation: int = 1, revision: str = "revision-1") -> RevisionApplicationReceipt:
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
    with pytest.raises(RuntimeError, match="older accepted generation"):
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

"""Durable CRUD contract for sinex_publication_obligations.

Uses a real source.db bootstrapped by ArchiveStore (via ``workspace_env``),
not a synthetic in-memory schema -- these tests fail if the migration/DDL
drifts from what ``obligations.py`` actually reads/writes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.sinex.models import ObligationStatus, PublicationMode, ReceiptState
from polylogue.sinex.obligations import get_obligation, list_obligations, mark_attempt, record_obligation


def _conn(source_db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(source_db_path)
    conn.row_factory = sqlite3.Row
    return conn


def test_record_obligation_is_idempotent_by_the_four_part_key(workspace_env: dict[str, Path]) -> None:
    source_db = workspace_env["archive_root"] / "source.db"
    conn = _conn(source_db)
    try:
        first = record_obligation(
            conn,
            object_id="claude-code-session:s1",
            protocol_version="polylogue.material-protocol/v1",
            revision_id="rev-1",
            manifest_digest="digest-1",
            mode=PublicationMode.MIRROR,
            now_ms=1000,
        )
        second = record_obligation(
            conn,
            object_id="claude-code-session:s1",
            protocol_version="polylogue.material-protocol/v1",
            revision_id="rev-1",
            manifest_digest="digest-1",
            mode=PublicationMode.MIRROR,
            now_ms=9999,  # a later retry must NOT overwrite created_at_ms
        )
        conn.commit()
        assert first == second
        assert first.created_at_ms == 1000
        assert first.status is ObligationStatus.PENDING
        assert first.attempt_count == 0
        rows = conn.execute("SELECT COUNT(*) FROM sinex_publication_obligations").fetchone()[0]
        assert rows == 1

        # A different revision_id is a genuinely new obligation, not merged
        # with the first -- proves the idempotency key is the full 4-tuple,
        # not just object_id.
        record_obligation(
            conn,
            object_id="claude-code-session:s1",
            protocol_version="polylogue.material-protocol/v1",
            revision_id="rev-2",
            manifest_digest="digest-2",
            mode=PublicationMode.MIRROR,
            now_ms=2000,
        )
        conn.commit()
        rows = conn.execute("SELECT COUNT(*) FROM sinex_publication_obligations").fetchone()[0]
        assert rows == 2
    finally:
        conn.close()


def test_record_obligation_rejects_off_mode(workspace_env: dict[str, Path]) -> None:
    source_db = workspace_env["archive_root"] / "source.db"
    conn = _conn(source_db)
    try:
        with pytest.raises(ValueError, match="off mode"):
            record_obligation(
                conn,
                object_id="claude-code-session:s1",
                protocol_version="polylogue.material-protocol/v1",
                revision_id="rev-1",
                manifest_digest="digest-1",
                mode=PublicationMode.OFF,
                now_ms=1000,
            )
    finally:
        conn.close()


def test_mark_attempt_increments_count_and_sets_retired_only_on_terminal_status(
    workspace_env: dict[str, Path],
) -> None:
    source_db = workspace_env["archive_root"] / "source.db"
    conn = _conn(source_db)
    try:
        obligation = record_obligation(
            conn,
            object_id="claude-code-session:s1",
            protocol_version="polylogue.material-protocol/v1",
            revision_id="rev-1",
            manifest_digest="digest-1",
            mode=PublicationMode.PRIMARY,
            now_ms=1000,
        )
        conn.commit()

        after_pending = mark_attempt(
            conn,
            obligation,
            status=ObligationStatus.PENDING,
            receipt_state=ReceiptState.RAW_ACCEPTED,
            error=None,
            now_ms=2000,
        )
        conn.commit()
        assert after_pending.attempt_count == 1
        assert after_pending.retired_at_ms is None
        assert after_pending.status is ObligationStatus.PENDING

        after_confirmed = mark_attempt(
            conn,
            after_pending,
            status=ObligationStatus.CONFIRMED,
            receipt_state=ReceiptState.PERSISTED_CONFIRMED,
            error=None,
            now_ms=3000,
        )
        conn.commit()
        assert after_confirmed.attempt_count == 2
        assert after_confirmed.retired_at_ms == 3000
        assert after_confirmed.status is ObligationStatus.CONFIRMED
        assert after_confirmed.last_receipt_state is ReceiptState.PERSISTED_CONFIRMED
    finally:
        conn.close()


def test_list_obligations_filters_by_status_and_object(workspace_env: dict[str, Path]) -> None:
    source_db = workspace_env["archive_root"] / "source.db"
    conn = _conn(source_db)
    try:
        pending = record_obligation(
            conn,
            object_id="claude-code-session:s1",
            protocol_version="polylogue.material-protocol/v1",
            revision_id="rev-1",
            manifest_digest="digest-1",
            mode=PublicationMode.MIRROR,
            now_ms=1000,
        )
        confirmed = record_obligation(
            conn,
            object_id="claude-code-session:s2",
            protocol_version="polylogue.material-protocol/v1",
            revision_id="rev-1",
            manifest_digest="digest-1",
            mode=PublicationMode.MIRROR,
            now_ms=1000,
        )
        mark_attempt(
            conn,
            confirmed,
            status=ObligationStatus.CONFIRMED,
            receipt_state=ReceiptState.PERSISTED_CONFIRMED,
            error=None,
            now_ms=2000,
        )
        conn.commit()

        only_pending = list_obligations(conn, statuses=(ObligationStatus.PENDING,))
        assert [o.object_id for o in only_pending] == [pending.object_id]

        only_s2 = list_obligations(conn, object_id="claude-code-session:s2")
        assert len(only_s2) == 1
        assert only_s2[0].status is ObligationStatus.CONFIRMED

        assert (
            get_obligation(
                conn,
                object_id="claude-code-session:does-not-exist",
                protocol_version="polylogue.material-protocol/v1",
                revision_id="rev-1",
                manifest_digest="digest-1",
            )
            is None
        )
    finally:
        conn.close()

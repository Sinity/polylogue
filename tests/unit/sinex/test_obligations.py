"""Real-source.db exact outbox, transaction, and migration contracts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.sinex.models import (
    ObligationStatus,
    PublicationMode,
    PublicationReceipt,
    ReceiptState,
)
from polylogue.sinex.obligations import (
    PublicationPayloadConflictError,
    PublicationPayloadInvalidError,
    list_obligations,
    load_payload,
    mark_attempt,
    stage_payload,
)
from tests.unit.sinex._fixtures import publication_payload


def _conn(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def test_acceptance_marker_and_exact_payload_share_commit_or_rollback(
    workspace_env: dict[str, Path],
) -> None:
    conn = _conn(workspace_env["archive_root"] / "source.db")
    conn.execute("CREATE TABLE IF NOT EXISTS test_raw_acceptance(raw_id TEXT PRIMARY KEY)")
    payload = publication_payload()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO test_raw_acceptance VALUES ('rollback')")
        stage_payload(conn, payload=payload, mode=PublicationMode.MIRROR, now_ms=1_000)
        conn.rollback()
        assert conn.execute("SELECT COUNT(*) FROM test_raw_acceptance").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM sinex_publication_obligations").fetchone()[0] == 0

        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO test_raw_acceptance VALUES ('commit')")
        stage_payload(conn, payload=payload, mode=PublicationMode.MIRROR, now_ms=1_001)
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM test_raw_acceptance").fetchone()[0] == 1
        assert load_payload(conn, list_obligations(conn)[0]) == payload
    finally:
        conn.close()


def test_duplicate_is_idempotent_mode_only_elevates_and_changed_revision_is_history(
    workspace_env: dict[str, Path],
) -> None:
    conn = _conn(workspace_env["archive_root"] / "source.db")
    first = publication_payload()
    second = publication_payload(revision_id="rev-2", marker="two")
    try:
        conn.execute("BEGIN IMMEDIATE")
        stage_payload(conn, payload=first, mode=PublicationMode.MIRROR, now_ms=1_000)
        stage_payload(conn, payload=first, mode=PublicationMode.PRIMARY, now_ms=2_000)
        stage_payload(conn, payload=first, mode=PublicationMode.MIRROR, now_ms=3_000)
        stage_payload(conn, payload=second, mode=PublicationMode.PRIMARY, now_ms=4_000)
        conn.commit()
        rows = list_obligations(conn)
        assert len(rows) == 2
        assert rows[0].mode is PublicationMode.PRIMARY
        assert conn.execute("SELECT COUNT(*) FROM sinex_publication_payloads").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM sinex_publication_segments").fetchone()[0] == 4
    finally:
        conn.close()


def test_exact_byte_corruption_and_same_key_collision_are_detected(
    workspace_env: dict[str, Path],
) -> None:
    conn = _conn(workspace_env["archive_root"] / "source.db")
    payload = publication_payload()
    try:
        conn.execute("BEGIN IMMEDIATE")
        obligation = stage_payload(conn, payload=payload, mode=PublicationMode.MIRROR, now_ms=1_000)
        conn.commit()
        conn.execute(
            "UPDATE sinex_publication_segments SET segment_bytes=X'00' WHERE object_id=? AND position=0",
            (payload.object_id,),
        )
        conn.commit()
        with pytest.raises(PublicationPayloadInvalidError):
            load_payload(conn, obligation)
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(PublicationPayloadConflictError):
            stage_payload(conn, payload=payload, mode=PublicationMode.MIRROR, now_ms=2_000)
        conn.rollback()
    finally:
        conn.close()


def test_attempt_receipt_history_and_retry_schedule_are_durable(
    workspace_env: dict[str, Path],
) -> None:
    conn = _conn(workspace_env["archive_root"] / "source.db")
    payload = publication_payload()
    try:
        conn.execute("BEGIN IMMEDIATE")
        obligation = stage_payload(conn, payload=payload, mode=PublicationMode.PRIMARY, now_ms=1_000)
        updated = mark_attempt(
            conn,
            obligation,
            status=ObligationStatus.DURABLE_DEBT,
            receipt=PublicationReceipt(obligation.request_id, ReceiptState.DURABLE_DEBT, "spooled"),
            error_code=None,
            now_ms=2_000,
            next_attempt_at_ms=5_000,
        )
        conn.commit()
        assert updated.attempt_count == 1
        assert updated.next_attempt_at_ms == 5_000
        receipt = conn.execute(
            "SELECT request_id, receipt_state, receipt_detail FROM sinex_publication_receipts"
        ).fetchone()
        assert tuple(receipt) == (obligation.request_id, "durable_debt", "spooled")
    finally:
        conn.close()


def test_source_schema_v12_contains_outbox_payload_receipt_tables(
    workspace_env: dict[str, Path],
) -> None:
    conn = _conn(workspace_env["archive_root"] / "source.db")
    try:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert {
            "sinex_publication_obligations",
            "sinex_publication_payloads",
            "sinex_publication_segments",
            "sinex_publication_receipts",
        } <= names
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sinex_publication_obligations)")}
        assert "next_attempt_at_ms" in columns
    finally:
        conn.close()

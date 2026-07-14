"""Tests for mirror/primary lifecycle mechanics (polylogue-27m).

These tests exercise the durable request/outbox mechanism against
``SinexContractFake`` -- a fault-injecting versioned contract fake. Per the
bead's explicit non-goal, none of these tests claim a real Sinex purge; they
prove the *local mechanism* (pending-through-faults, ops.db-independence,
rejection-cannot-report-success, confirm-gated invalidation) that
polylogue-303r.6 will later bind to a real Sinex client.

Anti-vacuity: ``test_ops_db_deletion_does_not_erase_the_request`` actually
deletes the ops.db file created via ``initialize_archive_database`` and
re-reads the request from ``user.db`` afterward -- if
``submit_lifecycle_request``/``read_lifecycle_request`` were changed to
store state in ops.db, this test would fail (FileNotFoundError or a missing
row) rather than trivially passing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.security.lifecycle import (
    ContractResponse,
    SinexContractFake,
    apply_primary_invalidation_if_confirmed,
    drive_lifecycle_request,
    mirror_may_hide_locally,
    primary_may_invalidate_locally,
    read_lifecycle_request,
    submit_lifecycle_request,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


@pytest.fixture
def user_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "user.db"
    initialize_archive_database(db_path, ArchiveTier.USER)
    return db_path


class TestSubmitLifecycleRequest:
    def test_creates_pending_request(self, user_db: Path) -> None:
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                assertion_id = submit_lifecycle_request(
                    conn, target_ref="session:codex-session:x", mode="mirror", reason="r", now_ms=1
                )
            row = read_lifecycle_request(conn, assertion_id)
            assert row is not None
            assert row.state == "pending"
            assert row.mode == "mirror"
            assert row.attempt_count == 0
        finally:
            conn.close()

    def test_idempotent_for_same_target_and_mode(self, user_db: Path) -> None:
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                first = submit_lifecycle_request(
                    conn, target_ref="session:codex-session:x", mode="primary", reason="r1", now_ms=1
                )
            with conn:
                second = submit_lifecycle_request(
                    conn, target_ref="session:codex-session:x", mode="primary", reason="r2 (ignored)", now_ms=2
                )
            assert first == second
            row = read_lifecycle_request(conn, first)
            assert row is not None
            assert row.reason == "r1"  # the second submit did not reset the row
        finally:
            conn.close()


class TestFaultInjection:
    def test_network_loss_keeps_request_pending(self, user_db: Path) -> None:
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                assertion_id = submit_lifecycle_request(
                    conn, target_ref="session:codex-session:x", mode="primary", reason="r", now_ms=1
                )
            fake = SinexContractFake(drop_next_n=3)
            with conn:
                row = drive_lifecycle_request(conn, fake, assertion_id, now_ms=2)
            assert row.state == "pending"
            assert row.attempt_count == 1
            assert row.history[-1]["reachable"] is False
        finally:
            conn.close()

    def test_restart_recovers_state_from_the_durable_row_not_the_client(self, user_db: Path) -> None:
        """A fresh contract fake instance (simulated process restart) must not
        change the outcome of driving the SAME durable request -- the durable
        row, not client memory, is authoritative."""
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                assertion_id = submit_lifecycle_request(
                    conn, target_ref="session:codex-session:x", mode="primary", reason="r", now_ms=1
                )
            fake_before_restart = SinexContractFake(drop_next_n=1)
            with conn:
                row = drive_lifecycle_request(conn, fake_before_restart, assertion_id, now_ms=2)
            assert row.state == "pending"

            # "process restart": brand new fake, no shared state with the one above.
            fake_after_restart = SinexContractFake()
            with conn:
                row_after = drive_lifecycle_request(conn, fake_after_restart, assertion_id, now_ms=3)
            assert row_after.state == "acknowledged"
            assert row_after.attempt_count == 2  # the dropped attempt still counted
        finally:
            conn.close()

    def test_ops_db_deletion_does_not_erase_the_request(self, user_db: Path, tmp_path: Path) -> None:
        ops_db = tmp_path / "ops.db"
        initialize_archive_database(ops_db, ArchiveTier.OPS)
        assert ops_db.exists()

        conn = sqlite3.connect(user_db)
        try:
            with conn:
                assertion_id = submit_lifecycle_request(
                    conn, target_ref="session:codex-session:x", mode="primary", reason="r", now_ms=1
                )
            fake = SinexContractFake()
            with conn:
                drive_lifecycle_request(conn, fake, assertion_id, now_ms=2)  # -> acknowledged

            ops_db.unlink()
            assert not ops_db.exists()

            row = read_lifecycle_request(conn, assertion_id)
            assert row is not None
            assert row.state == "acknowledged"
        finally:
            conn.close()

    def test_rejection_cannot_report_success(self, user_db: Path, tmp_path: Path) -> None:
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                assertion_id = submit_lifecycle_request(
                    conn, target_ref="session:codex-session:x", mode="primary", reason="r", now_ms=1
                )
            fake = SinexContractFake(always_reject=True)
            with conn:
                row = drive_lifecycle_request(conn, fake, assertion_id, now_ms=2)
            assert row.state == "rejected"

            outcome = apply_primary_invalidation_if_confirmed(tmp_path, conn, assertion_id)
            assert outcome.success is False
            assert outcome.reason == "rejected"

            # Driving a terminal (rejected) request again is a no-op, not a retry.
            with conn:
                row_again = drive_lifecycle_request(conn, fake, assertion_id, now_ms=3)
            assert row_again.state == "rejected"
            assert row_again.attempt_count == row.attempt_count
        finally:
            conn.close()

    def test_pending_and_acknowledged_cannot_report_success(self, user_db: Path, tmp_path: Path) -> None:
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                assertion_id = submit_lifecycle_request(
                    conn, target_ref="session:codex-session:x", mode="primary", reason="r", now_ms=1
                )
            # still pending: no drive at all yet
            outcome_pending = apply_primary_invalidation_if_confirmed(tmp_path, conn, assertion_id)
            assert outcome_pending.success is False
            assert outcome_pending.reason == "pending_confirmation"

            fake = SinexContractFake()
            with conn:
                drive_lifecycle_request(conn, fake, assertion_id, now_ms=2)  # -> acknowledged
            outcome_ack = apply_primary_invalidation_if_confirmed(tmp_path, conn, assertion_id)
            assert outcome_ack.success is False
            assert outcome_ack.reason == "pending_confirmation"
        finally:
            conn.close()


class TestModeGates:
    def test_mirror_may_hide_before_confirmation(self, user_db: Path) -> None:
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                assertion_id = submit_lifecycle_request(
                    conn, target_ref="session:codex-session:x", mode="mirror", reason="r", now_ms=1
                )
            row = read_lifecycle_request(conn, assertion_id)
            assert row is not None
            assert mirror_may_hide_locally(row) is True
            assert primary_may_invalidate_locally(row) is False  # wrong mode
        finally:
            conn.close()

    def test_primary_never_hides_before_confirmed(self, user_db: Path) -> None:
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                assertion_id = submit_lifecycle_request(
                    conn, target_ref="session:codex-session:x", mode="primary", reason="r", now_ms=1
                )
            row = read_lifecycle_request(conn, assertion_id)
            assert row is not None
            assert primary_may_invalidate_locally(row) is False

            fake = SinexContractFake()
            with conn:
                drive_lifecycle_request(conn, fake, assertion_id, now_ms=2)  # acknowledged
            row_ack = read_lifecycle_request(conn, assertion_id)
            assert row_ack is not None
            assert primary_may_invalidate_locally(row_ack) is False

            with conn:
                drive_lifecycle_request(conn, fake, assertion_id, now_ms=3)  # confirmed
            row_confirmed = read_lifecycle_request(conn, assertion_id)
            assert row_confirmed is not None
            assert primary_may_invalidate_locally(row_confirmed) is True
        finally:
            conn.close()


class TestPrimaryInvalidatesOnlyAfterConfirmation:
    def test_confirmed_request_actually_excises_the_local_replica(self, tmp_path: Path, user_db: Path) -> None:
        source_db = tmp_path / "source.db"
        index_db = tmp_path / "index.db"
        initialize_archive_database(source_db, ArchiveTier.SOURCE)
        initialize_archive_database(index_db, ArchiveTier.INDEX)

        source_conn = sqlite3.connect(source_db)
        source_conn.execute("PRAGMA foreign_keys = ON")
        try:
            raw_id = write_source_raw_session(
                source_conn,
                origin="codex-session",
                source_path="/fake/x.jsonl",
                source_index=0,
                payload=b"hello",
                acquired_at_ms=1,
                native_id="native-x",
            )
            source_conn.commit()
        finally:
            source_conn.close()

        index_conn = sqlite3.connect(index_db)
        index_conn.execute("PRAGMA foreign_keys = ON")
        try:
            index_conn.execute(
                "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms) "
                "VALUES (?, 'codex-session', ?, ?, zeroblob(32), 1, 2)",
                ("native-x", raw_id, "T"),
            )
            index_conn.commit()
            session_id = index_conn.execute("SELECT session_id FROM sessions").fetchone()[0]
        finally:
            index_conn.close()

        conn = sqlite3.connect(user_db)
        try:
            with conn:
                assertion_id = submit_lifecycle_request(
                    conn, target_ref=f"session:{session_id}", mode="primary", reason="leak", now_ms=1
                )
            fake = SinexContractFake()
            with conn:
                drive_lifecycle_request(conn, fake, assertion_id, now_ms=2)  # acknowledged
            with conn:
                row = drive_lifecycle_request(conn, fake, assertion_id, now_ms=3)  # confirmed
            assert row.state == "confirmed"

            outcome = apply_primary_invalidation_if_confirmed(tmp_path, conn, assertion_id)
            assert outcome.success is True
        finally:
            conn.close()

        index_conn = sqlite3.connect(index_db)
        try:
            remaining = index_conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        finally:
            index_conn.close()
        assert remaining == 0

    def test_unknown_request_id_cannot_report_success(self, tmp_path: Path, user_db: Path) -> None:
        conn = sqlite3.connect(user_db)
        try:
            outcome = apply_primary_invalidation_if_confirmed(
                tmp_path, conn, "assertion-excision_request:does-not-exist"
            )
            assert outcome.success is False
            assert outcome.reason == "unknown_request"
        finally:
            conn.close()


class TestContractResponseShape:
    def test_contract_response_is_typed(self) -> None:
        response = ContractResponse(reachable=True, outcome="acknowledged", detail="queued")
        assert response.reachable is True
        assert response.outcome == "acknowledged"

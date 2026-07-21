"""Real-dependency tests for the t46.9/kwsb.2 named-route actuators.

Anti-vacuity: these drive ``SessionDeleteActuator``/``IdentityResetActuator``
against a real seeded ``index.db``/``user.db`` pair (real schema, real SQL),
not a toy replica. ``test_session_delete_actuator_removes_the_session_row``
fails if ``SessionDeleteActuator.apply`` stops calling
``ArchiveStore.delete_sessions``; ``test_identity_reset_actuator_suppresses_
and_deletes`` fails if ``IdentityResetActuator.apply`` stops writing the
``user.db`` suppression row or stops deleting the ``index.db`` session row.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.operations.mutation_actuators import (
    IdentityResetActuator,
    IdentityResetArgs,
    SessionDeleteActuator,
    SessionDeleteArgs,
)
from polylogue.operations.mutation_transaction import ConfirmationRequiredError, OperationExecutor, PlanStaleError
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _seed_archive_session(archive_root: Path, *, native_id: str) -> str:
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    session_id = f"codex-session:{native_id}"
    raw_id = f"raw-{native_id}"
    with sqlite3.connect(source_db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms
            )
            VALUES (?, 'codex-session', ?, ?, zeroblob(32), 0, 1000)
            """,
            (raw_id, native_id, str(archive_root / f"{native_id}.jsonl")),
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms
            )
            VALUES (?, 'codex-session', ?, ?, zeroblob(32), 1000, 2000)
            """,
            (native_id, raw_id, f"Session {native_id}"),
        )
    return session_id


class TestSessionDeleteActuator:
    def test_prepare_only_plans_currently_existing_sessions(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="alpha")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = SessionDeleteActuator()
            plan = actuator.prepare(SessionDeleteArgs(archive=archive, session_ids=(session_id, "does-not-exist")))

        assert plan.target_refs == (f"session:{session_id}",)
        # PREPARE performed zero mutation.
        with sqlite3.connect(archive_root / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1

    def test_full_lifecycle_deletes_the_session_row(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="beta")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = SessionDeleteActuator()
            executor = OperationExecutor()
            args = SessionDeleteArgs(archive=archive, session_ids=(session_id,))
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="confirm_flag"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.status == "applied"
        assert receipt.affected_count == 1
        with sqlite3.connect(archive_root / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0

    def test_execute_without_authorization_confirm_flag_refuses(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="gamma")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = SessionDeleteActuator()
            executor = OperationExecutor()
            args = SessionDeleteArgs(archive=archive, session_ids=(session_id,))
            plan = executor.prepare(actuator, args)
            with pytest.raises(ConfirmationRequiredError):
                executor.authorize(
                    actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
                )

        # Refused before mutation: the session row is untouched.
        with sqlite3.connect(archive_root / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1

    def test_stale_plan_after_concurrent_delete_refuses(self, tmp_path: Path) -> None:
        """The "excision bypass" regression class applied to session delete.

        A plan/authorization prepared while the session existed must not
        apply after the session was removed by another actor in the
        meantime (simulated here by deleting it directly between AUTHORIZE
        and EXECUTE).
        """
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="delta")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = SessionDeleteActuator()
            executor = OperationExecutor()
            args = SessionDeleteArgs(archive=archive, session_ids=(session_id,))
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="confirm_flag"
            )
            # Concurrent removal out from under the held authorization.
            archive.delete_sessions((session_id,))

            with pytest.raises(PlanStaleError):
                executor.execute(actuator, plan, authorization, args)


class TestIdentityResetActuator:
    def test_full_lifecycle_suppresses_and_deletes(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="epsilon")

        actuator = IdentityResetActuator()
        executor = OperationExecutor()
        args = IdentityResetArgs(archive_root=archive_root, session_ids=(session_id,), reason="test reset")
        plan = executor.prepare(actuator, args)
        assert plan.target_refs == (f"session:{session_id}",)
        authorization = executor.authorize(
            actuator, plan, actor="test", role="write", capability="test", confirmation_strength="confirm_flag"
        )
        receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.status == "applied"
        assert receipt.affected_count == 1
        with sqlite3.connect(archive_root / "user.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'suppression'").fetchone()[0] == 1
        with sqlite3.connect(archive_root / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0

    def test_nonexistent_session_plans_zero_targets(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        _seed_archive_session(archive_root, native_id="real-one")

        actuator = IdentityResetActuator()
        plan = actuator.prepare(
            IdentityResetArgs(archive_root=archive_root, session_ids=("codex-session:typo",), reason="x")
        )

        assert plan.target_refs == ()
        with sqlite3.connect(archive_root / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1

"""Real-dependency tests for the t46.9/kwsb.2 named-route actuators.

Anti-vacuity: these drive the actuators against a real seeded
``index.db``/``user.db`` pair (real schema, real SQL), not a toy replica.
``test_session_delete_actuator_removes_the_session_row`` fails if
``SessionDeleteActuator.apply`` stops calling ``ArchiveStore.delete_sessions``;
``test_identity_reset_actuator_suppresses_and_deletes`` fails if
``IdentityResetActuator.apply`` stops writing the ``user.db`` suppression row
or stops deleting the ``index.db`` session row.

Phase 2 (t46.9/kwsb.2) additions below (``TagAddActuator``/``TagRemoveActuator``/
``BulkTagActuator``/``MetadataSetActuator``/``MetadataDeleteActuator``/
``MarkAddActuator``/``MarkRemoveActuator``) drive the real
``ArchiveStore.add_user_tags``/``remove_user_tags``/``set_user_metadata``/
``delete_user_metadata``/``add_mark``/``remove_mark`` primitives against the
same real ``user.db`` schema, and additionally prove AC4 (reversible writes
do not require interactive confirmation): every ``role_only``-strength
``authorize`` call below succeeds where the phase-1 delete/reset actuators
above require ``confirm_flag`` and refuse ``role_only``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.operations.mutation_actuators import (
    BulkTagActuator,
    BulkTagArgs,
    IdentityResetActuator,
    IdentityResetArgs,
    MarkAddActuator,
    MarkArgs,
    MarkRemoveActuator,
    MetadataDeleteActuator,
    MetadataDeleteArgs,
    MetadataSetActuator,
    MetadataSetArgs,
    SessionDeleteActuator,
    SessionDeleteArgs,
    TagAddActuator,
    TagAddArgs,
    TagRemoveActuator,
    TagRemoveArgs,
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


class TestTagAddActuator:
    def test_full_lifecycle_writes_the_tag_assertion(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="tag-add")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = TagAddActuator()
            executor = OperationExecutor()
            args = TagAddArgs(archive=archive, session_id=session_id, tag="Review")
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.status == "applied"
        assert receipt.affected_count == 1
        with sqlite3.connect(archive_root / "user.db") as conn:
            rows = conn.execute("SELECT key FROM assertions WHERE kind = 'tag' AND status != 'deleted'").fetchall()
        assert [r[0] for r in rows] == ["review"]

    def test_duplicate_add_is_already_satisfied(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="tag-dup")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = TagAddActuator()
            executor = OperationExecutor()
            args = TagAddArgs(archive=archive, session_id=session_id, tag="dup")
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            executor.execute(actuator, plan, authorization, args)
            # Second round trip against the same live state.
            plan2 = executor.prepare(actuator, args)
            authorization2 = executor.authorize(
                actuator, plan2, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt2 = executor.execute(actuator, plan2, authorization2, args)

        assert receipt2.status == "already_satisfied"
        assert receipt2.affected_count == 0

    def test_nonexistent_session_raises_keyerror_at_prepare(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        _seed_archive_session(archive_root, native_id="tag-real")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = TagAddActuator()
            args = TagAddArgs(archive=archive, session_id="codex-session:typo", tag="x")
            with pytest.raises(KeyError):
                actuator.prepare(args)


class TestTagRemoveActuator:
    def test_full_lifecycle_marks_the_tag_deleted(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="tag-remove")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            archive.add_user_tags((session_id,), ("keep",))
            actuator = TagRemoveActuator()
            executor = OperationExecutor()
            args = TagRemoveArgs(archive=archive, session_id=session_id, tag="KEEP")
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.status == "applied"
        with sqlite3.connect(archive_root / "user.db") as conn:
            status = conn.execute("SELECT status FROM assertions WHERE kind = 'tag'").fetchone()[0]
        assert status == "deleted"

    def test_missing_tag_is_already_satisfied(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="tag-remove-missing")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = TagRemoveActuator()
            executor = OperationExecutor()
            args = TagRemoveArgs(archive=archive, session_id=session_id, tag="absent")
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.status == "already_satisfied"
        assert receipt.detail == "tag_not_present"


class TestBulkTagActuator:
    def test_skips_unresolved_sessions_and_tags_the_rest(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="bulk")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = BulkTagActuator()
            executor = OperationExecutor()
            args = BulkTagArgs(archive=archive, session_ids=(session_id, "does-not-exist"), tags=("a", "b"))
            plan = executor.prepare(actuator, args)
            assert plan.target_refs == (f"session:{session_id}",)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.status == "applied"
        assert receipt.domain_receipt["affected_count"] == 1
        assert receipt.domain_receipt["session_count"] == 2
        assert receipt.domain_receipt["skipped_count"] == 1
        with sqlite3.connect(archive_root / "user.db") as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE kind = 'tag' AND status != 'deleted'"
            ).fetchone()[0]
        assert count == 2


class TestMetadataSetActuator:
    def test_full_lifecycle_writes_the_metadata_assertion(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="meta-set")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = MetadataSetActuator()
            executor = OperationExecutor()
            args = MetadataSetArgs(archive=archive, session_id=session_id, key="priority", value="high")
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.status == "applied"
        with sqlite3.connect(archive_root / "user.db") as conn:
            row = conn.execute("SELECT key, value_json FROM assertions WHERE kind = 'metadata'").fetchone()
        assert row[0] == "priority"
        assert "high" in row[1]

    def test_unchanged_value_is_already_satisfied(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="meta-unchanged")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            archive.set_user_metadata((session_id,), (("status", "open"),))
            actuator = MetadataSetActuator()
            executor = OperationExecutor()
            args = MetadataSetArgs(archive=archive, session_id=session_id, key="status", value="open")
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.status == "already_satisfied"
        assert receipt.detail == "value_unchanged"


class TestMetadataDeleteActuator:
    def test_full_lifecycle_marks_the_metadata_deleted(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="meta-delete")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            archive.set_user_metadata((session_id,), (("status", "open"),))
            actuator = MetadataDeleteActuator()
            executor = OperationExecutor()
            args = MetadataDeleteArgs(archive=archive, session_id=session_id, key="status")
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.status == "applied"
        with sqlite3.connect(archive_root / "user.db") as conn:
            status = conn.execute("SELECT status FROM assertions WHERE kind = 'metadata'").fetchone()[0]
        assert status == "deleted"

    def test_missing_key_is_already_satisfied(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="meta-delete-missing")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = MetadataDeleteActuator()
            executor = OperationExecutor()
            args = MetadataDeleteArgs(archive=archive, session_id=session_id, key="absent")
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.status == "already_satisfied"
        assert receipt.detail == "key_not_found"


class TestMarkActuators:
    def test_add_then_remove_round_trips_through_user_db(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="mark")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()

            add_actuator = MarkAddActuator()
            add_args = MarkArgs(archive=archive, target_type="session", target_id=session_id, mark_type="star")
            add_plan = executor.prepare(add_actuator, add_args)
            add_authorization = executor.authorize(
                add_actuator, add_plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            add_receipt = executor.execute(add_actuator, add_plan, add_authorization, add_args)
            assert add_receipt.status == "applied"

            with sqlite3.connect(archive_root / "user.db") as conn:
                status = conn.execute("SELECT status FROM assertions WHERE kind = 'mark'").fetchone()[0]
            assert status != "deleted"

            remove_actuator = MarkRemoveActuator()
            remove_plan = executor.prepare(remove_actuator, add_args)
            remove_authorization = executor.authorize(
                remove_actuator,
                remove_plan,
                actor="test",
                role="write",
                capability="test",
                confirmation_strength="role_only",
            )
            remove_receipt = executor.execute(remove_actuator, remove_plan, remove_authorization, add_args)
            assert remove_receipt.status == "applied"

            with sqlite3.connect(archive_root / "user.db") as conn:
                status = conn.execute("SELECT status FROM assertions WHERE kind = 'mark'").fetchone()[0]
            assert status == "deleted"

    def test_duplicate_add_is_already_satisfied(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="mark-dup")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            actuator = MarkAddActuator()
            args = MarkArgs(archive=archive, target_type="session", target_id=session_id, mark_type="pin")
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            executor.execute(actuator, plan, authorization, args)

            plan2 = executor.prepare(actuator, args)
            authorization2 = executor.authorize(
                actuator, plan2, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt2 = executor.execute(actuator, plan2, authorization2, args)

        assert receipt2.status == "already_satisfied"


class TestReversibleActuatorsAcceptTheWeakestConfirmation:
    """AC4: reversible writes must not require confirm_flag/bound_token."""

    def test_role_only_never_raises_confirmation_required(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="ac4")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()

            tag_actuator = TagAddActuator()
            tag_args = TagAddArgs(archive=archive, session_id=session_id, tag="t")
            tag_plan = executor.prepare(tag_actuator, tag_args)
            executor.authorize(
                tag_actuator, tag_plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )

            metadata_actuator = MetadataSetActuator()
            metadata_args = MetadataSetArgs(archive=archive, session_id=session_id, key="k", value="v")
            metadata_plan = executor.prepare(metadata_actuator, metadata_args)
            executor.authorize(
                metadata_actuator,
                metadata_plan,
                actor="test",
                role="write",
                capability="test",
                confirmation_strength="role_only",
            )

            mark_actuator = MarkAddActuator()
            mark_args = MarkArgs(archive=archive, target_type="session", target_id=session_id, mark_type="star")
            mark_plan = executor.prepare(mark_actuator, mark_args)
            # None of the three role_only authorize calls above raised
            # ConfirmationRequiredError -- that is the AC4 assertion.
            executor.authorize(
                mark_actuator,
                mark_plan,
                actor="test",
                role="write",
                capability="test",
                confirmation_strength="role_only",
            )

    def test_delete_class_actuator_refuses_role_only(self, tmp_path: Path) -> None:
        """Contrast: the phase-1 delete actuator still requires confirm_flag."""
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="ac4-contrast")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            actuator = SessionDeleteActuator()
            args = SessionDeleteArgs(archive=archive, session_ids=(session_id,))
            plan = executor.prepare(actuator, args)
            with pytest.raises(ConfirmationRequiredError):
                executor.authorize(
                    actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
                )

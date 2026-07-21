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

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.operations.mutation_actuators import (
    AnnotationDeleteActuator,
    AnnotationDeleteArgs,
    AnnotationSaveActuator,
    AnnotationSaveArgs,
    BlockerResolveActuator,
    BlockerResolveArgs,
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


def _seed_raw_authority_blocker(
    archive_root: Path,
    *,
    blocker_id: str,
    plan_id: str,
    census_id: str,
    frontier: bool = False,
    reason: str = "immutable source/index preconditions changed after the census",
) -> None:
    """Seed one real, unresolved ``raw_authority_blockers`` row (plus the plan/census
    rows it references and one real ``raw_sessions`` row so a non-frontier
    resolution can genuinely replan it)."""
    raw_id = f"raw-{blocker_id}"
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        payload = (
            b'{"type":"session_meta","payload":{"id":"' + blocker_id.encode() + b'"}}\n'
            b'{"type":"response_item","payload":{"type":"message","id":"m-1",'
            b'"role":"user","content":[{"type":"input_text","text":"hi"}]}}\n'
        )
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path=f"{blocker_id}.jsonl",
            acquired_at_ms=1000,
            raw_id=raw_id,
        )
    witness_schema = "polylogue.raw-authority-frontier-plan.v1" if frontier else "polylogue.raw-authority-plan.v1"
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        next_sequence_no = int(
            conn.execute("SELECT COALESCE(MAX(sequence_no), 0) + 1 FROM raw_authority_censuses").fetchone()[0]
        )
        conn.execute(
            """
            INSERT INTO raw_authority_censuses (
                census_id, sequence_no, scope_json, residual_json, parser_fingerprint,
                mode, lifecycle_status, quiescent, inventory_digest, residual_digest,
                plan_count, post_inventory_digest, post_residual_json, post_residual_digest,
                post_plan_count, postflight_at_ms, executable_plan_count, residual_plan_count,
                predecessor_census_id, fixed_point, created_at_ms, completed_at_ms
            ) VALUES (?, ?, '{}', '{}', 'actuator-test-fp', 'apply', 'completed', 1, ?, ?, 1,
                      ?, '{}', ?, 0, 1000, 1, 0, NULL, 0, 1000, 1000)
            """,
            (census_id, next_sequence_no, "a" * 64, "b" * 64, "c" * 64, "d" * 64),
        )
        conn.execute(
            """
            INSERT INTO raw_authority_plans (
                plan_id, input_digest, input_raw_ids_json, logical_keys_json,
                authority_witness_json, source_preconditions_json, index_preconditions_json,
                created_at_ms
            ) VALUES (?, ?, ?, '[]', ?, '{}', '{}', 1000)
            """,
            (plan_id, "e" * 64, json.dumps([raw_id]), json.dumps({"schema": witness_schema})),
        )
        conn.execute(
            """
            INSERT INTO raw_authority_blockers (
                blocker_id, plan_id, census_id, reason, expected_json, observed_json, created_at_ms
            ) VALUES (?, ?, ?, ?, '{}', '{}', 1000)
            """,
            (blocker_id, plan_id, census_id, reason),
        )
        conn.commit()


class TestAnnotationActuators:
    def test_save_then_delete_round_trips_through_user_db(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="annotation")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()

            save_actuator = AnnotationSaveActuator()
            save_args = AnnotationSaveArgs(
                archive=archive,
                annotation_id="note-1",
                target_type="session",
                target_id=session_id,
                note_text="first note",
            )
            save_plan = executor.prepare(save_actuator, save_args)
            save_authorization = executor.authorize(
                save_actuator,
                save_plan,
                actor="test",
                role="write",
                capability="test",
                confirmation_strength="role_only",
            )
            save_receipt = executor.execute(save_actuator, save_plan, save_authorization, save_args)
            assert save_receipt.status == "applied"
            assert save_receipt.domain_receipt["created"] is True

            with sqlite3.connect(archive_root / "user.db") as conn:
                row = conn.execute("SELECT status, body_text FROM assertions WHERE kind = 'annotation'").fetchone()
            assert row[0] != "deleted"
            assert row[1] == "first note"

            # Updating an existing annotation is a real write (not a no-op),
            # and `created` reflects update-vs-create honestly.
            update_args = AnnotationSaveArgs(
                archive=archive,
                annotation_id="note-1",
                target_type="session",
                target_id=session_id,
                note_text="updated note",
            )
            update_plan = executor.prepare(save_actuator, update_args)
            update_authorization = executor.authorize(
                save_actuator,
                update_plan,
                actor="test",
                role="write",
                capability="test",
                confirmation_strength="role_only",
            )
            update_receipt = executor.execute(save_actuator, update_plan, update_authorization, update_args)
            assert update_receipt.status == "applied"
            assert update_receipt.domain_receipt["created"] is False
            with sqlite3.connect(archive_root / "user.db") as conn:
                body = conn.execute("SELECT body_text FROM assertions WHERE kind = 'annotation'").fetchone()[0]
            assert body == "updated note"

            delete_actuator = AnnotationDeleteActuator()
            delete_args = AnnotationDeleteArgs(archive=archive, annotation_id="note-1")
            delete_plan = executor.prepare(delete_actuator, delete_args)
            delete_authorization = executor.authorize(
                delete_actuator,
                delete_plan,
                actor="test",
                role="write",
                capability="test",
                confirmation_strength="role_only",
            )
            delete_receipt = executor.execute(delete_actuator, delete_plan, delete_authorization, delete_args)
            assert delete_receipt.status == "applied"

            with sqlite3.connect(archive_root / "user.db") as conn:
                status = conn.execute("SELECT status FROM assertions WHERE kind = 'annotation'").fetchone()[0]
            assert status == "deleted"

    def test_delete_missing_annotation_is_already_satisfied(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        _seed_archive_session(archive_root, native_id="annotation-missing")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            actuator = AnnotationDeleteActuator()
            args = AnnotationDeleteArgs(archive=archive, annotation_id="does-not-exist")
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.status == "already_satisfied"

    def test_role_only_authorize_succeeds(self, tmp_path: Path) -> None:
        """AC4: the annotation family is reversible class -- role_only, not confirm_flag."""
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="annotation-ac4")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            actuator = AnnotationSaveActuator()
            args = AnnotationSaveArgs(
                archive=archive, annotation_id="n", target_type="session", target_id=session_id, note_text="x"
            )
            plan = executor.prepare(actuator, args)
            # Does not raise ConfirmationRequiredError.
            executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )


class TestBlockerResolveActuator:
    def test_prepare_finds_a_real_unresolved_blocker(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore(archive_root):
            pass
        _seed_raw_authority_blocker(
            archive_root,
            blocker_id="blocker-prepare",
            plan_id="raw-replay:prepare-plan",
            census_id="raw-authority-census:prepare",
        )

        actuator = BlockerResolveActuator()
        args = BlockerResolveArgs(archive_root=archive_root, blocker_id="blocker-prepare", resolution="ack")
        plan = actuator.prepare(args)

        assert plan.target_refs == ("raw-authority-blocker:blocker-prepare",)
        assert plan.context["kind"] == "stale_plan"
        # PREPARE performed zero mutation.
        with sqlite3.connect(archive_root / "source.db") as conn:
            resolved = conn.execute(
                "SELECT resolved_at_ms FROM raw_authority_blockers WHERE blocker_id = ?", ("blocker-prepare",)
            ).fetchone()[0]
        assert resolved is None

    def test_prepare_on_unknown_blocker_yields_empty_plan(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore(archive_root):
            pass

        actuator = BlockerResolveActuator()
        args = BlockerResolveArgs(archive_root=archive_root, blocker_id="does-not-exist", resolution="ack")
        plan = actuator.prepare(args)

        assert plan.target_refs == ()
        assert plan.context["found"] is False

    def test_execute_resolves_and_reopens_replanning(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore(archive_root):
            pass
        _seed_raw_authority_blocker(
            archive_root,
            blocker_id="blocker-execute",
            plan_id="raw-replay:execute-plan",
            census_id="raw-authority-census:execute",
        )

        actuator = BlockerResolveActuator()
        executor = OperationExecutor()
        args = BlockerResolveArgs(
            archive_root=archive_root, blocker_id="blocker-execute", resolution="current path is authoritative"
        )
        plan = executor.prepare(actuator, args)
        authorization = executor.authorize(
            actuator, plan, actor="test", role="write", capability="test", confirmation_strength="confirm_flag"
        )
        receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.status == "applied"
        with sqlite3.connect(archive_root / "source.db") as conn:
            row = conn.execute(
                "SELECT resolved_at_ms, resolution FROM raw_authority_blockers WHERE blocker_id = ?",
                ("blocker-execute",),
            ).fetchone()
        assert row[0] is not None
        assert "current path is authoritative" in str(row[1])

    def test_role_only_authorize_refuses(self, tmp_path: Path) -> None:
        """Contrast with the reversible-class actuators: reset-class needs confirm_flag."""
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore(archive_root):
            pass
        _seed_raw_authority_blocker(
            archive_root,
            blocker_id="blocker-ac4-contrast",
            plan_id="raw-replay:ac4-contrast-plan",
            census_id="raw-authority-census:ac4-contrast",
        )

        actuator = BlockerResolveActuator()
        executor = OperationExecutor()
        args = BlockerResolveArgs(archive_root=archive_root, blocker_id="blocker-ac4-contrast", resolution="ack")
        plan = executor.prepare(actuator, args)
        with pytest.raises(ConfirmationRequiredError):
            executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )

    def test_concurrent_resolution_between_prepare_and_execute_raises_plan_stale(self, tmp_path: Path) -> None:
        """TOCTOU: another actor resolving the same blocker between PREPARE and
        EXECUTE must be caught by the executor's fresh-PREPARE revalidation,
        not silently double-applied."""
        from polylogue.storage.raw_authority import resolve_raw_authority_blocker

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore(archive_root):
            pass
        _seed_raw_authority_blocker(
            archive_root,
            blocker_id="blocker-stale",
            plan_id="raw-replay:stale-plan",
            census_id="raw-authority-census:stale",
        )

        actuator = BlockerResolveActuator()
        executor = OperationExecutor()
        args = BlockerResolveArgs(archive_root=archive_root, blocker_id="blocker-stale", resolution="ack")
        plan = executor.prepare(actuator, args)
        authorization = executor.authorize(
            actuator, plan, actor="test", role="write", capability="test", confirmation_strength="confirm_flag"
        )

        # A concurrent actor resolves the same blocker out-of-band.
        resolve_raw_authority_blocker(archive_root, "blocker-stale", resolution="resolved elsewhere first")

        with pytest.raises(PlanStaleError):
            executor.execute(actuator, plan, authorization, args)

    def test_frontier_judgment_kind_is_classified(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore(archive_root):
            pass
        _seed_raw_authority_blocker(
            archive_root,
            blocker_id="blocker-frontier",
            plan_id="raw-replay:frontier-plan",
            census_id="raw-authority-census:frontier",
            frontier=True,
            reason="conflicting canonical authority",
        )

        actuator = BlockerResolveActuator()
        args = BlockerResolveArgs(archive_root=archive_root, blocker_id="blocker-frontier", resolution="ack")
        plan = actuator.prepare(args)

        assert plan.context["kind"] == "frontier_judgment"

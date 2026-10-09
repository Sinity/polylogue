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

import hashlib
import json
import sqlite3
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from contextvars import Context
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.analysis.feedback import LearningCorrection
from polylogue.archive.write_gateway import ArchiveWriteGateway, WriteOperation
from polylogue.core.enums import AssertionKind, Provider
from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.mutation_actuators import (
    AnnotationDeleteActuator,
    AnnotationDeleteArgs,
    AnnotationSaveActuator,
    AnnotationSaveArgs,
    BlackboardPostActuator,
    BlackboardPostArgs,
    BlobPublicationAbandonActuator,
    BlobPublicationAbandonArgs,
    BlockerResolveActuator,
    BlockerResolveArgs,
    BulkMetadataSetActuator,
    BulkMetadataSetArgs,
    BulkTagActuator,
    BulkTagArgs,
    CaptureAssertionCandidateActuator,
    CaptureAssertionCandidateArgs,
    CorrectionDeleteActuator,
    CorrectionDeleteArgs,
    CorrectionRecordActuator,
    CorrectionRecordArgs,
    CorrectionsClearActuator,
    CorrectionsClearArgs,
    IdentityResetActuator,
    IdentityResetArgs,
    InsightsRebuildActuator,
    InsightsRebuildArgs,
    MarkAddActuator,
    MarkArgs,
    MarkRemoveActuator,
    MetadataDeleteActuator,
    MetadataDeleteArgs,
    MetadataSetActuator,
    MetadataSetArgs,
    RecallPackDeleteActuator,
    RecallPackDeleteArgs,
    RecallPackSaveActuator,
    RecallPackSaveArgs,
    SavedViewDeleteActuator,
    SavedViewDeleteArgs,
    SavedViewSaveActuator,
    SavedViewSaveArgs,
    SessionDeleteActuator,
    SessionDeleteArgs,
    SessionExcisionActuator,
    SessionExcisionArgs,
    TagAddActuator,
    TagAddArgs,
    TagRemoveActuator,
    TagRemoveArgs,
    WorkspaceDeleteActuator,
    WorkspaceDeleteArgs,
    WorkspaceSaveActuator,
    WorkspaceSaveArgs,
)
from polylogue.operations.mutation_transaction import (
    ConfirmationRequiredError,
    MutationAuthorization,
    MutationPlan,
    MutationPreview,
    MutationPrincipal,
    MutationReceipt,
    MutationTransactionError,
    OperationExecutor,
    PlanStaleError,
    RecoveryDeferredError,
    build_plan,
)
from polylogue.storage.accepted_marker_inputs import persist_pending_marker_input_sync, prepare_accepted_marker_input
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.user_write import (
    assertion_id_for_saved_view,
    assertion_id_for_workspace,
)
from tests.infra.operation_recovery import recover_on_admitted_owner


def test_blob_abandon_refuses_blocked_liveness_before_deleting_any_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Anti-vacuity: BLOCKED cannot be folded into the deletable unreferenced set."""
    actuator = BlobPublicationAbandonActuator()
    plan = build_plan(
        operation=actuator.operation,
        destructive_class=actuator.destructive_class,
        target_refs=("source:blob-publication:blocked",),
        affected_tiers=("source", "audit"),
        reversible=False,
        context={"requested": ["blocked"], "blocked": ["blocked"]},
    )
    monkeypatch.setattr(
        "polylogue.storage.blob_publication.abandon_blob_publication_receipts",
        lambda *args, **kwargs: pytest.fail("blocked evidence must stop before deletion"),
    )
    with pytest.raises(RecoveryDeferredError, match="blocked"):
        actuator.apply(plan, BlobPublicationAbandonArgs(tmp_path, ("blocked",)))


def _seed_archive_session(archive_root: Path, *, native_id: str) -> str:
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    initialize_active_archive_root(archive_root)
    session_id = f"codex-session:{native_id}"
    raw_id = f"raw-{native_id}"
    with closing(sqlite3.connect(source_db)) as conn, conn:
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
    with closing(sqlite3.connect(index_db)) as conn, conn:
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


def _seed_prefix_lineage(archive_root: Path) -> tuple[str, str]:
    parent_id = _seed_archive_session(archive_root, native_id="excision-parent")
    child_id = _seed_archive_session(archive_root, native_id="excision-child")
    with closing(sqlite3.connect(archive_root / "index.db")) as conn, conn:
        conn.execute(
            "INSERT INTO messages (session_id, native_id, position, role, content_hash) "
            "VALUES (?, 'm1', 0, 'user', zeroblob(32))",
            (parent_id,),
        )
        branch_point = conn.execute("SELECT message_id FROM messages WHERE session_id = ?", (parent_id,)).fetchone()[0]
        conn.execute(
            "INSERT INTO session_links (src_session_id, dst_origin, dst_native_id, link_type, "
            "resolved_dst_session_id, branch_point_message_id, inheritance, status, method, confidence, "
            "evidence_json, observed_at_ms, resolved_at_ms) "
            "VALUES (?, 'codex-session', 'excision-parent', 'branch', ?, ?, 'prefix-sharing', "
            "NULL, NULL, 1.0, '[]', 1000, NULL)",
            (child_id, parent_id, branch_point),
        )
    return parent_id, child_id


class TestSessionExcisionActuator:
    def test_prepare_and_apply_expose_marker_carrier_terminal_evidence(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="excision-marker")
        raw_id = "raw-excision-marker"
        marker = prepare_accepted_marker_input(
            raw_id, [{"session_id": session_id, "candidates": [{"body": "marker secret"}]}]
        )
        with closing(sqlite3.connect(archive_root / "source.db")) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            persist_pending_marker_input_sync(conn, marker, expected_incarnation_id=str(uuid.uuid4()))

        actuator = SessionExcisionActuator()
        args = SessionExcisionArgs(
            archive_root=archive_root,
            session_id=session_id,
            reason="r",
            actor="user:test",
            cascade_lineage=False,
        )
        plan = actuator.prepare(args)
        assert plan.context["source_marker_inputs_pending"] == 1
        assert plan.context["source_marker_inputs_accepted"] == 0
        assert plan.context["marker_input_digests"] == [marker.payload_sha256]

        from tests.infra.excision_execution import execute_excision

        domain_receipt = execute_excision(archive_root, session_id, reason="r", actor="user:test")
        counts = cast("dict[str, int]", domain_receipt["counts"])
        assert counts["source_marker_inputs_pending"] == 1
        assert domain_receipt["marker_input_digests"] == [marker.payload_sha256]

    def test_prepare_without_cascade_refuses_a_lineage_parent(self, tmp_path: Path) -> None:
        from polylogue.security.excision import LineageDependentsError

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        parent_id, child_id = _seed_prefix_lineage(archive_root)
        with pytest.raises(LineageDependentsError) as excinfo:
            SessionExcisionActuator().prepare(
                SessionExcisionArgs(
                    archive_root=archive_root,
                    session_id=parent_id,
                    reason="r",
                    actor="user:test",
                    cascade_lineage=False,
                )
            )
        assert excinfo.value.dependent_session_ids == (child_id,)


class TestSessionDeleteActuator:
    def test_production_delete_runs_post_commit_archive_effects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real delete actuator reaches the gateway after index mutation.

        Anti-vacuity: bypassing ``ArchiveStore.delete_sessions``' gateway
        commit leaves this cache-invalidation observation empty even though
        the session row was deleted.
        """
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="delete-effects")
        invalidated: list[bool] = []

        monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.ensure_fts_triggers_sync", lambda _conn: None)
        monkeypatch.setattr("polylogue.storage.search.cache.invalidate_search_cache", lambda: invalidated.append(True))

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = SessionDeleteActuator()
            executor = OperationExecutor()
            args = SessionDeleteArgs(archive=archive, session_ids=(session_id,))
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="confirm_flag"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.affected_count == 1
        assert invalidated == [True]
        # F614: the invalidation is part of the admitted writer transaction, so
        # there is no independent deferred writer left to race the coordinator.

    def test_prepare_only_plans_currently_existing_sessions(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="alpha")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = SessionDeleteActuator()
            plan = actuator.prepare(SessionDeleteArgs(archive=archive, session_ids=(session_id, "does-not-exist")))

        assert plan.target_refs == (f"session:{session_id}",)
        # PREPARE performed zero mutation.
        with closing(sqlite3.connect(archive_root / "index.db")) as conn, conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1

    def test_full_lifecycle_deletes_the_session_row(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        initialize_active_archive_root(archive_root)
        session_id = _seed_archive_session(archive_root, native_id="beta")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = SessionDeleteActuator()
            executor = OperationExecutor.for_archive_root(archive_root)
            args = SessionDeleteArgs(archive=archive, session_ids=(session_id,))
            binding = runtime_operation_binding(actuator)
            principal = MutationPrincipal("test", frozenset({"archive.delete_session"}), "api", "write")
            preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=archive_root)
            authorization = executor.authorize_bound(binding, preview, principal)
            receipt = executor.execute_bound(binding, preview, authorization, args)

        assert receipt.status == "applied"
        assert receipt.affected_count == 1
        with closing(sqlite3.connect(archive_root / "index.db")) as conn, conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        with closing(sqlite3.connect(archive_root / "audit.db")) as conn, conn:
            assert conn.execute("SELECT state FROM operation_previews").fetchone()[0] == "consumed"
            assert conn.execute("SELECT status FROM operation_runs").fetchone()[0] == "completed"

    def test_startup_replays_a_dead_delete_attempt_to_completion(self, tmp_path: Path) -> None:
        """Recovery startup re-applies the interrupted delete; nothing is left unknown.

        Anti-vacuity: skip resolution in ``recover_interrupted_operations`` and
        the run stays ``interrupted`` with the session still indexed.
        """

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        initialize_active_archive_root(archive_root)
        session_id = _seed_archive_session(archive_root, native_id="restart-delete")
        actuator = SessionDeleteActuator()
        binding = runtime_operation_binding(actuator)
        principal = MutationPrincipal("test", frozenset({"archive.delete_session"}), "api", "write")
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor.for_archive_root(archive_root)
            args = SessionDeleteArgs(archive=archive, session_ids=(session_id,))
            preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=archive_root)
            authorization = executor.authorize_bound(binding, preview, principal)
            assert executor._audit is not None
            operation_id = executor._audit.consume_authorization_and_start(preview, authorization)
        with closing(sqlite3.connect(archive_root / "audit.db")) as conn, conn:
            conn.execute(
                "UPDATE operation_attempts SET worker_id = 'pid:999999999:0' WHERE operation_id = ?", (operation_id,)
            )
            conn.commit()

        recover_on_admitted_owner(archive_root)

        with closing(sqlite3.connect(archive_root / "audit.db")) as conn, conn:
            assert conn.execute(
                "SELECT status, terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
            ).fetchone() == ("completed", "recovered_complete")
            assert conn.execute(
                "SELECT event_type FROM operation_events WHERE operation_id = ? ORDER BY sequence DESC LIMIT 1",
                (operation_id,),
            ).fetchone() == ("recovery_resolved",)
        with closing(sqlite3.connect(archive_root / "index.db")) as conn, conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)

    def test_startup_recovery_never_steals_an_attempt_a_live_owner_still_holds(self, tmp_path: Path) -> None:
        """Daemon startup is not a cross-process exclusion bypass.

        Anti-vacuity: removing the owner-liveness barrier from orphan
        discovery makes this classify -- and terminalize -- an attempt whose
        worker is still running and may still commit the delete.
        """

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        initialize_active_archive_root(archive_root)
        session_id = _seed_archive_session(archive_root, native_id="live-owner")
        binding = runtime_operation_binding(SessionDeleteActuator())
        principal = MutationPrincipal("test", frozenset({"archive.delete_session"}), "api", "write")
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor.for_archive_root(archive_root)
            args = SessionDeleteArgs(archive=archive, session_ids=(session_id,))
            preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=archive_root)
            authorization = executor.authorize_bound(binding, preview, principal)
            assert executor._audit is not None
            operation_id = executor._audit.consume_authorization_and_start(preview, authorization)

        # The attempt owner is this live process, exactly as a concurrent
        # writer in another process would look to the recovering daemon.
        recover_on_admitted_owner(archive_root)

        with closing(sqlite3.connect(archive_root / "audit.db")) as conn, conn:
            assert conn.execute(
                "SELECT status FROM operation_runs WHERE operation_id = ?", (operation_id,)
            ).fetchone() == ("running",)
            assert conn.execute(
                "SELECT COUNT(*) FROM operation_events WHERE operation_id = ? AND event_type = 'recovery_resolved'",
                (operation_id,),
            ).fetchone() == (0,)
        with closing(sqlite3.connect(archive_root / "index.db")) as conn, conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)

    def test_startup_recovery_is_a_noop_on_an_archive_with_no_interrupted_work(self, tmp_path: Path) -> None:
        """A quiet archive costs no durable events and no archive open."""

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        initialize_active_archive_root(archive_root)
        _seed_archive_session(archive_root, native_id="quiet")

        recover_on_admitted_owner(archive_root)
        recover_on_admitted_owner(archive_root)

        with closing(sqlite3.connect(archive_root / "audit.db")) as conn, conn:
            assert conn.execute("SELECT COUNT(*) FROM operation_events").fetchone() == (0,)

    def test_delete_recovery_replays_the_recorded_plan_not_preview_target_rows(self, tmp_path: Path) -> None:
        """Recovery re-applies the authorized plan, whatever the target index rows say.

        Anti-vacuity: rebuilding the replay plan from ``operation_preview_targets``
        instead of the recorded plan drops the second session, which then
        survives the recovered delete.
        """

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        initialize_active_archive_root(archive_root)
        first = _seed_archive_session(archive_root, native_id="lost-target-first")
        second = _seed_archive_session(archive_root, native_id="lost-target-second")
        actuator = SessionDeleteActuator()
        binding = runtime_operation_binding(actuator)
        principal = MutationPrincipal("test", frozenset({"archive.delete_session"}), "api", "write")
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor.for_archive_root(archive_root)
            preview = executor.prepare_bound_for_archive(
                binding,
                SessionDeleteArgs(archive=archive, session_ids=(first, second)),
                principal,
                archive_root=archive_root,
            )
            authorization = executor.authorize_bound(binding, preview, principal)
            assert executor._audit is not None
            operation_id = executor._audit.consume_authorization_and_start(preview, authorization)
        with closing(sqlite3.connect(archive_root / "audit.db")) as conn, conn:
            conn.execute(
                "DELETE FROM operation_preview_targets WHERE preview_id = ? AND ordinal = 1", (preview.preview_ref,)
            )
            conn.execute(
                "UPDATE operation_attempts SET worker_id = 'pid:999999999:0' WHERE operation_id = ?", (operation_id,)
            )
            conn.commit()

        recover_on_admitted_owner(archive_root)

        with closing(sqlite3.connect(archive_root / "audit.db")) as conn, conn:
            assert conn.execute(
                "SELECT status, unknown_count FROM operation_runs WHERE operation_id = ?", (operation_id,)
            ).fetchone() == ("completed", 0)
        with closing(sqlite3.connect(archive_root / "index.db")) as conn, conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)

    def test_zero_target_interruption_completes_once_and_does_not_poison_later_noop(self, tmp_path: Path) -> None:
        """An interrupted empty delete plan resolves once and leaves no barrier.

        Anti-vacuity: resolving it twice appends a second ``recovery_resolved``
        event; leaving it nonterminal refuses the final no-op execution.
        """

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        initialize_active_archive_root(archive_root)
        session_id = _seed_archive_session(archive_root, native_id="zero-target")
        actuator = SessionDeleteActuator()
        binding = runtime_operation_binding(actuator)
        principal = MutationPrincipal("test", frozenset({"archive.delete_session"}), "api", "write")
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            archive.delete_sessions((session_id,))
            executor = OperationExecutor.for_archive_root(archive_root)
            args = SessionDeleteArgs(archive=archive, session_ids=(session_id,))
            preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=archive_root)
            assert preview.plan.target_count == 0
            authorization = executor.authorize_bound(binding, preview, principal)
            assert executor._audit is not None
            operation_id = executor._audit.consume_authorization_and_start(preview, authorization)
        with closing(sqlite3.connect(archive_root / "audit.db")) as conn, conn:
            conn.execute(
                "UPDATE operation_attempts SET worker_id = 'pid:999999999:0' WHERE operation_id = ?", (operation_id,)
            )
            conn.commit()

        recover_on_admitted_owner(archive_root)
        recover_on_admitted_owner(archive_root)

        with closing(sqlite3.connect(archive_root / "audit.db")) as conn, conn:
            assert conn.execute(
                "SELECT status, terminal_reason, target_count FROM operation_runs WHERE operation_id = ?",
                (operation_id,),
            ).fetchone() == ("completed", "recovered_complete", 0)
            assert conn.execute(
                "SELECT COUNT(*) FROM operation_events WHERE operation_id = ? AND event_type = 'recovery_resolved'",
                (operation_id,),
            ).fetchone() == (1,)
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            retry = OperationExecutor.for_archive_root(archive_root)
            retry_args = SessionDeleteArgs(archive=archive, session_ids=(session_id,))
            retry_preview = retry.prepare_bound_for_archive(binding, retry_args, principal, archive_root=archive_root)
            retry_authorization = retry.authorize_bound(binding, retry_preview, principal)
            assert (
                retry.execute_bound(binding, retry_preview, retry_authorization, retry_args).status
                == "already_satisfied"
            )

    def test_version_drift_is_terminal_not_replayable_and_does_not_repeat_at_restart(self, tmp_path: Path) -> None:
        """A retired operation version is terminalized once and never replayed.

        Anti-vacuity: drop the version check and today's actuator replays the
        v99 plan, deleting the session.
        """

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        initialize_active_archive_root(archive_root)
        session_id = _seed_archive_session(archive_root, native_id="version-drift")
        actuator = SessionDeleteActuator()
        binding = runtime_operation_binding(actuator)
        principal = MutationPrincipal("test", frozenset({"archive.delete_session"}), "api", "write")
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor.for_archive_root(archive_root)
            args = SessionDeleteArgs(archive=archive, session_ids=(session_id,))
            preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=archive_root)
            authorization = executor.authorize_bound(binding, preview, principal)
            assert executor._audit is not None
            operation_id = executor._audit.consume_authorization_and_start(preview, authorization)
        with closing(sqlite3.connect(archive_root / "audit.db")) as conn, conn:
            conn.execute("UPDATE operation_runs SET operation_version = 99 WHERE operation_id = ?", (operation_id,))
            conn.execute(
                "UPDATE operation_attempts SET worker_id = 'pid:999999999:0' WHERE operation_id = ?", (operation_id,)
            )
            conn.commit()

        recover_on_admitted_owner(archive_root)
        recover_on_admitted_owner(archive_root)

        with closing(sqlite3.connect(archive_root / "audit.db")) as conn, conn:
            assert conn.execute(
                "SELECT status, terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
            ).fetchone() == ("failed", "recovery_not_replayable")
            assert conn.execute(
                "SELECT COUNT(*) FROM operation_events WHERE operation_id = ? AND event_type = 'recovery_resolved'",
                (operation_id,),
            ).fetchone() == (1,)
        with closing(sqlite3.connect(archive_root / "index.db")) as conn, conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)

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
        with closing(sqlite3.connect(archive_root / "index.db")) as conn, conn:
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
    def test_production_reset_commits_user_and_index_policies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Identity reset commits its durable suppression and index delete separately.

        Anti-vacuity: deleting the index gateway call still removes the row
        but leaves the required post-commit cache invalidation absent.
        """
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="reset-effects")
        invalidated: list[bool] = []
        routes: list[tuple[WriteOperation, str]] = []
        original_commit = ArchiveWriteGateway.commit_write_sync

        def observe_commit(self: ArchiveWriteGateway, op: WriteOperation, payload: dict[str, object]) -> object:
            routes.append((op, str(payload.get("effect_scope", "archive-index"))))
            return original_commit(self, op, payload)

        monkeypatch.setattr("polylogue.storage.fts.fts_lifecycle.ensure_fts_triggers_sync", lambda _conn: None)
        monkeypatch.setattr("polylogue.storage.search.cache.invalidate_search_cache", lambda: invalidated.append(True))
        monkeypatch.setattr(ArchiveWriteGateway, "commit_write_sync", observe_commit)

        actuator = IdentityResetActuator()
        executor = OperationExecutor()
        args = IdentityResetArgs(archive_root=archive_root, session_ids=(session_id,), reason="test reset")
        plan = executor.prepare(actuator, args)
        authorization = executor.authorize(
            actuator, plan, actor="test", role="write", capability="test", confirmation_strength="confirm_flag"
        )
        receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.affected_count == 1
        assert routes == [(WriteOperation.RESET, "user-overlay"), (WriteOperation.RESET, "archive-index")]
        assert invalidated == [True]

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
        with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
            assert conn.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'suppression'").fetchone()[0] == 1
        with closing(sqlite3.connect(archive_root / "index.db")) as conn, conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0

    def test_a_session_absent_from_the_index_is_still_tombstoned(self, tmp_path: Path) -> None:
        """A missing index row must not cancel the durable tombstone.

        This previously asserted the opposite (``plan.target_refs == ()``):
        ``prepare`` re-filtered the caller's already-resolved ids against
        ``index.db``. A session that disappeared between CLI resolution and
        PREPARE therefore produced an empty plan, ``apply`` returned
        ``already_satisfied`` without writing any suppression, and the CLI
        still reported ``ok`` -- so the next ingest or index rebuild made the
        content the operator deleted visible again. The durable user.db
        suppression does not depend on the rebuildable row existing, and a
        suppression for an id that never existed is inert, so the asymmetry
        resolves toward writing it.

        Anti-vacuity: restoring the existence filter in
        ``IdentityResetActuator.prepare`` makes ``target_refs`` empty and the
        suppression count zero. Leaving the index untouched does not make it
        green either -- the surviving session's row is asserted below.
        """
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        _seed_archive_session(archive_root, native_id="real-one")
        vanished = "codex-session:vanished"

        actuator = IdentityResetActuator()
        executor = OperationExecutor()
        args = IdentityResetArgs(archive_root=archive_root, session_ids=(vanished,), reason="x")
        plan = actuator.prepare(args)
        assert plan.target_refs == (f"session:{vanished}",)

        authorization = executor.authorize(
            actuator, plan, actor="test", role="write", capability="test", confirmation_strength="confirm_flag"
        )
        receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.status == "applied"
        assert receipt.affected_count == 1
        # The receipt names what had no rebuildable row, so a zero deleted-row
        # count can never be read as "nothing was tombstoned".
        from polylogue.operations.machine_receipts import IdentityResetHistoricalReceipt

        history = receipt.historical_receipt
        assert isinstance(history, IdentityResetHistoricalReceipt)
        assert history.deleted_archive_rows == 0
        assert history.tombstoned_without_index_row_count == 1
        assert receipt.target_refs == (f"session:{vanished}",)
        with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM assertions WHERE kind = 'suppression' AND target_ref = ?",
                    (f"session:{vanished}",),
                ).fetchone()[0]
                == 1
            )
        with closing(sqlite3.connect(archive_root / "index.db")) as conn, conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


class TestTagAddActuator:
    def test_production_tag_and_metadata_writes_use_user_overlay_policy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """User assertions commit through the gateway without stale-index work.

        Anti-vacuity: routing either writer as an archive-index write invokes
        the failing FTS stub; removing either gateway call makes its recorded
        production route disappear.
        """
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="user-overlay-effects")
        routes: list[tuple[WriteOperation, str]] = []
        cache_invalidations: list[bool] = []
        original_commit = ArchiveWriteGateway.commit_write_sync

        def observe_commit(self: ArchiveWriteGateway, op: WriteOperation, payload: dict[str, object]) -> object:
            routes.append((op, str(payload.get("effect_scope", "archive-index"))))
            return original_commit(self, op, payload)

        monkeypatch.setattr(
            "polylogue.storage.fts.fts_lifecycle.ensure_fts_triggers_sync",
            lambda _conn: pytest.fail("user-overlay mutation touched index FTS"),
        )
        monkeypatch.setattr(
            "polylogue.storage.search.cache.invalidate_search_cache", lambda: cache_invalidations.append(True)
        )
        monkeypatch.setattr(ArchiveWriteGateway, "commit_write_sync", observe_commit)

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            assert archive.add_user_tags((session_id,), ("review",)) == 1
            assert archive.set_user_metadata((session_id,), (("priority", "high"),)) == 1
            assert archive.remove_user_tags((session_id,), ("review",)) == 1
            assert archive.delete_user_metadata(session_id, "priority") == 1
            assert archive.add_user_tags((session_id,), ("review",)) == 1
            assert archive.add_user_tags((session_id,), ("review",)) == 0

        assert routes == [
            (WriteOperation.TAG_UPDATE, "user-overlay"),
            (WriteOperation.METADATA_UPDATE, "user-overlay"),
            (WriteOperation.TAG_UPDATE, "user-overlay"),
            (WriteOperation.METADATA_UPDATE, "user-overlay"),
            (WriteOperation.TAG_UPDATE, "user-overlay"),
            (WriteOperation.TAG_UPDATE, "user-overlay"),
        ]
        # Cached search results carry session tags and answer tag filters, so
        # each tag write that changed a row drops the result cache (#5727);
        # metadata writes and the no-op re-add leave it alone.
        assert cache_invalidations == [True, True, True]

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
        with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
            rows = conn.execute("SELECT key FROM assertions WHERE kind = 'tag' AND status != 'deleted'").fetchall()
        assert [r[0] for r in rows] == ["review"]

    def test_qualified_origin_prefix_keeps_read_and_tag_target_in_its_namespace(self, tmp_path: Path) -> None:
        from polylogue.operations.daemon_reads import execute_read_operation

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        codex_session_id = _seed_archive_session(archive_root, native_id="abcdef")
        unknown_raw_id = "raw-unknown-collision"
        with closing(sqlite3.connect(archive_root / "source.db")) as conn, conn:
            conn.execute(
                "INSERT INTO raw_sessions (raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms) "
                "VALUES (?, 'unknown-export', 'abc', ?, zeroblob(32), 0, 1000)",
                (unknown_raw_id, str(archive_root / "unknown.json")),
            )
        with closing(sqlite3.connect(archive_root / "index.db")) as conn, conn:
            conn.execute(
                "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash, created_at_ms, updated_at_ms) "
                "VALUES ('abc', 'unknown-export', ?, 'Unknown collision', zeroblob(32), 1000, 2000)",
                (unknown_raw_id,),
            )

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            read = execute_read_operation(
                "session.read", {"ref": "session:codex-session:abc"}, archive=archive, serving_identity="test"
            )
            assert read["session_id"] == codex_session_id
            assert archive.resolve_session_id("codex:abcdef") == codex_session_id
            assert archive.resolve_session_id(codex_session_id) == codex_session_id
            actuator = TagAddActuator()
            args = TagAddArgs(archive=archive, session_id="codex-session:abc", tag="right-session")
            plan = actuator.prepare(args)
            assert plan.target_refs == (f"session:{codex_session_id}",)
            actuator.apply(plan, args)
            with pytest.raises(KeyError):
                archive.resolve_session_id("nonsense:abc")

        _seed_archive_session(archive_root, native_id="abc-other")
        with ArchiveStore.open_existing(archive_root) as archive:
            with pytest.raises(ValueError, match="ambiguous"):
                archive.resolve_session_id("codex-session:abc")

        with closing(sqlite3.connect(archive_root / "user.db")) as conn:
            targets = conn.execute("SELECT target_ref FROM assertions WHERE kind = 'tag'").fetchall()
        assert targets == [(f"session:{codex_session_id}",)]

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

    def test_shared_executor_keeps_concurrent_bound_execution_scopes_isolated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_ids = (
            _seed_archive_session(archive_root, native_id="concurrent-first"),
            _seed_archive_session(archive_root, native_id="concurrent-second"),
        )
        actuators = (TagAddActuator(), TagAddActuator())
        bindings = tuple(runtime_operation_binding(actuator) for actuator in actuators)
        principal = MutationPrincipal("test", frozenset({"archive.add_tag"}), "api", "write")
        executor = OperationExecutor.for_archive_root(archive_root)
        previews: list[MutationPreview] = []
        authorizations: list[MutationAuthorization] = []
        for index, (session_id, binding) in enumerate(zip(session_ids, bindings, strict=True)):
            with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
                args = TagAddArgs(archive=archive, session_id=session_id, tag=f"concurrent-{index}")
                preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=archive_root)
            previews.append(preview)
            authorizations.append(executor.authorize_bound(binding, preview, principal))

        execute_barrier = threading.Barrier(2)
        first_scope_installed = threading.Event()
        first_execute_done = threading.Event()
        original_execute = OperationExecutor.execute

        def synchronized_execute(
            self: OperationExecutor,
            actuator: Any,
            plan: MutationPlan,
            authorization: MutationAuthorization,
            args: Any,
        ) -> MutationReceipt:
            if actuator is actuators[0]:
                first_scope_installed.set()
            execute_barrier.wait(timeout=10)
            if actuator is actuators[0]:
                try:
                    return original_execute(self, actuator, plan, authorization, args)
                finally:
                    first_execute_done.set()
            if not first_execute_done.wait(timeout=10):
                raise TimeoutError("first executor route did not complete")
            return original_execute(self, actuator, plan, authorization, args)

        monkeypatch.setattr(OperationExecutor, "execute", synchronized_execute)

        def execute_tag(index: int) -> MutationReceipt:
            if index == 1 and not first_scope_installed.wait(timeout=10):
                raise TimeoutError("first bound execution did not install its scope")
            with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
                args = TagAddArgs(
                    archive=archive,
                    session_id=session_ids[index],
                    tag=f"concurrent-{index}",
                )
                return executor.execute_bound(bindings[index], previews[index], authorizations[index], args)

        with ThreadPoolExecutor(max_workers=2) as pool:
            receipts = tuple(pool.map(execute_tag, range(2)))

        assert [receipt.status for receipt in receipts] == ["applied", "applied"]
        assert all(receipt.operation_id is not None for receipt in receipts)
        with closing(sqlite3.connect(archive_root / "user.db")) as connection, connection:
            tags = connection.execute(
                "SELECT key FROM assertions WHERE kind = 'tag' AND status != 'deleted' ORDER BY key"
            ).fetchall()
        assert tags == [("concurrent-0",), ("concurrent-1",)]
        with closing(sqlite3.connect(archive_root / "audit.db")) as connection, connection:
            audit_rows = connection.execute(
                "SELECT status, unknown_count FROM operation_runs ORDER BY operation_id"
            ).fetchall()
        assert audit_rows == [("completed", 0), ("completed", 0)]

    def test_bound_prevalidation_does_not_escape_into_deferred_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="deferred-context")
        actuator = TagAddActuator()
        binding = runtime_operation_binding(actuator)
        principal = MutationPrincipal("test", frozenset({"archive.add_tag"}), "api", "write")
        executor = OperationExecutor.for_archive_root(archive_root)
        deferred: list[tuple[Context, MutationPlan, TagAddArgs]] = []
        original_apply = TagAddActuator.apply

        def capture_context(self: TagAddActuator, plan: MutationPlan, args: TagAddArgs) -> MutationReceipt:
            from contextvars import copy_context

            deferred.append((copy_context(), plan, args))
            return original_apply(self, plan, args)

        monkeypatch.setattr(TagAddActuator, "apply", capture_context)
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            args = TagAddArgs(archive=archive, session_id=session_id, tag="once")
            preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=archive_root)
            authorization = executor.authorize_bound(binding, preview, principal)
            assert executor.execute_bound(binding, preview, authorization, args).status == "applied"
            inherited, applied_plan, applied_args = deferred.pop()
            with pytest.raises(PlanStaleError):
                inherited.run(executor.execute, actuator, applied_plan, authorization, applied_args)


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
        with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
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
    def test_startup_replays_partial_tags_with_the_original_author(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dropping authors from the durable plan misattributes the recovered suffix."""
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        first_id = _seed_archive_session(archive_root, native_id="bulk-restart-first")
        second_id = _seed_archive_session(archive_root, native_id="bulk-restart-second")
        actuator = BulkTagActuator()
        binding = runtime_operation_binding(actuator)
        principal = MutationPrincipal("test", frozenset({"archive.bulk_tag_sessions"}), "api", "write")
        author_ref, author_kind = "agent:synthetic", "agent"

        class InterruptedApply(BaseException):
            pass

        original_add = ArchiveStore.add_user_tags

        def interrupt_second(
            archive: ArchiveStore, session_ids: tuple[str, ...], tags: tuple[str, ...], **kwargs: Any
        ) -> int:
            if session_ids == (second_id,):
                raise InterruptedApply
            return original_add(archive, session_ids, tags, **kwargs)

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor.for_archive_root(archive_root)
            args = BulkTagArgs(
                archive,
                (first_id, second_id, "missing-neutral"),
                ("neutral-tag",),
                author_ref=author_ref,
                author_kind=author_kind,
            )
            preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=archive_root)
            authorization = executor.authorize_bound(binding, preview, principal, confirmation_strength="bound_token")
            assert executor._audit is not None
            operation_id = executor._audit.consume_authorization_and_start(preview, authorization)
            with monkeypatch.context() as patch:
                patch.setattr(ArchiveStore, "add_user_tags", interrupt_second)
                with pytest.raises(InterruptedApply):
                    actuator.apply(preview.plan, args)

        with closing(sqlite3.connect(archive_root / "user.db")) as conn:
            assert conn.execute("SELECT target_ref FROM assertions WHERE kind='tag'").fetchall() == [
                (f"session:{first_id}",)
            ]
        with closing(sqlite3.connect(archive_root / "audit.db")) as conn, conn:
            conn.execute(
                "UPDATE operation_attempts SET worker_id='pid:999999999:0' WHERE operation_id=?", (operation_id,)
            )

        recover_on_admitted_owner(archive_root)

        with closing(sqlite3.connect(archive_root / "user.db")) as conn:
            rows = conn.execute(
                "SELECT target_ref, author_ref, author_kind FROM assertions WHERE kind='tag' ORDER BY target_ref"
            ).fetchall()
        assert rows == [(f"session:{sid}", author_ref, author_kind) for sid in (first_id, second_id)]
        with closing(sqlite3.connect(archive_root / "audit.db")) as conn:
            assert conn.execute(
                "SELECT status, terminal_reason FROM operation_runs WHERE operation_id=?", (operation_id,)
            ).fetchone() == ("completed", "recovered_complete")

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
        with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE kind = 'tag' AND status != 'deleted'"
            ).fetchone()[0]
        assert count == 2

    def test_unresolved_session_id_is_a_degraded_outcome(self, tmp_path: Path) -> None:
        """A caller-named id the archive cannot resolve is a named gap.

        Anti-vacuity: restore the silent ``except KeyError: continue`` split
        (dropping the id instead of returning it as unresolved) and the
        receipt reports ``ok`` over the smaller set -- this test is then red
        on the outcome state, the gap reason, and the named id.
        """

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="bulk-degraded")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = BulkTagActuator()
            executor = OperationExecutor()
            args = BulkTagArgs(archive=archive, session_ids=(session_id, "excised-mid-flight"), tags=("a",))
            plan = executor.prepare(actuator, args)
            assert plan.target_refs == (f"session:{session_id}",)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

        outcome = cast("dict[str, Any]", receipt.domain_receipt["outcome"])
        assert outcome["state"] == "degraded"
        assert outcome["reason"] == "unresolved_session_ids"
        assert outcome["detail"]["unresolved_session_ids"] == ["excised-mid-flight"]
        assert receipt.domain_receipt["unresolved_session_ids"] == ["excised-mid-flight"]
        assert receipt.detail == "unresolved_session_ids"

    def test_every_id_unresolved_is_degraded_not_empty(self, tmp_path: Path) -> None:
        """Zero rows behind a named gap is never reported as an empty scope.

        Anti-vacuity: decide the outcome from ``matched`` alone (dropping the
        gap) and this returns ``empty``.
        """

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        _seed_archive_session(archive_root, native_id="bulk-all-gone")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = BulkMetadataSetActuator()
            executor = OperationExecutor()
            args = BulkMetadataSetArgs(archive=archive, session_ids=("gone-a", "gone-b"), pairs=(("k", "v"),))
            plan = executor.prepare(actuator, args)
            assert plan.target_refs == ()
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

        outcome = cast("dict[str, Any]", receipt.domain_receipt["outcome"])
        assert outcome["state"] == "degraded"
        assert receipt.domain_receipt["unresolved_session_ids"] == ["gone-a", "gone-b"]


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
        with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
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
        with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
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
    def test_block_target_is_stored_and_revalidated_by_stable_id(self, tmp_path: Path) -> None:
        from tests.infra.storage_records import SessionBuilder

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        builder = SessionBuilder(archive_root / "index.db", "stable-mark")
        builder.provider("codex").add_message(
            "message-1", role="user", text="hello", blocks=[{"type": "text", "text": "hello"}]
        ).save()
        session_id = builder.native_session_id()

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            row = archive._conn.execute("SELECT block_id FROM blocks WHERE session_id=?", (session_id,)).fetchone()
            assert row is not None
            block_id = str(row[0])
            source_guard_calls: list[str] = []
            args = MarkArgs(
                archive,
                "block",
                block_id,
                "star",
                session_id,
                source_guard=lambda: source_guard_calls.append("checked"),
            )
            actuator = MarkAddActuator()
            plan = actuator.prepare(args)
            actuator.apply(plan, args)
            assert source_guard_calls == ["checked", "checked"]
            target_ref = archive._conn.execute("SELECT target_ref FROM assertions WHERE kind='mark'").fetchone()[0]
            assert target_ref == f"block:{block_id}"

            with pytest.raises(ValueError, match="must be resolved to a stable block_id"):
                actuator.prepare(MarkArgs(archive, "block", "message-1:0", "star", session_id))

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

            with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
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

            with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
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
    observed_pass_id: str,
    frontier: bool = False,
    reason: str = "immutable source/index preconditions changed after the inspection pass",
) -> None:
    """Seed one real, unresolved ``raw_authority_blockers`` row plus the one
    real ``raw_sessions`` row a non-frontier resolution needs to replan it.

    ``frontier`` seeds a ``frontier_obligation``-kind blocker: the current
    frontier plan shape the prepared inspector writes. Without
    it the row is a ``stale_plan`` -- a durable snapshot predating that shape,
    which the resolver re-derives from live evidence instead of trusting.
    """
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
            canonical_source_path=f"{blocker_id}.jsonl",
            acquired_at_ms=1000,
            raw_id=raw_id,
        )
    witness_schema = "polylogue.raw-authority-frontier-plan.v1" if frontier else "polylogue.raw-authority-plan.v1"
    input_digest = hashlib.sha256(plan_id.encode("utf-8")).hexdigest()
    observed_json = "{}"
    with closing(sqlite3.connect(archive_root / "source.db")) as conn, conn:
        conn.execute("PRAGMA foreign_keys = ON")
        # The blocker is keyed on the plan's content address and carries the
        # plan snapshot itself: that snapshot, not a join into a plan ledger,
        # is what every reader resolves against.
        conn.execute(
            """
            INSERT INTO raw_authority_blockers (
                blocker_id, plan_input_digest, observed_pass_id, reason, expected_json,
                observed_json, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, 1000)
            """,
            (
                blocker_id,
                input_digest,
                observed_pass_id,
                reason,
                json.dumps(
                    {
                        "plan_id": plan_id,
                        "input_digest": input_digest,
                        "input_raw_ids": [raw_id],
                        "logical_keys": [],
                        "authority_witness": {"schema": witness_schema},
                        "source_preconditions": {},
                        "index_preconditions": {},
                    }
                ),
                observed_json,
            ),
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

            with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
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
            with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
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

            with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
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


class TestCaptureAssertionCandidateActuator:
    """Real user-tier proof for the terminal candidate capture route."""

    def test_executor_lifecycle_writes_and_replays_idempotently(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        _seed_archive_session(archive_root, native_id="candidate-capture")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            actuator = CaptureAssertionCandidateActuator()
            args = CaptureAssertionCandidateArgs(
                archive=archive,
                body_text="candidate body",
                kind=AssertionKind.LESSON,
                refs=(),
                scope_refs=("repo:polylogue",),
                cwd=None,
                author_ref="agent:test",
                author_kind="agent",
                idempotency_key="candidate-key",
                assertion_id="assertion-terminal-note:" + hashlib.sha256(b"agent:test\0candidate-key").hexdigest(),
                ttl_seconds=60,
            )
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            first = executor.execute(actuator, plan, authorization, args)
            replay_plan = executor.prepare(actuator, args)
            replay_authorization = executor.authorize(
                actuator,
                replay_plan,
                actor="test",
                role="write",
                capability="test",
                confirmation_strength="role_only",
            )
            replay = executor.execute(actuator, replay_plan, replay_authorization, args)

        assert first.status == "applied"
        assert replay.status == "already_satisfied"
        with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
            row = conn.execute(
                "SELECT kind, status, body_text, author_ref, scope_ref FROM assertions WHERE key = 'terminal-note'"
            ).fetchone()
        assert row == ("lesson", "candidate", "candidate body", "agent:test", "repo:polylogue")

    def test_role_only_authorize_succeeds(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        _seed_archive_session(archive_root, native_id="candidate-ac4")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = CaptureAssertionCandidateActuator()
            args = CaptureAssertionCandidateArgs(
                archive=archive,
                body_text="candidate body",
                kind=AssertionKind.NOTE,
                refs=(),
                scope_refs=(),
                cwd=None,
                author_ref="agent:test",
                author_kind="agent",
                idempotency_key=None,
                assertion_id="assertion-terminal-note:one-shot",
                ttl_seconds=None,
            )
            plan = OperationExecutor().prepare(actuator, args)
            # Reversible candidate capture does not require an interactive token.
            OperationExecutor().authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )


class TestBlackboardPostActuator:
    """Phase 6 (t46.9/kwsb.2): closes the `blackboard_post` declared-not-routed row.

    ``test_full_lifecycle_writes_the_note_assertion`` fails if
    ``BlackboardPostActuator.apply`` stops calling
    ``ArchiveStore.post_blackboard_note``. Unlike every other actuator in
    this module, ``prepare`` never reads live state (the note id is minted
    by the caller before the actuator runs, mirroring
    ``AnnotationSaveArgs.annotation_id``), so there is no live-state race to
    revalidate -- the plan is a pure function of the caller-supplied args.
    """

    def test_full_lifecycle_writes_the_note_assertion(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="blackboard")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            actuator = BlackboardPostActuator()
            args = BlackboardPostArgs(
                archive=archive,
                note_id="note-bb-1",
                body="kind=blocker\ntitle=T\ncontent=some finding",
                target_type="session",
                target_id=session_id,
                author_ref="agent:test",
                author_kind="agent",
                evidence_refs=(),
                staleness=None,
                context_policy=None,
            )
            plan = executor.prepare(actuator, args)
            assert plan.target_refs == ("blackboard:note-bb-1",)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

            assert receipt.status == "applied"
            assert receipt.domain_receipt["note_id"] == "note-bb-1"

            with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
                row = conn.execute(
                    "SELECT status, body_text, author_ref, author_kind FROM assertions WHERE kind = 'note'"
                ).fetchone()
            assert row[0] != "deleted"
            assert row[1] == args.body
            assert row[2] == "agent:test"
            assert row[3] == "agent"

    def test_role_only_authorize_succeeds(self, tmp_path: Path) -> None:
        """AC4: blackboard post is reversible class -- role_only, not confirm_flag."""
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        _seed_archive_session(archive_root, native_id="blackboard-ac4")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            actuator = BlackboardPostActuator()
            args = BlackboardPostArgs(
                archive=archive,
                note_id="note-bb-ac4",
                body="x",
                target_type=None,
                target_id=None,
                author_ref=None,
                author_kind="user",
                evidence_refs=(),
                staleness=None,
                context_policy=None,
            )
            plan = executor.prepare(actuator, args)
            # Does not raise ConfirmationRequiredError.
            executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )

    def test_repeated_calls_mint_distinct_notes(self, tmp_path: Path) -> None:
        """Append-only semantics: two posts with distinct caller-minted ids never collapse."""
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        _seed_archive_session(archive_root, native_id="blackboard-distinct")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            actuator = BlackboardPostActuator()
            for note_id in ("note-a", "note-b"):
                args = BlackboardPostArgs(
                    archive=archive,
                    note_id=note_id,
                    body="same body",
                    target_type=None,
                    target_id=None,
                    author_ref=None,
                    author_kind="user",
                    evidence_refs=(),
                    staleness=None,
                    context_policy=None,
                )
                plan = executor.prepare(actuator, args)
                authorization = executor.authorize(
                    actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
                )
                executor.execute(actuator, plan, authorization, args)

            with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
                count = conn.execute("SELECT COUNT(*) FROM assertions WHERE kind = 'note'").fetchone()[0]
            assert count == 2


class TestBlockerResolveActuator:
    def test_prepare_finds_a_real_unresolved_blocker(self, tmp_path: Path) -> None:
        root = tmp_path / "archive"

        def check(args: BlockerResolveArgs) -> None:
            plan = BlockerResolveActuator().prepare(args)
            assert plan.target_refs == ("raw-authority-blocker:blocker-prepare",)
            assert plan.context["kind"] == "stale_plan"
            with closing(sqlite3.connect(root / "source.db")) as conn, conn:
                assert (
                    conn.execute(
                        "SELECT resolved_at_ms FROM raw_authority_blockers WHERE blocker_id=?", ("blocker-prepare",)
                    ).fetchone()[0]
                    is None
                )

        _run_prepared_blocker_case(
            root,
            {
                "blocker_id": "blocker-prepare",
                "plan_id": "raw-replay:prepare-plan",
                "observed_pass_id": "raw-authority-frontier-pass:prepare",
            },
            lambda phase: phase("blocker-prepare", "ack", check),
        )

    def test_prepare_on_unknown_blocker_yields_empty_plan(self, tmp_path: Path) -> None:
        def check(args: BlockerResolveArgs) -> None:
            plan = BlockerResolveActuator().prepare(args)
            assert plan.target_refs == ()
            assert plan.context["found"] is False

        _run_prepared_blocker_case(tmp_path / "archive", None, lambda phase: phase("does-not-exist", "ack", check))

    def test_execute_resolves_and_reopens_replanning(self, tmp_path: Path) -> None:
        root = tmp_path / "archive"

        def check(args: BlockerResolveArgs) -> None:
            actuator = BlockerResolveActuator()
            executor = OperationExecutor()
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="confirm_flag"
            )
            receipt = executor.execute(actuator, plan, authorization, args)
            assert receipt.status == "applied"

        _run_prepared_blocker_case(
            root,
            {
                "blocker_id": "blocker-execute",
                "plan_id": "raw-replay:execute-plan",
                "observed_pass_id": "raw-authority-frontier-pass:execute",
            },
            lambda phase: phase("blocker-execute", "current path is authoritative", check),
        )
        with closing(sqlite3.connect(root / "source.db")) as conn, conn:
            row = conn.execute(
                "SELECT resolved_at_ms,resolution FROM raw_authority_blockers WHERE blocker_id=?", ("blocker-execute",)
            ).fetchone()
        assert row[0] is not None and "current path is authoritative" in str(row[1])

    def test_role_only_authorize_refuses(self, tmp_path: Path) -> None:
        """The reset-class actuator still requires confirm_flag authorization."""

        def check(args: BlockerResolveArgs) -> None:
            actuator = BlockerResolveActuator()
            executor = OperationExecutor()
            plan = executor.prepare(actuator, args)
            with pytest.raises(ConfirmationRequiredError):
                executor.authorize(
                    actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
                )

        _run_prepared_blocker_case(
            tmp_path / "archive",
            {
                "blocker_id": "blocker-ac4-contrast",
                "plan_id": "raw-replay:ac4-contrast-plan",
                "observed_pass_id": "raw-authority-frontier-pass:ac4-contrast",
            },
            lambda phase: phase("blocker-ac4-contrast", "ack", check),
        )

    def test_concurrent_resolution_between_prepare_and_execute_raises_plan_stale(self, tmp_path: Path) -> None:
        """A real intervening acknowledgement invalidates the original prepared plan."""
        actuator = BlockerResolveActuator()
        executor = OperationExecutor()

        def prepare(args: BlockerResolveArgs) -> tuple[Any, Any]:
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="confirm_flag"
            )
            return plan, authorization

        def exercise(phase: _BlockerPhase) -> None:
            plan, authorization = phase("blocker-stale", "ack", prepare)

            def acknowledge(args: BlockerResolveArgs) -> None:
                intervening, token = prepare(args)
                assert executor.execute(actuator, intervening, token, args).status == "applied"

            phase("blocker-stale", "resolved elsewhere first", acknowledge)

            def stale(args: BlockerResolveArgs) -> None:
                with pytest.raises(PlanStaleError):
                    executor.execute(actuator, plan, authorization, args)

            phase("blocker-stale", "ack", stale)

        _run_prepared_blocker_case(
            tmp_path / "archive",
            {
                "blocker_id": "blocker-stale",
                "plan_id": "raw-replay:stale-plan",
                "observed_pass_id": "raw-authority-frontier-pass:stale",
            },
            exercise,
        )

    def test_frontier_obligation_resolves_with_no_extra_operator_authority(self, tmp_path: Path) -> None:
        """The original frontier obligation needs no second judgment or assertion contract."""
        root = tmp_path / "archive"

        def check(args: BlockerResolveArgs) -> None:
            actuator = BlockerResolveActuator()
            executor = OperationExecutor()
            plan = actuator.prepare(args)
            assert plan.context["kind"] == "frontier_obligation"
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="confirm_flag"
            )
            assert executor.execute(actuator, plan, authorization, args).status == "applied"

        _run_prepared_blocker_case(
            root,
            {
                "blocker_id": "blocker-obligation",
                "plan_id": "raw-replay:obligation-plan",
                "observed_pass_id": "raw-authority-frontier-pass:obligation",
                "frontier": True,
                "reason": "missing bytes require reacquisition",
            },
            lambda phase: phase("blocker-obligation", "ack", check),
        )
        with closing(sqlite3.connect(root / "source.db")) as conn, conn:
            assert (
                conn.execute(
                    "SELECT resolved_at_ms FROM raw_authority_blockers WHERE blocker_id=?", ("blocker-obligation",)
                ).fetchone()[0]
                is not None
            )


class TestListUnresolvedRawAuthorityBlockersPagination:
    """Anti-vacuity for the Codex-flagged hard 500-row cap (PR #3258): prove
    truncated/total_count/next_offset let a caller page past --limit rather
    than silently losing rows beyond the first page."""

    def test_pagination_metadata_reflects_a_second_page(self, tmp_path: Path) -> None:
        from polylogue.storage.raw_authority import list_unresolved_raw_authority_blockers

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore(archive_root):
            pass
        for index in range(3):
            _seed_raw_authority_blocker(
                archive_root,
                blocker_id=f"blocker-page-{index}",
                plan_id=f"raw-replay:page-plan-{index}",
                observed_pass_id=f"raw-authority-frontier-pass:page-{index}",
            )

        first_page = list_unresolved_raw_authority_blockers(archive_root, limit=2, offset=0)
        assert first_page["total_count"] == 3
        assert first_page["returned_count"] == 2
        assert first_page["truncated"] is True
        assert first_page["next_offset"] == 2

        second_page = list_unresolved_raw_authority_blockers(archive_root, limit=2, offset=2)
        assert second_page["returned_count"] == 1
        assert second_page["truncated"] is False
        assert second_page["next_offset"] is None

        first_blockers = cast("list[dict[str, object]]", first_page["blockers"])
        second_blockers = cast("list[dict[str, object]]", second_page["blockers"])
        first_ids = {row["blocker_id"] for row in first_blockers}
        second_ids = {row["blocker_id"] for row in second_blockers}
        assert first_ids.isdisjoint(second_ids)
        assert first_ids | second_ids == {"blocker-page-0", "blocker-page-1", "blocker-page-2"}

    def test_no_blockers_reports_untruncated_empty_page(self, tmp_path: Path) -> None:
        from polylogue.storage.raw_authority import list_unresolved_raw_authority_blockers

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore(archive_root):
            pass

        payload = list_unresolved_raw_authority_blockers(archive_root)
        assert payload == {
            "blockers": [],
            "offset": 0,
            "limit": 100,
            "returned_count": 0,
            "total_count": 0,
            "truncated": False,
            "next_offset": None,
        }


class TestSavedViewActuators:
    def test_save_then_delete_round_trips_through_user_db(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()

            save_actuator = SavedViewSaveActuator()
            save_args = SavedViewSaveArgs(
                archive=archive, view_id="view-1", name="my view", query_json='{"origin":"codex-session"}'
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

            with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
                row = conn.execute("SELECT status, key FROM assertions WHERE kind = 'saved_query'").fetchone()
            assert row[0] != "deleted"
            assert row[1] == "my view"

            delete_actuator = SavedViewDeleteActuator()
            delete_args = SavedViewDeleteArgs(archive=archive, view_id="view-1")
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

            with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
                status = conn.execute("SELECT status FROM assertions WHERE kind = 'saved_query'").fetchone()[0]
            assert status == "deleted"

    def test_delete_missing_view_is_already_satisfied(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            actuator = SavedViewDeleteActuator()
            args = SavedViewDeleteArgs(archive=archive, view_id="does-not-exist")
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.status == "already_satisfied"

    def test_role_only_authorize_succeeds(self, tmp_path: Path) -> None:
        """AC4: saved views are reversible class -- role_only, not confirm_flag."""
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            actuator = SavedViewSaveActuator()
            args = SavedViewSaveArgs(archive=archive, view_id="v", name="n", query_json="{}")
            plan = executor.prepare(actuator, args)
            executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )

    def test_name_collision_resolves_victim_during_prepare(self, tmp_path: Path) -> None:
        """CodeRabbit #3262 P2: ArchiveStore.save_view soft-deletes whichever
        row currently owns the target name when a save uses a new id -- the
        plan/receipt must name and count that victim too, not just the new id."""
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            first_actuator = SavedViewSaveActuator()
            first_args = SavedViewSaveArgs(archive=archive, view_id="view-old", name="shared name", query_json="{}")
            first_plan = executor.prepare(first_actuator, first_args)
            first_authorization = executor.authorize(
                first_actuator,
                first_plan,
                actor="test",
                role="write",
                capability="test",
                confirmation_strength="role_only",
            )
            executor.execute(first_actuator, first_plan, first_authorization, first_args)

            second_actuator = SavedViewSaveActuator()
            second_args = SavedViewSaveArgs(archive=archive, view_id="view-new", name="shared name", query_json="{}")
            second_plan = executor.prepare(second_actuator, second_args)

            assert second_plan.target_refs == ("saved_view:view-new", "saved_view:view-old")
            assert second_plan.context["collision_view_id"] == "view-old"

            second_authorization = executor.authorize(
                second_actuator,
                second_plan,
                actor="test",
                role="write",
                capability="test",
                confirmation_strength="role_only",
            )
            receipt = executor.execute(second_actuator, second_plan, second_authorization, second_args)

            assert receipt.affected_count == 2
            assert receipt.domain_receipt["collision_view_id"] == "view-old"

            with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
                statuses = dict(
                    conn.execute("SELECT assertion_id, status FROM assertions WHERE kind = 'saved_query'").fetchall()
                )
            assert statuses[assertion_id_for_saved_view("view-old")] == "deleted"
            assert statuses[assertion_id_for_saved_view("view-new")] != "deleted"


class TestRecallPackActuators:
    def test_save_then_delete_round_trips_through_user_db(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()

            save_actuator = RecallPackSaveActuator()
            save_args = RecallPackSaveArgs(
                archive=archive,
                pack_id="pack-1",
                label="my pack",
                session_ids_json="[]",
                payload_json='{"schema_version":1,"label":"my pack","items":[]}',
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

            with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
                row = conn.execute("SELECT status, key FROM assertions WHERE kind = 'recall_pack'").fetchone()
            assert row[0] != "deleted"
            assert row[1] == "my pack"

            delete_actuator = RecallPackDeleteActuator()
            delete_args = RecallPackDeleteArgs(archive=archive, pack_id="pack-1")
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

            with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
                status = conn.execute("SELECT status FROM assertions WHERE kind = 'recall_pack'").fetchone()[0]
            assert status == "deleted"

    def test_delete_missing_pack_is_already_satisfied(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            actuator = RecallPackDeleteActuator()
            args = RecallPackDeleteArgs(archive=archive, pack_id="does-not-exist")
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.status == "already_satisfied"


class TestWorkspaceActuators:
    def test_save_then_delete_round_trips_through_user_db(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()

            save_actuator = WorkspaceSaveActuator()
            save_args = WorkspaceSaveArgs(
                archive=archive,
                workspace_id="ws-1",
                name="my workspace",
                mode="tabs",
                open_targets_json="[]",
                layout_json="{}",
                active_target_json="{}",
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

            with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
                row = conn.execute("SELECT status, key FROM assertions WHERE kind = 'workspace_note'").fetchone()
            assert row[0] != "deleted"
            assert row[1] == "my workspace"

            delete_actuator = WorkspaceDeleteActuator()
            delete_args = WorkspaceDeleteArgs(archive=archive, workspace_id="ws-1")
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

            with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
                status = conn.execute("SELECT status FROM assertions WHERE kind = 'workspace_note'").fetchone()[0]
            assert status == "deleted"

    def test_delete_missing_workspace_is_already_satisfied(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            actuator = WorkspaceDeleteActuator()
            args = WorkspaceDeleteArgs(archive=archive, workspace_id="does-not-exist")
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.status == "already_satisfied"

    def test_role_only_authorize_succeeds(self, tmp_path: Path) -> None:
        """AC4: workspaces are reversible class -- role_only, not confirm_flag."""
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            actuator = WorkspaceSaveActuator()
            args = WorkspaceSaveArgs(
                archive=archive,
                workspace_id="ws",
                name="n",
                mode="tabs",
                open_targets_json="[]",
                layout_json="{}",
                active_target_json="{}",
            )
            plan = executor.prepare(actuator, args)
            executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )

    def test_name_collision_resolves_victim_during_prepare(self, tmp_path: Path) -> None:
        """CodeRabbit #3262 P2: ArchiveStore.save_workspace soft-deletes whichever
        row currently owns the target name when a save uses a new id -- the
        plan/receipt must name and count that victim too, not just the new id."""
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            first_actuator = WorkspaceSaveActuator()
            first_args = WorkspaceSaveArgs(
                archive=archive,
                workspace_id="ws-old",
                name="shared name",
                mode="tabs",
                open_targets_json="[]",
                layout_json="{}",
                active_target_json="{}",
            )
            first_plan = executor.prepare(first_actuator, first_args)
            first_authorization = executor.authorize(
                first_actuator,
                first_plan,
                actor="test",
                role="write",
                capability="test",
                confirmation_strength="role_only",
            )
            executor.execute(first_actuator, first_plan, first_authorization, first_args)

            second_actuator = WorkspaceSaveActuator()
            second_args = WorkspaceSaveArgs(
                archive=archive,
                workspace_id="ws-new",
                name="shared name",
                mode="tabs",
                open_targets_json="[]",
                layout_json="{}",
                active_target_json="{}",
            )
            second_plan = executor.prepare(second_actuator, second_args)

            assert second_plan.target_refs == ("workspace:ws-new", "workspace:ws-old")
            assert second_plan.context["collision_workspace_id"] == "ws-old"

            second_authorization = executor.authorize(
                second_actuator,
                second_plan,
                actor="test",
                role="write",
                capability="test",
                confirmation_strength="role_only",
            )
            receipt = executor.execute(second_actuator, second_plan, second_authorization, second_args)

            assert receipt.affected_count == 2
            assert receipt.domain_receipt["collision_workspace_id"] == "ws-old"

            with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
                statuses = dict(
                    conn.execute("SELECT assertion_id, status FROM assertions WHERE kind = 'workspace_note'").fetchall()
                )
            assert statuses[assertion_id_for_workspace("ws-old")] == "deleted"
            assert statuses[assertion_id_for_workspace("ws-new")] != "deleted"


class TestCorrectionActuators:
    """Phase 5 (t46.9/kwsb.2): learning-corrections family.

    Anti-vacuity: ``test_record_then_delete_round_trips_through_user_db``
    fails if ``CorrectionRecordActuator``/``CorrectionDeleteActuator`` stop
    calling ``ArchiveStore.record_correction``/``delete_correction``.
    ``test_concurrent_record_between_authorize_and_execute_makes_clear_plan_stale``
    fails if ``CorrectionsClearActuator.prepare`` stops re-resolving the
    exact live set of correction kinds (i.e. regresses to trusting a
    stale caller-supplied target set, the excision-bypass regression
    class).
    """

    def test_record_then_delete_round_trips_through_user_db(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="correction")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()

            record_actuator = CorrectionRecordActuator()
            record_args = CorrectionRecordArgs(
                archive=archive,
                session_id=session_id,
                kind="tag_reject",
                payload={"tag": "todo"},
            )
            record_plan = executor.prepare(record_actuator, record_args)
            record_authorization = executor.authorize(
                record_actuator,
                record_plan,
                actor="test",
                role="write",
                capability="test",
                confirmation_strength="role_only",
            )
            record_receipt = executor.execute(record_actuator, record_plan, record_authorization, record_args)
            assert record_receipt.status == "applied"
            correction = cast("LearningCorrection", record_receipt.domain_receipt["correction"])
            assert correction.kind.value == "tag_reject"
            assert correction.payload == {"tag": "todo"}

            with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
                row = conn.execute("SELECT status, key FROM assertions WHERE kind = 'correction'").fetchone()
            assert row[0] != "deleted"
            assert row[1] == "tag_reject"

            delete_actuator = CorrectionDeleteActuator()
            delete_args = CorrectionDeleteArgs(archive=archive, session_id=session_id, kind="tag_reject")
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

            with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
                status = conn.execute("SELECT status FROM assertions WHERE kind = 'correction'").fetchone()[0]
            assert status == "deleted"

    def test_delete_missing_correction_is_already_satisfied(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="correction-missing")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            actuator = CorrectionDeleteActuator()
            args = CorrectionDeleteArgs(archive=archive, session_id=session_id, kind="tag_accept")
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )
            receipt = executor.execute(actuator, plan, authorization, args)

        assert receipt.status == "already_satisfied"
        assert receipt.detail == "correction_not_found"

    def test_role_only_authorize_succeeds(self, tmp_path: Path) -> None:
        """AC4: the correction family is reversible class -- role_only, not confirm_flag."""
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="correction-ac4")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            actuator = CorrectionRecordActuator()
            args = CorrectionRecordArgs(
                archive=archive, session_id=session_id, kind="summary_override", payload={"summary": "x"}
            )
            plan = executor.prepare(actuator, args)
            # Does not raise ConfirmationRequiredError.
            executor.authorize(
                actuator, plan, actor="test", role="write", capability="test", confirmation_strength="role_only"
            )

    def test_clear_removes_every_correction_for_the_session_only(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="correction-clear")
        other_session_id = _seed_archive_session(archive_root, native_id="correction-clear-other")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            executor = OperationExecutor()
            record_actuator = CorrectionRecordActuator()
            for kind, payload in (("tag_reject", {"tag": "a"}), ("tag_accept", {"tag": "b"})):
                args = CorrectionRecordArgs(archive=archive, session_id=session_id, kind=kind, payload=payload)
                plan = executor.prepare(record_actuator, args)
                authorization = executor.authorize(
                    record_actuator,
                    plan,
                    actor="test",
                    role="write",
                    capability="test",
                    confirmation_strength="role_only",
                )
                executor.execute(record_actuator, plan, authorization, args)
            other_args = CorrectionRecordArgs(
                archive=archive, session_id=other_session_id, kind="tag_reject", payload={"tag": "c"}
            )
            other_plan = executor.prepare(record_actuator, other_args)
            other_authorization = executor.authorize(
                record_actuator,
                other_plan,
                actor="test",
                role="write",
                capability="test",
                confirmation_strength="role_only",
            )
            executor.execute(record_actuator, other_plan, other_authorization, other_args)

            clear_actuator = CorrectionsClearActuator()
            clear_args = CorrectionsClearArgs(archive=archive, session_id=session_id)
            clear_plan = executor.prepare(clear_actuator, clear_args)
            assert set(clear_plan.target_refs) == {
                f"correction:{session_id}:tag_accept",
                f"correction:{session_id}:tag_reject",
            }
            clear_authorization = executor.authorize(
                clear_actuator,
                clear_plan,
                actor="test",
                role="write",
                capability="test",
                confirmation_strength="role_only",
            )
            clear_receipt = executor.execute(clear_actuator, clear_plan, clear_authorization, clear_args)
            assert clear_receipt.status == "applied"
            assert clear_receipt.affected_count == 2

            remaining = archive.list_corrections(session_id=session_id)
            assert remaining == []
            other_remaining = archive.list_corrections(session_id=other_session_id)
            assert len(other_remaining) == 1

    def test_concurrent_record_between_authorize_and_execute_makes_clear_plan_stale(self, tmp_path: Path) -> None:
        """The "excision bypass" regression class applied to bulk clear.

        A plan/authorization prepared against one live set of correction
        kinds must not silently apply to a *different* live set after a
        concurrent ``record_correction`` adds a new kind in the meantime --
        clearing a kind the caller never previewed would be exactly the
        TOCTOU gap AC3/AC5 exist to close.
        """
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="correction-stale")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            record_actuator = CorrectionRecordActuator()
            executor = OperationExecutor()
            seed_args = CorrectionRecordArgs(
                archive=archive, session_id=session_id, kind="tag_reject", payload={"tag": "a"}
            )
            seed_plan = executor.prepare(record_actuator, seed_args)
            seed_authorization = executor.authorize(
                record_actuator,
                seed_plan,
                actor="test",
                role="write",
                capability="test",
                confirmation_strength="role_only",
            )
            executor.execute(record_actuator, seed_plan, seed_authorization, seed_args)

            clear_actuator = CorrectionsClearActuator()
            clear_args = CorrectionsClearArgs(archive=archive, session_id=session_id)
            clear_plan = executor.prepare(clear_actuator, clear_args)
            assert clear_plan.target_refs == (f"correction:{session_id}:tag_reject",)
            clear_authorization = executor.authorize(
                clear_actuator,
                clear_plan,
                actor="test",
                role="write",
                capability="test",
                confirmation_strength="role_only",
            )

            # Concurrent addition out from under the held authorization.
            concurrent_args = CorrectionRecordArgs(
                archive=archive, session_id=session_id, kind="tag_accept", payload={"tag": "b"}
            )
            concurrent_plan = executor.prepare(record_actuator, concurrent_args)
            concurrent_authorization = executor.authorize(
                record_actuator,
                concurrent_plan,
                actor="test",
                role="write",
                capability="test",
                confirmation_strength="role_only",
            )
            executor.execute(record_actuator, concurrent_plan, concurrent_authorization, concurrent_args)

            with pytest.raises(PlanStaleError):
                executor.execute(clear_actuator, clear_plan, clear_authorization, clear_args)

            # Refused before mutation: both corrections are still present.
            remaining = archive.list_corrections(session_id=session_id)
            assert {correction.kind.value for correction in remaining} == {"tag_reject", "tag_accept"}


class TestDerivedMaintenanceActuators:
    """Real derived-tier effects remain behind the shared executor."""

    def test_insights_rebuild_actuator_rejects_unsealed_direct_execution(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        session_id = _seed_archive_session(archive_root, native_id="maintenance-insights")

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator = InsightsRebuildActuator()
            args = InsightsRebuildArgs(archive=archive, session_ids=(session_id,))
            executor = OperationExecutor()
            plan = executor.prepare(actuator, args)
            authorization = executor.authorize(
                actuator,
                plan,
                actor="test",
                role="write",
                capability="archive.rebuild_insights",
                confirmation_strength="role_only",
            )
            with pytest.raises(
                MutationTransactionError,
                match="only through a sealed accepted machine part owner",
            ):
                executor.execute(actuator, plan, authorization, args)


class TestFilesystemResetActuator:
    """polylogue-4fbgw: the product's largest destructive surface must write
    its audit rows before it deletes anything.

    Anti-vacuity: revert ``maintenance_reset`` to unlinking inline (or point
    the binding at a spec whose ``executor_status`` is ``declared-not-routed``)
    and ``test_the_reset_writes_preview_and_run_rows_before_deleting`` goes red,
    because audit.db carries no preview/run row for the deletion that happened.
    """

    def _actuator_args(self, tmp_path: Path) -> tuple[Path, Any]:
        from polylogue.operations.mutation_actuators import FilesystemResetArgs

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        initialize_active_archive_root(archive_root)
        doomed_file = archive_root / "scratch.cache"
        doomed_file.write_bytes(b"cache bytes")
        doomed_tree = archive_root / "blob"
        doomed_tree.mkdir()
        (doomed_tree / "aa").mkdir()
        (doomed_tree / "aa" / "blob.bin").write_bytes(b"payload")
        args = FilesystemResetArgs(
            archive_root=archive_root,
            targets=(("scratch cache", doomed_file), ("blob store", doomed_tree)),
        )
        return archive_root, args

    def test_prepare_plans_every_target_and_mutates_nothing(self, tmp_path: Path) -> None:
        from polylogue.operations.mutation_actuators import FilesystemResetActuator

        archive_root, args = self._actuator_args(tmp_path)

        plan = FilesystemResetActuator().prepare(args)

        assert plan.target_refs == (
            f"path:{archive_root / 'scratch.cache'}",
            f"path:{archive_root / 'blob'}",
        )
        assert (archive_root / "scratch.cache").exists()
        assert (archive_root / "blob" / "aa" / "blob.bin").exists()

    def test_the_reset_writes_preview_and_run_rows_before_deleting(self, tmp_path: Path) -> None:
        from polylogue.operations.mutation_actuators import FilesystemResetActuator

        archive_root, args = self._actuator_args(tmp_path)
        actuator = FilesystemResetActuator()
        binding = runtime_operation_binding(actuator)
        principal = MutationPrincipal("test", frozenset({"archive.reset"}), "cli", "write")
        executor = OperationExecutor.for_archive_root(archive_root)

        preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=archive_root)
        authorization = executor.authorize_bound(binding, preview, principal, confirmation_strength="bound_token")
        receipt = executor.execute_bound(binding, preview, authorization, args)

        assert receipt.status == "applied"
        assert receipt.affected_count == 2
        assert not (archive_root / "scratch.cache").exists()
        assert not (archive_root / "blob").exists()
        with closing(sqlite3.connect(archive_root / "audit.db")) as conn, conn:
            assert conn.execute("SELECT state FROM operation_previews").fetchone()[0] == "consumed"
            assert conn.execute("SELECT status FROM operation_runs").fetchone()[0] == "completed"
            kinds = {row[0] for row in conn.execute("SELECT target_kind FROM operation_targets").fetchall()}
            assert kinds == {"path"}
            assert conn.execute("SELECT COUNT(*) FROM operation_attempts").fetchone()[0] >= 1

    def test_a_target_absent_at_apply_is_named_not_counted(self, tmp_path: Path) -> None:
        from polylogue.operations.mutation_actuators import FilesystemResetActuator, FilesystemResetArgs

        archive_root, args = self._actuator_args(tmp_path)
        args = FilesystemResetArgs(
            archive_root=archive_root,
            targets=(*args.targets, ("ops database", archive_root / "never-existed.db")),
        )
        actuator = FilesystemResetActuator()

        receipt = actuator.apply(actuator.prepare(args), args)

        assert receipt.affected_count == 2
        assert receipt.domain_receipt["absent_at_apply"] == ["ops database"]


class TestReplayActsOnTheRecordedIdsExactly:
    """Crash-recovery replay writes only the full ids its plan recorded.

    Each case prepares a plan on ``codex-session:prefix``, lets that session
    vanish (as an interrupted apply or a concurrent delete leaves it), and
    replays the plan while ``codex-session:prefix-sibling`` survives.
    Anti-vacuity: resolve a recorded id through ``resolve_session_id`` at
    apply and its prefix fallback re-points the vanished id at the sibling --
    the replay then deletes, tags or annotates the sibling, and each case is
    red on the sibling assertion.
    """

    @staticmethod
    def _sibling_state(archive_root: Path, sibling: str) -> tuple[bool, int]:
        with closing(sqlite3.connect(archive_root / "index.db")) as conn, conn:
            exists = conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (sibling,)).fetchone() is not None
        # Every overlay the families write names its session inside
        # ``target_ref`` (``session:<id>``, ``insight:<id>``), and only the
        # sibling's own id contains ``-sibling``.
        with closing(sqlite3.connect(archive_root / "user.db")) as conn, conn:
            assertions = conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE instr(target_ref, ?) > 0 AND status != 'deleted'",
                (sibling,),
            ).fetchone()[0]
        return exists, int(assertions)

    @pytest.mark.parametrize(
        "family",
        ["delete", "tag", "bulk-tag", "metadata", "bulk-metadata", "correction"],
    )
    def test_replay_after_the_target_vanished_never_reaches_a_prefix_sibling(self, tmp_path: Path, family: str) -> None:
        from polylogue.operations.mutation_transaction import ReplayHandles

        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        target = _seed_archive_session(archive_root, native_id="prefix")
        sibling = _seed_archive_session(archive_root, native_id="prefix-sibling")
        assert sibling.startswith(target)

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            actuator: Any
            args: Any
            if family == "delete":
                actuator, args = SessionDeleteActuator(), SessionDeleteArgs(archive=archive, session_ids=(target,))
            elif family == "tag":
                actuator, args = TagAddActuator(), TagAddArgs(archive=archive, session_id=target, tag="replayed")
            elif family == "bulk-tag":
                actuator, args = BulkTagActuator(), BulkTagArgs(archive=archive, session_ids=(target,), tags=("t",))
            elif family == "metadata":
                actuator, args = (
                    MetadataSetActuator(),
                    MetadataSetArgs(archive=archive, session_id=target, key="k", value="v"),
                )
            elif family == "bulk-metadata":
                actuator, args = (
                    BulkMetadataSetActuator(),
                    BulkMetadataSetArgs(archive=archive, session_ids=(target,), pairs=(("k", "v"),)),
                )
            else:
                actuator, args = (
                    CorrectionRecordActuator(),
                    CorrectionRecordArgs(archive=archive, session_id=target, kind="tag_reject", payload={"tag": "x"}),
                )
            plan = actuator.prepare(args)
            assert target in plan.target_refs[0]
            assert archive.delete_sessions((target,)) == 1

        handles = ReplayHandles(archive_root)
        try:
            if family == "delete":
                resolution = actuator.recover(handles, plan)
                assert resolution.outcome == "complete"
                assert resolution.receipt is not None
                assert resolution.receipt.affected_count == 0
            else:
                # The recorded session is not stored: the write waits for
                # convergence rather than landing on the sibling.
                with pytest.raises(RecoveryDeferredError):
                    actuator.recover(handles, plan)
        finally:
            handles.close()

        assert self._sibling_state(archive_root, sibling) == (True, 0)


def _wal_tier(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS reset_probe(value TEXT)")
    conn.execute("INSERT INTO reset_probe VALUES ('retained')")
    conn.commit()
    return conn


@pytest.mark.parametrize("target_name", ["source.db", "user.db", "embeddings.db", "audit.db", "."])
def test_reset_never_deletes_a_durable_tier_or_a_directory_holding_tiers(tmp_path: Path, target_name: str) -> None:
    """No route deletes these: bootstrap refuses an archive missing a durable tier (07.F012).

    Anti-vacuity: drop the ``unresettable`` checks in ``apply``/``recover`` and
    the tier (or the archive directory holding it) is deleted at the seam.
    """
    from polylogue.operations.mutation_actuators import FilesystemResetActuator, FilesystemResetArgs
    from polylogue.operations.mutation_transaction import ReplayHandles
    from polylogue.operations.reset_safety import UnresettableArchiveTierError, archive_tiers_closed

    initialize_active_archive_root(tmp_path)
    db_path = tmp_path / ("user.db" if target_name == "." else target_name)
    conn = _wal_tier(db_path)
    try:
        cache = tmp_path / "reset-cache"
        cache.write_bytes(b"also retained on refusal")
        args = FilesystemResetArgs(tmp_path, (("cache", cache), ("tier", tmp_path / target_name)))
        actuator = FilesystemResetActuator()
        plan = actuator.prepare(args)
        with pytest.raises(UnresettableArchiveTierError):
            actuator.apply(plan, args)
        handles = ReplayHandles(tmp_path)
        try:
            with archive_tiers_closed(tmp_path):
                resolution = actuator.recover(handles, plan)
        finally:
            handles.close()
        assert resolution.outcome == "replay-failed"
        assert resolution.receipt is None
        assert cache.read_bytes() == b"also retained on refusal"
        assert db_path.exists()
        assert conn.execute("SELECT value FROM reset_probe").fetchone() == ("retained",)
    finally:
        conn.close()


@pytest.mark.parametrize("target_name", ["index.db", "ops.db"])
def test_derived_tier_reset_waits_for_the_closed_seam_then_deletes_the_database_and_sidecars(
    tmp_path: Path, target_name: str
) -> None:
    """A live APPLY or ordinary recovery never unlinks a derived tier (07.F012).

    Outside ``archive_tiers_closed`` the open handle keeps its data and the
    plan stays pending; inside it the database, every sidecar beside it and
    the plan's other targets are deleted.

    Anti-vacuity: drop the deferral and ordinary recovery unlinks the file
    under the open connection; drop the sidecar sweep and ``-wal`` survives.
    """
    from polylogue.operations.mutation_actuators import FilesystemResetActuator, FilesystemResetArgs
    from polylogue.operations.mutation_transaction import RecoveryDeferredError, ReplayHandles
    from polylogue.operations.reset_safety import LiveArchiveTierResetError, archive_tiers_closed

    initialize_active_archive_root(tmp_path)
    db_path = tmp_path / target_name
    conn = _wal_tier(db_path)
    wal = db_path.with_name(db_path.name + "-wal")
    cache = tmp_path / "reset-cache"
    cache.write_bytes(b"deleted with the plan")
    targets = [("cache", cache), ("tier", db_path)]
    targets += [(f"tier {suffix}", db_path.with_name(db_path.name + suffix)) for suffix in ("-wal", "-shm")]
    args = FilesystemResetArgs(tmp_path, tuple(targets))
    actuator = FilesystemResetActuator()
    plan = actuator.prepare(args)
    try:
        with pytest.raises(LiveArchiveTierResetError):
            actuator.apply(plan, args)
        handles = ReplayHandles(tmp_path)
        try:
            with pytest.raises(RecoveryDeferredError):
                actuator.recover(handles, plan)
        finally:
            handles.close()
        assert cache.exists()
        assert wal.exists()
        assert conn.execute("SELECT COUNT(*) FROM reset_probe").fetchone() == (1,)
    finally:
        conn.close()
    # Closing the last connection may checkpoint the WAL away; leave a stale
    # one behind, as a killed daemon would.
    wal.write_bytes(b"stale frames")

    handles = ReplayHandles(tmp_path)
    try:
        with archive_tiers_closed(tmp_path):
            resolution = actuator.recover(handles, plan)
    finally:
        handles.close()

    assert resolution.outcome == "complete"
    assert not db_path.exists()
    assert not wal.exists()
    assert not db_path.with_name(db_path.name + "-shm").exists()
    assert not cache.exists()


def test_seam_keeps_a_recreated_tier_and_clears_sidecars_of_a_vanished_one(tmp_path: Path) -> None:
    """Identity decides the database; a vanished database's stale sidecars still go.

    Anti-vacuity: drop the identity check and the recreated ``ops.db`` is
    deleted; tie sidecars to the database's own identity and the stale
    ``index.db-wal`` stays beside the file bootstrap creates next.
    """
    from polylogue.operations.mutation_actuators import FilesystemResetActuator, FilesystemResetArgs
    from polylogue.operations.mutation_transaction import ReplayHandles
    from polylogue.operations.reset_safety import archive_tiers_closed

    initialize_active_archive_root(tmp_path)
    index_db, ops_db = tmp_path / "index.db", tmp_path / "ops.db"
    args = FilesystemResetArgs(tmp_path, (("index database", index_db), ("ops database", ops_db)))
    actuator = FilesystemResetActuator()
    plan = actuator.prepare(args)
    index_db.unlink()
    index_db.with_name("index.db-wal").write_bytes(b"stale frames")
    ops_db.unlink()
    ops_db.write_bytes(b"recreated after the preview")
    ops_db.with_name("ops.db-wal").write_bytes(b"belongs to the new file")

    handles = ReplayHandles(tmp_path)
    try:
        with archive_tiers_closed(tmp_path):
            resolution = actuator.recover(handles, plan)
    finally:
        handles.close()

    assert resolution.outcome == "complete"
    assert not index_db.with_name("index.db-wal").exists()
    assert ops_db.read_bytes() == b"recreated after the preview"
    assert ops_db.with_name("ops.db-wal").exists()


@pytest.mark.parametrize("user_reference", ["none", "deleted_target_assertion", "surviving_evidence"])
def test_audited_excision_recovery_keeps_exact_removal_authority(
    tmp_path: Path, user_reference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovery finishes the one recorded excision attempt and nothing more.

    A surviving session's assertion whose evidence names the excised session
    keeps that reference unchanged (2e9aae50de: User anchors to an authorized,
    now-absent session survive excision); an assertion targeting the excised
    session is excised content and goes with it.
    """
    from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion
    from polylogue.storage.sqlite.reference_seal import _PreparedExcisionEmbeddingsChild
    from polylogue.storage.sqlite.write_lease import permitted_session_removals, write_lease
    from tests.infra.excision_embeddings import seed_excision_session
    from tests.infra.excision_execution import execute_excision, recover_excision

    session_id = seed_excision_session(tmp_path, native_id="recorded-excision-recovery", with_embedding=True)
    if user_reference != "none":
        survivor = seed_excision_session(tmp_path, native_id="recovery-survivor")
        with closing(sqlite3.connect(tmp_path / "user.db")) as user, user:
            upsert_assertion(
                user,
                assertion_id="recovery-protection",
                target_ref=f"session:{survivor if user_reference == 'surviving_evidence' else session_id}",
                kind=AssertionKind.NOTE,
                value={"body": "Retained content"},
                author_ref="user:local",
                author_kind="user",
                evidence_refs=(f"session:{session_id}",),
                now_ms=1,
            )

    class RecoveryBoundaryError(Exception):
        pass

    interruption = RecoveryBoundaryError("actual Source commit before paid commit")
    original_apply = _PreparedExcisionEmbeddingsChild.apply
    reached = False

    def stop_before_paid(child: _PreparedExcisionEmbeddingsChild) -> None:
        nonlocal reached
        if not reached:
            reached = True
            with closing(sqlite3.connect(tmp_path / "source.db")) as source, source:
                assert source.execute(
                    "SELECT count(*) FROM raw_sessions WHERE source_path='/fake/recorded-excision-recovery.jsonl'"
                ).fetchone() == (0,)
            raise interruption
        original_apply(child)

    monkeypatch.setattr(_PreparedExcisionEmbeddingsChild, "apply", stop_before_paid)
    retained = user_reference == "surviving_evidence"
    with pytest.raises(RecoveryBoundaryError) as caught:
        execute_excision(tmp_path, session_id, reason="recorded reason", actor="user:recorded")
    assert caught.value is interruption
    assert reached
    with closing(sqlite3.connect(tmp_path / "audit.db")) as audit, audit:
        original_attempts = audit.execute("SELECT operation_id,attempt_id FROM operation_attempts").fetchall()
        assert len(original_attempts) == 1
        assert audit.execute("SELECT status FROM operation_runs").fetchall() == [("interrupted",)]
    recover_excision(tmp_path)
    recover_excision(tmp_path)
    with closing(sqlite3.connect(tmp_path / "audit.db")) as audit, audit:
        assert audit.execute("SELECT operation_id,attempt_id FROM operation_attempts").fetchall() == original_attempts
        assert audit.execute("SELECT status FROM operation_runs").fetchall() == [("completed",)]
    with write_lease("test.recovery-terminal-permission", archive_root=tmp_path):
        assert permitted_session_removals(archive_root=tmp_path) == frozenset()
    with closing(sqlite3.connect(tmp_path / "index.db")) as index, index:
        exists = index.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    assert exists is None
    with closing(sqlite3.connect(tmp_path / "user.db")) as user, user:
        assertion = user.execute(
            "SELECT evidence_refs_json FROM assertions WHERE assertion_id='recovery-protection'"
        ).fetchone()
    if retained:
        assert assertion is not None and json.loads(assertion[0]) == [f"session:{session_id}"]
    else:
        assert assertion is None
    with closing(sqlite3.connect(tmp_path / "embeddings.db")) as paid, paid:
        assert paid.execute("SELECT count(*) FROM message_embeddings_meta").fetchone() == (0,)
        assert paid.execute("SELECT count(*) FROM excision_embedding_completions").fetchone() == (1,)


_BlockerPhase = Callable[[str, str, Callable[[BlockerResolveArgs], Any]], Any]


def _run_prepared_blocker_case(
    archive_root: Path, seed: dict[str, Any] | None, work: Callable[[_BlockerPhase], object]
) -> object:
    """Run the original actuator law on a real supplied preparation/writer owner."""
    import asyncio

    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.storage.frontier_inspection import prepared_frontier_blocker_acknowledgement
    from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    async def run() -> object:
        def initialize() -> None:
            bootstrap_archive_root(archive_root)
            if seed is not None:
                _seed_raw_authority_blocker(archive_root, **seed)

        await run_archive_fixture_write(archive_root, initialize)
        async with prepared_live_convergence_owner(archive_root) as owner:

            def exercise() -> object:
                def phase(blocker_id: str, resolution: str, assertion: Callable[[BlockerResolveArgs], Any]) -> Any:
                    with prepared_frontier_blocker_acknowledgement(
                        archive_root,
                        blocker_id,
                        resolution=resolution,
                        input_demand=owner._compute_adapter.amend_current_input_demand,
                    ) as prepared:
                        args = BlockerResolveArgs(archive_root, blocker_id, resolution, prepared)
                        return admit_stage_write("fixture.blocker.actuator", lambda: assertion(args))

                return work(phase)

            return await owner.run_convergence_sync("fixture.blocker.law", exercise)

    return asyncio.run(run())

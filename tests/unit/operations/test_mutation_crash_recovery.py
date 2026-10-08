"""Every executor-routed mutation survives a kill at any point of its apply.

Each scenario runs one production actuator through the real executor route to
its durable intent, "kills" the process either before or after the domain
apply, and restarts: daemon startup recovery must leave the mutation complete
(or, for an atomic apply that never committed, absent), terminalize the run,
and leave nothing behind that a later request on the same targets would be
refused by. ``unknown`` is not an outcome any scenario may reach.

Anti-vacuity: restoring an inspector that reports ``unknown`` for a family, or
dropping a family from ``recoverable_actuators``, turns its rows red on the
terminal reason; breaking an actuator's convergence (for example, letting
``SessionDeleteActuator.apply`` pass an already-deleted id to
``delete_sessions``) turns its crash-after-apply row red on ``replay-failed``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest

from polylogue.core.enums import AssertionKind
from polylogue.operations import mutation_actuators as actuators
from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.mutation_replay import (
    RECOVERY_SERVICE_ACTOR_REF,
    recover_interrupted_operations,
    recoverable_actuators,
)
from polylogue.operations.mutation_transaction import (
    MutationPrincipal,
    OperationExecutor,
)
from polylogue.operations.specs import build_runtime_operation_catalog
from polylogue.security.lifecycle import _request_assertion_id
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.audit_leaf import open_verified_sqlite_read_connection
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.operation_recovery import recover_on_admitted_owner
from tests.unit.operations.test_mutation_actuators import _seed_archive_session, _seed_raw_authority_blocker

Crash = Literal["before-apply", "after-apply"]


@dataclass(frozen=True)
class _Scenario:
    name: str
    actuator: Any
    args: Callable[[Path, ArchiveStore], Any] | None
    applied: Callable[[Path, ArchiveStore], bool]


def _session(root: Path) -> str:
    return _seed_archive_session(root, native_id="crash-target")


def _indexed(root: Path, session_id: str) -> bool:
    with closing(sqlite3.connect(root / "index.db")) as conn, conn:
        return conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone() is not None


def _assertion(root: Path, assertion_id: str) -> bool:
    with closing(sqlite3.connect(root / "user.db")) as conn, conn:
        row = conn.execute("SELECT status FROM assertions WHERE assertion_id = ?", (assertion_id,)).fetchone()
    return row is not None and row[0] != "deleted"


def _always(_root: Path, _archive: ArchiveStore) -> bool:
    return True


_SID = "codex-session:crash-target"
_BLACKBOARD_BODY = "kind=blocker\ntitle=T\ncontent=crash finding"
_CAPTURE_ID = "assertion-terminal-note:" + hashlib.sha256(b"agent:test\0crash-key").hexdigest()

_SCENARIOS: tuple[_Scenario, ...] = (
    _Scenario(
        "session-delete",
        actuators.SessionDeleteActuator(),
        lambda root, archive: actuators.SessionDeleteArgs(archive, (_session(root),)),
        lambda root, _archive: not _indexed(root, _SID),
    ),
    _Scenario(
        "session-excision",
        actuators.SessionExcisionActuator(),
        lambda root, _archive: actuators.SessionExcisionArgs(root, _session(root), "crash test", "user:test", False),
        lambda root, _archive: not _indexed(root, _SID),
    ),
    _Scenario(
        "session-lifecycle-request",
        actuators.SessionLifecycleRequestActuator(),
        lambda root, _archive: actuators.SessionLifecycleRequestArgs(
            root, _session(root), "mirror", "crash test", "user:test", 1_000
        ),
        lambda root, _archive: _assertion(root, _request_assertion_id(f"session:{_SID}", "mirror")),
    ),
    _Scenario(
        "identity-reset",
        actuators.IdentityResetActuator(),
        lambda root, _archive: actuators.IdentityResetArgs(root, (_session(root),), "crash test"),
        lambda root, _archive: not _indexed(root, _SID),
    ),
    _Scenario(
        "filesystem-reset",
        actuators.FilesystemResetActuator(),
        lambda root, _archive: actuators.FilesystemResetArgs(root, (("scratch", _scratch_file(root)),)),
        lambda root, _archive: not (root / "scratch.bin").exists(),
    ),
    _Scenario(
        "blob-publication-abandon",
        actuators.BlobPublicationAbandonActuator(),
        lambda root, _archive: actuators.BlobPublicationAbandonArgs(root, ("publication:absent",)),
        _always,
    ),
    _Scenario(
        "tag-add",
        actuators.TagAddActuator(),
        lambda root, archive: actuators.TagAddArgs(archive, _session(root), "crash", "agent:test", "agent"),
        lambda _root, archive: "crash" in archive.list_user_tags(),
    ),
    _Scenario(
        "tag-remove",
        actuators.TagRemoveActuator(),
        lambda root, archive: actuators.TagRemoveArgs(archive, _tagged_session(root, archive), "crash"),
        lambda _root, archive: "crash" not in archive.list_user_tags(),
    ),
    _Scenario(
        "bulk-tag",
        actuators.BulkTagActuator(),
        lambda root, archive: actuators.BulkTagArgs(archive, (_session(root),), ("crash", "bulk")),
        lambda _root, archive: {"crash", "bulk"} <= set(archive.list_user_tags()),
    ),
    _Scenario(
        "metadata-set",
        actuators.MetadataSetActuator(),
        lambda root, archive: actuators.MetadataSetArgs(archive, _session(root), "crash", "set"),
        lambda _root, archive: archive.read_user_metadata(_SID).get("crash") == "set",
    ),
    _Scenario(
        "bulk-metadata-set",
        actuators.BulkMetadataSetActuator(),
        lambda root, archive: actuators.BulkMetadataSetArgs(archive, (_session(root),), (("crash", "bulk"),)),
        lambda _root, archive: archive.read_user_metadata(_SID).get("crash") == "bulk",
    ),
    _Scenario(
        "metadata-delete",
        actuators.MetadataDeleteActuator(),
        lambda root, archive: actuators.MetadataDeleteArgs(archive, _metadata_session(root, archive), "crash"),
        lambda _root, archive: "crash" not in archive.read_user_metadata(_SID),
    ),
    _Scenario(
        "mark-add",
        actuators.MarkAddActuator(),
        lambda root, archive: actuators.MarkArgs(archive, "session", _session(root), "star"),
        lambda _root, archive: len(archive.list_marks(session_id=_SID)) == 1,
    ),
    _Scenario(
        "mark-remove",
        actuators.MarkRemoveActuator(),
        lambda root, archive: actuators.MarkArgs(archive, "session", _marked_session(root, archive), "star"),
        lambda _root, archive: archive.list_marks(session_id=_SID) == [],
    ),
    _Scenario(
        "capture-assertion-candidate",
        actuators.CaptureAssertionCandidateActuator(),
        lambda _root, archive: actuators.CaptureAssertionCandidateArgs(
            archive,
            "candidate body",
            AssertionKind.LESSON,
            (),
            ("repo:polylogue",),
            None,
            "agent:test",
            "agent",
            "crash-key",
            _CAPTURE_ID,
            None,
        ),
        lambda root, _archive: _assertion(root, _CAPTURE_ID),
    ),
    _Scenario(
        "set-user-setting",
        actuators.SetUserSettingActuator(),
        lambda _root, archive: actuators.SetUserSettingArgs(archive, "subscription_tier", "max_5x", "user:test"),
        lambda root, _archive: _setting(root, "subscription_tier") is not None,
    ),
    _Scenario(
        "annotation-save",
        actuators.AnnotationSaveActuator(),
        lambda root, archive: actuators.AnnotationSaveArgs(archive, "note-crash", "session", _session(root), "text"),
        lambda _root, archive: archive.get_annotation("note-crash") is not None,
    ),
    _Scenario(
        "annotation-delete",
        actuators.AnnotationDeleteActuator(),
        lambda root, archive: actuators.AnnotationDeleteArgs(archive, _annotated(root, archive)),
        lambda _root, archive: archive.get_annotation("note-crash") is None,
    ),
    _Scenario(
        "raw-authority-blocker-resolve",
        actuators.BlockerResolveActuator(),
        None,
        lambda root, _archive: _blocker_resolved(root),
    ),
    _Scenario(
        "saved-view-save",
        actuators.SavedViewSaveActuator(),
        lambda _root, archive: actuators.SavedViewSaveArgs(archive, "view-crash", "Crash", '{"query": "x"}'),
        lambda _root, archive: archive.get_view("view-crash") is not None,
    ),
    _Scenario(
        "saved-view-delete",
        actuators.SavedViewDeleteActuator(),
        lambda _root, archive: actuators.SavedViewDeleteArgs(archive, _saved_view(archive)),
        lambda _root, archive: archive.get_view("view-crash") is None,
    ),
    _Scenario(
        "recall-pack-save",
        actuators.RecallPackSaveActuator(),
        lambda _root, archive: actuators.RecallPackSaveArgs(archive, "pack-crash", "Crash", "[]", "{}"),
        lambda _root, archive: archive.get_recall_pack("pack-crash") is not None,
    ),
    _Scenario(
        "recall-pack-delete",
        actuators.RecallPackDeleteActuator(),
        lambda _root, archive: actuators.RecallPackDeleteArgs(archive, _recall_pack(archive)),
        lambda _root, archive: archive.get_recall_pack("pack-crash") is None,
    ),
    _Scenario(
        "workspace-save",
        actuators.WorkspaceSaveActuator(),
        lambda _root, archive: actuators.WorkspaceSaveArgs(archive, "ws-crash", "Crash", "tabs", "[]", "{}", "{}"),
        lambda _root, archive: archive.get_workspace("ws-crash") is not None,
    ),
    _Scenario(
        "workspace-delete",
        actuators.WorkspaceDeleteActuator(),
        lambda _root, archive: actuators.WorkspaceDeleteArgs(archive, _workspace(archive)),
        lambda _root, archive: archive.get_workspace("ws-crash") is None,
    ),
    _Scenario(
        "correction-record",
        actuators.CorrectionRecordActuator(),
        lambda root, archive: actuators.CorrectionRecordArgs(
            archive, _session(root), "tag_reject", {"tag": "todo"}, None, "user:test", "user"
        ),
        lambda _root, archive: len(archive.list_corrections(session_id=_SID)) == 1,
    ),
    _Scenario(
        "correction-delete",
        actuators.CorrectionDeleteActuator(),
        lambda root, archive: actuators.CorrectionDeleteArgs(archive, _corrected_session(root, archive), "tag_reject"),
        lambda _root, archive: archive.list_corrections(session_id=_SID) == [],
    ),
    _Scenario(
        "corrections-clear",
        actuators.CorrectionsClearActuator(),
        lambda root, archive: actuators.CorrectionsClearArgs(archive, _corrected_session(root, archive)),
        lambda _root, archive: archive.list_corrections(session_id=_SID) == [],
    ),
    _Scenario(
        "blackboard-post",
        actuators.BlackboardPostActuator(),
        lambda _root, archive: actuators.BlackboardPostArgs(
            archive, "note-crash", _BLACKBOARD_BODY, None, None, "agent:test", "agent", (), None, None
        ),
        lambda root, _archive: _note_posted(root),
    ),
)


def _scratch_file(root: Path) -> Path:
    path = root / "scratch.bin"
    path.write_bytes(b"scratch")
    return path


def _tagged_session(root: Path, archive: ArchiveStore) -> str:
    session_id = _session(root)
    archive.add_user_tags((session_id,), ("crash",))
    return session_id


def _metadata_session(root: Path, archive: ArchiveStore) -> str:
    session_id = _session(root)
    archive.set_user_metadata((session_id,), (("crash", "set"),))
    return session_id


def _marked_session(root: Path, archive: ArchiveStore) -> str:
    session_id = _session(root)
    archive.add_mark("session", session_id, "star")
    return session_id


def _annotated(root: Path, archive: ArchiveStore) -> str:
    archive.save_annotation("note-crash", "session", _session(root), "text")
    return "note-crash"


def _saved_view(archive: ArchiveStore) -> str:
    archive.save_view("view-crash", "Crash", '{"query": "x"}')
    return "view-crash"


def _recall_pack(archive: ArchiveStore) -> str:
    archive.save_recall_pack("pack-crash", "Crash", "[]", "{}")
    return "pack-crash"


def _workspace(archive: ArchiveStore) -> str:
    archive.save_workspace(
        workspace_id="ws-crash",
        name="Crash",
        mode="tabs",
        open_targets_json="[]",
        layout_json="{}",
        active_target_json="{}",
    )
    return "ws-crash"


def _corrected_session(root: Path, archive: ArchiveStore) -> str:
    session_id = _session(root)
    archive.record_correction(session_id, "tag_reject", {"tag": "todo"})
    return session_id


def _note_posted(root: Path) -> bool:
    with closing(sqlite3.connect(root / "user.db")) as conn, conn:
        row = conn.execute("SELECT status, body_text FROM assertions WHERE kind = 'note'").fetchone()
    return row is not None and row[0] != "deleted" and row[1] == _BLACKBOARD_BODY


def _setting(root: Path, key: str) -> object:
    with closing(sqlite3.connect(root / "user.db")) as conn, conn:
        return conn.execute("SELECT 1 FROM user_settings WHERE setting_key = ?", (key,)).fetchone()


def _blocker(root: Path) -> str:
    _seed_raw_authority_blocker(root, blocker_id="blocker-crash", plan_id="plan-crash", observed_pass_id="pass-crash")
    return "blocker-crash"


def _blocker_resolved(root: Path) -> bool:
    with closing(sqlite3.connect(root / "source.db")) as conn, conn:
        row = conn.execute(
            "SELECT resolved_at_ms FROM raw_authority_blockers WHERE blocker_id = 'blocker-crash'"
        ).fetchone()
    return row is not None and row[0] is not None


def _principal(binding: Any) -> MutationPrincipal:
    spec = binding.spec
    capabilities = frozenset(
        capability for policy in spec.target_authority for capability in policy.required_capabilities
    )
    return MutationPrincipal("user:test", capabilities, spec.allowed_surfaces[0], "write")


def _crash_mid_mutation(root: Path, scenario: _Scenario, crash: Crash) -> str:
    """Drive the real route to durable intent, optionally apply, then die."""

    with ArchiveStore.open_existing(root, read_only=False) as archive:
        assert scenario.args is not None
        args = scenario.args(root, archive)
        executor = OperationExecutor.for_archive_root(root)
        binding = runtime_operation_binding(scenario.actuator)
        principal = _principal(binding)
        preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=root)
        authorization = executor.authorize_bound(binding, preview, principal, confirmation_strength="bound_token")
        assert executor._audit is not None
        # ``begin_bound`` journals the same durable intent ``execute_bound``
        # does; the after-apply leg then runs the executor's own apply phase
        # (removal authority, started-mutation transfer) and dies before
        # finalization. Calling ``actuator.apply`` bare would skip the removal
        # authority a real destructive apply always holds.
        started = executor.begin_bound(binding, preview, authorization, args)
        assert started.operation_id is not None
        operation_id = started.operation_id
        if crash == "after-apply":
            # The daemon applies under its admitted writer; the library route
            # takes the same root-bound lease.
            scope = executor._prevalidated_executions.set(((scenario.actuator, started),))
            try:
                with write_lease("test.crash-mid-mutation.apply", archive_root=root):
                    executor.execute(scenario.actuator, started.plan, authorization, args)
            finally:
                executor._prevalidated_executions.reset(scope)
    with closing(sqlite3.connect(root / "audit.db")) as conn, conn:
        conn.execute(
            "UPDATE operation_attempts SET worker_id = 'pid:999999999:0' WHERE operation_id = ?", (operation_id,)
        )
        conn.commit()
    return operation_id


# An Excision's physical apply runs only inside its prepared compute phase
# with the original input-demand owner and result sink, which this generic
# route does not construct. Its crash after a committed Source effect is
# ``test_audited_excision_recovery_keeps_exact_removal_authority``
# (``tests/unit/operations/test_mutation_actuators.py``), which interrupts the
# real prepared apply between its Source and paid commits and recovers it.
_CRASH_CASES = tuple(
    pytest.param(scenario, crash, id=f"{scenario.name}-{crash}")
    for scenario in _SCENARIOS
    for crash in ("before-apply", "after-apply")
    if not (scenario.name == "session-excision" and crash == "after-apply")
)


@pytest.mark.parametrize(("scenario", "crash"), _CRASH_CASES)
def test_restart_leaves_an_interrupted_mutation_complete_and_unblocked(
    tmp_path: Path, scenario: _Scenario, crash: Crash
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    if scenario.name == "raw-authority-blocker-resolve":
        operation_id = _prepared_blocker_crash_and_restart(root, crash)
    else:
        _seed_archive_session(root, native_id="bootstrap")
        operation_id = _crash_mid_mutation(root, scenario, crash)
        recover_on_admitted_owner(root)

    with open_verified_sqlite_read_connection(root / "audit.db") as conn:
        status, reason = conn.execute(
            "SELECT status, terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone()
        target_states = {
            str(row[0])
            for row in conn.execute("SELECT state FROM operation_targets WHERE operation_id = ?", (operation_id,))
        }
        blocking = conn.execute(
            "SELECT COUNT(*) FROM operation_runs WHERE status IN ('running', 'interrupted')"
        ).fetchone()[0]
    assert (status, reason) == ("completed", "recovered_complete")
    assert target_states <= {"applied", "already_satisfied"}
    assert blocking == 0
    if scenario.name == "identity-reset":
        from polylogue.operations.audit import AuditRepository
        from polylogue.operations.machine_receipts import IdentityResetHistoricalReceipt

        history = AuditRepository.for_archive_root(root).historical_machine_receipt(operation_id)
        assert isinstance(history, IdentityResetHistoricalReceipt)
        assert history.count_scope == "completing-apply"
        assert history.suppressed_count == 1
        assert history.deleted_archive_rows == (1 if crash == "before-apply" else 0)
        assert history.tombstoned_without_index_row_count == 0
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        assert scenario.applied(root, archive)


def test_every_executor_routed_family_declares_a_recovery_route() -> None:
    """No executor-routed operation can reach startup recovery without a route.

    Anti-vacuity: remove any entry from ``recoverable_actuators`` and the
    family's interrupted runs end ``not-replayable`` instead of complete.
    """

    routed = {
        spec.name for spec in build_runtime_operation_catalog().specs if spec.executor_status == "executor-routed"
    }
    assert routed - set(recoverable_actuators()) == set()


def test_every_crash_scenario_names_a_registered_family() -> None:
    """The crash matrix covers every family whose route it can drive generically.

    The rest have their own crash tests: the annotation batch import
    (``tests/unit/annotations/test_importer.py``), the sealed insight page
    (``tests/unit/operations/test_insight_acceptance.py``) and the
    owner-reconverged ingest.
    """

    covered = {scenario.actuator.operation for scenario in _SCENARIOS}
    registered = set(recoverable_actuators())
    assert covered <= registered
    assert registered - covered == {
        "ingest-archive-runtime",
        "mutate-import-annotation-batch",
        "mutate-rebuild-insights",
    }


def test_excision_is_not_reported_complete_from_a_missing_index_row(tmp_path: Path) -> None:
    """An interrupted excision whose index row is gone but that has no record fails visibly.

    The rebuildable index is where excision finds the session; its absence is
    not proof the durable content was removed. Startup recovery refuses with a
    typed ``RecoveryDeferredError`` and keeps the run's barrier: an Excision
    lacking settled exact-attempt evidence refuses startup instead of ending
    as a terminal replay failure (``recover_interrupted_operations``).
    Anti-vacuity: treat ``found=False`` as ``already_satisfied`` without
    checking the excision record and recovery completes the run with its
    source row intact.
    """
    from polylogue.operations.mutation_transaction import RecoveryDeferredError

    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    scenario = next(item for item in _SCENARIOS if item.name == "session-excision")
    operation_id = _crash_mid_mutation(root, scenario, "before-apply")
    with closing(sqlite3.connect(root / "index.db")) as conn, conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (_SID,))

    with pytest.raises(RecoveryDeferredError):
        recover_on_admitted_owner(root)

    with closing(sqlite3.connect(root / "audit.db")) as conn, conn:
        (status,) = conn.execute("SELECT status FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone()
    assert status in {"running", "interrupted"}
    with closing(sqlite3.connect(root / "source.db")) as conn, conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE native_id = 'crash-target'").fetchone() == (1,)


def test_replayed_setting_keeps_its_original_timestamp(tmp_path: Path) -> None:
    """Re-applying a committed setting converges without restamping it.

    Anti-vacuity: drop the unchanged-value check in
    ``SetUserSettingActuator.apply`` and ``updated_at_ms`` moves on replay.
    """
    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    scenario = next(item for item in _SCENARIOS if item.name == "set-user-setting")
    operation_id = _crash_mid_mutation(root, scenario, "after-apply")
    with closing(sqlite3.connect(root / "user.db")) as conn, conn:
        (before,) = conn.execute(
            "SELECT updated_at_ms FROM user_settings WHERE setting_key = 'subscription_tier'"
        ).fetchone()
    with closing(sqlite3.connect(root / "user.db")) as conn, conn:
        conn.execute(
            "UPDATE user_settings SET updated_at_ms = ? WHERE setting_key = 'subscription_tier'", (before - 1,)
        )
        conn.commit()

    recover_on_admitted_owner(root)

    with closing(sqlite3.connect(root / "user.db")) as conn, conn:
        assert conn.execute(
            "SELECT updated_at_ms FROM user_settings WHERE setting_key = 'subscription_tier'"
        ).fetchone() == (before - 1,)
    with closing(sqlite3.connect(root / "audit.db")) as conn, conn:
        assert conn.execute(
            "SELECT terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("recovered_complete",)


def test_filesystem_reset_recovery_leaves_a_recreated_path_alone(tmp_path: Path) -> None:
    """A path recreated after the reset committed is not the object the plan authorized.

    Anti-vacuity: replay deletion by pathname alone and the new file is gone.
    """
    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    scenario = next(item for item in _SCENARIOS if item.name == "filesystem-reset")
    operation_id = _crash_mid_mutation(root, scenario, "after-apply")
    (root / "scratch.bin").write_bytes(b"recreated after the reset")

    recover_on_admitted_owner(root)

    assert (root / "scratch.bin").read_bytes() == b"recreated after the reset"
    with closing(sqlite3.connect(root / "audit.db")) as conn, conn:
        assert conn.execute(
            "SELECT terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("recovered_complete",)


def test_saved_view_recovery_refuses_a_collision_the_plan_did_not_name(tmp_path: Path) -> None:
    """Replay does not tombstone a view that took the name after authorization.

    Anti-vacuity: drop ``SavedViewSaveActuator.replay_refusal`` and the later
    view is tombstoned by the replayed save.
    """
    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    scenario = next(item for item in _SCENARIOS if item.name == "saved-view-save")
    operation_id = _crash_mid_mutation(root, scenario, "before-apply")
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.save_view("view-later", "Crash", '{"query": "y"}')

    recover_on_admitted_owner(root)

    with ArchiveStore.open_existing(root, read_only=False) as archive:
        assert archive.get_view("view-later") is not None
        assert archive.get_view("view-crash") is None
    with closing(sqlite3.connect(root / "audit.db")) as conn, conn:
        assert conn.execute(
            "SELECT terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("recovery_replay_failed",)


def test_corrections_clear_recovery_keeps_a_kind_recorded_after_authorization(tmp_path: Path) -> None:
    """Replay clears only the planned kinds.

    Anti-vacuity: replay through ``clear_corrections`` and the later
    ``tag_accept`` correction is deleted too.
    """
    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    scenario = next(item for item in _SCENARIOS if item.name == "corrections-clear")
    _crash_mid_mutation(root, scenario, "before-apply")
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.record_correction(_SID, "tag_accept", {"tag": "keep"})

    recover_on_admitted_owner(root)

    with ArchiveStore.open_existing(root, read_only=False) as archive:
        assert [item.kind.value for item in archive.list_corrections(session_id=_SID)] == ["tag_accept"]


def test_replayed_annotation_keeps_its_original_timestamp(tmp_path: Path) -> None:
    """A committed annotation save is not restamped by recovery.

    Anti-vacuity: drop ``AnnotationSaveActuator.already_applied`` and
    ``updated_at`` moves to the recovery time.
    """
    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    scenario = next(item for item in _SCENARIOS if item.name == "annotation-save")
    _crash_mid_mutation(root, scenario, "after-apply")
    with closing(sqlite3.connect(root / "user.db")) as conn, conn:
        conn.execute("UPDATE assertions SET created_at_ms = 7, updated_at_ms = 7 WHERE key = 'note-crash'")
        conn.commit()

    recover_on_admitted_owner(root)

    with ArchiveStore.open_existing(root, read_only=False) as archive:
        stored = archive.get_annotation("note-crash")
    assert stored is not None and stored["updated_at"] == "7"


def test_committed_view_delete_is_not_replayed_over_a_new_watched_view(tmp_path: Path) -> None:
    """Replaying a committed delete would clear the watch of a view saved under the name since.

    Anti-vacuity: drop ``SavedViewDeleteActuator.already_applied`` and the new
    view's ``query_names.watch`` is cleared by the replayed delete.
    """
    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    scenario = next(item for item in _SCENARIOS if item.name == "saved-view-delete")
    _crash_mid_mutation(root, scenario, "after-apply")
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.save_view(
            "view-later", "Crash", '{"query": "sessions where origin:codex-session AND repo:polylogue"}', watch=True
        )

    recover_on_admitted_owner(root)

    with closing(sqlite3.connect(root / "user.db")) as conn, conn:
        assert conn.execute("SELECT watch FROM query_names WHERE name = 'Crash'").fetchone() == (1,)


def test_watched_save_interrupted_before_its_baseline_is_measured_on_recovery(tmp_path: Path) -> None:
    """A watched save whose view committed but whose baseline did not is not complete.

    The kill lands between ``save_view`` and ``establish_watch_baselines``
    inside ``SavedViewSaveActuator.apply``. Anti-vacuity: drop the baseline
    check from ``SavedViewSaveActuator.already_applied`` and recovery reports
    the run complete with no baseline, so the next evaluation absorbs the
    first changed session instead of reporting it.
    """
    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    query_json = '{"query": "sessions where origin:codex-session"}'
    actuator = actuators.SavedViewSaveActuator()
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        args = actuators.SavedViewSaveArgs(archive, "view-watch", "Watched", query_json, watch=True)
        executor = OperationExecutor.for_archive_root(root)
        binding = runtime_operation_binding(actuator)
        principal = _principal(binding)
        preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=root)
        authorization = executor.authorize_bound(binding, preview, principal, confirmation_strength="bound_token")
        assert executor._audit is not None
        operation_id = executor._audit.consume_authorization_and_start(preview, authorization)
        archive.save_view("view-watch", "Watched", query_json, watch=True)
    with closing(sqlite3.connect(root / "audit.db")) as conn, conn:
        conn.execute(
            "UPDATE operation_attempts SET worker_id = 'pid:999999999:0' WHERE operation_id = ?", (operation_id,)
        )
        conn.commit()
    with closing(sqlite3.connect(root / "user.db")) as conn, conn:
        assert conn.execute("SELECT COUNT(*) FROM watched_query_baselines").fetchone() == (0,)

    recover_on_admitted_owner(root)

    with closing(sqlite3.connect(root / "user.db")) as conn, conn:
        baselined = conn.execute(
            "SELECT rs.member_count FROM query_names AS n "
            "JOIN watched_query_baselines AS b ON b.query_hash = n.query_hash "
            "JOIN result_sets AS rs ON rs.result_set_id = b.result_set_id "
            "WHERE n.name = 'Watched' AND n.watch = 1"
        ).fetchall()
    assert baselined == [(1,)]
    with closing(sqlite3.connect(root / "audit.db")) as conn, conn:
        assert conn.execute(
            "SELECT terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("recovered_complete",)


def _execute(root: Path, actuator: Any, build_args: Callable[[Path, ArchiveStore], Any]) -> None:
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        args = build_args(root, archive)
        executor = OperationExecutor.for_archive_root(root)
        binding = runtime_operation_binding(actuator)
        principal = _principal(binding)
        preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=root)
        authorization = executor.authorize_bound(binding, preview, principal, confirmation_strength="bound_token")
        executor.execute_bound(binding, preview, authorization, args)


def test_a_new_mutation_first_lands_interrupted_work_it_cannot_see(tmp_path: Path) -> None:
    """An interrupted record of a new kind lands before a clear planned without it.

    The record's target did not exist when the clear was planned, so no
    target overlap links them. Anti-vacuity: resolve only overlapping dead
    work in ``_resolve_dead_operations`` and the first clear applies, after
    which the replayed record survives it.
    """
    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    record = next(item for item in _SCENARIOS if item.name == "correction-record")
    _crash_mid_mutation(root, record, "before-apply")

    from polylogue.operations.mutation_transaction import PlanStaleError

    def clear() -> None:
        _execute(
            root,
            actuators.CorrectionsClearActuator(),
            lambda _root, archive: actuators.CorrectionsClearArgs(archive, _SID),
        )

    # The interrupted record lands first, so the clear previewed without it is stale.
    with pytest.raises(PlanStaleError):
        clear()
    clear()

    with ArchiveStore.open_existing(root, read_only=False) as archive:
        assert archive.list_corrections(session_id=_SID) == []


def test_user_write_recovery_waits_while_its_session_does_not_resolve(tmp_path: Path) -> None:
    """A tag write whose session the index cannot resolve yet is deferred, not failed.

    Anti-vacuity: let ``KeyError`` reach the generic handler and the run is
    terminalized ``recovery_replay_failed`` while the tag never lands.
    """
    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    scenario = next(item for item in _SCENARIOS if item.name == "tag-add")
    operation_id = _crash_mid_mutation(root, scenario, "before-apply")
    with closing(sqlite3.connect(root / "index.db")) as conn, conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (_SID,))

    recover_on_admitted_owner(root)

    with closing(sqlite3.connect(root / "audit.db")) as conn, conn:
        (status,) = conn.execute("SELECT status FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone()
    assert status == "interrupted"


def test_a_request_is_revalidated_against_the_state_recovery_leaves(tmp_path: Path) -> None:
    """Interrupted work lands before the new request's freshness check, not after it.

    A save of view A under name N was previewed while an interrupted save of
    B under N had not applied. Landing B first makes A's preview stale, so A
    refuses instead of tombstoning B. Anti-vacuity: call
    ``_resolve_dead_operations`` after the fresh-plan comparison and A
    applies over B.
    """
    from polylogue.operations.mutation_transaction import PlanStaleError

    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    later = actuators.SavedViewSaveActuator()
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        args = actuators.SavedViewSaveArgs(archive, "view-other", "Crash", '{"query": "z"}')
        executor = OperationExecutor.for_archive_root(root)
        binding = runtime_operation_binding(later)
        principal = _principal(binding)
        preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=root)
        authorization = executor.authorize_bound(binding, preview, principal, confirmation_strength="bound_token")
    scenario = next(item for item in _SCENARIOS if item.name == "saved-view-save")
    _crash_mid_mutation(root, scenario, "before-apply")

    with ArchiveStore.open_existing(root, read_only=False) as archive:
        args = actuators.SavedViewSaveArgs(archive, "view-other", "Crash", '{"query": "z"}')
        with pytest.raises(PlanStaleError):
            OperationExecutor.for_archive_root(root).execute_bound(binding, preview, authorization, args)
        assert archive.get_view("view-crash") is not None


def test_corrections_clear_recovery_waits_while_its_session_does_not_resolve(tmp_path: Path) -> None:
    """The kind-scoped clear defers like every other user write while the index lacks its session.

    Anti-vacuity: let ``delete_correction``'s ``KeyError`` escape
    ``CorrectionsClearActuator.recover`` and the run ends
    ``recovery_replay_failed`` with the corrections still live.
    """
    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    scenario = next(item for item in _SCENARIOS if item.name == "corrections-clear")
    operation_id = _crash_mid_mutation(root, scenario, "before-apply")
    with closing(sqlite3.connect(root / "index.db")) as conn, conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (_SID,))

    recover_on_admitted_owner(root)

    with closing(sqlite3.connect(root / "audit.db")) as conn, conn:
        (status,) = conn.execute("SELECT status FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone()
    assert status == "interrupted"


def test_replayed_identity_reset_keeps_its_suppression_timestamp(tmp_path: Path) -> None:
    """A committed reset's suppression is not restamped by recovery.

    Anti-vacuity: drop the ``_suppression_matches`` check in
    ``IdentityResetActuator.apply`` and ``updated_at_ms`` moves to the
    recovery time.
    """
    from polylogue.storage.sqlite.archive_tiers.user_write import assertion_id_for_suppression

    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    scenario = next(item for item in _SCENARIOS if item.name == "identity-reset")
    operation_id = _crash_mid_mutation(root, scenario, "after-apply")
    suppression = assertion_id_for_suppression(_SID)
    with closing(sqlite3.connect(root / "user.db")) as conn, conn:
        conn.execute("UPDATE assertions SET updated_at_ms = 7 WHERE assertion_id = ?", (suppression,))
        conn.commit()

    recover_on_admitted_owner(root)

    with closing(sqlite3.connect(root / "user.db")) as conn, conn:
        assert conn.execute(
            "SELECT updated_at_ms FROM assertions WHERE assertion_id = ?", (suppression,)
        ).fetchone() == (7,)
    with closing(sqlite3.connect(root / "audit.db")) as conn, conn:
        assert conn.execute(
            "SELECT terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
        ).fetchone() == ("recovered_complete",)


def test_a_request_is_refused_after_recovering_a_file_reset_under_its_handles(tmp_path: Path) -> None:
    """A request whose handles predate a recovered file reset is reissued, not applied.

    Recovery inside ``begin_bound`` can unlink files the caller already has
    open. Anti-vacuity: drop the ``replaces_archive_files`` refusal in
    ``_resolve_dead_operations`` and the first tag write applies through the
    handles opened before the reset landed.
    """
    from polylogue.operations.mutation_transaction import RecoveryBlockedError

    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    _session(root)
    scenario = next(item for item in _SCENARIOS if item.name == "filesystem-reset")
    _crash_mid_mutation(root, scenario, "before-apply")

    def tag() -> None:
        _execute(
            root,
            actuators.TagAddActuator(),
            lambda _root, archive: actuators.TagAddArgs(archive, _SID, "after", "agent:test", "agent"),
        )

    with pytest.raises(RecoveryBlockedError, match="retry"):
        tag()
    assert not (root / "scratch.bin").exists()
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        assert "after" not in archive.list_user_tags()
    tag()

    with ArchiveStore.open_existing(root, read_only=False) as archive:
        assert "after" in archive.list_user_tags()


def test_a_request_is_refused_behind_an_unrouted_file_reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dead file-deleting work this process cannot route fences every write, not only overlapping ones.

    Anti-vacuity: drop the ``path:`` refusal in ``_resolve_dead_operations``
    and the tag lands in a database startup recovery would later unlink.
    """
    from polylogue.operations import mutation_transaction
    from polylogue.operations.mutation_transaction import RecoveryBlockedError

    root = tmp_path / "archive"
    root.mkdir()
    _seed_archive_session(root, native_id="bootstrap")
    _session(root)
    scenario = next(item for item in _SCENARIOS if item.name == "filesystem-reset")
    _crash_mid_mutation(root, scenario, "before-apply")
    monkeypatch.delitem(mutation_transaction._RECOVERY_ROUTES, "mutate-filesystem-reset")

    with pytest.raises(RecoveryBlockedError, match="startup recovery"):
        _execute(
            root,
            actuators.TagAddActuator(),
            lambda _root, archive: actuators.TagAddArgs(archive, _SID, "after", "agent:test", "agent"),
        )

    assert (root / "scratch.bin").exists()
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        assert "after" not in archive.list_user_tags()


def _prepared_blocker_crash_and_restart(root: Path, crash: Crash) -> str:
    """The original before/after-apply law crosses real preparation owners on restart."""
    import asyncio

    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.storage.frontier_inspection import prepared_frontier_blocker_acknowledgement
    from polylogue.storage.sqlite.audit_leaf import open_verified_audit_connection
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    async def run() -> str:
        def seed() -> None:
            _seed_archive_session(root, native_id="bootstrap")
            _blocker(root)

        await run_archive_fixture_write(root, seed)
        async with prepared_live_convergence_owner(root) as owner:

            def interrupted() -> str:
                with prepared_frontier_blocker_acknowledgement(
                    root,
                    "blocker-crash",
                    resolution="acknowledged in crash test",
                    input_demand=owner._compute_adapter.amend_current_input_demand,
                ) as prepared:
                    args = actuators.BlockerResolveArgs(root, "blocker-crash", "acknowledged in crash test", prepared)
                    actuator = actuators.BlockerResolveActuator()
                    executor = OperationExecutor.for_archive_root(root)
                    binding = runtime_operation_binding(actuator)
                    principal = _principal(binding)

                    def intent() -> tuple[str, Any]:
                        preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=root)
                        authorization = executor.authorize_bound(
                            binding, preview, principal, confirmation_strength="bound_token"
                        )
                        begun = executor.begin_bound(binding, preview, authorization, args)
                        assert begun.operation_id is not None
                        return begun.operation_id, begun.plan

                    operation_id, plan = admit_stage_write("fixture.blocker.interrupted", intent)
                if crash == "after-apply":
                    with prepared_frontier_blocker_acknowledgement(
                        root,
                        "blocker-crash",
                        resolution="acknowledged in crash test",
                        input_demand=owner._compute_adapter.amend_current_input_demand,
                    ) as prepared:
                        args = actuators.BlockerResolveArgs(
                            root, "blocker-crash", "acknowledged in crash test", prepared
                        )
                        receipt = admit_stage_write("fixture.blocker.apply", lambda: actuator.apply(plan, args))
                        assert receipt.status == "applied"
                return operation_id

            operation_id = await owner.run_convergence_sync("fixture.blocker.crash", interrupted)

        def mark_dead() -> None:
            with open_verified_audit_connection(root / "audit.db") as conn:
                conn.execute(
                    "UPDATE operation_attempts SET worker_id='pid:999999999:0' WHERE operation_id=?", (operation_id,)
                )
                conn.commit()

        await run_archive_fixture_write(root, mark_dead)
        # A different physically admitted owner performs the real recovery reduction.
        async with prepared_live_convergence_owner(root) as restarted:
            await restarted.run_convergence_sync(
                "fixture.blocker.restart",
                recover_interrupted_operations,
                root,
                resolver_actor_ref=RECOVERY_SERVICE_ACTOR_REF,
                input_demand=restarted._compute_adapter.amend_current_input_demand,
            )
        return operation_id

    return asyncio.run(run())


@pytest.mark.parametrize("family", ["bulk-tag", "bulk-metadata-set"])
@pytest.mark.parametrize("selection", ["partial", "missing", "present"])
@pytest.mark.parametrize("crash", ["before-apply", "after-apply"])
def test_bulk_recovery_retains_original_named_gaps_without_widening_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str, selection: str, crash: Crash
) -> None:
    """Losing missing IDs in the persisted plan turns recovered degraded into empty/ok."""
    root = tmp_path / "archive"
    root.mkdir()
    sid = _seed_archive_session(root, native_id="bootstrap")
    missing = "codex-session:arrives-later"
    ids = (sid, missing) if selection == "partial" else (missing,) if selection == "missing" else (sid,)
    scenario = next(item for item in _SCENARIOS if item.name == family)
    args = (
        (lambda _root, archive: actuators.BulkTagArgs(archive, ids, ("neutral",)))
        if family == "bulk-tag"
        else (lambda _root, archive: actuators.BulkMetadataSetArgs(archive, ids, (("neutral", "value"),)))
    )
    scenario = _Scenario(scenario.name, scenario.actuator, args, scenario.applied)
    operation_id = _crash_mid_mutation(root, scenario, crash)
    # Newly available request evidence must never become a new authorized target.
    _seed_archive_session(root, native_id="arrives-later")
    resolutions = []
    original = type(scenario.actuator).recover

    def observe(self: Any, handles: Any, plan: Any) -> Any:
        resolution = original(self, handles, plan)
        resolutions.append(resolution)
        return resolution

    monkeypatch.setattr(type(scenario.actuator), "recover", observe)
    recover_on_admitted_owner(root)
    assert len(resolutions) == 1
    receipt = resolutions[0].receipt
    assert receipt is not None
    expected_missing = [] if selection == "present" else [missing]
    assert receipt.domain_receipt["unresolved_session_ids"] == expected_missing
    assert receipt.domain_receipt["session_count"] == len(ids)
    assert receipt.domain_receipt["outcome"]["state"] == (
        "degraded" if expected_missing else "ok" if crash == "before-apply" else "empty"
    )
    assert receipt.target_refs == (() if selection == "missing" else (f"session:{sid}",))
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        assert archive.read_user_metadata(missing) == {}
    with closing(sqlite3.connect(root / "user.db")) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM assertions WHERE target_ref = ?", (f"session:{missing}",)).fetchone()[0]
            == 0
        )
    with open_verified_sqlite_read_connection(root / "audit.db") as conn:
        assert tuple(
            conn.execute(
                "SELECT status, terminal_reason FROM operation_runs WHERE operation_id = ?", (operation_id,)
            ).fetchone()
        ) == ("completed", "recovered_complete")


@pytest.mark.parametrize("crash", ["before-apply", "after-apply"])
@pytest.mark.parametrize("change", [None, "payload", "note", "author_ref", "author_kind"])
def test_correction_recovery_keeps_exact_committed_effect_and_timestamps(
    tmp_path: Path, frozen_clock: Any, crash: Crash, change: str | None
) -> None:
    """Exact replay must preserve time; any changed effect must still be applied."""
    root = tmp_path / "archive"
    root.mkdir()
    sid = _seed_archive_session(root, native_id="bootstrap")
    planned: dict[str, Any] = {
        "payload": {"tag": "neutral"},
        "note": "original",
        "author_ref": "user:fixture",
        "author_kind": "user",
    }

    def args(_root: Path, archive: ArchiveStore) -> actuators.CorrectionRecordArgs:
        return actuators.CorrectionRecordArgs(archive, sid, "tag_reject", **planned)

    scenario = _Scenario("correction-record", actuators.CorrectionRecordActuator(), args, _always)
    _crash_mid_mutation(root, scenario, crash)
    if crash == "after-apply" and change is not None:
        changed = dict(planned)
        changed[change] = (
            {"tag": "changed"}
            if change == "payload"
            else "service"
            if change == "author_kind"
            else "user:changed"
            if change == "author_ref"
            else "changed"
        )
        with ArchiveStore.open_existing(root, read_only=False) as archive:
            archive.record_correction(sid, "tag_reject", **changed)

    def rows() -> list[tuple[Any, ...]]:
        with closing(sqlite3.connect(root / "user.db")) as conn:
            return conn.execute(
                "SELECT value_json, author_ref, author_kind, created_at_ms, updated_at_ms FROM assertions WHERE kind='correction'"
            ).fetchall()

    before = rows()
    frozen_clock.advance(60)
    recover_on_admitted_owner(root)
    after = rows()
    assert len(after) == 1
    assert json.loads(after[0][0]) == {"payload": planned["payload"], "note": planned["note"]}
    assert after[0][1:3] == ("user:fixture", "user")
    if crash == "after-apply" and change is None:
        assert after == before
    elif before:
        assert after[0][3] == before[0][3]
        assert after[0][4] > before[0][4]

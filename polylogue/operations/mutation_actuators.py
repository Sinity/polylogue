"""Domain actuators for the t46.9/kwsb.2 named routes.

Each actuator wraps exactly one existing low-level mutation primitive
(``ArchiveStore.delete_sessions``, ``security.excision``, the identity-reset
tombstone helpers, the ``ArchiveStore`` tag/metadata/mark writers) behind the
:class:`~polylogue.operations.mutation_transaction.MutationActuator` protocol.
Actuators own target resolution and the real mutation; they never enforce
authorization -- every surface drives them through
:class:`~polylogue.operations.mutation_transaction.OperationExecutor`.

Phase 1 (PR #3249) shipped ``SessionDeleteActuator``/``SessionExcisionActuator``/
``IdentityResetActuator`` for the ``delete``/``excise``/``reset`` destructive
classes, all requiring ``confirm_flag``-strength authorization. Phase 2
(polylogue-t46.9/polylogue-kwsb.2) adds the ``reversible``-class tag, metadata,
and mark actuators below: their ``required_confirmation`` is ``role_only`` --
AC4 requires reversible writes not acquire unnecessary interactive
confirmation, since undo is another write through the same actuator.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, cast

from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.core.protocols import ProgressCallback
from polylogue.core.refs import ObjectRef, normalize_object_ref_text, parse_public_ref
from polylogue.operations.mutation_transaction import (
    ConfirmationStrength,
    ConvergentReplay,
    DestructiveClass,
    MutationPlan,
    MutationReceipt,
    MutationTargetStatus,
    PlanStaleError,
    RecoveryDeferredError,
    RecoveryResolution,
    ReplayHandles,
    build_plan,
    make_target_ref,
    register_recovery_route,
)
from polylogue.security.lifecycle import LifecycleMode
from polylogue.storage.sqlite.connection_profile import (
    open_connection,
    open_isolated_write_connection,
    open_readonly_connection,
)
from polylogue.surfaces.outcome import OutcomeEnvelope, decide_outcome

if TYPE_CHECKING:
    from polylogue.core.json import JSONValue
    from polylogue.storage.frontier_inspection import PreparedFrontierAcknowledgement
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.audit_continuity import CanonicalAuditLiteral


# ---------------------------------------------------------------------------
# Session delete (mutate-delete-session)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionDeleteArgs:
    """Shared prepare/apply argument shape for session delete."""

    archive: ArchiveStore
    session_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SessionDeleteActuator(ConvergentReplay):
    """Actuator for ``mutate-delete-session``: permanent, re-ingest-resurrectable removal.

    Real production mutation: ``ArchiveStore.delete_sessions`` -- the single
    low-level primitive CLI ``delete`` and MCP ``write(operation=
    'delete_session')`` both reach, via the resident selection executor and
    ``PolylogueArchiveMixin.delete_session_safe`` respectively. This actuator
    does not change that primitive; it makes the *authorization path* to it
    shared instead of independently reimplemented per surface.
    """

    operation: str = "mutate-delete-session"
    destructive_class: DestructiveClass = "delete"
    required_confirmation: ConfirmationStrength = "confirm_flag"

    def prepare(self, args: SessionDeleteArgs) -> MutationPlan:
        # Re-resolve existence against live state: a session id the caller
        # already matched via a query result set may have been deleted (by
        # a concurrent actor) between query and delete -- prepare only
        # plans the subset that still exists right now. Every caller hands
        # full ids it resolved once, so existence is exact: a vanished id
        # never widens to another session sharing its prefix.
        existing = args.archive.stored_session_ids(args.session_ids)
        return build_plan(
            operation=self.operation,
            destructive_class="delete",
            target_refs=tuple(make_target_ref("session", sid) for sid in existing),
            affected_tiers=("index",),
            reversible=False,
            context={"session_ids": list(existing)},
        )

    def apply(self, plan: MutationPlan, args: SessionDeleteArgs) -> MutationReceipt:
        if any(not target_ref.startswith("session:") for target_ref in plan.target_refs):
            raise ValueError("session delete plan contains a non-session target")
        planned = tuple(target_ref.removeprefix("session:") for target_ref in plan.target_refs)
        # A re-applied plan converges: a target an interrupted apply already
        # removed is satisfied. Only ids still stored exactly reach
        # ``delete_sessions``, whose own resolver would widen a missing id to a
        # prefix match and delete a different session (crash-recovery replay
        # runs exactly this after the first apply removed the target).
        session_ids = args.archive.stored_session_ids(planned)
        deleted = args.archive.delete_sessions(session_ids) if session_ids else 0
        status: MutationTargetStatus = "applied" if deleted else "already_satisfied"
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status=status,
            target_refs=plan.target_refs,
            affected_count=deleted,
            detail=None if deleted else "no_matching_sessions",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"deleted_count": deleted, "session_count": len(planned)},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> SessionDeleteArgs:
        return SessionDeleteArgs(
            archive=handles.archive, session_ids=tuple(cast("list[str]", plan.context["session_ids"]))
        )


# ---------------------------------------------------------------------------
# Session excision (mutate-session-excision)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionExcisionArgs:
    """Shared prepare/apply argument shape for session excision."""

    archive_root: Path
    session_id: str
    reason: str
    actor: str
    cascade_lineage: bool
    # Ephemeral original creator admission, excluded from durable plan data.
    input_demand: Callable[[int], None] | None = field(default=None, repr=False, compare=False)
    result_sink: Callable[[Mapping[str, object], CanonicalAuditLiteral], None] | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class SessionExcisionActuator(ConvergentReplay):
    """Actuator for ``mutate-session-excision``: durable, re-ingest-proof removal.

    Real production mutation: ``security.excision.plan_session_excision`` /
    the audited Excision operation -- the cross-tier (source/index/embeddings/
    user) removal that records a durable removed-hash marker so re-ingest of
    unmodified source files cannot resurrect the content. Unlike
    ``mutate-delete-session``, excision is not idempotent-silent: a stale
    plan (lineage dependents changed, or the target was concurrently
    excised) must refuse via ``PlanStaleError`` rather than partially apply.
    """

    operation: str = "mutate-session-excision"
    destructive_class: DestructiveClass = "excise"
    required_confirmation: ConfirmationStrength = "confirm_flag"

    def prepare(self, args: SessionExcisionArgs) -> MutationPlan:
        from polylogue.operations.operation_context import open_operation_read
        from polylogue.security.excision import excision_target_replay, plan_session_excision

        with open_operation_read(args.archive_root) as pinned:
            plan = plan_session_excision(pinned.archive, args.session_id, cascade_lineage=args.cascade_lineage)
        target_refs = ((make_target_ref("session", args.session_id),) if plan.found else ()) + tuple(
            make_target_ref("session", sid) for sid in plan.lineage_dependent_session_ids
        )
        return build_plan(
            operation=self.operation,
            destructive_class="excise",
            target_refs=target_refs,
            affected_tiers=("source", "index", "embeddings", "user"),
            reversible=False,
            context={
                "session_id": args.session_id,
                "actor": args.actor,
                "found": plan.found,
                "reason": args.reason,
                "cascade_lineage": args.cascade_lineage,
                "lineage_dependent_session_ids": list(plan.lineage_dependent_session_ids),
                "source_marker_inputs_pending": plan.source_marker_inputs_pending,
                "source_marker_inputs_accepted": plan.source_marker_inputs_accepted,
                "marker_input_digests": list(plan.marker_input_digests),
                "targets": [excision_target_replay(target) for target in plan.targets],
                "user_frame_epoch": plan.user_frame_epoch,
            },
        )

    def apply(self, plan: MutationPlan, args: SessionExcisionArgs) -> MutationReceipt:
        from polylogue.operations.mutation_transaction import take_started_bound_mutation
        from polylogue.security.excision import (
            ExcisionBlobReferenceUnknownError,
            LineageDependentsError,
            _apply_started_session_excision,
            _deliver_started_no_effect_excision,
        )

        started = take_started_bound_mutation(self, plan)
        if not plan.context.get("found"):
            summary = _deliver_started_no_effect_excision(started, args)
            return MutationReceipt(
                operation=self.operation,
                plan_hash=plan.plan_hash,
                status="already_satisfied",
                target_refs=plan.target_refs,
                affected_count=0,
                detail=None,
                receipt_ref=None,
                applied_at=plan.prepared_at,
                domain_receipt=summary,
            )
        try:
            receipt = _apply_started_session_excision(started, args, actuator=self)
        except (LineageDependentsError, ExcisionBlobReferenceUnknownError) as exc:
            # Both refusals roll the excision back before any write.
            return MutationReceipt(
                operation=self.operation,
                plan_hash=plan.plan_hash,
                status="blocked",
                target_refs=plan.target_refs,
                affected_count=0,
                detail=str(exc),
                receipt_ref=None,
                applied_at=plan.prepared_at,
                domain_receipt={
                    "refusal_kind": (
                        "lineage_dependents_unresolved"
                        if isinstance(exc, LineageDependentsError)
                        else "blob_references_undecided"
                    )
                },
            )
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied" if receipt["found"] else "already_satisfied",
            target_refs=plan.target_refs,
            affected_count=cast("Mapping[str, int]", receipt["counts"]).get("index_sessions", 0),
            detail=None,
            receipt_ref=cast("str | None", receipt["receipt_assertion_id"]),
            applied_at=plan.prepared_at,
            domain_receipt=receipt,
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> SessionExcisionArgs:
        return SessionExcisionArgs(
            archive_root=handles.archive_root,
            session_id=str(plan.context["session_id"]),
            reason=str(plan.context["reason"]),
            actor=str(plan.context["actor"]),
            cascade_lineage=bool(plan.context["cascade_lineage"]),
        )

    def recover(self, handles: ReplayHandles, plan: MutationPlan) -> RecoveryResolution:
        """Finish the exact recorded attempt using its Source and atomic paid proof."""
        from dataclasses import replace

        from polylogue.core.stage_admission import admit_stage_write
        from polylogue.operations.audit import AuditRepository
        from polylogue.security.excision import _apply_original_session_excision
        from polylogue.storage.sqlite.literal_cells import owned_literal_stream
        from polylogue.storage.sqlite.reference_seal import ReferenceSealError

        original = handles.recovery_operation
        if (
            original is None
            or original.operation != self.operation
            or original.plan_hash != plan.plan_hash
            or original.attempt_id is None
            or not original.target_evidence_complete
            or original.reconstructed_target_count != original.expected_target_count
            or original.expected_target_count != len(plan.target_refs)
            or tuple(target.ref for target in original.targets) != plan.target_refs
        ):
            raise ReferenceSealError("Excision recovery requires its exact recorded operation and attempt")
        if handles.input_demand is None:
            raise ReferenceSealError("Excision recovery requires its original prepared compute admission")
        if plan.context["found"] is False:
            return RecoveryResolution("absent", "the original no-effect Excision plan has no selected effect")
        if plan.context["found"] is not True:
            raise ReferenceSealError("Excision recovery has no canonical original found state")

        def consume_effect_product(_summary: Mapping[str, object], literal: CanonicalAuditLiteral) -> None:
            # Internal recovery promises a domain effect resolution. It does
            # not certify installation in a daemon request's result owner.
            with owned_literal_stream(literal.verified_chunks()) as chunks:
                for _chunk in chunks:
                    pass

        args = replace(
            self.replay_args(handles, plan),
            input_demand=handles.input_demand,
            result_sink=consume_effect_product,
        )
        admit_stage_write(
            "operation.session-excision.recovery-continuity",
            lambda: AuditRepository.for_archive_root(handles.archive_root).reconcile_continuity(),
        )
        summary = _apply_original_session_excision(
            plan,
            original.operation_id,
            args,
            actuator=self,
            recovery=original,
        )
        return RecoveryResolution(
            "complete",
            "the original Excision attempt is physically settled",
            MutationReceipt(
                operation=self.operation,
                plan_hash=plan.plan_hash,
                status="applied",
                target_refs=plan.target_refs,
                affected_count=cast("Mapping[str, int]", summary["counts"]).get("index_sessions", 0),
                detail=None,
                receipt_ref=cast("str | None", summary["receipt_assertion_id"]),
                applied_at=plan.prepared_at,
                domain_receipt=summary,
            ),
        )


# ---------------------------------------------------------------------------
# Lifecycle request (mutate-session-lifecycle-request)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionLifecycleRequestArgs:
    """Arguments for the durable mirror/primary excision-request outbox row."""

    archive_root: Path
    session_id: str
    mode: LifecycleMode
    reason: str
    actor: str
    now_ms: int


@dataclass(frozen=True, slots=True)
class SessionLifecycleRequestActuator(ConvergentReplay):
    """Create the local lifecycle request through the audit-backed executor."""

    operation: str = "mutate-session-lifecycle-request"
    destructive_class: DestructiveClass = "additive"
    required_confirmation: ConfirmationStrength = "confirm_flag"

    def prepare(self, args: SessionLifecycleRequestArgs) -> MutationPlan:
        return build_plan(
            operation=self.operation,
            destructive_class=self.destructive_class,
            target_refs=(make_target_ref("session", args.session_id),),
            affected_tiers=("user",),
            reversible=True,
            context={"mode": args.mode, "reason": args.reason, "actor": args.actor, "now_ms": args.now_ms},
        )

    def apply(self, plan: MutationPlan, args: SessionLifecycleRequestArgs) -> MutationReceipt:
        from polylogue.security.lifecycle import submit_lifecycle_request_with_outcome

        user_db = args.archive_root / "user.db"
        connection = open_isolated_write_connection(
            user_db,
            purpose="operation.mutate-session-lifecycle-request",
            archive_root=args.archive_root,
        )
        try:
            submission = submit_lifecycle_request_with_outcome(
                connection,
                target_ref=make_target_ref("session", args.session_id),
                mode=args.mode,
                reason=args.reason,
                actor=args.actor,
                now_ms=args.now_ms,
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied" if submission.created else "already_satisfied",
            target_refs=plan.target_refs,
            affected_count=1 if submission.created else 0,
            detail=None,
            receipt_ref=submission.assertion_id,
            applied_at=plan.prepared_at,
            domain_receipt={"assertion_id": submission.assertion_id, "mode": args.mode},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> SessionLifecycleRequestArgs:
        return SessionLifecycleRequestArgs(
            archive_root=handles.archive_root,
            session_id=plan.target_refs[0].removeprefix("session:"),
            mode=cast(LifecycleMode, plan.context["mode"]),
            reason=str(plan.context["reason"]),
            actor=str(plan.context["actor"]),
            now_ms=int(cast(int, plan.context["now_ms"])),
        )


# ---------------------------------------------------------------------------
# Derived reset / identity tombstone (mutate-identity-reset)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IdentityResetArgs:
    """Shared prepare/apply argument shape for identity reset."""

    archive_root: Path
    session_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class IdentityResetActuator(ConvergentReplay):
    """Actuator for ``mutate-identity-reset``: tombstone + rebuildable-row delete.

    Real production mutation: the ``polylogue ops reset --session/--source``
    tombstone helpers (``_suppress_archive_sessions`` writes the durable
    user.db suppression, ``_delete_archive_sessions`` drops the rebuildable
    index.db rows). Target resolution (token -> exact session ids) happens
    once by the resident preview owner before ``prepare`` is invoked. The
    client receives a sealed preview-batch request reference. Confirmation
    authorizes its exact bounded plans, and apply submits only the authorization
    batch reference; each part reconstructs arguments from its audited plan. ``prepare`` re-verifies the exact
    recorded IDs so a concurrent target change is caught before APPLY.
    """

    operation: str = "mutate-identity-reset"
    destructive_class: DestructiveClass = "reset"
    required_confirmation: ConfirmationStrength = "confirm_flag"

    def prepare(self, args: IdentityResetArgs) -> MutationPlan:
        # Every caller-resolved id is a tombstone target, whether or not it
        # still has a row in the rebuildable index. The durable user.db
        # suppression is what makes the deletion survive, and a session that
        # vanished from index.db between resolution and PREPARE is exactly
        # the case where dropping it writes no suppression and lets the next
        # ingest or rebuild make the content visible again. ``present`` is
        # carried only so the receipt can say how many rebuildable rows the
        # apply expects to delete.
        requested = tuple(dict.fromkeys(args.session_ids))
        present = tuple(dict.fromkeys(_resolve_existing_session_ids(args.archive_root, requested)))
        return build_plan(
            operation=self.operation,
            destructive_class="reset",
            target_refs=tuple(make_target_ref("session", sid) for sid in requested),
            affected_tiers=("index", "user"),
            reversible=True,
            context={
                "session_ids": list(requested),
                "present_in_index": list(present),
                "reason": args.reason,
            },
        )

    def apply(self, plan: MutationPlan, args: IdentityResetArgs) -> MutationReceipt:
        from polylogue.operations.machine_receipts import IdentityResetHistoricalReceipt
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
        from polylogue.storage.sqlite.archive_tiers.user_write import upsert_suppression

        session_ids: tuple[str, ...] = tuple(cast("list[str]", plan.context.get("session_ids") or ()))
        if not session_ids:
            return MutationReceipt(
                operation=self.operation,
                plan_hash=plan.plan_hash,
                status="already_satisfied",
                target_refs=plan.target_refs,
                affected_count=0,
                detail="no_matching_sessions",
                receipt_ref=None,
                applied_at=plan.prepared_at,
                historical_receipt=IdentityResetHistoricalReceipt(
                    suppressed_count=0, deleted_archive_rows=0, tombstoned_without_index_row_count=0
                ),
            )

        user_db = args.archive_root / "user.db"
        initialize_archive_database(user_db, ArchiveTier.USER)
        conn = open_isolated_write_connection(
            user_db,
            purpose="operation.mutate-identity-reset",
            archive_root=args.archive_root,
        )
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for session_id in session_ids:
                    # A replayed apply leaves an identical suppression untouched
                    # rather than restamping it as a fresh edit.
                    if not _suppression_matches(conn, session_id, reason=args.reason, mode="hide"):
                        upsert_suppression(conn, session_id=session_id, reason=args.reason, mode="hide")
                from polylogue.archive.write_gateway import ArchiveWriteGateway, WriteOperation

                ArchiveWriteGateway(user_db).commit_write_sync(
                    WriteOperation.RESET,
                    {"_connection": conn, "changed_session_ids": (), "effect_scope": "user-overlay"},
                )
            except BaseException:
                conn.rollback()
                raise
        finally:
            conn.close()
        suppressed = len(session_ids)

        # Only ids PREPARE saw in the index reach ``delete_sessions``: that
        # call resolves every id it is given and raises ``KeyError`` for one
        # with no row (#5154), which would abort the apply *after* the durable
        # suppressions were committed -- the receipt would then be lost for a
        # tombstone that had already landed. The absent ids are exactly the
        # ``tombstoned_without_index_row`` set the receipt names below, and
        # deleting a row that is not there is a no-op anyway.
        prepared_present = set(cast("list[str]", plan.context.get("present_in_index") or ()))
        # Re-resolve against the live index so a re-applied plan converges:
        # rows an interrupted apply already dropped are not deleted twice.
        live_present = set(_resolve_existing_session_ids(args.archive_root, session_ids))
        present_in_index = tuple(sid for sid in session_ids if sid in prepared_present and sid in live_present)
        deleted = 0
        if present_in_index and _index_db_path(args.archive_root).exists():
            from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

            with ArchiveStore.open_existing(args.archive_root, read_only=False) as archive:
                deleted = archive.delete_sessions(present_in_index, write_operation=WriteOperation.RESET)

        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied",
            target_refs=plan.target_refs,
            affected_count=suppressed,
            detail=None,
            receipt_ref=None,
            applied_at=plan.prepared_at,
            historical_receipt=IdentityResetHistoricalReceipt(
                suppressed_count=suppressed,
                deleted_archive_rows=deleted,
                tombstoned_without_index_row_count=sum(sid not in prepared_present for sid in session_ids),
            ),
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> IdentityResetArgs:
        return IdentityResetArgs(
            archive_root=handles.archive_root,
            session_ids=tuple(cast("list[str]", plan.context["session_ids"])),
            reason=str(plan.context["reason"]),
        )


def _suppression_matches(conn: sqlite3.Connection, session_id: str, *, reason: str, mode: str) -> bool:
    from polylogue.storage.sqlite.archive_tiers.user_write import read_archive_suppression_envelope

    try:
        existing = read_archive_suppression_envelope(conn, session_id)
    except KeyError:
        return False
    return existing.reason == reason and existing.mode == mode


# ---------------------------------------------------------------------------
# Filesystem reset (polylogue-4fbgw)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FilesystemResetArgs:
    """Exact set of already-resolved filesystem targets one reset will delete."""

    archive_root: Path
    #: ``(display name, absolute path)`` pairs, resolved by the daemon from
    #: the request flags before PREPARE. Carrying the resolved set (rather
    #: than the flags) keeps PREPARE and APPLY over the identical targets.
    targets: tuple[tuple[str, Path], ...]


@dataclass(frozen=True, slots=True)
class FilesystemResetActuator(ConvergentReplay):
    """Actuator for ``mutate-filesystem-reset``: delete archive files and trees.

    This is the largest destructive surface in the product -- it unlinks
    archive databases and ``shutil.rmtree``s the blob/assets/cache trees.
    Before polylogue-4fbgw it ran straight out of the daemon handler with the
    ``audit``/``snapshot`` arguments unused, so an operator who lost their
    archive to it had no durable record that the operation was attempted, by
    whom, against which targets, or how far it got -- and an interruption left
    no attempt row for ``recover_abandoned_attempts`` to sweep. Routing it
    through :class:`OperationExecutor` writes the preview, authorization and
    attempt rows *before* the first ``unlink``.

    ``prepare`` re-stats every target so a path that disappeared between
    resolution and APPLY is reported rather than counted, and never mutates.
    """

    operation: str = "mutate-filesystem-reset"
    destructive_class: DestructiveClass = "reset"
    required_confirmation: ConfirmationStrength = "bound_token"
    replaces_archive_files: ClassVar[bool] = True

    def prepare(self, args: FilesystemResetArgs) -> MutationPlan:
        # ``lexists`` semantics: apply deletes a dangling symlink too.
        present = tuple((name, path) for name, path in args.targets if path.is_symlink() or path.exists())
        return build_plan(
            operation=self.operation,
            destructive_class="reset",
            target_refs=tuple(make_target_ref("path", str(path)) for _name, path in args.targets),
            affected_tiers=("filesystem",),
            reversible=False,
            context={
                "targets": [[name, str(path)] for name, path in args.targets],
                "present": [str(path) for _name, path in present],
                # The authorized objects, not just their names: recovery must
                # not delete something recreated at the same path afterwards.
                "identities": {str(path): _path_identity(path) for _name, path in present},
            },
        )

    def apply(self, plan: MutationPlan, args: FilesystemResetArgs) -> MutationReceipt:
        import shutil

        # APPLY runs in a live process that may hold tier connections, so it
        # never unlinks a tier file. The daemon handler stages such a reset
        # for the next start instead (``recover`` under ``archive_tiers_closed``).
        self._refuse_live_tier_deletion(args)

        deleted: list[str] = []
        missing: list[str] = []
        for name, path in args.targets:
            if path.is_file() or path.is_symlink():
                path.unlink()
                deleted.append(name)
            elif path.is_dir():
                shutil.rmtree(path)
                deleted.append(name)
            else:
                missing.append(name)

        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied" if deleted else "already_satisfied",
            target_refs=plan.target_refs,
            affected_count=len(deleted),
            detail=None if deleted else "no_existing_targets",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={
                "deleted": len(deleted),
                "targets": deleted,
                # Named rather than silently folded into the count: a target
                # that vanished between PREPARE and APPLY is evidence about
                # the archive, not a successful deletion.
                "absent_at_apply": missing,
            },
        )

    @staticmethod
    def _refuse_live_tier_deletion(args: FilesystemResetArgs) -> None:
        from polylogue.operations.reset_safety import (
            LiveArchiveTierResetError,
            UnresettableArchiveTierError,
            classify_reset_targets,
        )

        classes = classify_reset_targets(
            args.archive_root, args.targets, served_index_path=_index_db_path(args.archive_root)
        )
        if classes.unresettable:
            raise UnresettableArchiveTierError(classes.unresettable_names)
        if classes.derived_tier_files:
            raise LiveArchiveTierResetError(classes.derived_tier_names)

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> FilesystemResetArgs:
        return FilesystemResetArgs(
            archive_root=handles.archive_root,
            targets=tuple(
                (str(name), Path(str(path))) for name, path in cast("list[list[str]]", plan.context["targets"])
            ),
        )

    def recover(self, handles: ReplayHandles, plan: MutationPlan) -> RecoveryResolution:
        """Finish deleting only the objects the plan authorized.

        A path whose authorized object is gone was reset; one recreated since
        (a new login token, a fresh cache) holds content nobody previewed and
        is left alone. Identity is checked immediately before each deletion.

        A plan naming ``index.db``/``ops.db`` files is a reset the live
        daemon staged: it runs only inside :func:`archive_tiers_closed`, the
        daemon's startup seam, and stays pending anywhere else. There each
        tier database is deleted with every sidecar beside it. The sidecars
        are not identity-checked: with no connection open they belong to the
        database at their path, and a stale ``-wal`` left beside the file
        bootstrap creates next would be replayed into it.
        """
        import shutil

        from polylogue.operations.reset_safety import (
            UnresettableArchiveTierError,
            archive_tiers_are_closed,
            classify_reset_targets,
            discard_closed_derived_tier,
            sqlite_primary,
        )

        args = self.replay_args(handles, plan)
        classes = classify_reset_targets(
            args.archive_root, args.targets, served_index_path=_index_db_path(args.archive_root)
        )
        if classes.unresettable:
            return RecoveryResolution("replay-failed", str(UnresettableArchiveTierError(classes.unresettable_names)))
        tier_files = {path for _name, path in classes.derived_tier_files}
        if tier_files:
            if not archive_tiers_are_closed(args.archive_root):
                raise RecoveryDeferredError(
                    "a reset of archive tier files is applied when polylogued next starts, before any tier opens"
                )
            if (args.archive_root / ".index-active-pointer").exists():
                return RecoveryResolution(
                    "replay-failed",
                    "a managed index generation was promoted after the reset was staged; "
                    "the pointer-managed index is never deleted in place; no files were deleted",
                )
        identities = cast("dict[str, list[int]]", plan.context["identities"])
        names = {path: name for name, path in args.targets}
        deleted: list[str] = []
        databases: dict[Path, str] = {}
        for name, path in args.targets:
            if path in tier_files:
                primary = sqlite_primary(path) or path
                databases.setdefault(primary, names.get(primary, name))
                continue
            if str(path) not in identities or _path_identity(path) != identities[str(path)]:
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
            deleted.append(name)
        for database, name in databases.items():
            current = _path_identity(database)
            if current is not None and current != identities.get(str(database)):
                # Recreated after the preview: not the authorized object.
                continue
            removed = discard_closed_derived_tier(args.archive_root, database)
            for removed_path in removed:
                suffix = removed_path.name.removeprefix(database.name)
                deleted.append(name if not suffix else f"{name} {suffix}")
        receipt = MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied" if deleted else "already_satisfied",
            target_refs=plan.target_refs,
            affected_count=len(deleted),
            detail=None if deleted else "authorized_objects_already_gone",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"deleted": len(deleted), "targets": deleted},
        )
        return RecoveryResolution("complete", "deleted the authorized objects that remained", receipt)


def _view_watched(handles: ReplayHandles, name: str) -> bool:
    with closing(open_readonly_connection(handles.archive_root / "user.db", timeout_class="background-read")) as conn:
        row = conn.execute("SELECT watch FROM query_names WHERE name = ?", (name,)).fetchone()
    return row is not None and bool(row[0])


def _view_watch_baselined(handles: ReplayHandles, name: str) -> bool:
    """Whether the definition this name watches has a measured baseline."""
    with closing(open_readonly_connection(handles.archive_root / "user.db", timeout_class="background-read")) as conn:
        row = conn.execute(
            "SELECT 1 FROM query_names AS n JOIN watched_query_baselines AS b ON b.query_hash = n.query_hash "
            "WHERE n.name = ? AND n.watch = 1",
            (name,),
        ).fetchone()
    return row is not None


def _path_identity(path: Path) -> list[int] | None:
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return None
    return [stat.st_dev, stat.st_ino]


# ---------------------------------------------------------------------------
# Blob publication receipt abandonment
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BlobPublicationAbandonArgs:
    """Exact operator disposition for a named set of publication receipts."""

    archive_root: Path
    publication_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BlobPublicationAbandonActuator(ConvergentReplay):
    """Discharge named publication-reservation debt without touching blob bytes.

    Publication reconciliation has no TTL and never treats age as proof that a
    publisher died, so a retained receipt for a blob nothing references is
    debt only an operator can terminalize (``docs/internals.md``). That made it
    the last durable mutation ``ops maintenance`` performed in the CLI's own
    process: ``blob_publications_command`` called
    ``abandon_blob_publication_receipts`` directly, which ``DELETE``s from and
    commits against ``source.db`` with no daemon involved at all. The decision
    stays an operator's; the write is the daemon's.

    ``prepare`` re-classifies every requested receipt against current
    reference evidence and never mutates. APPLY re-checks liveness itself
    under archive-wide publisher exclusion, so a receipt that became
    referenced between PREPARE and APPLY is retained, not deleted.
    """

    operation: str = "mutate-abandon-blob-publication-receipts"
    destructive_class: DestructiveClass = "reset"
    required_confirmation: ConfirmationStrength = "confirm_flag"

    def prepare(self, args: BlobPublicationAbandonArgs) -> MutationPlan:
        from polylogue.storage.blob_liveness import LivenessState
        from polylogue.storage.blob_publication import inspect_blob_publication_receipts

        requested = set(args.publication_ids)
        receipts = inspect_blob_publication_receipts(
            args.archive_root / "source.db",
            args.archive_root / "blob",
            index_db_path=_index_db_path(args.archive_root),
            publication_ids=tuple(args.publication_ids),
        )
        present = {item.publication_id: item for item in receipts if item.publication_id in requested}
        unreferenced = sorted(pid for pid, item in present.items() if item.liveness.state is LivenessState.UNREFERENCED)
        referenced = sorted(pid for pid, item in present.items() if item.liveness.state is LivenessState.LIVE)
        blocked = sorted(pid for pid, item in present.items() if item.liveness.state is LivenessState.BLOCKED)
        return build_plan(
            operation=self.operation,
            destructive_class=self.destructive_class,
            target_refs=tuple(make_target_ref("source", f"blob-publication:{pid}") for pid in args.publication_ids),
            affected_tiers=("source", "audit"),
            reversible=False,
            context={
                "requested": list(args.publication_ids),
                "unreferenced": unreferenced,
                "referenced": referenced,
                "blocked": blocked,
                "missing": sorted(requested - set(present)),
            },
        )

    def apply(self, plan: MutationPlan, args: BlobPublicationAbandonArgs) -> MutationReceipt:
        from polylogue.core.stage_admission import admit_stage_write
        from polylogue.storage.blob_publication import abandon_blob_publication_receipts

        blocked = cast("list[str]", plan.context.get("blocked", []))
        if blocked:
            raise RecoveryDeferredError(f"blob publication liveness is blocked for receipts: {blocked}")
        # The abandonment writes Source. A daemon request already holds the
        # writer (admission falls through); startup recovery replays this
        # apply outside it, where an unadmitted Source transaction is denied.
        abandonment = admit_stage_write(
            "operation.blob-publication-abandon.apply",
            lambda: abandon_blob_publication_receipts(
                args.archive_root / "source.db",
                args.archive_root / "blob",
                args.publication_ids,
                confirmed=True,
                index_db_path=_index_db_path(args.archive_root),
            ),
        )
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied" if abandonment.abandoned else "already_satisfied",
            target_refs=plan.target_refs,
            affected_count=abandonment.abandoned,
            detail=None if abandonment.abandoned else "no_unreferenced_receipts",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={
                "abandoned": abandonment.abandoned,
                # Named rather than folded into the count: a receipt retained
                # because it became referenced, and one that was never there,
                # are different evidence about the archive.
                "skipped_referenced": abandonment.skipped_referenced,
                "missing_receipts": abandonment.missing_receipts,
                "blob_effect": "none",
            },
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> BlobPublicationAbandonArgs:
        return BlobPublicationAbandonArgs(
            archive_root=handles.archive_root,
            publication_ids=tuple(cast("list[str]", plan.context["requested"])),
        )


def _index_db_path(archive_root: Path) -> Path:
    from polylogue.storage.archive_identity import ArchiveLocation

    return ArchiveLocation.resolve(archive_root).active_index_path


def _resolve_existing_session_ids(archive_root: Path, session_ids: tuple[str, ...]) -> tuple[str, ...]:
    index_db = _index_db_path(archive_root)
    if not index_db.exists():
        # No archive tier: suppressions in user.db are still valid tombstone
        # targets for ids the caller already resolved (mirrors reset.py's
        # existing "archive tier absent" allowance).
        return session_ids
    conn = open_readonly_connection(index_db, timeout_class="background-read")
    try:
        if not session_ids:
            return ()
        found: set[str] = set()
        batch_size = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
        for offset in range(0, len(session_ids), batch_size):
            batch = session_ids[offset : offset + batch_size]
            placeholders = ",".join("?" for _ in batch)
            found.update(
                str(row[0])
                for row in conn.execute(f"SELECT session_id FROM sessions WHERE session_id IN ({placeholders})", batch)
            )
        return tuple(sid for sid in session_ids if sid in found)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tag mutations (mutate-add-tag / mutate-remove-tag / mutate-bulk-tag-sessions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TagAddArgs:
    """Shared prepare/apply argument shape for single-session tag add."""

    archive: ArchiveStore
    session_id: str
    tag: str
    author_ref: str | None = None
    author_kind: str | None = None


@dataclass(frozen=True, slots=True)
class TagAddActuator(ConvergentReplay):
    """Actuator for ``mutate-add-tag``: reversible user.db tag assertion.

    Real production mutation: ``ArchiveStore.add_user_tags`` -- the same
    primitive ``PolylogueArchiveMixin.add_tag`` (reached by CLI's
    ``apply_modifiers``/query-mutation path and MCP's
    ``write(operation='add_tag')``) already calls. Undo is
    ``mutate-remove-tag`` through the same primitive family, so this is
    ``reversible``-class and requires only ``role_only`` confirmation (AC4).
    """

    operation: str = "mutate-add-tag"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: TagAddArgs) -> MutationPlan:
        resolved = args.archive.resolve_session_id(args.session_id)
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(make_target_ref("session", resolved),),
            affected_tiers=("user",),
            reversible=True,
            context={
                "session_id": resolved,
                "tag": args.tag,
                "author_ref": args.author_ref,
                "author_kind": args.author_kind,
            },
        )

    def apply(self, plan: MutationPlan, args: TagAddArgs) -> MutationReceipt:
        session_id = str(plan.context["session_id"])
        tag = str(plan.context["tag"])
        changed = args.archive.add_user_tags(
            (session_id,),
            (tag,),
            author_ref=cast("str | None", plan.context["author_ref"]),
            author_kind=cast("str | None", plan.context["author_kind"]),
        )
        status: MutationTargetStatus = "applied" if changed else "already_satisfied"
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status=status,
            target_refs=plan.target_refs,
            affected_count=changed,
            detail=None if changed else "already_present",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"changed": changed},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> TagAddArgs:
        return TagAddArgs(
            archive=handles.archive,
            session_id=str(plan.context["session_id"]),
            tag=str(plan.context["tag"]),
            author_ref=cast("str | None", plan.context["author_ref"]),
            author_kind=cast("str | None", plan.context["author_kind"]),
        )


@dataclass(frozen=True, slots=True)
class TagRemoveArgs:
    """Shared prepare/apply argument shape for single-session tag remove."""

    archive: ArchiveStore
    session_id: str
    tag: str


@dataclass(frozen=True, slots=True)
class TagRemoveActuator(ConvergentReplay):
    """Actuator for ``mutate-remove-tag``: reversible user.db tag retraction.

    Real production mutation: ``ArchiveStore.remove_user_tags`` -- marks the
    tag assertion deleted rather than physically removing it, so this is
    itself reversible via ``mutate-add-tag``.
    """

    operation: str = "mutate-remove-tag"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: TagRemoveArgs) -> MutationPlan:
        resolved = args.archive.resolve_session_id(args.session_id)
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(make_target_ref("session", resolved),),
            affected_tiers=("user",),
            reversible=True,
            context={"session_id": resolved, "tag": args.tag},
        )

    def apply(self, plan: MutationPlan, args: TagRemoveArgs) -> MutationReceipt:
        session_id = str(plan.context["session_id"])
        tag = str(plan.context["tag"])
        changed = args.archive.remove_user_tags((session_id,), (tag,))
        status: MutationTargetStatus = "applied" if changed else "already_satisfied"
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status=status,
            target_refs=plan.target_refs,
            affected_count=changed,
            detail=None if changed else "tag_not_present",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"changed": changed},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> TagRemoveArgs:
        return TagRemoveArgs(
            archive=handles.archive, session_id=str(plan.context["session_id"]), tag=str(plan.context["tag"])
        )


@dataclass(frozen=True, slots=True)
class BulkTagArgs:
    """Shared prepare/apply argument shape for multi-session bulk tagging."""

    archive: ArchiveStore
    session_ids: tuple[str, ...]
    tags: tuple[str, ...]
    author_ref: str | None = None
    author_kind: str | None = None


@dataclass(frozen=True, slots=True)
class BulkTagActuator(ConvergentReplay):
    """Actuator for ``mutate-bulk-tag-sessions``: reversible multi-target tagging.

    Real production mutation: ``ArchiveStore.add_user_tags`` applied per
    resolved session. ``prepare`` plans the sessions that resolve against live
    state right now and records the ones that do not as a named gap, so an id
    excised between selection and execution makes the receipt ``degraded``
    instead of a success over a silently smaller set.
    """

    operation: str = "mutate-bulk-tag-sessions"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: BulkTagArgs) -> MutationPlan:
        resolved, unresolved = _partition_requested_session_ids(args.archive, args.session_ids)
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=tuple(make_target_ref("session", sid) for sid in resolved),
            affected_tiers=("user",),
            reversible=True,
            context={
                "session_ids": list(resolved),
                "tags": list(args.tags),
                "requested_session_count": len(args.session_ids),
                "requested_session_ids": list(args.session_ids),
                "unresolved_session_ids": list(unresolved),
                "author_ref": args.author_ref,
                "author_kind": args.author_kind,
            },
        )

    def apply(self, plan: MutationPlan, args: BulkTagArgs) -> MutationReceipt:
        session_ids: tuple[str, ...] = tuple(cast("list[str]", plan.context.get("session_ids") or ()))
        tags: tuple[str, ...] = tuple(cast("list[str]", plan.context.get("tags") or ()))
        requested_count = cast("int", plan.context["requested_session_count"])
        # Every planned id is stored exactly or the apply stops before its
        # first write; none is re-resolved to a prefix sibling.
        args.archive.require_stored_session_ids(session_ids)
        affected = 0
        assertions = 0
        for session_id in session_ids:
            changed = args.archive.add_user_tags(
                (session_id,),
                tags,
                author_ref=cast("str | None", plan.context["author_ref"]),
                author_kind=cast("str | None", plan.context["author_kind"]),
            )
            assertions += changed
            if changed > 0:
                affected += 1
        unresolved = tuple(cast("list[str]", plan.context["unresolved_session_ids"]))
        outcome = _narrowed_plan_outcome(matched=affected, unresolved=unresolved)
        status: MutationTargetStatus = "applied" if affected else "already_satisfied"
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status=status,
            target_refs=plan.target_refs,
            affected_count=affected,
            detail=_narrowed_plan_detail(affected=affected, unresolved=unresolved),
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={
                "session_count": requested_count,
                "tag_count": len(tags),
                "affected_count": affected,
                "skipped_count": requested_count - affected,
                "unresolved_session_ids": list(unresolved),
                "outcome": outcome.to_dict(),
                # Sessions changed (``affected_count``) and session/tag pairs
                # written differ whenever more than one tag is applied; a
                # surface that reports pairs needs the second number.
                "assertion_count": assertions,
            },
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> BulkTagArgs:
        return BulkTagArgs(
            archive=handles.archive,
            session_ids=tuple(cast("list[str]", plan.context["requested_session_ids"])),
            tags=tuple(cast("list[str]", plan.context["tags"])),
            author_ref=cast("str | None", plan.context["author_ref"]),
            author_kind=cast("str | None", plan.context["author_kind"]),
        )


# ---------------------------------------------------------------------------
# Metadata mutations (mutate-set-metadata / mutate-delete-metadata)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetadataSetArgs:
    """Shared prepare/apply argument shape for session metadata set."""

    archive: ArchiveStore
    session_id: str
    key: str
    value: object


@dataclass(frozen=True, slots=True)
class MetadataSetActuator(ConvergentReplay):
    """Actuator for ``mutate-set-metadata``: reversible user.db metadata write.

    Real production mutation: ``ArchiveStore.set_user_metadata``. Key
    validation (``validate_metadata_key``) stays in the adapter layer, run
    before the actuator is constructed, matching the existing
    ``PolylogueArchiveMixin.set_metadata`` contract.
    """

    operation: str = "mutate-set-metadata"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: MetadataSetArgs) -> MutationPlan:
        resolved = args.archive.resolve_session_id(args.session_id)
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(make_target_ref("session", resolved),),
            affected_tiers=("user",),
            reversible=True,
            context={"session_id": resolved, "key": args.key, "value": args.value},
        )

    def apply(self, plan: MutationPlan, args: MetadataSetArgs) -> MutationReceipt:
        session_id = str(plan.context["session_id"])
        key = str(plan.context["key"])
        value = plan.context["value"]
        changed = args.archive.set_user_metadata((session_id,), ((key, value),))
        status: MutationTargetStatus = "applied" if changed else "already_satisfied"
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status=status,
            target_refs=plan.target_refs,
            affected_count=changed,
            detail=None if changed else "value_unchanged",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"changed": changed},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> MetadataSetArgs:
        return MetadataSetArgs(
            archive=handles.archive,
            session_id=str(plan.context["session_id"]),
            key=str(plan.context["key"]),
            value=plan.context["value"],
        )


@dataclass(frozen=True, slots=True)
class BulkMetadataSetArgs:
    """Shared prepare/apply argument shape for multi-session metadata set."""

    archive: ArchiveStore
    session_ids: tuple[str, ...]
    pairs: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class BulkMetadataSetActuator(ConvergentReplay):
    """Actuator for ``mutate-bulk-set-metadata``: reversible multi-target metadata.

    Real production mutation: ``ArchiveStore.set_user_metadata`` applied per
    resolved session, the multi-target sibling of ``MetadataSetActuator`` in
    the same relationship ``BulkTagActuator`` has to ``TagAddActuator``.
    ``prepare`` plans only sessions that resolve against live state and carries
    the unresolved ids as a named gap, so a session deleted between selection
    and execution degrades the receipt rather than failing the whole batch or
    vanishing from it.  Key validation stays in the adapter, matching the
    single-target actuator's contract.
    """

    operation: str = "mutate-bulk-set-metadata"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: BulkMetadataSetArgs) -> MutationPlan:
        resolved, unresolved = _partition_requested_session_ids(args.archive, args.session_ids)
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=tuple(make_target_ref("session", sid) for sid in resolved),
            affected_tiers=("user",),
            reversible=True,
            context={
                "session_ids": list(resolved),
                "pairs": [[key, value] for key, value in args.pairs],
                "requested_session_count": len(args.session_ids),
                "requested_session_ids": list(args.session_ids),
                "unresolved_session_ids": list(unresolved),
            },
        )

    def apply(self, plan: MutationPlan, args: BulkMetadataSetArgs) -> MutationReceipt:
        session_ids: tuple[str, ...] = tuple(cast("list[str]", plan.context.get("session_ids") or ()))
        planned_pairs = cast("list[list[object]]", plan.context.get("pairs") or [])
        pairs: tuple[tuple[str, object], ...] = tuple((str(pair[0]), pair[1]) for pair in planned_pairs)
        requested_count = cast("int", plan.context["requested_session_count"])
        args.archive.require_stored_session_ids(session_ids)
        affected = 0
        assertions = 0
        for session_id in session_ids:
            changed = args.archive.set_user_metadata((session_id,), pairs)
            assertions += changed
            if changed > 0:
                affected += 1
        unresolved = tuple(cast("list[str]", plan.context["unresolved_session_ids"]))
        outcome = _narrowed_plan_outcome(matched=affected, unresolved=unresolved)
        status: MutationTargetStatus = "applied" if affected else "already_satisfied"
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status=status,
            target_refs=plan.target_refs,
            affected_count=affected,
            detail=_narrowed_plan_detail(affected=affected, unresolved=unresolved),
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={
                "session_count": requested_count,
                "key_count": len(pairs),
                "affected_count": affected,
                "skipped_count": requested_count - affected,
                "unresolved_session_ids": list(unresolved),
                "outcome": outcome.to_dict(),
                # Sessions changed and session/key pairs written differ
                # whenever more than one key is set; see ``BulkTagActuator``.
                "assertion_count": assertions,
            },
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> BulkMetadataSetArgs:
        return BulkMetadataSetArgs(
            archive=handles.archive,
            session_ids=tuple(cast("list[str]", plan.context["requested_session_ids"])),
            pairs=tuple((str(pair[0]), pair[1]) for pair in cast("list[list[object]]", plan.context["pairs"])),
        )


@dataclass(frozen=True, slots=True)
class MetadataDeleteArgs:
    """Shared prepare/apply argument shape for session metadata delete."""

    archive: ArchiveStore
    session_id: str
    key: str


@dataclass(frozen=True, slots=True)
class MetadataDeleteActuator(ConvergentReplay):
    """Actuator for ``mutate-delete-metadata``: reversible user.db metadata retraction.

    Real production mutation: ``ArchiveStore.delete_user_metadata``, which
    marks the metadata assertion deleted (undo = ``mutate-set-metadata``).
    """

    operation: str = "mutate-delete-metadata"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: MetadataDeleteArgs) -> MutationPlan:
        resolved = args.archive.resolve_session_id(args.session_id)
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(make_target_ref("session", resolved),),
            affected_tiers=("user",),
            reversible=True,
            context={"session_id": resolved, "key": args.key},
        )

    def apply(self, plan: MutationPlan, args: MetadataDeleteArgs) -> MutationReceipt:
        session_id = str(plan.context["session_id"])
        key = str(plan.context["key"])
        changed = args.archive.delete_user_metadata(session_id, key)
        status: MutationTargetStatus = "applied" if changed else "already_satisfied"
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status=status,
            target_refs=plan.target_refs,
            affected_count=changed,
            detail=None if changed else "key_not_found",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"changed": changed},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> MetadataDeleteArgs:
        return MetadataDeleteArgs(
            archive=handles.archive, session_id=str(plan.context["session_id"]), key=str(plan.context["key"])
        )


# ---------------------------------------------------------------------------
# Mark mutations (mutate-add-mark / mutate-remove-mark) -- the first
# MCP no-spec mutation family (census "add_mark / remove_mark" row) to gain
# an OperationSpec and executor route.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarkArgs:
    """Shared prepare/apply argument shape for mark add/remove.

    ``target_type``/``target_id`` are resolved once by the caller (mirroring
    ``IdentityResetActuator``'s "resolve once, preview and mutate the
    identical set" pattern) via
    ``PolylogueArchiveMixin._resolve_user_state_target`` before the actuator
    ever runs -- a mark target can be a session, message, or block, and that
    resolution is async (may consult insight-derived indexes), while
    ``MutationActuator.prepare``/``apply`` are synchronous.
    """

    archive: ArchiveStore
    target_type: str
    target_id: str
    mark_type: str
    owner_session_id: str | None = None


@dataclass(frozen=True, slots=True)
class MarkAddActuator(ConvergentReplay):
    """Actuator for ``mutate-add-mark``: reversible user.db mark assertion.

    Real production mutation: ``ArchiveStore.add_mark``, the same primitive
    ``PolylogueArchiveMixin.add_mark`` (reached by MCP's
    ``write(operation='add_mark')``) already calls. Undo is
    ``mutate-remove-mark``.
    """

    operation: str = "mutate-add-mark"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: MarkArgs) -> MutationPlan:
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(f"{args.target_type}:{args.target_id}",),
            affected_tiers=("user",),
            reversible=True,
            context={
                "target_type": args.target_type,
                "target_id": args.target_id,
                "mark_type": args.mark_type,
                "owner_session_id": args.owner_session_id,
            },
        )

    def apply(self, plan: MutationPlan, args: MarkArgs) -> MutationReceipt:
        added = args.archive.add_mark(
            args.target_type,
            args.target_id,
            args.mark_type,
            owner_session_id=args.owner_session_id,
        )
        status: MutationTargetStatus = "applied" if added else "already_satisfied"
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status=status,
            target_refs=plan.target_refs,
            affected_count=1 if added else 0,
            detail=None if added else "already_present",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"added": added},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> MarkArgs:
        return MarkArgs(
            archive=handles.archive,
            target_type=str(plan.context["target_type"]),
            target_id=str(plan.context["target_id"]),
            mark_type=str(plan.context["mark_type"]),
            owner_session_id=cast("str | None", plan.context["owner_session_id"]),
        )

    def already_applied(self, handles: ReplayHandles, plan: MutationPlan) -> bool:
        owner = plan.context["owner_session_id"]
        return any(
            owner is None or mark["session_id"] == owner
            for mark in handles.archive.list_marks(
                mark_type=str(plan.context["mark_type"]),
                target_type=str(plan.context["target_type"]),
                target_id=str(plan.context["target_id"]),
            )
        )


@dataclass(frozen=True, slots=True)
class MarkRemoveActuator(ConvergentReplay):
    """Actuator for ``mutate-remove-mark``: reversible user.db mark retraction.

    Real production mutation: ``ArchiveStore.remove_mark`` (undo =
    ``mutate-add-mark``).
    """

    operation: str = "mutate-remove-mark"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: MarkArgs) -> MutationPlan:
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(f"{args.target_type}:{args.target_id}",),
            affected_tiers=("user",),
            reversible=True,
            context={
                "target_type": args.target_type,
                "target_id": args.target_id,
                "mark_type": args.mark_type,
                "owner_session_id": args.owner_session_id,
            },
        )

    def apply(self, plan: MutationPlan, args: MarkArgs) -> MutationReceipt:
        removed = args.archive.remove_mark(args.target_type, args.target_id, args.mark_type)
        status: MutationTargetStatus = "applied" if removed else "already_satisfied"
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status=status,
            target_refs=plan.target_refs,
            affected_count=1 if removed else 0,
            detail=None if removed else "not_present",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"removed": removed},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> MarkArgs:
        return MarkArgs(
            archive=handles.archive,
            target_type=str(plan.context["target_type"]),
            target_id=str(plan.context["target_id"]),
            mark_type=str(plan.context["mark_type"]),
            owner_session_id=cast("str | None", plan.context["owner_session_id"]),
        )


# ---------------------------------------------------------------------------
# Terminal assertion candidate capture (mutate-capture-assertion-candidate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CaptureAssertionCandidateArgs:
    """Inputs for one terminal-captured assertion candidate."""

    archive: ArchiveStore
    body_text: str
    kind: AssertionKind
    refs: tuple[str, ...]
    scope_refs: tuple[str, ...]
    cwd: Path | None
    author_ref: str
    author_kind: str
    idempotency_key: str | None
    assertion_id: str
    ttl_seconds: int | None
    #: Source evidence that is neither a target nor a scope, such as the
    #: capture artifact a browser selection was taken from.
    evidence_refs: tuple[str, ...] = ()


def resolve_assertion_candidate_refs(archive: ArchiveStore, refs: Sequence[str], *, cwd: Path | None) -> list[str]:
    """Resolve candidate target refs to archive identities.

    The facade and the daemon actuator share this so a ref shape one accepts is
    never refused by the other: ``last``, ``session:<id>``, a canonical
    ``message:<id>`` ref, or a ``<session>::<message>`` evidence ref.
    """
    resolved_refs: list[str] = []
    for ref in refs:
        if ref == "last":
            resolved_cwd = (cwd or Path.cwd()).resolve()
            repo_root = next(
                (candidate for candidate in (resolved_cwd, *resolved_cwd.parents) if (candidate / ".git").exists()),
                resolved_cwd,
            )
            summaries = archive.list_summaries(cwd_prefix=str(repo_root), limit=1)
            if not summaries:
                raise ValueError("--ref last found no archived session for the current repository/cwd")
            resolved_refs.append(f"session:{summaries[0].session_id}")
            continue
        parsed = parse_public_ref(ref)
        if isinstance(parsed, ObjectRef):
            if parsed.kind == "message":
                resolved_refs.append(parsed.format())
                continue
            if parsed.kind != "session":
                raise ValueError("--ref must be a session or message ref, or 'last'")
            try:
                session_id = archive.resolve_session_id(parsed.object_id)
            except KeyError:
                raise ValueError(f"session ref not found: {parsed.object_id}") from None
            resolved_refs.append(f"session:{session_id}")
            continue
        if parsed.message_id is None or parsed.block_index is not None:
            raise ValueError("--ref must identify a session or message")
        try:
            session_id = archive.resolve_session_id(parsed.session_id)
        except KeyError:
            raise ValueError(f"session ref not found: {parsed.session_id}") from None
        resolved_refs.append(f"{session_id}::{parsed.message_id}")
    return resolved_refs


def _capture_candidate_inputs(args: CaptureAssertionCandidateArgs) -> dict[str, object]:
    """Normalize and resolve capture inputs without writing state."""

    normalized_body = args.body_text.strip()
    if not normalized_body:
        raise ValueError("note text cannot be empty")
    normalized_author_ref = normalize_object_ref_text(args.author_ref)
    normalized_author_kind = args.author_kind.strip().lower()
    if not normalized_author_kind:
        raise ValueError("author_kind cannot be empty")
    if args.ttl_seconds is not None and args.ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")

    normalized_idempotency_key = None if args.idempotency_key is None else args.idempotency_key.strip()
    if args.idempotency_key is not None and not normalized_idempotency_key:
        raise ValueError("idempotency_key cannot be empty")
    if normalized_idempotency_key is not None and len(normalized_idempotency_key) > 240:
        raise ValueError("idempotency_key exceeds 240 characters")

    resolved_refs = resolve_assertion_candidate_refs(args.archive, args.refs, cwd=args.cwd)

    normalized_scope_refs = [parse_public_ref(ref).format() for ref in args.scope_refs]
    supplied_evidence_refs = [parse_public_ref(ref).format() for ref in args.evidence_refs]
    evidence_refs = list(dict.fromkeys((*resolved_refs, *normalized_scope_refs, *supplied_evidence_refs)))
    if normalized_idempotency_key is None:
        assertion_id = args.assertion_id
    else:
        identity = hashlib.sha256(
            f"{normalized_author_ref}\0{normalized_idempotency_key}".encode("utf-8", errors="surrogatepass")
        ).hexdigest()
        derived_assertion_id = f"assertion-terminal-note:{identity}"
        if args.assertion_id != derived_assertion_id:
            raise ValueError("assertion_id does not match idempotency_key")
        assertion_id = derived_assertion_id
    target_ref = resolved_refs[0] if resolved_refs else f"assertion:{assertion_id}"
    fingerprint_document = {
        "author_kind": normalized_author_kind,
        "author_ref": normalized_author_ref,
        "body_text": normalized_body,
        "evidence_refs": evidence_refs,
        "kind": args.kind.value,
        "scope_refs": normalized_scope_refs,
        "target_ref": target_ref,
    }
    capture_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="surrogatepass")
    ).hexdigest()
    return {
        "assertion_id": assertion_id,
        "author_kind": normalized_author_kind,
        "author_ref": normalized_author_ref,
        "body_text": normalized_body,
        "capture_fingerprint": capture_fingerprint,
        "evidence_refs": evidence_refs,
        "kind": args.kind.value,
        "resolved_refs": resolved_refs,
        "scope_refs": normalized_scope_refs,
        "supplied_evidence_refs": supplied_evidence_refs,
        "target_ref": target_ref,
        "ttl_seconds": args.ttl_seconds,
    }


@dataclass(frozen=True, slots=True)
class CaptureAssertionCandidateActuator(ConvergentReplay):
    """Actuator for the durable terminal assertion candidate capture."""

    operation: str = "mutate-capture-assertion-candidate"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: CaptureAssertionCandidateArgs) -> MutationPlan:
        context = _capture_candidate_inputs(args)
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(str(context["target_ref"]),),
            affected_tiers=("user",),
            reversible=True,
            context=context,
        )

    def apply(self, plan: MutationPlan, args: CaptureAssertionCandidateArgs) -> MutationReceipt:
        from polylogue.storage.sqlite.archive_tiers.user_write import read_assertion_envelope, upsert_assertion

        context = plan.context
        assertion_id = str(context["assertion_id"])
        user_db = args.archive.user_db_path
        try:
            conn = open_connection(user_db, archive_root=args.archive._write_lease_archive_root)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("BEGIN IMMEDIATE")
                existing = read_assertion_envelope(conn, assertion_id)
                if existing is not None:
                    existing_value = existing.value if isinstance(existing.value, dict) else {}
                    existing_scope_refs = existing_value.get("scope_refs")
                    existing_document = {
                        "author_kind": existing.author_kind,
                        "author_ref": existing.author_ref,
                        "body_text": existing.body_text,
                        "evidence_refs": existing.evidence_refs,
                        "kind": existing.kind.value,
                        "scope_refs": existing_scope_refs if isinstance(existing_scope_refs, list) else [],
                        "target_ref": existing.target_ref,
                    }
                    existing_fingerprint = hashlib.sha256(
                        json.dumps(
                            existing_document,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8", errors="surrogatepass")
                    ).hexdigest()
                    if existing_fingerprint == str(context["capture_fingerprint"]):
                        conn.commit()
                        envelope = existing
                    else:
                        raise ValueError("idempotency_key conflicts with a different assertion candidate capture")
                else:
                    capture_now_ms = int(datetime.now(UTC).timestamp() * 1000)
                    ttl_seconds = cast("int | None", context["ttl_seconds"])
                    staleness = None if ttl_seconds is None else {"expires_at_ms": capture_now_ms + ttl_seconds * 1000}
                    envelope = upsert_assertion(
                        conn,
                        assertion_id=assertion_id,
                        target_ref=str(context["target_ref"]),
                        scope_ref=cast("list[str]", context["scope_refs"])[0]
                        if cast("list[str]", context["scope_refs"])
                        else None,
                        kind=AssertionKind.from_string(str(context["kind"])),
                        key="terminal-note",
                        value={
                            "capture_surface": "terminal",
                            "scope_refs": cast("list[str]", context["scope_refs"]),
                            "unanchored": not bool(cast("list[str]", context["resolved_refs"])),
                        },
                        body_text=str(context["body_text"]),
                        author_ref=str(context["author_ref"]),
                        author_kind=str(context["author_kind"]),
                        evidence_refs=tuple(cast("list[str]", context["evidence_refs"])),
                        status=AssertionStatus.CANDIDATE,
                        staleness=staleness,
                        context_policy={"inject": False, "promotion_required": True},
                        now_ms=capture_now_ms,
                    )
                    conn.commit()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            raise RuntimeError(f"failed to capture assertion candidate: {exc}") from exc
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="already_satisfied" if existing is not None else "applied",
            target_refs=plan.target_refs,
            affected_count=0 if existing is not None else 1,
            detail="idempotent_replay" if existing is not None else None,
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"envelope": envelope},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> CaptureAssertionCandidateArgs:
        # ``apply`` reads only the archive handle and the resolved plan
        # context; the remaining fields restate that context.
        context = plan.context
        return CaptureAssertionCandidateArgs(
            archive=handles.archive,
            body_text=str(context["body_text"]),
            kind=AssertionKind.from_string(str(context["kind"])),
            refs=tuple(cast("list[str]", context["resolved_refs"])),
            scope_refs=tuple(cast("list[str]", context["scope_refs"])),
            cwd=None,
            author_ref=str(context["author_ref"]),
            author_kind=str(context["author_kind"]),
            idempotency_key=None,
            assertion_id=str(context["assertion_id"]),
            ttl_seconds=cast("int | None", context["ttl_seconds"]),
            evidence_refs=tuple(cast("list[str]", context["supplied_evidence_refs"])),
        )


@dataclass(frozen=True, slots=True)
class SetUserSettingArgs:
    """Inputs for one typed ``user_settings`` upsert."""

    archive: ArchiveStore
    setting_key: str
    value: object
    author_ref: str


@dataclass(frozen=True, slots=True)
class SetUserSettingActuator(ConvergentReplay):
    """Actuator for ``mutate-set-user-setting``: a reversible user.db setting upsert.

    polylogue-r29bv: ``polylogue setting set`` wrote ``user.db`` directly from
    the facade, with no preview, no authorization record and no audit row --
    a wrong-owner write to the archive's one irreplaceable tier. Routing it
    through the executor cycle gives it the same three records every other
    user-tier mutation produces.

    The key's own validator is the gate on whether the value is writable at
    all, and it runs in ``prepare``: an unknown key or a rejected value must
    refuse before an authorization is issued, not after.
    """

    operation: str = "mutate-set-user-setting"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: SetUserSettingArgs) -> MutationPlan:
        from polylogue.storage.sqlite.archive_tiers.user_settings_write import validate_user_setting

        setting_key = args.setting_key.strip()
        if not setting_key:
            raise ValueError("setting_key cannot be empty")
        author_ref = normalize_object_ref_text(args.author_ref)
        validate_user_setting(setting_key, cast("JSONValue", args.value))
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(f"setting:{setting_key}",),
            affected_tiers=("user",),
            reversible=True,
            context={
                "setting_key": setting_key,
                "value": args.value,
                "author_ref": author_ref,
            },
        )

    def apply(self, plan: MutationPlan, args: SetUserSettingArgs) -> MutationReceipt:
        from polylogue.storage.sqlite.archive_tiers.user_settings_write import get_user_setting, set_user_setting

        context = plan.context
        setting_key = str(context["setting_key"])
        user_db = args.archive.user_db_path
        if not user_db.exists():
            raise ValueError("user settings tier is not initialized")
        try:
            conn = open_connection(user_db, archive_root=args.archive._write_lease_archive_root)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("BEGIN IMMEDIATE")
                existing = get_user_setting(conn, setting_key)
                if (
                    existing is not None
                    and existing.value == context["value"]
                    and existing.author_ref == str(context["author_ref"])
                ):
                    # The setting already holds this value from this author:
                    # a re-applied plan converges without restamping it.
                    conn.commit()
                    return MutationReceipt(
                        operation=self.operation,
                        plan_hash=plan.plan_hash,
                        status="already_satisfied",
                        target_refs=plan.target_refs,
                        affected_count=0,
                        detail="setting_unchanged",
                        receipt_ref=None,
                        applied_at=plan.prepared_at,
                        domain_receipt={"envelope": existing},
                    )
                envelope = set_user_setting(
                    conn,
                    setting_key,
                    cast("JSONValue", context["value"]),
                    author_ref=str(context["author_ref"]),
                )
                conn.commit()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            raise RuntimeError(f"failed to set user setting {setting_key!r}: {exc}") from exc
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied",
            target_refs=plan.target_refs,
            affected_count=1,
            detail=None,
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"envelope": envelope},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> SetUserSettingArgs:
        return SetUserSettingArgs(
            archive=handles.archive,
            setting_key=str(plan.context["setting_key"]),
            value=plan.context["value"],
            author_ref=str(plan.context["author_ref"]),
        )


# ---------------------------------------------------------------------------
# Annotation mutations (mutate-save-annotation / mutate-delete-annotation) --
# phase 3 (t46.9/kwsb.2): the annotation family's second member (mark) was
# already routed in phase 2; this closes save/delete annotation, the census
# "save_annotation / delete_annotation" declared-not-routed row.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnnotationSaveArgs:
    """Shared prepare/apply argument shape for annotation create/update.

    ``target_type``/``target_id`` are resolved once by the caller (mirroring
    ``MarkArgs``) via ``PolylogueArchiveMixin._resolve_user_state_target``
    before the actuator runs.
    """

    archive: ArchiveStore
    annotation_id: str
    target_type: str
    target_id: str
    note_text: str
    owner_session_id: str | None = None


@dataclass(frozen=True, slots=True)
class AnnotationSaveActuator(ConvergentReplay):
    """Actuator for ``mutate-save-annotation``: reversible user.db annotation upsert.

    Real production mutation: ``ArchiveStore.save_annotation``, the same
    primitive ``PolylogueArchiveMixin.save_annotation`` already calls. Every
    apply writes (create-or-update is inherently a write, not a no-op-when-
    unchanged idempotency check like tags/metadata), so the receipt's
    ``created``-vs-``updated`` distinction lives in ``domain_receipt`` rather
    than in ``status``. Undo is ``mutate-delete-annotation`` (soft-delete via
    the same assertion row), so this is ``reversible``-class, role_only
    confirmation.
    """

    operation: str = "mutate-save-annotation"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: AnnotationSaveArgs) -> MutationPlan:
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(f"annotation:{args.annotation_id}",),
            affected_tiers=("user",),
            reversible=True,
            context={
                "annotation_id": args.annotation_id,
                "target_type": args.target_type,
                "target_id": args.target_id,
                "note_text": args.note_text,
                "owner_session_id": args.owner_session_id,
            },
        )

    def apply(self, plan: MutationPlan, args: AnnotationSaveArgs) -> MutationReceipt:
        annotation_id = str(plan.context["annotation_id"])
        target_type = str(plan.context["target_type"])
        target_id = str(plan.context["target_id"])
        note_text = str(plan.context["note_text"])
        created = args.archive.save_annotation(
            annotation_id,
            target_type,
            target_id,
            note_text,
            owner_session_id=cast(str | None, plan.context.get("owner_session_id")),
        )
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied",
            target_refs=plan.target_refs,
            affected_count=1,
            detail=None,
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"created": created},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> AnnotationSaveArgs:
        return AnnotationSaveArgs(
            archive=handles.archive,
            annotation_id=str(plan.context["annotation_id"]),
            target_type=str(plan.context["target_type"]),
            target_id=str(plan.context["target_id"]),
            note_text=str(plan.context["note_text"]),
            owner_session_id=cast("str | None", plan.context["owner_session_id"]),
        )

    def already_applied(self, handles: ReplayHandles, plan: MutationPlan) -> bool:
        stored = handles.archive.get_annotation(str(plan.context["annotation_id"]))
        owner = plan.context["owner_session_id"]
        return (
            stored is not None
            and (stored["target_type"], stored["target_id"], stored["note_text"])
            == (plan.context["target_type"], plan.context["target_id"], plan.context["note_text"])
            and (owner is None or stored["session_id"] == owner)
        )


@dataclass(frozen=True, slots=True)
class AnnotationDeleteArgs:
    """Shared prepare/apply argument shape for annotation deletion."""

    archive: ArchiveStore
    annotation_id: str


@dataclass(frozen=True, slots=True)
class AnnotationDeleteActuator(ConvergentReplay):
    """Actuator for ``mutate-delete-annotation``: reversible user.db annotation retraction.

    Real production mutation: ``ArchiveStore.delete_annotation``, which marks
    the annotation's assertion row deleted rather than physically removing
    it (undo = ``mutate-save-annotation``).
    """

    operation: str = "mutate-delete-annotation"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: AnnotationDeleteArgs) -> MutationPlan:
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(f"annotation:{args.annotation_id}",),
            affected_tiers=("user",),
            reversible=True,
            context={"annotation_id": args.annotation_id},
        )

    def apply(self, plan: MutationPlan, args: AnnotationDeleteArgs) -> MutationReceipt:
        annotation_id = str(plan.context["annotation_id"])
        deleted = args.archive.delete_annotation(annotation_id)
        status: MutationTargetStatus = "applied" if deleted else "already_satisfied"
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status=status,
            target_refs=plan.target_refs,
            affected_count=1 if deleted else 0,
            detail=None if deleted else "annotation_not_found",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"deleted": deleted},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> AnnotationDeleteArgs:
        return AnnotationDeleteArgs(archive=handles.archive, annotation_id=str(plan.context["annotation_id"]))


# ---------------------------------------------------------------------------
# Prepared blocker acknowledgement uses the same executor authorization and
# audit path as other mutations. Its Source capability stays on the original
# admitted creator through publication and physical settlement.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BlockerResolveArgs:
    """Shared prepare/apply argument shape for raw-authority blocker resolution."""

    archive_root: Path
    blocker_id: str
    resolution: str
    prepared: PreparedFrontierAcknowledgement


@dataclass(frozen=True, slots=True)
class BlockerResolveActuator(ConvergentReplay):
    """Actuator for ``mutate-resolve-raw-authority-blocker``: reopen raw replanning.

    Real production mutation: the prepared frontier acknowledgement publication
    -- the durable ``source.db`` acknowledgement that a frontier obligation
    was read and accepted, so the next census pass replans it against current
    evidence. Classified ``reset`` (not ``reversible``: an operator cannot
    literally un-resolve a blocker once acknowledged; not ``delete``/
    ``excise``: no archive content or evidence row is removed, only the
    blocked state is tombstoned so replanning can resume), requiring
    ``confirm_flag`` -- matching the CLI's pre-existing ``--yes`` gate this
    actuator now authorizes through instead of leaving unauthorized.

    Acknowledging a blocker repairs nothing: it records that the operator saw
    the obligation. Whatever the obligation names is discharged by ordinary
    acquisition or derivation, or it stays blocked.
    """

    operation: str = "mutate-resolve-raw-authority-blocker"
    destructive_class: DestructiveClass = "reset"
    required_confirmation: ConfirmationStrength = "confirm_flag"

    def prepare(self, args: BlockerResolveArgs) -> MutationPlan:
        prepared = args.prepared
        if (prepared.archive_root, prepared.blocker_id, prepared.resolution) != (
            args.archive_root,
            args.blocker_id,
            args.resolution,
        ):
            raise ValueError("blocker acknowledgement does not match its prepared authority")
        target_refs = (f"raw-authority-blocker:{args.blocker_id}",) if prepared.found else ()
        return build_plan(
            operation=self.operation,
            destructive_class="reset",
            target_refs=target_refs,
            affected_tiers=("source",),
            reversible=False,
            context={
                "blocker_id": args.blocker_id,
                "found": prepared.found,
                "kind": prepared.kind,
                "resolution": args.resolution,
            },
        )

    def apply(self, plan: MutationPlan, args: BlockerResolveArgs) -> MutationReceipt:
        if self.prepare(args).context != plan.context:
            raise PlanStaleError("blocker acknowledgement no longer matches its authorized target")
        if not plan.context.get("found"):
            return MutationReceipt(
                operation=self.operation,
                plan_hash=plan.plan_hash,
                status="already_satisfied",
                target_refs=plan.target_refs,
                affected_count=0,
                detail="blocker_not_found_or_already_resolved",
                receipt_ref=None,
                applied_at=plan.prepared_at,
            )
        try:
            receipt = args.prepared.publish()
        except KeyError:
            # Only an unresolved blocker is found, so a re-applied plan whose
            # first apply committed the resolution converges here.
            return MutationReceipt(
                operation=self.operation,
                plan_hash=plan.plan_hash,
                status="already_satisfied",
                target_refs=plan.target_refs,
                affected_count=0,
                detail="blocker_already_resolved",
                receipt_ref=None,
                applied_at=plan.prepared_at,
            )
        receipt_dict = dict(receipt)
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied",
            target_refs=plan.target_refs,
            affected_count=1,
            detail=None,
            receipt_ref=str(receipt_dict.get("detail_query_handle") or "") or None,
            applied_at=plan.prepared_at,
            domain_receipt=receipt_dict,
        )

    def recover(self, handles: ReplayHandles, plan: MutationPlan) -> RecoveryResolution:
        from polylogue.core.stage_admission import admit_stage_write
        from polylogue.storage.frontier_inspection import prepared_frontier_blocker_acknowledgement

        if handles.input_demand is None:
            raise RecoveryDeferredError("blocker acknowledgement requires its admitted preparation owner")
        with prepared_frontier_blocker_acknowledgement(
            handles.archive_root,
            str(plan.context["blocker_id"]),
            resolution=str(plan.context["resolution"]),
            input_demand=handles.input_demand,
        ) as prepared:
            args = BlockerResolveArgs(handles.archive_root, prepared.blocker_id, prepared.resolution, prepared)
            if not prepared.found:
                receipt = MutationReceipt(
                    operation=self.operation,
                    plan_hash=plan.plan_hash,
                    status="already_satisfied",
                    target_refs=plan.target_refs,
                    affected_count=0,
                    detail="blocker_not_found_or_already_resolved",
                    receipt_ref=None,
                    applied_at=plan.prepared_at,
                )
            else:
                receipt = admit_stage_write("operation.frontier.blocker.recover", lambda: self.apply(plan, args))
        if receipt.status in {"applied", "already_satisfied"}:
            return RecoveryResolution("complete", "re-applied the interrupted acknowledgement", receipt)
        return RecoveryResolution("replay-failed", f"acknowledgement recovery ended {receipt.status}")


# ---------------------------------------------------------------------------
# Saved-view mutations (mutate-save-saved-view / mutate-delete-saved-view) --
# phase 4 (t46.9/kwsb.2): the saved-view/recall-pack/workspace family (MCP
# operations save_saved_view/delete_saved_view, save_recall_pack/
# delete_recall_pack, save_workspace/delete_workspace) was declared-not-routed
# debt through phase 3. Each pair follows the exact reversible-class pattern
# already proven for tag/metadata/mark/annotation: the underlying ArchiveStore
# primitive is a create-or-update upsert paired with a soft-delete
# (mark_assertion_status ... "deleted"), so undo is always the paired actuator
# through the same primitive -- role_only confirmation per AC4.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SavedViewSaveArgs:
    """Shared prepare/apply argument shape for saved-view create/update."""

    archive: ArchiveStore
    view_id: str
    name: str
    query_json: str
    watch: bool = False


@dataclass(frozen=True, slots=True)
class SavedViewSaveActuator(ConvergentReplay):
    """Actuator for ``mutate-save-saved-view``: reversible user.db saved-view upsert.

    Real production mutation: ``ArchiveStore.save_view``, the same primitive
    ``PolylogueArchiveMixin.save_view`` already calls. Every apply writes
    (create-or-update), so ``created``-vs-``updated`` is a receipt detail
    (mirrors ``AnnotationSaveActuator``), not an idempotency short-circuit.
    """

    operation: str = "mutate-save-saved-view"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: SavedViewSaveArgs) -> MutationPlan:
        collision = args.archive.get_view_by_name(args.name)
        collision_view_id = (
            collision["view_id"] if collision is not None and collision["view_id"] != args.view_id else None
        )
        target_refs: tuple[str, ...] = (f"saved_view:{args.view_id}",)
        if collision_view_id is not None:
            target_refs = (*target_refs, f"saved_view:{collision_view_id}")
        if args.watch:
            # A watched view is also a durable standing-query definition. Compile
            # it here so a preview refuses an unwatchable selection instead of a
            # green plan whose apply fails (polylogue-pm8cj).
            import json as _json

            from polylogue.storage.sqlite.query_watch import validate_watch_definition

            parsed = _json.loads(args.query_json)
            if not isinstance(parsed, dict):
                raise ValueError("query_json must encode an object")
            validate_watch_definition(parsed)
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=target_refs,
            affected_tiers=("user",),
            reversible=True,
            context={
                "view_id": args.view_id,
                "name": args.name,
                "query_json": args.query_json,
                "collision_view_id": collision_view_id,
                "watch": args.watch,
            },
        )

    def apply(self, plan: MutationPlan, args: SavedViewSaveArgs) -> MutationReceipt:
        view_id = str(plan.context["view_id"])
        name = str(plan.context["name"])
        query_json = str(plan.context["query_json"])
        collision_view_id = plan.context["collision_view_id"]
        watch = bool(plan.context.get("watch"))
        created = args.archive.save_view(view_id, name, query_json, watch=watch)
        if watch:
            from polylogue.daemon.convergence_standing_queries import establish_watch_baselines

            establish_watch_baselines(args.archive.index_db_path, archive_root=args.archive.archive_root)
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied",
            target_refs=plan.target_refs,
            affected_count=len(plan.target_refs),
            detail=None,
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"created": created, "collision_view_id": collision_view_id, "watch": watch},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> SavedViewSaveArgs:
        return SavedViewSaveArgs(
            archive=handles.archive,
            view_id=str(plan.context["view_id"]),
            name=str(plan.context["name"]),
            query_json=str(plan.context["query_json"]),
            watch=bool(plan.context.get("watch")),
        )

    def already_applied(self, handles: ReplayHandles, plan: MutationPlan) -> bool:
        # The baseline is part of a watched save's effect: apply commits the
        # view first and measures after, so a crash between the two leaves a
        # durable watch with no baseline, and the next evaluation would absorb
        # the first changed session into it instead of reporting the delta.
        stored = handles.archive.get_view(str(plan.context["view_id"]))
        name = str(plan.context["name"])
        watch = bool(plan.context.get("watch"))
        return (
            stored is not None
            and stored["name"] == name
            and json.loads(stored["query_json"]) == json.loads(str(plan.context["query_json"]))
            and _view_watched(handles, name) == watch
            and (not watch or _view_watch_baselined(handles, name))
        )

    def replay_refusal(self, handles: ReplayHandles, plan: MutationPlan) -> str | None:
        collision = handles.archive.get_view_by_name(str(plan.context["name"]))
        live = (
            collision["view_id"] if collision is not None and collision["view_id"] != plan.context["view_id"] else None
        )
        if live != plan.context["collision_view_id"]:
            return f"saved view name is now held by {live!r}, which the authorized plan did not name"
        return None


@dataclass(frozen=True, slots=True)
class SavedViewDeleteArgs:
    """Shared prepare/apply argument shape for saved-view deletion."""

    archive: ArchiveStore
    view_id: str


@dataclass(frozen=True, slots=True)
class SavedViewDeleteActuator(ConvergentReplay):
    """Actuator for ``mutate-delete-saved-view``: reversible user.db saved-view retraction.

    Real production mutation: ``ArchiveStore.delete_view``, which marks the
    saved-view's assertion row deleted (undo = ``mutate-save-saved-view``).
    """

    operation: str = "mutate-delete-saved-view"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: SavedViewDeleteArgs) -> MutationPlan:
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(f"saved_view:{args.view_id}",),
            affected_tiers=("user",),
            reversible=True,
            context={"view_id": args.view_id},
        )

    def apply(self, plan: MutationPlan, args: SavedViewDeleteArgs) -> MutationReceipt:
        view_id = str(plan.context["view_id"])
        deleted = args.archive.delete_view(view_id)
        status: MutationTargetStatus = "applied" if deleted else "already_satisfied"
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status=status,
            target_refs=plan.target_refs,
            affected_count=1 if deleted else 0,
            detail=None if deleted else "saved_view_not_found",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"deleted": deleted},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> SavedViewDeleteArgs:
        return SavedViewDeleteArgs(archive=handles.archive, view_id=str(plan.context["view_id"]))

    def already_applied(self, handles: ReplayHandles, plan: MutationPlan) -> bool:
        # A committed delete must not run again: it clears the name-based
        # watch, which may now belong to a view saved under the name since.
        return handles.archive.get_view(str(plan.context["view_id"])) is None


# ---------------------------------------------------------------------------
# Recall-pack mutations (mutate-save-recall-pack / mutate-delete-recall-pack)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecallPackSaveArgs:
    """Shared prepare/apply argument shape for recall-pack create/update.

    ``session_ids_json``/``payload_json`` are the already-normalized values
    ``PolylogueArchiveMixin._build_recall_pack_payload`` produces (async item
    resolution happens once by the caller before the actuator runs, mirroring
    ``MarkArgs``/``AnnotationSaveArgs``'s "resolve once, plan and mutate the
    identical set" pattern).
    """

    archive: ArchiveStore
    pack_id: str
    label: str
    session_ids_json: str
    payload_json: str


@dataclass(frozen=True, slots=True)
class RecallPackSaveActuator(ConvergentReplay):
    """Actuator for ``mutate-save-recall-pack``: reversible user.db recall-pack upsert.

    Real production mutation: ``ArchiveStore.save_recall_pack``, the same
    primitive ``PolylogueArchiveMixin.create_recall_pack`` already calls.
    """

    operation: str = "mutate-save-recall-pack"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: RecallPackSaveArgs) -> MutationPlan:
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(f"recall_pack:{args.pack_id}",),
            affected_tiers=("user",),
            reversible=True,
            context={
                "pack_id": args.pack_id,
                "label": args.label,
                "session_ids_json": args.session_ids_json,
                "payload_json": args.payload_json,
            },
        )

    def apply(self, plan: MutationPlan, args: RecallPackSaveArgs) -> MutationReceipt:
        pack_id = str(plan.context["pack_id"])
        label = str(plan.context["label"])
        session_ids_json = str(plan.context["session_ids_json"])
        payload_json = str(plan.context["payload_json"])
        created = args.archive.save_recall_pack(pack_id, label, session_ids_json, payload_json)
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied",
            target_refs=plan.target_refs,
            affected_count=1,
            detail=None,
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"created": created},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> RecallPackSaveArgs:
        return RecallPackSaveArgs(
            archive=handles.archive,
            pack_id=str(plan.context["pack_id"]),
            label=str(plan.context["label"]),
            session_ids_json=str(plan.context["session_ids_json"]),
            payload_json=str(plan.context["payload_json"]),
        )

    def already_applied(self, handles: ReplayHandles, plan: MutationPlan) -> bool:
        stored = handles.archive.get_recall_pack(str(plan.context["pack_id"]))
        return (
            stored is not None
            and stored["label"] == plan.context["label"]
            and json.loads(stored["session_ids_json"]) == json.loads(str(plan.context["session_ids_json"]))
            and json.loads(stored["payload_json"]) == json.loads(str(plan.context["payload_json"]))
        )


@dataclass(frozen=True, slots=True)
class RecallPackDeleteArgs:
    """Shared prepare/apply argument shape for recall-pack deletion."""

    archive: ArchiveStore
    pack_id: str


@dataclass(frozen=True, slots=True)
class RecallPackDeleteActuator(ConvergentReplay):
    """Actuator for ``mutate-delete-recall-pack``: reversible user.db recall-pack retraction.

    Real production mutation: ``ArchiveStore.delete_recall_pack`` (undo =
    ``mutate-save-recall-pack``).
    """

    operation: str = "mutate-delete-recall-pack"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: RecallPackDeleteArgs) -> MutationPlan:
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(f"recall_pack:{args.pack_id}",),
            affected_tiers=("user",),
            reversible=True,
            context={"pack_id": args.pack_id},
        )

    def apply(self, plan: MutationPlan, args: RecallPackDeleteArgs) -> MutationReceipt:
        pack_id = str(plan.context["pack_id"])
        deleted = args.archive.delete_recall_pack(pack_id)
        status: MutationTargetStatus = "applied" if deleted else "already_satisfied"
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status=status,
            target_refs=plan.target_refs,
            affected_count=1 if deleted else 0,
            detail=None if deleted else "recall_pack_not_found",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"deleted": deleted},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> RecallPackDeleteArgs:
        return RecallPackDeleteArgs(archive=handles.archive, pack_id=str(plan.context["pack_id"]))


# ---------------------------------------------------------------------------
# Workspace mutations (mutate-save-workspace / mutate-delete-workspace)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkspaceSaveArgs:
    """Shared prepare/apply argument shape for reader-workspace create/update.

    ``open_targets_json``/``active_target_json`` are the already-normalized
    values ``PolylogueArchiveMixin._build_workspace_targets``/
    ``_build_workspace_active_target`` produce (async item resolution happens
    once by the caller before the actuator runs).
    """

    archive: ArchiveStore
    workspace_id: str
    name: str
    mode: str
    open_targets_json: str
    layout_json: str
    active_target_json: str


@dataclass(frozen=True, slots=True)
class WorkspaceSaveActuator(ConvergentReplay):
    """Actuator for ``mutate-save-workspace``: reversible user.db workspace upsert.

    Real production mutation: ``ArchiveStore.save_workspace``, the same
    primitive ``PolylogueArchiveMixin.save_workspace`` already calls.
    """

    operation: str = "mutate-save-workspace"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: WorkspaceSaveArgs) -> MutationPlan:
        collision = args.archive.get_workspace_by_name(args.name)
        collision_workspace_id = (
            collision["workspace_id"]
            if collision is not None and collision["workspace_id"] != args.workspace_id
            else None
        )
        target_refs: tuple[str, ...] = (f"workspace:{args.workspace_id}",)
        if collision_workspace_id is not None:
            target_refs = (*target_refs, f"workspace:{collision_workspace_id}")
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=target_refs,
            affected_tiers=("user",),
            reversible=True,
            context={
                "workspace_id": args.workspace_id,
                "name": args.name,
                "mode": args.mode,
                "open_targets_json": args.open_targets_json,
                "layout_json": args.layout_json,
                "active_target_json": args.active_target_json,
                "collision_workspace_id": collision_workspace_id,
            },
        )

    def apply(self, plan: MutationPlan, args: WorkspaceSaveArgs) -> MutationReceipt:
        created = args.archive.save_workspace(
            workspace_id=str(plan.context["workspace_id"]),
            name=str(plan.context["name"]),
            mode=str(plan.context["mode"]),
            open_targets_json=str(plan.context["open_targets_json"]),
            layout_json=str(plan.context["layout_json"]),
            active_target_json=str(plan.context["active_target_json"]),
        )
        collision_workspace_id = plan.context["collision_workspace_id"]
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied",
            target_refs=plan.target_refs,
            affected_count=len(plan.target_refs),
            detail=None,
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"created": created, "collision_workspace_id": collision_workspace_id},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> WorkspaceSaveArgs:
        return WorkspaceSaveArgs(
            archive=handles.archive,
            workspace_id=str(plan.context["workspace_id"]),
            name=str(plan.context["name"]),
            mode=str(plan.context["mode"]),
            open_targets_json=str(plan.context["open_targets_json"]),
            layout_json=str(plan.context["layout_json"]),
            active_target_json=str(plan.context["active_target_json"]),
        )

    def already_applied(self, handles: ReplayHandles, plan: MutationPlan) -> bool:
        stored = handles.archive.get_workspace(str(plan.context["workspace_id"]))
        return stored is not None and all(
            stored[key] == plan.context[key]
            for key in ("name", "mode", "open_targets_json", "layout_json", "active_target_json")
        )

    def replay_refusal(self, handles: ReplayHandles, plan: MutationPlan) -> str | None:
        collision = handles.archive.get_workspace_by_name(str(plan.context["name"]))
        live = (
            collision["workspace_id"]
            if collision is not None and collision["workspace_id"] != plan.context["workspace_id"]
            else None
        )
        if live != plan.context["collision_workspace_id"]:
            return f"workspace name is now held by {live!r}, which the authorized plan did not name"
        return None


@dataclass(frozen=True, slots=True)
class WorkspaceDeleteArgs:
    """Shared prepare/apply argument shape for reader-workspace deletion."""

    archive: ArchiveStore
    workspace_id: str


@dataclass(frozen=True, slots=True)
class WorkspaceDeleteActuator(ConvergentReplay):
    """Actuator for ``mutate-delete-workspace``: reversible user.db workspace retraction.

    Real production mutation: ``ArchiveStore.delete_workspace`` (undo =
    ``mutate-save-workspace``).
    """

    operation: str = "mutate-delete-workspace"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: WorkspaceDeleteArgs) -> MutationPlan:
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(f"workspace:{args.workspace_id}",),
            affected_tiers=("user",),
            reversible=True,
            context={"workspace_id": args.workspace_id},
        )

    def apply(self, plan: MutationPlan, args: WorkspaceDeleteArgs) -> MutationReceipt:
        workspace_id = str(plan.context["workspace_id"])
        deleted = args.archive.delete_workspace(workspace_id)
        status: MutationTargetStatus = "applied" if deleted else "already_satisfied"
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status=status,
            target_refs=plan.target_refs,
            affected_count=1 if deleted else 0,
            detail=None if deleted else "workspace_not_found",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"deleted": deleted},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> WorkspaceDeleteArgs:
        return WorkspaceDeleteArgs(archive=handles.archive, workspace_id=str(plan.context["workspace_id"]))


# ---------------------------------------------------------------------------
# Learning corrections (mutate-record-correction / mutate-delete-correction /
# mutate-clear-corrections) -- t46.9/kwsb.2 phase 5: the remaining named
# census family after saved-view/recall-pack/workspace (#3262) landed.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorrectionRecordArgs:
    """Shared prepare/apply argument shape for recording a learning correction."""

    archive: ArchiveStore
    session_id: str
    kind: str
    payload: dict[str, str]
    note: str | None = None
    author_ref: str | None = None
    author_kind: str | None = None


@dataclass(frozen=True, slots=True)
class CorrectionRecordActuator(ConvergentReplay):
    """Actuator for ``mutate-record-correction``: reversible user.db correction upsert.

    Real production mutation: ``ArchiveStore.record_correction``, the same
    primitive ``PolylogueArchiveMixin.record_correction`` already calls.
    Every apply writes (insert-or-replace on the ``(session, kind)`` unique
    key); undo is ``mutate-delete-correction``.
    """

    operation: str = "mutate-record-correction"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: CorrectionRecordArgs) -> MutationPlan:
        resolved = args.archive.resolve_session_id(args.session_id)
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(f"correction:{resolved}:{args.kind}",),
            affected_tiers=("user",),
            reversible=True,
            context={
                "session_id": resolved,
                "kind": args.kind,
                "payload": dict(args.payload),
                "note": args.note,
                "author_ref": args.author_ref,
                "author_kind": args.author_kind,
            },
        )

    def apply(self, plan: MutationPlan, args: CorrectionRecordArgs) -> MutationReceipt:
        session_id = str(plan.context["session_id"])
        kind = str(plan.context["kind"])
        payload = cast("dict[str, str]", plan.context["payload"])
        note = cast("str | None", plan.context["note"])
        author_ref = cast("str | None", plan.context["author_ref"])
        author_kind = cast("str | None", plan.context["author_kind"])
        correction = args.archive.record_correction(
            session_id, kind, payload, note=note, author_ref=author_ref, author_kind=author_kind
        )
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied",
            target_refs=plan.target_refs,
            affected_count=1,
            detail=None,
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"correction": correction},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> CorrectionRecordArgs:
        return CorrectionRecordArgs(
            archive=handles.archive,
            session_id=str(plan.context["session_id"]),
            kind=str(plan.context["kind"]),
            payload=dict(cast("dict[str, str]", plan.context["payload"])),
            note=cast("str | None", plan.context["note"]),
            author_ref=cast("str | None", plan.context["author_ref"]),
            author_kind=cast("str | None", plan.context["author_kind"]),
        )

    def already_applied(self, handles: ReplayHandles, plan: MutationPlan) -> bool:
        from polylogue.storage.sqlite.archive_tiers.user_write import correction_effect_matches

        with closing(
            open_readonly_connection(handles.archive_root / "user.db", timeout_class="background-read")
        ) as conn:
            return correction_effect_matches(
                conn,
                "insight",
                str(plan.context["session_id"]),
                str(plan.context["kind"]),
                {"payload": plan.context["payload"], "note": plan.context["note"]},
                author_ref=cast("str | None", plan.context["author_ref"]),
                author_kind=cast("str | None", plan.context["author_kind"]),
            )


@dataclass(frozen=True, slots=True)
class CorrectionDeleteArgs:
    """Shared prepare/apply argument shape for deleting one learning correction."""

    archive: ArchiveStore
    session_id: str
    kind: str


@dataclass(frozen=True, slots=True)
class CorrectionDeleteActuator(ConvergentReplay):
    """Actuator for ``mutate-delete-correction``: reversible user.db correction retraction.

    Real production mutation: ``ArchiveStore.delete_correction``, which
    marks one correction's assertion row deleted (undo =
    ``mutate-record-correction``).
    """

    operation: str = "mutate-delete-correction"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: CorrectionDeleteArgs) -> MutationPlan:
        resolved = args.archive.resolve_session_id(args.session_id)
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(f"correction:{resolved}:{args.kind}",),
            affected_tiers=("user",),
            reversible=True,
            context={"session_id": resolved, "kind": args.kind},
        )

    def apply(self, plan: MutationPlan, args: CorrectionDeleteArgs) -> MutationReceipt:
        session_id = str(plan.context["session_id"])
        kind = str(plan.context["kind"])
        deleted = args.archive.delete_correction(session_id, kind)
        status: MutationTargetStatus = "applied" if deleted else "already_satisfied"
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status=status,
            target_refs=plan.target_refs,
            affected_count=1 if deleted else 0,
            detail=None if deleted else "correction_not_found",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"deleted": deleted},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> CorrectionDeleteArgs:
        return CorrectionDeleteArgs(
            archive=handles.archive, session_id=str(plan.context["session_id"]), kind=str(plan.context["kind"])
        )


@dataclass(frozen=True, slots=True)
class CorrectionsClearArgs:
    """Shared prepare/apply argument shape for clearing every correction on a session."""

    archive: ArchiveStore
    session_id: str


@dataclass(frozen=True, slots=True)
class CorrectionsClearActuator(ConvergentReplay):
    """Actuator for ``mutate-clear-corrections``: reversible bulk user.db retraction.

    Real production mutation: ``ArchiveStore.clear_corrections``. Unlike
    ``CorrectionDeleteActuator``, ``prepare`` re-resolves the *exact
    currently live set* of correction kinds for the session (mirroring
    ``SessionDeleteActuator``'s "re-resolve existence against live state"
    pattern), so a concurrent ``mutate-record-correction`` that adds a new
    kind between AUTHORIZE and EXECUTE changes the plan hash and forces a
    replan (``PlanStaleError``) instead of silently clearing a kind the
    caller never previewed.
    """

    operation: str = "mutate-clear-corrections"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: CorrectionsClearArgs) -> MutationPlan:
        resolved = args.archive.resolve_session_id(args.session_id)
        existing_kinds = tuple(
            sorted({correction.kind.value for correction in args.archive.list_corrections(session_id=resolved)})
        )
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=tuple(f"correction:{resolved}:{kind}" for kind in existing_kinds),
            affected_tiers=("user",),
            reversible=True,
            context={"session_id": resolved, "kinds": list(existing_kinds)},
        )

    def apply(self, plan: MutationPlan, args: CorrectionsClearArgs) -> MutationReceipt:
        session_id = str(plan.context["session_id"])
        cleared = args.archive.clear_corrections(session_id)
        status: MutationTargetStatus = "applied" if cleared else "already_satisfied"
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status=status,
            target_refs=plan.target_refs,
            affected_count=cleared,
            detail=None if cleared else "no_corrections",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"cleared_count": cleared},
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> CorrectionsClearArgs:
        return CorrectionsClearArgs(archive=handles.archive, session_id=str(plan.context["session_id"]))

    def recover(self, handles: ReplayHandles, plan: MutationPlan) -> RecoveryResolution:
        """Clear only the kinds the plan authorized, not any recorded since."""
        session_id = str(plan.context["session_id"])
        try:
            cleared = sum(
                bool(handles.archive.delete_correction(session_id, kind))
                for kind in cast("list[str]", plan.context["kinds"])
            )
        except KeyError as exc:
            # The session is not in the rebuildable index yet; the durable
            # corrections stay live until convergence restores it.
            raise RecoveryDeferredError(f"{plan.operation} target does not resolve yet: {exc}") from exc
        return RecoveryResolution(
            "complete",
            "cleared the authorized correction kinds that remained",
            MutationReceipt(
                operation=self.operation,
                plan_hash=plan.plan_hash,
                status="applied" if cleared else "already_satisfied",
                target_refs=plan.target_refs,
                affected_count=cleared,
                detail=None if cleared else "no_corrections",
                receipt_ref=None,
                applied_at=plan.prepared_at,
                domain_receipt={"cleared_count": cleared},
            ),
        )


# ---------------------------------------------------------------------------
# Blackboard post (mutate-blackboard-post) -- phase 6 (t46.9/kwsb.2): closes
# the "blackboard_post" declared-not-routed census row. This is an additive
# append-only write (a fresh note id is minted per call, so ``prepare`` never
# reads live state -- there is nothing to race), but it had NO OperationSpec
# entry at all before this phase, so it still lacked the shared capability
# declaration and MutationReceipt audit trail every other user.db write gets.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BlackboardPostArgs:
    """Shared prepare/apply argument shape for one blackboard note append.

    ``note_id`` is minted by the caller (mirroring ``AnnotationSaveArgs``'s
    caller-resolved ``annotation_id``) so the plan hash is stable across the
    executor's PREPARE -> fresh-PREPARE revalidation at EXECUTE time --
    ``prepare`` never mints its own id, or two calls would never agree on a
    plan hash.
    """

    archive: ArchiveStore
    note_id: str
    body: str
    target_type: str | None
    target_id: str | None
    author_ref: str | None
    author_kind: str
    evidence_refs: tuple[str, ...]
    staleness: dict[str, object] | None
    context_policy: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class BlackboardPostActuator(ConvergentReplay):
    """Actuator for ``mutate-blackboard-post``: append-only user.db note insert.

    Real production mutation: ``ArchiveStore.post_blackboard_note``, the same
    primitive ``PolylogueArchiveMixin.post_blackboard_note`` already calls.
    Classified ``reversible`` (least-severe class -- there is no destructive
    counterpart today) with ``role_only`` confirmation, consistent with the
    other additive user.db writes (annotation save, mark add).
    """

    operation: str = "mutate-blackboard-post"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: BlackboardPostArgs) -> MutationPlan:
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=(f"blackboard:{args.note_id}",),
            affected_tiers=("user",),
            reversible=True,
            context={
                "note_id": args.note_id,
                "body": args.body,
                "target_type": args.target_type,
                "target_id": args.target_id,
                "author_ref": args.author_ref,
                "author_kind": args.author_kind,
                "evidence_refs": list(args.evidence_refs),
                "staleness": args.staleness,
                "context_policy": args.context_policy,
            },
        )

    def apply(self, plan: MutationPlan, args: BlackboardPostArgs) -> MutationReceipt:
        envelope = args.archive.post_blackboard_note(
            str(plan.context["body"]),
            target_type=cast("str | None", plan.context["target_type"]),
            target_id=cast("str | None", plan.context["target_id"]),
            note_id=str(plan.context["note_id"]),
            author_ref=cast("str | None", plan.context["author_ref"]),
            author_kind=str(plan.context["author_kind"]),
            evidence_refs=tuple(cast("list[str]", plan.context["evidence_refs"])),
            staleness=cast("dict[str, object] | None", plan.context["staleness"]),
            context_policy=cast("dict[str, object] | None", plan.context["context_policy"]),
        )
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied",
            target_refs=plan.target_refs,
            affected_count=1,
            detail=None,
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={
                "note_id": envelope.note_id,
                "target_type": envelope.target_type,
                "target_id": envelope.target_id,
                "body": envelope.body,
                "created_at_ms": envelope.created_at_ms,
                "updated_at_ms": envelope.updated_at_ms,
            },
        )

    def replay_args(self, handles: ReplayHandles, plan: MutationPlan) -> BlackboardPostArgs:
        context = plan.context
        return BlackboardPostArgs(
            archive=handles.archive,
            note_id=str(context["note_id"]),
            body=str(context["body"]),
            target_type=cast("str | None", context["target_type"]),
            target_id=cast("str | None", context["target_id"]),
            author_ref=cast("str | None", context["author_ref"]),
            author_kind=str(context["author_kind"]),
            evidence_refs=tuple(cast("list[str]", context["evidence_refs"])),
            staleness=cast("dict[str, object] | None", context["staleness"]),
            context_policy=cast("dict[str, object] | None", context["context_policy"]),
        )

    def already_applied(self, handles: ReplayHandles, plan: MutationPlan) -> bool:
        stored = next(
            (note for note in handles.archive.list_blackboard_notes() if note.note_id == plan.context["note_id"]), None
        )
        return stored is not None and (stored.body, stored.target_type, stored.target_id) == (
            plan.context["body"],
            plan.context["target_type"],
            plan.context["target_id"],
        )


@dataclass(frozen=True, slots=True)
class InsightsRebuildArgs:
    """Archive handle, exact session scope, and optional progress observer."""

    archive: ArchiveStore
    session_ids: tuple[str, ...] | None = None
    progress_callback: ProgressCallback | None = None


@dataclass(frozen=True, slots=True)
class InsightsRebuildActuator(ConvergentReplay):
    """Actuator for the canonical durable session-insight materializer."""

    operation: str = "mutate-rebuild-insights"
    destructive_class: DestructiveClass = "maintenance"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: InsightsRebuildArgs) -> MutationPlan:
        from polylogue.operations.insight_acceptance import AcceptedInsightTarget, build_insight_page_context
        from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION

        if args.session_ids is None:
            # A full rebuild is identified by its scope, not by enumerating
            # every session (polylogue-buuxr): apply() rebuilds the whole index
            # from ``scope_kind`` alone, and a per-session target list both
            # cost four archive-sized collections per prepare and exceeded the
            # plan's target ceiling on a large archive.
            resolved: tuple[str, ...] = ()
        else:
            # An explicitly named session that does not resolve is a caller
            # error, not a smaller sweep: every single-target actuator lets
            # ``resolve_session_id`` raise here, and dropping the id would
            # rebuild a narrower scope than the caller authorized.
            resolved = tuple(dict.fromkeys(args.archive.resolve_session_id(sid) for sid in args.session_ids))
        scope_kind: Literal["explicit", "full"] = "full" if args.session_ids is None else "explicit"
        manifest_entries: list[object] = (
            [("scope", "full", str(args.archive.index_db_path), str(SESSION_INSIGHT_MATERIALIZER_VERSION))]
            if scope_kind == "full"
            else [(f"session:{session_id}", "required") for session_id in resolved]
        )
        manifest_digest = hashlib.sha256(json.dumps(manifest_entries, separators=(",", ":")).encode()).hexdigest()
        context = build_insight_page_context(
            scope_kind=scope_kind,
            index_generation=f"index-generation:{args.archive.index_db_path}",
            recipe_version=str(SESSION_INSIGHT_MATERIALIZER_VERSION),
            ordinal=0,
            page_count=1,
            manifest_digest=manifest_digest,
            previous_preview_ref=None,
            targets=tuple(AcceptedInsightTarget(f"session:{session_id}", "required") for session_id in resolved),
        )
        return build_plan(
            operation=self.operation,
            destructive_class="maintenance",
            target_refs=tuple(make_target_ref("session", session_id) for session_id in resolved),
            affected_tiers=("index",),
            reversible=False,
            context=context,
        )

    def apply(self, plan: MutationPlan, args: InsightsRebuildArgs) -> MutationReceipt:
        from polylogue.storage.derived.session.rebuild import rebuild_session_insights_sync
        from polylogue.storage.derived.session.runtime import SessionInsightCounts

        session_ids = tuple(
            str(target["target_ref"]).removeprefix("session:")
            for target in cast("list[dict[str, object]]", plan.context.get("targets") or ())
            if target.get("disposition") == "required"
        )
        full_rebuild = plan.context.get("scope_kind") == "full"
        if not session_ids and not full_rebuild:
            counts = SessionInsightCounts()
        else:
            counts = rebuild_session_insights_sync(
                args.archive._conn,
                session_ids=None if full_rebuild else session_ids,
                progress_callback=args.progress_callback,
                # A declared index rebuild owns canonical usage too: the
                # rollup is an index-tier input of the profiles this
                # operation replaces, and the plan's affected tier already
                # names index. Stated rather than inherited so the one route
                # that may reconcile usage says so (polylogue-bp12n.1 AC2).
                reconcile_usage_rollup=True,
            )
        affected_count = counts.total()
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied" if affected_count else "already_satisfied",
            target_refs=plan.target_refs,
            affected_count=affected_count,
            detail=None if affected_count else "no_insight_rows_rebuilt",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt=counts.to_dict(),
        )

    def recover(self, _handles: ReplayHandles, _plan: MutationPlan) -> RecoveryResolution:
        """An interrupted insight page is terminalized, not re-derived here.

        Every run of this family is a sealed machine part whose completion is
        an ``InsightPartHistoricalReceipt`` only the staged owner produces, and
        a page's ``scope_kind`` can still say ``full``. The derived rows it was
        rebuilding reconverge through ordinary convergence or a new rebuild
        request; the interrupted part reports failed instead of hanging.
        """
        return RecoveryResolution(
            "not-replayable", "an interrupted insight page is re-derived by convergence or a new rebuild request"
        )


#: The one named gap a multi-target mutation reports when the caller named a
#: session the archive could not resolve at PREPARE time.
UNRESOLVED_SESSION_GAP = "unresolved_session_ids"


def _partition_requested_session_ids(
    archive: ArchiveStore, session_ids: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split caller-named ids into (resolved, unresolved), preserving order.

    A multi-target mutation may not silently shrink to the ids that still
    resolve: an id excised between selection and execution is a *named gap*,
    not an absence of intent. PREPARE freezes the resolved half as effect
    targets and retains original IDs and named gaps in the closed machine
    context. APPLY and recovery report that same gap without resolving it
    into newly available targets.

    A multi-target selection is a set of full session ids its caller already
    resolved once (the CLI from its query, a facade caller from its own
    reads), so membership is exact. An abbreviated or vanished id is named in
    the gap; it is never widened to whichever session shares its prefix.
    """

    requested = tuple(dict.fromkeys(session_ids))
    resolved = archive.stored_session_ids(requested)
    present = set(resolved)
    return resolved, tuple(sid for sid in requested if sid not in present)


def _narrowed_plan_outcome(*, matched: int, unresolved: Sequence[str]) -> OutcomeEnvelope:
    """Decide the terminal outcome of a mutation whose plan named a gap.

    ``degraded`` outranks ``empty``: a bulk mutation that changed nothing
    because every named id had been excised is not an empty scope.
    """

    return decide_outcome(
        matched=matched,
        degraded=(UNRESOLVED_SESSION_GAP,) if unresolved else (),
        detail={UNRESOLVED_SESSION_GAP: list(unresolved)} if unresolved else None,
    )


def _narrowed_plan_detail(*, affected: int, unresolved: Sequence[str]) -> str | None:
    if unresolved:
        return UNRESOLVED_SESSION_GAP
    return None if affected else "no_sessions_changed"


register_recovery_route(
    SessionDeleteActuator(),
    SessionExcisionActuator(),
    SessionLifecycleRequestActuator(),
    IdentityResetActuator(),
    FilesystemResetActuator(),
    BlobPublicationAbandonActuator(),
    TagAddActuator(),
    TagRemoveActuator(),
    BulkTagActuator(),
    MetadataSetActuator(),
    BulkMetadataSetActuator(),
    MetadataDeleteActuator(),
    MarkAddActuator(),
    MarkRemoveActuator(),
    CaptureAssertionCandidateActuator(),
    SetUserSettingActuator(),
    AnnotationSaveActuator(),
    AnnotationDeleteActuator(),
    BlockerResolveActuator(),
    SavedViewSaveActuator(),
    SavedViewDeleteActuator(),
    RecallPackSaveActuator(),
    RecallPackDeleteActuator(),
    WorkspaceSaveActuator(),
    WorkspaceDeleteActuator(),
    CorrectionRecordActuator(),
    CorrectionDeleteActuator(),
    CorrectionsClearActuator(),
    BlackboardPostActuator(),
    InsightsRebuildActuator(),
)

__all__ = [
    "UNRESOLVED_SESSION_GAP",
    "AnnotationDeleteActuator",
    "AnnotationDeleteArgs",
    "AnnotationSaveActuator",
    "AnnotationSaveArgs",
    "BlackboardPostActuator",
    "BlackboardPostArgs",
    "BlockerResolveActuator",
    "BlockerResolveArgs",
    "BulkTagActuator",
    "BulkTagArgs",
    "CaptureAssertionCandidateActuator",
    "CaptureAssertionCandidateArgs",
    "CorrectionDeleteActuator",
    "CorrectionDeleteArgs",
    "CorrectionRecordActuator",
    "CorrectionRecordArgs",
    "CorrectionsClearActuator",
    "CorrectionsClearArgs",
    "IdentityResetActuator",
    "IdentityResetArgs",
    "InsightsRebuildActuator",
    "InsightsRebuildArgs",
    "MarkAddActuator",
    "MarkArgs",
    "MarkRemoveActuator",
    "MetadataDeleteActuator",
    "MetadataDeleteArgs",
    "MetadataSetActuator",
    "MetadataSetArgs",
    "RecallPackDeleteActuator",
    "RecallPackDeleteArgs",
    "RecallPackSaveActuator",
    "RecallPackSaveArgs",
    "SavedViewDeleteActuator",
    "SavedViewDeleteArgs",
    "SavedViewSaveActuator",
    "SavedViewSaveArgs",
    "SessionDeleteActuator",
    "SessionDeleteArgs",
    "SessionExcisionActuator",
    "SessionExcisionArgs",
    "SessionLifecycleRequestActuator",
    "SessionLifecycleRequestArgs",
    "TagAddActuator",
    "TagAddArgs",
    "TagRemoveActuator",
    "TagRemoveArgs",
    "WorkspaceDeleteActuator",
    "WorkspaceDeleteArgs",
    "WorkspaceSaveActuator",
    "WorkspaceSaveArgs",
]

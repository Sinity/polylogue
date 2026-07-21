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

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from polylogue.operations.mutation_transaction import (
    ConfirmationStrength,
    DestructiveClass,
    MutationPlan,
    MutationReceipt,
    MutationTargetStatus,
    build_plan,
    make_target_ref,
)

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _session_exists(archive: ArchiveStore, session_id: str) -> bool:
    try:
        archive.resolve_session_id(session_id)
    except KeyError:
        return False
    return True


# ---------------------------------------------------------------------------
# Session delete (mutate-delete-session)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionDeleteArgs:
    """Shared prepare/apply argument shape for session delete."""

    archive: ArchiveStore
    session_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SessionDeleteActuator:
    """Actuator for ``mutate-delete-session``: permanent, re-ingest-resurrectable removal.

    Real production mutation: ``ArchiveStore.delete_sessions`` -- the single
    low-level primitive CLI ``delete`` and MCP ``write(operation=
    'delete_session')`` both already reach, via ``_emit_delete`` and
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
        # plans the subset that still exists right now.
        existing = tuple(sid for sid in dict.fromkeys(args.session_ids) if _session_exists(args.archive, sid))
        return build_plan(
            operation=self.operation,
            destructive_class="delete",
            target_refs=tuple(make_target_ref("session", sid) for sid in existing),
            affected_tiers=("index",),
            reversible=False,
            context={"session_ids": list(existing)},
        )

    def apply(self, plan: MutationPlan, args: SessionDeleteArgs) -> MutationReceipt:
        session_ids: tuple[str, ...] = tuple(cast("list[str]", plan.context.get("session_ids") or ()))
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
            domain_receipt={"deleted_count": deleted, "session_count": len(session_ids)},
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


@dataclass(frozen=True, slots=True)
class SessionExcisionActuator:
    """Actuator for ``mutate-session-excision``: durable, re-ingest-proof removal.

    Real production mutation: ``security.excision.plan_session_excision`` /
    ``apply_session_excision`` -- the cross-tier (source/index/embeddings/
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
        from polylogue.security.excision import plan_session_excision

        plan = plan_session_excision(args.archive_root, args.session_id)
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
                "found": plan.found,
                "reason": args.reason,
                "cascade_lineage": args.cascade_lineage,
                "lineage_dependent_session_ids": list(plan.lineage_dependent_session_ids),
            },
        )

    def apply(self, plan: MutationPlan, args: SessionExcisionArgs) -> MutationReceipt:
        from polylogue.security.excision import LineageDependentsError, apply_session_excision

        if not plan.context.get("found"):
            return MutationReceipt(
                operation=self.operation,
                plan_hash=plan.plan_hash,
                status="unknown",
                target_refs=plan.target_refs,
                affected_count=0,
                detail="session_not_found",
                receipt_ref=None,
                applied_at=plan.prepared_at,
            )
        try:
            receipt = apply_session_excision(
                args.archive_root,
                args.session_id,
                reason=args.reason,
                actor=args.actor,
                cascade_lineage=args.cascade_lineage,
            )
        except LineageDependentsError as exc:
            return MutationReceipt(
                operation=self.operation,
                plan_hash=plan.plan_hash,
                status="blocked",
                target_refs=plan.target_refs,
                affected_count=0,
                detail=str(exc),
                receipt_ref=None,
                applied_at=plan.prepared_at,
            )
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied" if receipt.found else "unknown",
            target_refs=plan.target_refs,
            affected_count=receipt.counts.get("index_sessions", 0),
            detail=None,
            receipt_ref=receipt.receipt_assertion_id,
            applied_at=plan.prepared_at,
            domain_receipt=receipt.as_dict(),
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
class IdentityResetActuator:
    """Actuator for ``mutate-identity-reset``: tombstone + rebuildable-row delete.

    Real production mutation: the ``polylogue ops reset --session/--source``
    tombstone helpers (``_suppress_archive_sessions`` writes the durable
    user.db suppression, ``_delete_archive_sessions`` drops the rebuildable
    index.db rows). Target resolution (token -> exact session ids) happens
    once by the caller before ``prepare`` is invoked, mirroring #jnj.5's
    "resolve once, preview and mutate the identical set" fix; ``prepare``
    only re-verifies that the resolved set still exists so a concurrent
    change is caught before APPLY.
    """

    operation: str = "mutate-identity-reset"
    destructive_class: DestructiveClass = "reset"
    required_confirmation: ConfirmationStrength = "confirm_flag"

    def prepare(self, args: IdentityResetArgs) -> MutationPlan:
        existing = tuple(dict.fromkeys(_resolve_existing_session_ids(args.archive_root, args.session_ids)))
        return build_plan(
            operation=self.operation,
            destructive_class="reset",
            target_refs=tuple(make_target_ref("session", sid) for sid in existing),
            affected_tiers=("index", "user"),
            reversible=True,
            context={"session_ids": list(existing), "reason": args.reason},
        )

    def apply(self, plan: MutationPlan, args: IdentityResetArgs) -> MutationReceipt:
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
            )

        user_db = args.archive_root / "user.db"
        initialize_archive_database(user_db, ArchiveTier.USER)
        conn = sqlite3.connect(user_db)
        try:
            with conn:
                for session_id in session_ids:
                    upsert_suppression(conn, session_id=session_id, reason=args.reason, mode="hide")
        finally:
            conn.close()
        suppressed = len(session_ids)

        index_db = _index_db_path(args.archive_root)
        deleted = 0
        if index_db.exists():
            index_conn = sqlite3.connect(index_db)
            index_conn.execute("PRAGMA foreign_keys = ON")
            try:
                with index_conn:
                    for session_id in session_ids:
                        cursor = index_conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                        deleted += max(int(cursor.rowcount), 0)
            finally:
                index_conn.close()

        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied",
            target_refs=plan.target_refs,
            affected_count=suppressed,
            detail=None,
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={"suppressed_count": suppressed, "deleted_archive_rows": deleted},
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
    conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
    try:
        if not session_ids:
            return ()
        placeholders = ",".join("?" for _ in session_ids)
        rows = conn.execute(
            f"SELECT session_id FROM sessions WHERE session_id IN ({placeholders})",
            session_ids,
        ).fetchall()
        found = {str(row[0]) for row in rows}
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
class TagAddActuator:
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
            context={"session_id": resolved, "tag": args.tag},
        )

    def apply(self, plan: MutationPlan, args: TagAddArgs) -> MutationReceipt:
        session_id = str(plan.context["session_id"])
        tag = str(plan.context["tag"])
        changed = args.archive.add_user_tags(
            (session_id,), (tag,), author_ref=args.author_ref, author_kind=args.author_kind
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


@dataclass(frozen=True, slots=True)
class TagRemoveArgs:
    """Shared prepare/apply argument shape for single-session tag remove."""

    archive: ArchiveStore
    session_id: str
    tag: str


@dataclass(frozen=True, slots=True)
class TagRemoveActuator:
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


@dataclass(frozen=True, slots=True)
class BulkTagArgs:
    """Shared prepare/apply argument shape for multi-session bulk tagging."""

    archive: ArchiveStore
    session_ids: tuple[str, ...]
    tags: tuple[str, ...]
    author_ref: str | None = None
    author_kind: str | None = None


@dataclass(frozen=True, slots=True)
class BulkTagActuator:
    """Actuator for ``mutate-bulk-tag-sessions``: reversible multi-target tagging.

    Real production mutation: ``ArchiveStore.add_user_tags`` applied per
    resolved session, mirroring ``PolylogueArchiveMixin.bulk_tag_sessions``'s
    existing skip-unresolved behavior -- ``prepare`` only plans sessions that
    resolve against live state right now, same pattern as
    ``SessionDeleteActuator``.
    """

    operation: str = "mutate-bulk-tag-sessions"
    destructive_class: DestructiveClass = "reversible"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: BulkTagArgs) -> MutationPlan:
        resolved: list[str] = []
        for session_id in dict.fromkeys(args.session_ids):
            try:
                resolved.append(args.archive.resolve_session_id(session_id))
            except KeyError:
                continue
        return build_plan(
            operation=self.operation,
            destructive_class="reversible",
            target_refs=tuple(make_target_ref("session", sid) for sid in resolved),
            affected_tiers=("user",),
            reversible=True,
            context={
                "session_ids": resolved,
                "tags": list(args.tags),
                "requested_session_count": len(args.session_ids),
            },
        )

    def apply(self, plan: MutationPlan, args: BulkTagArgs) -> MutationReceipt:
        session_ids: tuple[str, ...] = tuple(cast("list[str]", plan.context.get("session_ids") or ()))
        tags: tuple[str, ...] = tuple(cast("list[str]", plan.context.get("tags") or ()))
        requested_count = int(cast("int", plan.context.get("requested_session_count") or len(session_ids)))
        affected = 0
        for session_id in session_ids:
            if (
                args.archive.add_user_tags(
                    (session_id,), tags, author_ref=args.author_ref, author_kind=args.author_kind
                )
                > 0
            ):
                affected += 1
        status: MutationTargetStatus = "applied" if affected else "already_satisfied"
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status=status,
            target_refs=plan.target_refs,
            affected_count=affected,
            detail=None if affected else "no_sessions_changed",
            receipt_ref=None,
            applied_at=plan.prepared_at,
            domain_receipt={
                "session_count": requested_count,
                "tag_count": len(tags),
                "affected_count": affected,
                "skipped_count": requested_count - affected,
            },
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
class MetadataSetActuator:
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


@dataclass(frozen=True, slots=True)
class MetadataDeleteArgs:
    """Shared prepare/apply argument shape for session metadata delete."""

    archive: ArchiveStore
    session_id: str
    key: str


@dataclass(frozen=True, slots=True)
class MetadataDeleteActuator:
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


@dataclass(frozen=True, slots=True)
class MarkAddActuator:
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
            context={"target_type": args.target_type, "target_id": args.target_id, "mark_type": args.mark_type},
        )

    def apply(self, plan: MutationPlan, args: MarkArgs) -> MutationReceipt:
        added = args.archive.add_mark(args.target_type, args.target_id, args.mark_type)
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


@dataclass(frozen=True, slots=True)
class MarkRemoveActuator:
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
            context={"target_type": args.target_type, "target_id": args.target_id, "mark_type": args.mark_type},
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


__all__ = [
    "BulkTagActuator",
    "BulkTagArgs",
    "IdentityResetActuator",
    "IdentityResetArgs",
    "MarkAddActuator",
    "MarkArgs",
    "MarkRemoveActuator",
    "MetadataDeleteActuator",
    "MetadataDeleteArgs",
    "MetadataSetActuator",
    "MetadataSetArgs",
    "SessionDeleteActuator",
    "SessionDeleteArgs",
    "SessionExcisionActuator",
    "SessionExcisionArgs",
    "TagAddActuator",
    "TagAddArgs",
    "TagRemoveActuator",
    "TagRemoveArgs",
]

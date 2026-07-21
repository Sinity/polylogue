"""Domain actuators for the two t46.9/kwsb.2 named routes.

Each actuator wraps exactly one existing low-level mutation primitive
(``ArchiveStore.delete_sessions``, ``security.excision``, the identity-reset
tombstone helpers) behind the :class:`~polylogue.operations.mutation_transaction
.MutationActuator` protocol. Actuators own target resolution and the real
mutation; they never enforce authorization -- every surface drives them
through :class:`~polylogue.operations.mutation_transaction.OperationExecutor`.
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


__all__ = [
    "IdentityResetActuator",
    "IdentityResetArgs",
    "SessionDeleteActuator",
    "SessionDeleteArgs",
    "SessionExcisionActuator",
    "SessionExcisionArgs",
]

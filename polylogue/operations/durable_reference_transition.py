"""Operations façade for the rebuild's durable-reference rebind step.

The surfaces reach the transition through here: planning needs the archive's
own topology, and applying it needs the archive lease and a proven backup for
both durable tiers. Every storage-tier failure leaves as one typed error.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path

from polylogue.maintenance.assertion_transition import ObjectRefReconciliationError
from polylogue.maintenance.durable_reference_transition import (
    DurableReferenceTransition,
    apply_durable_reference_transition,
    plan_durable_reference_transition,
    publish_transition_receipt,
    source_session_claims,
)
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.version import POLYLOGUE_VERSION


class DurableReferenceTransitionError(RuntimeError):
    """A transition could not be planned or applied against this archive."""


def resolve_archive_location(root: Path) -> tuple[Path, Path]:
    """The configured durable root and its active index generation."""
    location = ArchiveLocation.resolve(root.absolute())
    return location.configured_root, location.active_index_path


def _readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def plan_transition(
    *,
    archive_root: Path,
    candidate_index: Path,
    predecessor_index: Path,
    producer: str,
    user_conn: sqlite3.Connection | None = None,
    audit_conn: sqlite3.Connection | None = None,
) -> DurableReferenceTransition:
    """Classify every durable reference against a candidate index. Read-only.

    ``user_conn``/``audit_conn`` are supplied by :func:`apply_transition`,
    which already holds them read-write under the archive lease.
    """
    try:
        with ExitStack() as stack:
            user = (
                user_conn
                if user_conn is not None
                else stack.enter_context(closing(_readonly(archive_root / "user.db")))
            )
            audit = (
                audit_conn
                if audit_conn is not None
                else stack.enter_context(closing(_readonly(archive_root / "audit.db")))
            )
            source = stack.enter_context(closing(_readonly(archive_root / "source.db")))
            candidate = stack.enter_context(closing(_readonly(candidate_index)))
            predecessor = stack.enter_context(closing(_readonly(predecessor_index)))
            return plan_durable_reference_transition(
                user_conn=user,
                audit_conn=audit,
                candidate_index_conn=candidate,
                predecessor_index_conn=predecessor,
                source_claims=source_session_claims(source),
                package_version=POLYLOGUE_VERSION,
                producer=producer,
                candidate_index_path=str(candidate_index),
                predecessor_index_path=str(predecessor_index),
            )
    except (ObjectRefReconciliationError, sqlite3.Error) as exc:
        raise DurableReferenceTransitionError(str(exc)) from exc


def apply_transition(
    *,
    archive_root: Path,
    candidate_index: Path,
    predecessor_index: Path,
    producer: str,
    backup_manifest: Path,
    expected_plan_digest: str,
) -> tuple[DurableReferenceTransition, str]:
    """Apply the authorized transition and publish its receipt.

    Refuses unless the archive lease is free, the backup manifest authenticates
    against both durable tiers, and the plan recomputed here still has the
    digest being authorized.
    """
    from polylogue.operations.durable_change_train import ArchiveOwnershipError, acquire_durable_archive_ownership
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.migration_runner import MigrationError, validate_migration_backup_manifest

    try:
        with acquire_durable_archive_ownership(archive_root, owner_id=f"durable-reference-transition:{os.getpid()}"):
            with (
                closing(sqlite3.connect(archive_root / "user.db")) as user,
                closing(sqlite3.connect(archive_root / "audit.db")) as audit,
            ):
                validate_migration_backup_manifest(backup_manifest, ArchiveTier.USER, connection=user)
                validate_migration_backup_manifest(backup_manifest, ArchiveTier.AUDIT, connection=audit)
                # Recompute rather than deserialize: the plan document is an
                # authorization, and the durable tiers may have moved since.
                transition = plan_transition(
                    archive_root=archive_root,
                    candidate_index=candidate_index,
                    predecessor_index=predecessor_index,
                    producer=producer,
                    user_conn=user,
                    audit_conn=audit,
                )
                if transition.plan.digest() != expected_plan_digest:
                    raise DurableReferenceTransitionError("archive state no longer matches the authorized plan")
                apply_durable_reference_transition(
                    user_conn=user, audit_conn=audit, transition=transition, verified_backup=True
                )
            receipt = publish_transition_receipt(archive_root, transition, applied=True)
    except (ObjectRefReconciliationError, ArchiveOwnershipError, MigrationError, sqlite3.Error) as exc:
        raise DurableReferenceTransitionError(str(exc)) from exc
    return transition, receipt


__all__ = [
    "DurableReferenceTransition",
    "DurableReferenceTransitionError",
    "apply_transition",
    "plan_transition",
    "resolve_archive_location",
]

"""Resolve interrupted mutations from durable state, never from operator testimony.

An operation whose owner died between its durable intent and its terminal
record is resolved at daemon startup (here) or when a later request overlaps
its targets (the executor). Its actuator either re-applies the recorded plan
convergently or, for an atomic apply, reports whether the commit is present.
The outcome is a terminal :class:`RecoveryResolution`; none leaves an
``unknown`` run or a barrier over the targets.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.storage.archive_identity import OwnedArchiveLocation

from polylogue.operations.mutation_transaction import (
    RecoverableActuator,
    registered_recovery_routes,
    resolve_interrupted_operations,
)


def recoverable_actuators() -> Mapping[str, RecoverableActuator]:
    """Every executor-routed mutation family's recovery route, by operation name.

    Importing each actuator module registers its families
    (``register_recovery_route``); this is the one place that imports them
    all, so startup recovery sees the complete set.
    """

    import polylogue.annotations.importer
    import polylogue.operations.ingest_acceptance
    import polylogue.operations.mutation_actuators  # noqa: F401

    return registered_recovery_routes()


def recover_interrupted_operations(archive_root: Path) -> None:
    """Resolve dead operations at the daemon's single-writer startup seam.

    This is deliberately not executor composition.  Request handlers construct
    executors frequently; only daemon startup owns the writer lease that makes
    classifying an interrupted effect safe.

    The work is bounded and one-pass: abandoned attempts are terminalized
    first, then every remaining orphan is classified exactly once, so a
    restart over an already-recovered archive appends no further durable
    events.

    Every orphan is resolved from durable state by its actuator: re-applied
    convergently, or found committed or absent for an atomic apply.  No
    outcome is ``unknown`` and none leaves a barrier over its targets
    (:class:`RecoveryResolution`).  The exception is an accepted ingest whose
    request was never stopped: it stays nonterminal for the daemon's ingest
    owner, which re-drives it once this recovery has run
    (``DaemonOperationRuntime.start_accepted_ingest_redrive``).
    """

    if not (archive_root / "audit.db").is_file():
        return
    from polylogue.operations.audit import AuditRepository

    audit = AuditRepository.for_archive_root(
        archive_root,
        attempt_owner_id=AuditRepository.current_process_attempt_owner(),
    )
    audit.reconcile_continuity()
    # No handler can still be appending pages to a paged machine batch.
    audit.fence_staged_machine_pages()
    # Startup is the single-writer point where a dead ingest cannot still be
    # preparing pages. Continuity has promoted every accepted generation or
    # refused startup, so the remaining unpromoted headers are pre-accept work.
    if (archive_root / "source.db").is_file():
        from contextlib import closing

        from polylogue.storage.sqlite.archive_tiers.source_items import reconcile_unaccepted_prepared_source_manifests
        from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection

        with (
            closing(
                open_isolated_write_connection(
                    archive_root / "source.db", purpose="startup ingest preparation recovery", archive_root=archive_root
                )
            ) as source,
            source,
        ):
            source.execute("BEGIN IMMEDIATE")
            reconcile_unaccepted_prepared_source_manifests(source)
    # Terminalize dead attempts *before* discovering orphans so one startup
    # converges: otherwise a run this call marks interrupted would only be
    # classified by the next restart.
    audit.recover_abandoned_attempts()
    orphans = audit.orphaned_operations()
    if not orphans:
        return
    recoverable_actuators()
    resolve_interrupted_operations(audit, archive_root, orphans)


def apply_staged_archive_resets(archive_root: Path) -> tuple[str, ...]:
    """Apply the resets a previous daemon staged, before any tier is opened.

    ``maintenance.reset`` naming ``index.db``/``ops.db`` records its authorized
    plan and a running attempt owned by the daemon that received it, and
    deletes nothing. Once that process has exited, this seam resolves every
    such plan whose owner is dead through its actuator's ``recover`` inside
    :func:`~polylogue.operations.reset_safety.archive_tiers_closed`. The
    caller holds exclusive archive ownership and has opened no tier, so no
    handle can be left on an unlinked file.

    Only operations whose recovery route replaces archive files are touched;
    everything else stays for :func:`recover_interrupted_operations`. Returns
    the ids of the operations resolved here.
    """

    if not (archive_root / "audit.db").is_file():
        return ()
    from polylogue.operations.audit import AuditRepository
    from polylogue.operations.reset_safety import archive_tiers_closed

    audit = AuditRepository.for_archive_root(
        archive_root,
        attempt_owner_id=AuditRepository.current_process_attempt_owner(),
    )
    audit.reconcile_continuity()
    routes = recoverable_actuators()
    staged = tuple(
        operation
        for operation in audit.orphaned_operations()
        if getattr(routes.get(operation.operation), "replaces_archive_files", False)
    )
    if not staged:
        return ()
    with archive_tiers_closed(archive_root):
        deferred = set(resolve_interrupted_operations(audit, archive_root, staged))
    return tuple(operation.operation_id for operation in staged if operation.operation_id not in deferred)


def reconverge_disposable_ops_on_startup(archive_root: Path, *, archive_owner: OwnedArchiveLocation) -> bool:
    """Replace a stale ops tier under exclusive ownership, with no live handles.

    Inspection failures propagate: a lock or unreadable page is not evidence
    of schema drift. Only typed schema skew authorizes discarding disposable
    state. Durable tiers, index generations and purchased vectors are untouched.
    """
    from contextlib import closing

    from polylogue.core.errors import SchemaSkew
    from polylogue.operations.durable_change_train import assert_holds_archive_ownership
    from polylogue.operations.reset_safety import LiveArchiveTierResetError, archive_tiers_are_closed
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.connection_profile import assert_tier_schema_supported, open_readonly_connection
    from polylogue.storage.sqlite.write_lease import require_write_lease

    assert_holds_archive_ownership(archive_owner, archive_root)
    require_write_lease("daemon ops reconvergence", archive_root=archive_root)
    if not archive_tiers_are_closed(archive_root):
        raise LiveArchiveTierResetError(("ops.db",))
    path = archive_owner.location.configured_tier("ops").resolved_path
    if not path.exists():
        return False
    try:
        with closing(open_readonly_connection(path, validate_schema=False)) as conn:
            assert_tier_schema_supported(conn, path, ArchiveTier.OPS)
    except SchemaSkew:
        # Close the inspection handle before unlinking any member of the file
        # family. A restart interrupted here sees absence and bootstraps fresh.
        from polylogue.operations.reset_safety import discard_closed_derived_tier

        discard_closed_derived_tier(archive_root, path)
        initialize_archive_database(path, ArchiveTier.OPS)
        return True
    return False


__all__ = [
    "apply_staged_archive_resets",
    "recover_interrupted_operations",
    "recoverable_actuators",
    "reconverge_disposable_ops_on_startup",
]

"""Resolve interrupted mutations from durable state, never from operator testimony.

An operation whose owner died between its durable intent and its terminal
record is resolved at daemon startup (here) or when a later request overlaps
its targets (the executor). Its actuator either re-applies the recorded plan
convergently or, for an atomic apply, reports whether the commit is present.
An established effect produces a terminal :class:`RecoveryResolution`.
Missing or conflicting exact-attempt evidence defers recovery and preserves
the interrupted run and its target barrier.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.storage.archive_identity import OwnedArchiveLocation

from polylogue.operations.mutation_transaction import (
    RecoverableActuator,
    RecoveryDeferredError,
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


# Automated recovery has no human request principal. Keep its event identity stable.
RECOVERY_SERVICE_ACTOR_REF = "daemon:recovery"


def recover_interrupted_operations(
    archive_root: Path, *, resolver_actor_ref: str, input_demand: Callable[[int], None]
) -> None:
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
    completed resolution leaves an unknown run or a barrier over its targets.
    Excision with missing or conflicting original-attempt evidence refuses
    startup and preserves its barrier. An accepted ingest whose
    request was never stopped stays nonterminal for the daemon's ingest
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
    from polylogue.core.stage_admission import admit_stage_write

    admit_stage_write("daemon.operation_recovery.continuity", audit.reconcile_continuity)
    # No handler can still be appending pages to a paged machine batch.
    admit_stage_write("daemon.operation_recovery.machine-pages", audit.fence_staged_machine_pages)
    # Startup is the single-writer point where a dead ingest cannot still be
    # preparing pages. Continuity has promoted every accepted generation or
    # refused startup, so the remaining unpromoted headers are pre-accept work.
    _reconcile_startup_source_preparation(archive_root, input_demand=input_demand)
    # Terminalize dead attempts *before* discovering orphans so one startup
    # converges: otherwise a run this call marks interrupted would only be
    # classified by the next restart.
    admit_stage_write("daemon.operation_recovery.abandoned", audit.recover_abandoned_attempts)
    with audit.settled_machine_read():
        orphans = audit.orphaned_operations()
    if not orphans:
        return
    recoverable_actuators()
    deferred = resolve_interrupted_operations(
        audit, archive_root, orphans, resolver_actor_ref=resolver_actor_ref, input_demand=input_demand
    )
    if any(
        operation.operation == "mutate-session-excision" and operation.operation_id in deferred for operation in orphans
    ):
        raise RecoveryDeferredError("startup Excision recovery lacks settled exact-attempt evidence")


def _reconcile_startup_source_preparation(archive_root: Path, *, input_demand: Callable[[int], None]) -> None:
    """Retain the canonical pre-accept cleanup under one original Source witness."""
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.storage.io_phase_metrics import connection_cursor
    from polylogue.storage.sqlite.connection_profile import readonly_connection_context
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealError

    source_path = archive_root / "source.db"
    if not source_path.is_file():
        raise ReferenceSealError("startup Source reconciliation lacks its declared durable tier")
    orphan = (
        "SELECT source_generation_id, publisher_id FROM prepared_source_manifests p "
        "WHERE NOT EXISTS (SELECT 1 FROM source_generations g "
        "WHERE g.source_generation_id=p.source_generation_id)"
    )
    # A genuine empty pre-accept set requires neither a writer nor a claim.
    # This probe never certifies a nonempty cleanup: the original witness below
    # independently selects and validates its exact rows before publication.
    with (
        readonly_connection_context(source_path) as source,
        connection_cursor(source, f"SELECT 1 FROM ({orphan}) LIMIT 1") as cursor,
    ):
        if cursor.fetchone() is None:
            return
    with PreparedIndexMutation.source_only(archive_root=archive_root, input_demand=input_demand) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            selected = (
                ("blob_publication_reservations", f"publisher_id IN (SELECT publisher_id FROM ({orphan}))"),
                (
                    "prepared_source_manifest_members",
                    f"source_generation_id IN (SELECT source_generation_id FROM ({orphan}))",
                ),
                (
                    "prepared_source_manifests",
                    "NOT EXISTS (SELECT 1 FROM source_generations g "
                    "WHERE g.source_generation_id=prepared_source_manifests.source_generation_id)",
                ),
            )
            for table, predicate in selected:
                columns, keys = seal._known_tier_table_shape("source", table)
                after: int | None = None
                while True:
                    with seal.original_rows(
                        "source",
                        f'SELECT rowid FROM "{table}" WHERE ({predicate}) '
                        "AND (? IS NULL OR rowid>?) ORDER BY rowid LIMIT 256",
                        (after, after),
                    ) as rows:
                        page = tuple(int(row[0]) for row in rows)
                    if not page:
                        break
                    for rowid in page:
                        image = seal.retain_tier_row("source", table, rowid)
                        if image is None:
                            raise ReferenceSealError("original startup Source dependency disappeared")
                        seal.load_source_row(image)
                        expressions = [seal.source_literal_expression(image.cells[key]) for key in keys]
                        where = " AND ".join(
                            f'"{columns[key]}" IS {expression}'
                            for key, (expression, _operands) in zip(keys, expressions, strict=True)
                        )
                        parameters = tuple(value for _expression, operands in expressions for value in operands)
                        with seal.source_statement(
                            f'DELETE FROM "{table}" WHERE {where}',
                            parameters,
                            table=table,
                            writable_targets=((table, tuple(image.cells[key] for key in keys)),),
                        ):
                            pass
                    after = page[-1]
        permit = seal.prepare_source_mutation()

        def publish() -> None:
            with permit.hold_authority(), permit.mutation_connection() as source:
                with seal._owned_cursor(source, "BEGIN IMMEDIATE"):
                    pass
                permit.apply_source_statements(source)
                permit.allow_commit(source)
                source.commit()
                seal.accept_known_tier_commit(permit.committed())

        admit_stage_write("daemon.operation_recovery.source-preparation", publish)


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
        deferred = set(
            resolve_interrupted_operations(audit, archive_root, staged, resolver_actor_ref=RECOVERY_SERVICE_ACTOR_REF)
        )
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
    try:
        path.lstat()
    except FileNotFoundError:
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

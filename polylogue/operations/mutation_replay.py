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


__all__ = ["recover_interrupted_operations", "recoverable_actuators"]

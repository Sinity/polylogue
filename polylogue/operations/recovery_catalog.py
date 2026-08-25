"""Closed production recovery composition for executor-routed operations.

This is deliberately a small literal catalog, not reflection over actuator
classes. A new executor-routed family has to be named here before the daemon
can start, which prevents an unclassified destructive route from inheriting a
permissive recovery default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from polylogue.operations.mutation_transaction import (
    RecoveryDeclaration,
    RecoveryDisposition,
    RecoveryOperation,
)

RecoveryInspector = Callable[[RecoveryOperation, Path], RecoveryDisposition]


@dataclass(frozen=True, slots=True)
class RecoveryBinding:
    operation: str
    operation_version: int
    declaration: RecoveryDeclaration
    inspector: RecoveryInspector | None = None


OPERATOR_BLOCKING_RECOVERY = RecoveryDeclaration(
    target_identity="typed-targets-v1",
    plan_binding="plan-hash-and-target-digest-v1",
    precondition_inspection="domain-owned",
    postcondition_inspection="domain-owned",
    exact_retry=False,
    partial_actions=(),
    capability="operator-blocking",
)

DELETE_RECOVERY = RecoveryDeclaration(
    target_identity="typed-targets-v1",
    plan_binding="plan-hash-and-target-digest-v1",
    precondition_inspection="domain-owned",
    postcondition_inspection="domain-owned",
    exact_retry=True,
    partial_actions=("retry-exact", "forward"),
    capability="inspect-and-retry",
)


def _inspect_session_delete(operation: RecoveryOperation, archive_root: Path) -> RecoveryDisposition:
    from polylogue.operations.mutation_actuators import SessionDeleteActuator, SessionDeleteArgs
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
        return SessionDeleteActuator().inspect_recovery(operation, SessionDeleteArgs(archive=archive, session_ids=()))


# One explicit entry per executor-routed family. Keep this list boring: a
# family earns automatic continuation only with a domain inspector.
_DECLARATIONS: dict[str, RecoveryDeclaration] = {
    "mutate-add-tag": OPERATOR_BLOCKING_RECOVERY,
    "mutate-remove-tag": OPERATOR_BLOCKING_RECOVERY,
    "mutate-bulk-tag-sessions": OPERATOR_BLOCKING_RECOVERY,
    "mutate-set-metadata": OPERATOR_BLOCKING_RECOVERY,
    "mutate-delete-metadata": OPERATOR_BLOCKING_RECOVERY,
    "mutate-add-mark": OPERATOR_BLOCKING_RECOVERY,
    "mutate-remove-mark": OPERATOR_BLOCKING_RECOVERY,
    "mutate-save-annotation": OPERATOR_BLOCKING_RECOVERY,
    "mutate-delete-annotation": OPERATOR_BLOCKING_RECOVERY,
    "mutate-blackboard-post": OPERATOR_BLOCKING_RECOVERY,
    "mutate-capture-assertion-candidate": OPERATOR_BLOCKING_RECOVERY,
    "mutate-import-annotation-batch": OPERATOR_BLOCKING_RECOVERY,
    "mutate-rebuild-index": OPERATOR_BLOCKING_RECOVERY,
    "mutate-update-index": OPERATOR_BLOCKING_RECOVERY,
    "mutate-rebuild-insights": OPERATOR_BLOCKING_RECOVERY,
    "mutate-resolve-raw-authority-blocker": OPERATOR_BLOCKING_RECOVERY,
    "mutate-reset-raw-authority-census": OPERATOR_BLOCKING_RECOVERY,
    "mutate-prune-orphaned-index-revision-seeds": OPERATOR_BLOCKING_RECOVERY,
    "mutate-save-saved-view": OPERATOR_BLOCKING_RECOVERY,
    "mutate-delete-saved-view": OPERATOR_BLOCKING_RECOVERY,
    "mutate-save-recall-pack": OPERATOR_BLOCKING_RECOVERY,
    "mutate-delete-recall-pack": OPERATOR_BLOCKING_RECOVERY,
    "mutate-save-workspace": OPERATOR_BLOCKING_RECOVERY,
    "mutate-delete-workspace": OPERATOR_BLOCKING_RECOVERY,
    "mutate-record-correction": OPERATOR_BLOCKING_RECOVERY,
    "mutate-delete-correction": OPERATOR_BLOCKING_RECOVERY,
    "mutate-clear-corrections": OPERATOR_BLOCKING_RECOVERY,
    "mutate-delete-session": DELETE_RECOVERY,
    "mutate-session-excision": OPERATOR_BLOCKING_RECOVERY,
    "mutate-session-lifecycle-request": OPERATOR_BLOCKING_RECOVERY,
    "mutate-identity-reset": OPERATOR_BLOCKING_RECOVERY,
}


def recovery_declaration_for(operation: str) -> RecoveryDeclaration:
    try:
        return _DECLARATIONS[operation]
    except KeyError as exc:
        raise ValueError(f"executor-routed operation has no explicit recovery declaration: {operation}") from exc


def build_recovery_catalog() -> tuple[RecoveryBinding, ...]:
    """Resolve the complete runtime inventory and reject undeclared families."""

    from polylogue.operations.specs import build_runtime_operation_catalog

    bindings: list[RecoveryBinding] = []
    required: set[str] = set()
    for spec in build_runtime_operation_catalog().specs:
        if not (spec.mutates_state and spec.executor_status == "executor-routed"):
            continue
        required.add(spec.name)
        declaration = recovery_declaration_for(spec.name)
        if spec.recovery != declaration:
            raise ValueError(f"runtime recovery declaration drift for {spec.name}")
        bindings.append(
            RecoveryBinding(
                spec.name,
                spec.operation_version,
                declaration,
                _inspect_session_delete if spec.name == "mutate-delete-session" else None,
            )
        )
    unexpected = sorted(set(_DECLARATIONS) - required)
    if unexpected:
        raise ValueError(f"recovery declarations have no executor-routed operation: {unexpected}")
    return tuple(bindings)


def resolve_recovery_binding(operation: str, operation_version: int) -> RecoveryBinding | None:
    for binding in build_recovery_catalog():
        if binding.operation == operation and binding.operation_version == operation_version:
            return binding
    return None


def recover_archive_operations(archive_root: Path) -> None:
    """Run bounded classified recovery at the daemon's writer-owned startup seam."""

    from polylogue.operations.mutation_transaction import OperationExecutor

    # First-run daemon bootstraps can publish the archive tiers later in the
    # startup sequence. There is no durable operation evidence to recover
    # before audit.db exists, so avoid creating or probing a synthetic tier.
    if not (archive_root / "audit.db").is_file():
        return
    OperationExecutor.for_archive_root(archive_root)


__all__ = [
    "RecoveryBinding",
    "RecoveryInspector",
    "build_recovery_catalog",
    "recover_archive_operations",
    "recovery_declaration_for",
    "resolve_recovery_binding",
]

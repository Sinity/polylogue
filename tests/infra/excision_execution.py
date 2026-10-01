"""Canonical excision execution and validated low-level fault ownership."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.mutation_actuators import SessionExcisionActuator, SessionExcisionArgs
from polylogue.operations.mutation_transaction import MutationPrincipal, OperationExecutor, _authorized_removal_apply
from polylogue.security.excision import ExcisionReceipt, apply_session_excision
from polylogue.storage.sqlite.write_lease import write_lease


def execute_excision(
    archive_root: Path, session_id: str, *, reason: str, actor: str = "user:local", cascade_lineage: bool = False
) -> Mapping[str, Any]:
    """Return the real audited executor's canonical domain receipt."""
    args = SessionExcisionArgs(archive_root, session_id, reason, actor, cascade_lineage)
    binding = runtime_operation_binding(SessionExcisionActuator())
    principal = MutationPrincipal(actor, frozenset({"archive.excise_session"}), "api", "test-excision")
    with write_lease("test.excision-execution", archive_root=archive_root):
        executor = OperationExecutor.for_archive_root(archive_root)
        preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=archive_root)
        authorization = executor.authorize_bound(binding, preview, principal, confirmation_strength="bound_token")
        receipt = executor.execute_bound(binding, preview, authorization, args)
        assert receipt.status in {"applied", "already_satisfied"}, receipt
        return cast(Mapping[str, Any], receipt.domain_receipt)


def apply_excision_fault_control(
    archive_root: Path,
    session_id: str,
    *,
    reason: str,
    actor: str = "user:local",
    now_ms: int | None = None,
    cascade_lineage: bool = False,
) -> ExcisionReceipt:
    """Exercise the primitive under a real begun library apply owner.

    The library executor validates and consumes an actual bound preview. It
    intentionally has no Audit repository: these controls isolate primitive
    fault/time/retry behavior and do not claim audited publication coverage.
    """
    args = SessionExcisionArgs(archive_root, session_id, reason, actor, cascade_lineage)
    actuator = SessionExcisionActuator()
    binding = runtime_operation_binding(actuator)
    principal = MutationPrincipal(actor, frozenset({"archive.excise_session"}), "api", "test-excision-fault")
    with write_lease("test.excision-fault", archive_root=archive_root):
        executor = OperationExecutor(archive_root=archive_root)
        preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=archive_root)
        authorization = executor.authorize_bound(binding, preview, principal, confirmation_strength="bound_token")
        started = executor.begin_bound(binding, preview, authorization, args)
        with _authorized_removal_apply(started.plan, archive_root, actuator, args):
            return apply_session_excision(
                archive_root, session_id, reason=reason, actor=actor, now_ms=now_ms, cascade_lineage=cascade_lineage
            )

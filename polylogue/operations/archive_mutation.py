"""Archive-owned execution boundary for embedded mutation adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from polylogue.config import Config, active_archive_root
from polylogue.core.errors import PolylogueError, SessionNotFoundError
from polylogue.operations.mutation_transaction import (
    MutationActuator,
    MutationPlan,
    MutationPrincipal,
    MutationReceipt,
    OperationExecutor,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

_Args = TypeVar("_Args")


class MutationBlockedError(PolylogueError):
    """The mutation plan was refused rather than applied."""

    http_status_code = 409

    def __init__(self, operation: str, detail: str | None, target_refs: tuple[str, ...] = ()) -> None:
        super().__init__(f"{operation} was blocked: {detail or 'no reason recorded'}")
        self.operation = operation
        self.detail = detail
        self.target_refs = target_refs


class MutationTargetVanishedError(PolylogueError):
    """The target disappeared between authorization and apply."""

    http_status_code = 409


def require_archive_write_authority(config: Config, purpose: str) -> None:
    """Permit a scoped daemon lease or a proven exclusive offline owner."""
    from polylogue.core.write_lease import coordinator_write_lease_active, require_write_lease
    from polylogue.maintenance.offline_guard import (
        ArchiveWriterOwnershipError,
        ArchiveWriterOwnershipUndecidableError,
        DaemonResidencyUndecidableError,
        offline_writer_block_reason,
        resident_daemon_pid,
    )

    root = active_archive_root(config)
    try:
        block_reason = offline_writer_block_reason(config)
    except DaemonResidencyUndecidableError as exc:
        raise ArchiveWriterOwnershipUndecidableError(
            f"{purpose} cannot prove whether a resident daemon owns {root}: {exc}. "
            "Refusing rather than opening a writable archive tier beside an unseen writer",
            archive_root=root,
        ) from exc

    if block_reason is not None and not coordinator_write_lease_active():
        daemon_pid = resident_daemon_pid(root)
        resident_writer = (
            f"polylogued PID {daemon_pid} is running for this archive" if daemon_pid is not None else block_reason
        )
        raise ArchiveWriterOwnershipError(
            f"{purpose} may not write {root}: {resident_writer}. Route the mutation through the "
            "resident daemon, or stop it and run this operation as the archive's exclusive offline owner",
            archive_root=root,
            resident_writer=resident_writer,
        )

    require_write_lease(purpose, archive_root=root)


def execute_archive_mutation(
    config: Config,
    actuator: MutationActuator[_Args],
    build_args: Callable[[ArchiveStore], _Args],
    *,
    capability: str,
    session_id: str | None = None,
) -> tuple[MutationReceipt, MutationPlan]:
    """Prepare, authorize, and execute one bound mutation at the archive owner."""
    from polylogue.operations.bindings import runtime_operation_binding
    from polylogue.storage.sqlite.write_lease import write_lease

    require_archive_write_authority(config, "api.facade_mutation")
    root = active_archive_root(config)
    # The operation owner takes cross-process archive custody before the first
    # writable tier is opened; daemon callers re-enter the coordinator's lease.
    with (
        write_lease("api.facade_mutation", archive_root=root),
        ArchiveStore.open_existing(root, read_only=False) as archive,
    ):
        executor = OperationExecutor.for_archive_root(root)
        binding = runtime_operation_binding(actuator)
        principal = MutationPrincipal("facade", frozenset({capability}), "api", "write")
        try:
            args = build_args(archive)
            preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=root)
            authorization = executor.authorize_bound(binding, preview, principal)
        except KeyError:
            if session_id is None:
                raise
            raise SessionNotFoundError(session_id) from None
        try:
            receipt = executor.execute_bound(binding, preview, authorization, args)
        except KeyError as exc:
            raise MutationTargetVanishedError(
                f"{actuator.operation!r} target disappeared during execution: {exc}"
            ) from exc
    if receipt.status == "blocked":
        raise MutationBlockedError(receipt.operation, receipt.detail, receipt.target_refs)
    return receipt, preview.plan


__all__ = [
    "MutationBlockedError",
    "MutationTargetVanishedError",
    "execute_archive_mutation",
    "require_archive_write_authority",
]

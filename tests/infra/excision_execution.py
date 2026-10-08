"""Canonical excision execution and validated low-level fault ownership."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from polylogue.operations.audit import AuditRepository
from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.mutation_actuators import SessionExcisionActuator, SessionExcisionArgs
from polylogue.operations.mutation_transaction import (
    MutationPrincipal,
    OperationExecutor,
    StartedBoundMutation,
)
from polylogue.storage.sqlite.write_lease import write_lease

_T = TypeVar("_T")


ExcisionProductSink = Callable[[Any, Any], None]


def run_in_excision_owner(
    archive_root: Path,
    operation: Callable[[Callable[[int], None], ExcisionProductSink], _T],
    *,
    result_sink: Any = None,
) -> tuple[_T, list[Mapping[str, Any]]]:
    """Run ``operation`` inside a real Excision operation owner.

    ``operation`` receives the owner's original creator byte-demand admission
    and its result delivery sink; it returns its own value beside every
    delivered product, each independently decoded for assertions. This
    fixture proves effects and the sink contract, not daemon delivery.
    """
    import asyncio
    import hashlib
    import json
    from math import inf

    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.daemon.operation_runtime import DaemonOperationRuntime
    from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
    from polylogue.storage.sqlite.literal_cells import owned_literal_stream

    products: list[Mapping[str, Any]] = []

    def sink(summary: Any, literal: Any) -> None:
        with owned_literal_stream(literal.chunks()) as chunks:
            data = b"".join(chunks)
        assert len(data) == literal.byte_length
        assert hashlib.sha256(data).hexdigest() == literal.sha256
        document = json.loads(data)
        assert document["counts"] == summary["counts"]
        if result_sink is not None:
            result_sink(summary, literal)
        products.append(document)

    async def run() -> _T:
        coordinator = DaemonWriteCoordinator(archive_root=archive_root)
        kernel = BoundedComputeAdapter(max_workers=1, queue_units=1, queue_bytes=0)
        bridge = DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop())
        runtime = DaemonOperationRuntime(
            archive_root,
            write_bridge=bridge,
            execution_kernel=kernel,
            raw_observation_owner=RawObservationConvergenceOwner(
                archive_root, compute_adapter=kernel, write_bridge=bridge, write_coordinator=coordinator
            ),
        )
        try:
            return await runtime.prepared_phase(
                "test-excision-effects",
                lambda: operation(kernel.amend_current_input_demand, sink),
                estimated_bytes=0,
                exclusive_bytes=True,
            )
        finally:
            await runtime.shutdown()
            kernel.shutdown(wait=True)
            assert await coordinator.shutdown(timeout=inf)

    return asyncio.run(run()), products


def execute_excision(
    archive_root: Path,
    session_id: str,
    *,
    reason: str,
    actor: str = "user:local",
    cascade_lineage: bool = False,
    result_sink: Any = None,
) -> Mapping[str, Any]:
    """Run the real prepared creator and consume a complete synthetic product."""
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.operations.mutation_transaction import compute_parameter_digest
    from polylogue.storage.archive_identity import ArchiveIdentity

    def execute(input_demand: Callable[[int], None], sink: ExcisionProductSink) -> None:
        args = SessionExcisionArgs(
            archive_root,
            session_id,
            reason,
            actor,
            cascade_lineage,
            input_demand=input_demand,
            result_sink=sink,
        )
        binding = runtime_operation_binding(SessionExcisionActuator())
        principal = MutationPrincipal(actor, frozenset({"archive.excise_session"}), "cli", "test-excision")
        executor = OperationExecutor.for_archive_root(archive_root)
        raw = binding.actuator.prepare(args)
        preview = admit_stage_write(
            "test.excision-preview",
            lambda: executor.prepare_bound(
                binding,
                args,
                principal,
                raw_plan=raw,
                archive_instance_id=_required_audit(executor).ensure_archive_authority(now_ms=executor._now_ms()),
                archive_identity_digest=ArchiveIdentity.resolve(archive_root).authority_identity_digest,
                parameter_digest=compute_parameter_digest(raw),
            ),
        )
        authorization = admit_stage_write(
            "test.excision-authorize",
            lambda: executor.authorize_bound(
                binding,
                preview,
                principal,
                confirmation_strength="bound_token",
            ),
        )
        receipt = executor.execute_bound(binding, preview, authorization, args)
        assert receipt.status in {"applied", "already_satisfied"}, receipt

    _none, products = run_in_excision_owner(archive_root, execute, result_sink=result_sink)
    assert len(products) == 1
    return products[0]


def begin_excision_control(
    archive_root: Path,
    session_id: str,
    *,
    reason: str,
    actor: str = "user:local",
    cascade_lineage: bool = False,
) -> tuple[StartedBoundMutation, SessionExcisionArgs]:
    """Freeze and begin the actual audited plan without entering physical apply."""
    args = SessionExcisionArgs(archive_root, session_id, reason, actor, cascade_lineage)
    binding = runtime_operation_binding(SessionExcisionActuator())
    principal = MutationPrincipal(actor, frozenset({"archive.excise_session"}), "cli", "test-excision-preparation")
    with write_lease("test.excision-begin", archive_root=archive_root):
        executor = OperationExecutor.for_archive_root(archive_root)
        preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=archive_root)
        authorization = executor.authorize_bound(binding, preview, principal, confirmation_strength="bound_token")
        return executor.begin_bound(binding, preview, authorization, args), args


def recover_excision(archive_root: Path) -> None:
    """Run the real startup recovery owner without inventing another attempt.

    This resolves domain effects. It does not install a daemon result document.
    """
    import asyncio
    from math import inf

    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.daemon.http import _recover_startup_with_compute
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge

    async def run() -> None:
        coordinator = DaemonWriteCoordinator(archive_root=archive_root)
        kernel = BoundedComputeAdapter(max_workers=1, queue_units=1, queue_bytes=0)
        bridge = DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop())
        try:
            await _recover_startup_with_compute(bridge, kernel, archive_root)
        finally:
            kernel.shutdown(wait=True)
            assert await coordinator.shutdown(timeout=inf)

    asyncio.run(run())


def _required_audit(executor: Any) -> AuditRepository:
    """The fixture executor is archive-bound; its audit repository must exist."""
    audit = executor._audit
    assert isinstance(audit, AuditRepository)
    return audit

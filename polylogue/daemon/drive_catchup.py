"""Daemon admission for the existing Drive acquisition and parsing services."""

from __future__ import annotations

import asyncio
import builtins
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING, TypeVar

from polylogue.core import compute
from polylogue.core.compute import BoundedComputeAdapter, CancellationHandle, DaemonOperationCancelled
from polylogue.core.compute_cancel import compute_cancel
from polylogue.core.write_lease import adopt_write_lease, current_write_lease
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.logging import propagate

if TYPE_CHECKING:
    from polylogue.core.sql_settlement import SQLCustodyOwner

T = TypeVar("T")
P = TypeVar("P")


class DriveCatchupExecution:
    """Keep acquisition and completed parse preparation outside writer ownership."""

    def __init__(
        self,
        coordinator: DaemonWriteCoordinator,
        *,
        compute_adapter: BoundedComputeAdapter | None = None,
    ) -> None:
        self.coordinator = coordinator
        self._compute_adapter = compute_adapter or compute.compute_adapter()
        self._bridge = DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop())

    async def settle(
        self, pending: Awaitable[T], *, label: str = "settle", cancel_requested: Callable[[], None] | None = None
    ) -> T:
        """Wait for cancellation to settle before closing operation resources.

        The task is named because the supervisor is the daemon's sole *service*
        owner, and every bounded child beside it has to carry an identity an
        inventory can attribute. An anonymous ``Task-N`` is indistinguishable
        from an orphan.
        """
        task = asyncio.ensure_future(pending)
        if isinstance(task, asyncio.Task):
            # ``ensure_future`` only *creates* something for a coroutine. A
            # Future handed in (``loop.run_in_executor``) is already owned by
            # whoever scheduled it and carries no task identity to set.
            task.set_name(f"polylogue-drive-catchup:{label}")
        cancellation = None
        while not task.done():
            try:
                await asyncio.wait((task,))
            except asyncio.CancelledError as exc:
                if cancellation is None and cancel_requested is not None:
                    cancel_requested()
                cancellation = cancellation or exc
        try:
            result = task.result()
        except (asyncio.CancelledError, DaemonOperationCancelled):
            if cancellation is not None:
                raise cancellation from None
            raise
        except BaseException as failure:
            if cancellation is not None:
                raise builtins.BaseExceptionGroup(
                    "Drive cancellation and physical worker failure", [cancellation, failure]
                ) from None
            raise
        if cancellation is not None:
            raise cancellation
        return result

    async def prepare(self, operation: Callable[[], T]) -> T:
        if current_write_lease() is not None:
            raise RuntimeError("Drive preparation cannot run inside writer admission")
        cancellation = CancellationHandle()
        submitted = self._compute_adapter.submit(
            propagate(operation), admission_class="incremental-background", cancellation=cancellation
        )
        return await self.settle(
            asyncio.wrap_future(submitted.future), label="prepare", cancel_requested=cancellation.cancel
        )

    async def publish(self, actor: str, operation: Callable[[], Awaitable[T]]) -> T:
        return await self.settle(self.coordinator.run(f"maintenance.drive_catchup.{actor}", operation), label=actor)

    async def publish_sync(self, actor: str, operation: Callable[[], T]) -> T:
        if current_write_lease() is not None:
            return await self.coordinator.run_sync(f"maintenance.drive_catchup.{actor}", operation)
        return await self.settle(self.coordinator.run_sync(f"maintenance.drive_catchup.{actor}", operation))

    async def publish_prepared_sync(
        self,
        actor: str,
        prepare: Callable[[], P],
        operation: Callable[[P], T],
        *,
        estimated_bytes: int = 0,
    ) -> T:
        """Prepare off-gate and publish on that same managed compute worker.

        A reference seal owns live SQLite observers, so it cannot be moved
        between threads. This one compute unit retains those handles from its
        read-only census through bridge admission, writer settlement, and
        cleanup.
        """
        if current_write_lease() is not None:
            raise RuntimeError("prepared publication cannot begin inside writer admission")

        retained_prepared: list[SQLCustodyOwner] = []

        def prepare_and_publish() -> T:
            prepared = prepare()

            class PreparedSettlement:
                def close(self) -> None:
                    close = getattr(prepared, "close", None)
                    if callable(close):
                        close()
                    retained_prepared.clear()

            settlement = PreparedSettlement()
            retained_prepared.append(settlement)
            with (
                self._bridge.hold(f"maintenance.drive_catchup.{actor}") as delegation,
                adopt_write_lease(delegation),
            ):
                try:
                    result = operation(prepared)
                except BaseException as primary:
                    try:
                        settlement.close()
                    except BaseException as cleanup:
                        raise builtins.BaseExceptionGroup(
                            "Drive publication and prepared cleanup failed", [primary, cleanup]
                        ) from None
                    raise
                else:
                    # Terminal parent cleanup and creator-thread native drain
                    # retain this exact writer delegation through hold exit.
                    settlement.close()
                    return result

        def settlement_owners() -> tuple[SQLCustodyOwner, ...]:
            # The prepared carrier stays on this worker until close succeeds.
            # A successfully closed carrier disappears even when publication
            # raised; an unsettled one is kept for original-thread cleanup.
            return tuple(retained_prepared)

        def submit_worker(worker: Callable[[], None]) -> Future[None]:
            return self._compute_adapter.submit(
                propagate(worker),
                admission_class="incremental-background",
                estimated_bytes=estimated_bytes,
            ).future

        cancelled = threading.Event()
        cancellation_token = compute_cancel.set(cancelled)
        try:
            pending = asyncio.create_task(
                self.coordinator.run_prepared_sync(
                    f"maintenance.drive_catchup.{actor}",
                    prepare_and_publish,
                    submit_worker=submit_worker,
                    settlement_owners=settlement_owners,
                ),
                name=f"polylogue-drive-prepared:{actor}",
            )
        finally:
            compute_cancel.reset(cancellation_token)
        return await self.settle(pending, label=actor, cancel_requested=cancelled.set)

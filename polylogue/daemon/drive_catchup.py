"""Daemon admission for the existing Drive acquisition and parsing services."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from polylogue.core.write_lease import adopt_write_lease, current_write_lease
from polylogue.daemon.execution import BoundedComputeAdapter, daemon_compute_adapter
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.logging import propagate

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
        self._compute_adapter = compute_adapter or daemon_compute_adapter()
        self._bridge = DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop())

    async def settle(self, pending: Awaitable[T], *, label: str = "settle") -> T:
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
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()
            raise

    async def prepare(self, operation: Callable[[], T]) -> T:
        if current_write_lease() is not None:
            raise RuntimeError("Drive preparation cannot run inside writer admission")
        return await self.settle(asyncio.to_thread(operation), label="prepare")

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

        def prepare_and_publish() -> T:
            prepared = prepare()
            try:
                with (
                    self._bridge.hold(f"maintenance.drive_catchup.{actor}") as delegation,
                    adopt_write_lease(delegation),
                ):
                    return operation(prepared)
            finally:
                close = getattr(prepared, "close", None)
                if callable(close):
                    close()

        submitted = self._compute_adapter.submit(
            propagate(prepare_and_publish),
            admission_class="incremental-background",
            estimated_bytes=estimated_bytes,
        )
        pending = asyncio.wrap_future(submitted.future)
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            # Once admitted, the observer-owning callable must settle on its
            # worker before this owner unwinds or the bridge can be released.
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not pending.cancelled():
                pending.exception()
            raise

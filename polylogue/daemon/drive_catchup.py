"""Daemon admission for the existing Drive acquisition and parsing services."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from polylogue.core.write_lease import current_write_lease
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator

T = TypeVar("T")


class DriveCatchupExecution:
    """Keep acquisition and completed parse preparation outside writer ownership."""

    def __init__(self, coordinator: DaemonWriteCoordinator) -> None:
        self.coordinator = coordinator

    async def settle(self, pending: Awaitable[T]) -> T:
        """Wait for cancellation to settle before closing operation resources."""
        task = asyncio.ensure_future(pending)
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
        return await self.settle(asyncio.to_thread(operation))

    async def publish(self, actor: str, operation: Callable[[], Awaitable[T]]) -> T:
        return await self.settle(self.coordinator.run(f"maintenance.drive_catchup.{actor}", operation))

    async def publish_sync(self, actor: str, operation: Callable[[], T]) -> T:
        if current_write_lease() is not None:
            return await self.coordinator.run_sync(f"maintenance.drive_catchup.{actor}", operation)
        return await self.settle(self.coordinator.run_sync(f"maintenance.drive_catchup.{actor}", operation))

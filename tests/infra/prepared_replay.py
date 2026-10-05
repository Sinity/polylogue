"""Run storage-law fixtures on the canonical admitted preparation owner."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from polylogue.core.compute import BoundedComputeAdapter

T = TypeVar("T")


def run_on_convergence_owner(root: Path, actor: str, operation: Callable[[BoundedComputeAdapter], T]) -> T:
    """Run one synchronous law body on the real daemon preparation worker.

    Raw preparation refuses any thread other than its admitted compute
    creator, so a law body that computes, prepares or publishes Raw
    observations runs here with that owner's adapter and writer bridge.
    """
    from tests.infra.live_ingest import prepared_live_convergence_owner

    async def run() -> T:
        async with prepared_live_convergence_owner(root) as owner:
            return await owner.run_convergence_sync(actor, operation, owner._compute_adapter)

    return asyncio.run(run())

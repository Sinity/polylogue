"""Run startup operation recovery on its admitted owner, as the daemon does."""

from __future__ import annotations

import asyncio
from pathlib import Path

from polylogue.operations.mutation_replay import RECOVERY_SERVICE_ACTOR_REF, recover_interrupted_operations
from tests.infra.archive_templates import run_off_event_loop


def recover_on_admitted_owner(archive_root: Path) -> None:
    """Recover interrupted operations with the original input admission.

    Production recovery always runs inside an exclusive compute phase that
    charges the original rows it reads (``DaemonOperationRuntime`` and the
    HTTP server's startup). This runs the same function on the live
    convergence owner with its compute adapter's input demand; an async test
    calls it off its own event loop.
    """
    from tests.infra.live_ingest import prepared_live_convergence_owner

    async def run() -> None:
        async with prepared_live_convergence_owner(archive_root) as owner:
            await owner.run_convergence_sync(
                "fixture.operation-recovery",
                recover_interrupted_operations,
                archive_root,
                resolver_actor_ref=RECOVERY_SERVICE_ACTOR_REF,
                input_demand=owner._compute_adapter.amend_current_input_demand,
                startup=True,
            )

    run_off_event_loop(lambda: asyncio.run(run()))


__all__ = ["recover_on_admitted_owner"]

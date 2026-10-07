"""Fixture composition of the supplied acquisition and retained publication owners."""

from __future__ import annotations

import asyncio
import sys
import traceback
from builtins import BaseExceptionGroup
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from polylogue import Polylogue
from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.raw_failure_evidence import RetainedRawDecodeRefusalError
from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.sqlite_capture import LiveSQLiteCaptureStage
from polylogue.sources.revision_backfill import RetainedReplayOutcome
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.live_ingest import prepared_live_convergence_owner


@asynccontextmanager
async def prepared_live_batch_processor(
    root: Path,
    sources: Sequence[WatchSource],
    *,
    parser_fingerprint: str,
    failure_details: list[str] | None = None,
    converger: DaemonConverger | None = None,
    compute_adapter: BoundedComputeAdapter | None = None,
) -> AsyncIterator[LiveBatchProcessor]:
    """Keep one real kernel, coordinator, capture stage and Raw owner for the pass.

    ``converger`` is handed to the processor unchanged, for a law that also
    observes the per-file convergence stages after intake. A converger built
    from ``make_default_convergence_stages`` must share this pass's compute
    adapter, as the daemon's stages share its own: pass that adapter as
    ``compute_adapter``. The caller then owns its shutdown; otherwise the
    fixture creates and joins one.
    """
    owns_compute = compute_adapter is None
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1) if compute_adapter is None else compute_adapter
    coordinator = DaemonWriteCoordinator(archive_root=root)
    stage = LiveSQLiteCaptureStage(compute_adapter=compute)
    # Archive custody admits only an existing root directory.
    root.mkdir(parents=True, exist_ok=True)
    try:
        await coordinator.run_sync("fixture.live.bootstrap", lambda: bootstrap_archive_root(root))
        async with prepared_live_convergence_owner(
            root, compute_adapter=compute, write_coordinator=coordinator
        ) as owner:

            async def retained_runner(
                raw_ids: Sequence[str],
                *,
                on_terminal_refusal: Callable[[tuple[str, ...], RetainedRawDecodeRefusalError], None] | None = None,
            ) -> RetainedReplayOutcome:
                try:
                    outcome = await owner.ingest_retained_raw_ids(raw_ids, on_terminal_refusal=on_terminal_refusal)
                except BaseException as failure:
                    if failure_details is not None:
                        failure_details.append("".join(traceback.format_exception(failure)))
                    raise
                if failure_details is not None:
                    failure_details.extend(
                        "".join(traceback.format_exception(failure.error)) for failure in outcome.failures
                    )
                return outcome

            def open_cursor() -> CursorStore:
                return CursorStore(root / "index.db", ops_db_path=root / "ops.db")

            cursor = await coordinator.run_sync("fixture.live.cursor", open_cursor)
            archive = Polylogue(archive_root=root)
            try:
                yield LiveBatchProcessor(
                    archive,
                    sources,
                    cursor=cursor,
                    parser_fingerprint=parser_fingerprint,
                    sqlite_capture_stage=stage,
                    sync_runner=coordinator.run_sync,
                    append_runner=owner.ingest_append_plans,
                    retained_runner=retained_runner,
                    convergence_runner=owner.run_convergence_sync,
                    converger=converger,
                )
            finally:
                archive_primary = sys.exception()
                try:
                    await archive.close()
                except BaseException as cleanup:
                    if archive_primary is not None:
                        raise BaseExceptionGroup(
                            "live fixture body and archive close failed", [archive_primary, cleanup]
                        ) from archive_primary
                    raise
    finally:
        primary = sys.exception()
        failures: list[BaseException] = []
        coordinator_settled = False
        try:
            stage.shutdown()
        except BaseException as failure:
            failures.append(failure)
        try:
            coordinator_settled = await coordinator.shutdown(timeout=float("inf"))
            if not coordinator_settled:
                raise RuntimeError("live fixture coordinator did not physically settle")
        except BaseException as failure:
            failures.append(failure)
        try:
            if owns_compute:
                closing = asyncio.create_task(asyncio.to_thread(compute.shutdown, wait=coordinator_settled))
                while not closing.done():
                    try:
                        await asyncio.shield(closing)
                    except asyncio.CancelledError as failure:
                        failures.append(failure)
                closing.result()
        except BaseException as failure:
            failures.append(failure)
        if failures:
            if primary is not None:
                failures.insert(0, primary)
            raise BaseExceptionGroup("live fixture physical settlement failed", failures) from primary

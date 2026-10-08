"""The periodic daemon owner derives missing frontier work from durable coverage."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path
from types import MethodType
from typing import Any, cast

import pytest

from polylogue.core import compute
from polylogue.core.compute import BoundedComputeAdapter, DaemonBackpressureError
from polylogue.daemon import cli as daemon_cli
from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
from polylogue.operations import raw_frontier_inspection
from polylogue.sources.live import WatchSource
from polylogue.sources.source_layout import export_drop_layout
from polylogue.storage.frontier_inspection import (
    inspect_prepared_raw_authority_frontier,
    read_frontier_coverage_for_archive,
)
from tests.infra.live_batch import prepared_live_batch_processor


@pytest.mark.asyncio
@pytest.mark.uses_real_clock
@pytest.mark.parametrize("prior_inspection", [True, False])
async def test_periodic_owner_inspects_after_full_convergence_admission_rejected(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    bounded_compute_adapter: BoundedComputeAdapter,
    prior_inspection: bool,
) -> None:
    """Dropping the periodic coverage check leaves authority unknown forever.

    Saturation is a real exclusive compute reservation, not a fabricated stage
    failure. Intake publishes first; its unchanged retry no longer has session
    changes to schedule full convergence. The ordinary periodic owner must
    inspect even though no failed stage ever entered the debt ledger.
    """
    root = workspace_env["archive_root"]
    source_root = workspace_env["data_root"] / "projects" / "neutral"
    source_root.mkdir(parents=True)
    source_path = source_root / "session.jsonl"
    source_path.write_bytes(
        (Path(__file__).parents[2] / "fixtures/origin-capability/claude-code-session.jsonl").read_bytes()
    )
    adapter = bounded_compute_adapter
    stage = raw_frontier_inspection.make_raw_frontier_inspection_stage(root / "index.db", compute_adapter=adapter)
    inspections = 0
    original_inspection = inspect_prepared_raw_authority_frontier

    def observe_inspection(*args: Any, **kwargs: Any) -> Any:
        nonlocal inspections
        inspections += 1
        return original_inspection(*args, **kwargs)

    monkeypatch.setattr(raw_frontier_inspection, "inspect_prepared_raw_authority_frontier", observe_inspection)
    async with prepared_live_batch_processor(
        root,
        (WatchSource(name="test", root=source_root, layout=export_drop_layout((".jsonl",))),),
        parser_fingerprint="frontier-wakeup",
        converger=DaemonConverger((stage,)),
        compute_adapter=adapter,
    ) as processor:
        original_runner = processor._convergence_runner
        assert original_runner is not None
        assert isinstance(original_runner, MethodType)
        owner = cast(RawObservationConvergenceOwner, original_runner.__self__)
        if not prior_inspection:
            # Model fresh ops with no census, rather than the reusable fixture's
            # already-inspected empty archive. The real writer owns this setup.
            def remove_seed_mark() -> None:
                with sqlite3.connect(root / "ops.db") as conn:
                    conn.execute("DELETE FROM raw_frontier_inspection")

            await owner._write_coordinator.run_sync("fixture.frontier.uninspected", remove_seed_mark)

        async def saturated_runner(*args: Any, **kwargs: Any) -> Any:
            release = threading.Event()
            blocker = adapter.submit(release.wait, exclusive_bytes=True, estimated_bytes=0)
            try:
                return await original_runner(*args, **kwargs)
            finally:
                release.set()
                await asyncio.wrap_future(blocker.future)

        processor._convergence_runner = saturated_runner
        with pytest.raises(DaemonBackpressureError):
            await processor.ingest_files([source_path], emit_event=False)
        processor._convergence_runner = original_runner
        with sqlite3.connect(root / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        retry = await processor.ingest_files([source_path], emit_event=False)
        assert retry.failed_file_count == 0
        assert inspections == 0
        with sqlite3.connect(root / "ops.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM convergence_debt").fetchone()[0] == 0
        assert not read_frontier_coverage_for_archive(root)["current"]

        monkeypatch.setattr(compute, "compute_adapter", lambda: adapter)
        monkeypatch.setattr(daemon_cli, "daemon_write_coordinator", lambda: owner._write_coordinator)
        assert await daemon_cli._retry_convergence_debt_once(root / "index.db") is None
        coverage = read_frontier_coverage_for_archive(root)
        assert coverage["current"] and coverage["healthy"], coverage
        assert inspections == 1
        # With unchanged authority, another ordinary tick does not census again.
        assert await daemon_cli._retry_convergence_debt_once(root / "index.db") is None
        assert inspections == 1

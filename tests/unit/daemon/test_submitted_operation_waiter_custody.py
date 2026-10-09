"""Admitted compute work remains owned through daemon cancellation."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path

import pytest

from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.daemon.services import ServiceProfile
from polylogue.daemon.supervisor import DaemonSupervisor
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge


@pytest.mark.contract
@pytest.mark.uses_real_clock("compute worker and supervisor cancellation require OS scheduling")
@pytest.mark.asyncio
async def test_shutdown_retains_admitted_writer_until_compute_physically_settles(tmp_path: Path) -> None:
    """Cancellation must reach physical compute before shutdown retires its child."""
    database = tmp_path / "candidate.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE published (value TEXT NOT NULL)")

    loop = asyncio.get_running_loop()
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    bridge = DaemonWriteThreadBridge(coordinator, loop, timeout=2.0)
    entered = threading.Event()
    release = threading.Event()
    wrote: list[bool] = []

    def physical_operation() -> None:
        entered.set()
        assert release.wait(5.0)
        check_compute_cancelled()

        def publish() -> None:
            with sqlite3.connect(database) as connection:
                connection.execute("INSERT INTO published VALUES ('late')")
            wrote.append(True)

        bridge.run_sync("candidate.publish", publish)

    submitted = compute.submit(physical_operation, admission_class="incremental-background")

    async def service() -> None:
        await submitted.wait()

    supervisor = DaemonSupervisor(profile=ServiceProfile.INTAKE)
    child = supervisor.start("fair_intake", service)
    assert child is not None
    shutdown: asyncio.Task[object] | None = None
    try:
        assert await asyncio.to_thread(entered.wait, 5.0)
        shutdown = asyncio.create_task(supervisor.shutdown())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        child.cancel()
        await asyncio.sleep(0)

        assert not child.done()
        assert not shutdown.done()
        assert compute.snapshot().active_units == 1
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT COUNT(*) FROM published").fetchone() == (0,)

        release.set()
        report = await asyncio.wait_for(shutdown, timeout=5.0)
        assert report.orphaned == ()
        assert child.done()
        assert wrote == []
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT COUNT(*) FROM published").fetchone() == (0,)
        assert compute.snapshot().active_units == 0
    finally:
        release.set()
        await asyncio.gather(*(task for task in (shutdown, child) if task is not None), return_exceptions=True)
        assert await asyncio.to_thread(compute.close, join_timeout_s=5.0) == ()
        await coordinator.shutdown(timeout=2.0)


def test_async_submitted_operation_waiters_use_physical_settlement_contract() -> None:
    """The daemon's async compute owners must not wrap raw submitted futures."""
    from polylogue.daemon import cli, drive_catchup, operation_runtime

    for module in (cli, drive_catchup, operation_runtime):
        assert module.__file__ is not None
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "wrap_future(submitted.future)" not in source
    assert "return await submitted.wait()" in Path(cli.__file__).read_text(encoding="utf-8")
    assert 'submitted.wait(), label="prepare"' in Path(drive_catchup.__file__).read_text(encoding="utf-8")
    assert "return await submitted.wait()" in Path(operation_runtime.__file__).read_text(encoding="utf-8")

"""Identify physical shutdown ownership on the original small material route."""

from __future__ import annotations

import asyncio
import faulthandler
import json
import sqlite3
import sys
import threading
import traceback
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
from polylogue.sources import codex_state_evidence
from polylogue.storage.materials import MaterialSourceProducer, PreparedMaterial
from tests.infra.daemon_service_harness import record_private_lifecycle_probe
from tests.unit.sources.test_codex_state_bounds_and_excision import _materialize, _write_goals_db


@pytest.mark.timeout(120)
@pytest.mark.parametrize("exit_kind", ["success", "failure", "cancel"])
def test_small_material_owner_physically_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_kind: str
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    state = tmp_path / "goals_1.sqlite"
    _write_goals_db(state, [("thread-small", "goal-small", "objective")])
    observation = tmp_path / "material-shutdown-owner.jsonl"
    guard = threading.Lock()

    def record(phase: str, owner: object) -> None:
        frames = sys._current_frames()
        row: dict[str, object] = {
            "phase": phase,
            "owner_type": type(owner).__name__,
            "owner_id": id(owner),
            "thread_id": threading.get_ident(),
            "active_exception": traceback.format_exception(sys.exception()) if sys.exception() is not None else [],
            "threads": [
                {
                    "name": thread.name,
                    "ident": thread.ident,
                    "alive": thread.is_alive(),
                    "stack": traceback.format_stack(frames[thread.ident]) if thread.ident in frames else [],
                }
                for thread in threading.enumerate()
            ],
        }
        if isinstance(owner, BoundedComputeAdapter):
            row["executor_id"] = id(owner.executor)
            row["executor_shutdown"] = owner.executor._shutdown
            row["executor_queue_id"] = id(owner.executor._work_queue)
            row["executor_queue_size"] = owner.executor._work_queue.qsize()
            row["compute_snapshot"] = repr(owner.snapshot())
        if isinstance(owner, DaemonWriteCoordinator):
            row["terminal_workers"] = [
                {
                    "id": id(worker),
                    "retired": worker.retired,
                    "settling": worker.settling,
                    "mailbox_id": id(worker._requests),
                    "mailbox_size": worker._requests.qsize(),
                }
                for worker in owner._retained_workers()
            ]
        record_private_lifecycle_probe(f"material-{exit_kind}-{phase}", row)
        with guard, observation.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()

    original_coordinator_shutdown = DaemonWriteCoordinator.shutdown
    original_compute_shutdown = BoundedComputeAdapter.shutdown

    async def coordinator_shutdown(self: DaemonWriteCoordinator, *, timeout: float) -> bool:
        record("coordinator.before", self)
        try:
            return await original_coordinator_shutdown(self, timeout=timeout)
        finally:
            record("coordinator.after", self)

    def compute_shutdown(self: BoundedComputeAdapter, *, wait: bool = False, cancel_futures: bool = True) -> None:
        record("compute.before", self)
        try:
            original_compute_shutdown(self, wait=wait, cancel_futures=cancel_futures)
        finally:
            record("compute.after", self)

    monkeypatch.setattr(DaemonWriteCoordinator, "shutdown", coordinator_shutdown)
    monkeypatch.setattr(BoundedComputeAdapter, "shutdown", compute_shutdown)
    with (tmp_path / "material-shutdown-stacks.txt").open("w") as stacks:
        faulthandler.dump_traceback_later(45, file=stacks)
        try:
            if exit_kind == "success":
                receipt = _materialize(root, state)
                assert receipt is not None and receipt.rows_materialized == 1
            else:
                original_upsert = codex_state_evidence._upsert_codex_material_source
                entered = threading.Event()
                release = threading.Event()

                def upsert(
                    producer: MaterialSourceProducer, *, raw_id: str, prepared: PreparedMaterial, observed_at_ms: int
                ) -> None:
                    original_upsert(producer, raw_id=raw_id, prepared=prepared, observed_at_ms=observed_at_ms)
                    if exit_kind == "failure":
                        raise RuntimeError("neutral material producer failure")
                    entered.set()
                    release.wait()
                    check_compute_cancelled()

                monkeypatch.setattr(codex_state_evidence, "_upsert_codex_material_source", upsert)
                if exit_kind == "failure":
                    with pytest.raises(BaseException) as failure:
                        _materialize(root, state)
                    assert "neutral material producer failure" in "".join(traceback.format_exception(failure.value))
                else:

                    async def cancel_material() -> None:
                        from tests.unit.sources.test_codex_state_bounds_and_excision import _materialize_async

                        task = asyncio.create_task(_materialize_async(root, state))
                        started = asyncio.create_task(asyncio.to_thread(entered.wait))
                        try:
                            done, _ = await asyncio.wait((task, started), return_when=asyncio.FIRST_COMPLETED)
                            if task in done:
                                task.result()
                                raise AssertionError("material completed before cancellation rendezvous")
                            task.cancel()
                        finally:
                            release.set()
                            entered.set()
                            await started
                        with pytest.raises(asyncio.CancelledError):
                            await task

                    asyncio.run(cancel_material())
        finally:
            faulthandler.cancel_dump_traceback_later()
    if exit_kind != "success":
        with closing(sqlite3.connect(f"file:{root / 'source.db'}?mode=ro", uri=True)) as connection:
            assert connection.execute("SELECT COUNT(*) FROM material_observations").fetchone()[0] == 0
    phases = [json.loads(line)["phase"] for line in observation.read_text().splitlines()]
    assert phases == ["coordinator.before", "coordinator.after", "compute.before", "compute.after"]

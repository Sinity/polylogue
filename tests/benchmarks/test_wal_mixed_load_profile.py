"""Managed mixed-load profile for WAL checkpoint and read-frame calibration.

Runs interactive reads, incremental ingest and inactive-candidate construction
against one archive tier concurrently, then the recurring checkpoint, and
records the declared metric set. It is the calibration route for the numeric
policy in ``storage/sqlite/connection_profile.py``: WAL thresholds, the
checkpoint hold budget, and the maximum read-frame age.

The workload is synthetic and self-contained. Archive-scale trajectory and
restart evidence are a separate, larger measurement (polylogue-74kj3); this
profile is what makes that measurement reproducible rather than ad hoc.
"""

from __future__ import annotations

import asyncio
import resource
import sqlite3
import threading
from pathlib import Path
from time import perf_counter
from typing import cast

import pytest

from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
from polylogue.storage.sqlite.connection_profile import (
    CHECKPOINT_HOLD_BUDGET_S,
    arm_recurring_checkpoint_owner,
    read_frame,
)
from polylogue.storage.sqlite.wal_checkpoint import WalCheckpointObservation, checkpoint_wal
from tests.benchmarks.helpers import BenchmarkFixture, benchmark_one_shot
from tests.benchmarks.wal_mixed_load_profile import (
    MIXED_LOAD_WORKLOADS,
    PROFILE_METRICS,
    profile_manifest,
    record_metrics,
)

pytestmark = pytest.mark.uses_real_clock(
    "The profile measures wall-clock writer hold, checkpoint elapsed time and read-frame age."
)

_INGEST_BATCHES = 8
_ROWS_PER_BATCH = 96
_READ_FRAMES = 4
_BODY = b"m" * 4096


def test_mixed_load_profile_declares_every_workload_and_metric() -> None:
    manifest = profile_manifest()
    workloads = cast(list[dict[str, object]], manifest["workloads"])
    assert [workload["name"] for workload in workloads] == [w.name for w in MIXED_LOAD_WORKLOADS]
    assert manifest["metrics"] == list(PROFILE_METRICS)


def _seed(db: Path) -> None:
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE payload (id INTEGER PRIMARY KEY, body BLOB NOT NULL)")
        conn.executemany("INSERT INTO payload (body) VALUES (?)", [(_BODY,) for _ in range(_ROWS_PER_BATCH)])
        conn.commit()
    finally:
        conn.close()


def _wal_bytes(db: Path) -> int:
    wal = db.with_suffix(".db-wal")
    return wal.stat().st_size if wal.exists() else 0


class _MixedLoad:
    """One run of the declared workload set against one tier."""

    def __init__(self, db: Path) -> None:
        self.db = db
        self.wal_peak = 0
        self.publication_hold_s = 0.0
        self.checkpoint_hold_s = 0.0
        self.rebinds = 0
        self.max_frame_age_s = 0.0
        self.observation: WalCheckpointObservation | None = None
        self.over_budget_holds = 0

    def _publish_batch(self) -> None:
        conn = sqlite3.connect(self.db)
        try:
            conn.execute("PRAGMA wal_autocheckpoint = 0")
            conn.executemany("INSERT INTO payload (body) VALUES (?)", [(_BODY,) for _ in range(_ROWS_PER_BATCH)])
            conn.commit()
        finally:
            conn.close()
        self.wal_peak = max(self.wal_peak, _wal_bytes(self.db))

    def _build_candidate(self, destination: Path) -> None:
        """An owned inactive generation built while the live tier is written."""
        conn = sqlite3.connect(destination)
        try:
            conn.execute("PRAGMA journal_mode=MEMORY")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("CREATE TABLE candidate (id INTEGER PRIMARY KEY, body BLOB NOT NULL)")
            source = read_frame(self.db, timeout_class="offline-bulk")
            try:
                rows = source.connection.execute("SELECT id, body FROM payload").fetchall()
            finally:
                source.close()
            conn.executemany("INSERT INTO candidate (id, body) VALUES (?, ?)", [tuple(row) for row in rows])
            conn.commit()
        finally:
            conn.close()

    def _interactive_reads(self, stop: threading.Event) -> None:
        """What a bounded interactive reader actually does: read, then rebind.

        A frame is rebound both when the generation it read moved and when it
        reaches its declared age, so no reader in this profile pins WAL frames
        for longer than the policy allows.
        """
        frames = [read_frame(self.db, timeout_class="interactive-read") for _ in range(_READ_FRAMES)]
        try:
            while not stop.wait(0.01):
                for frame in frames:
                    self.max_frame_age_s = max(self.max_frame_age_s, frame.age_s)
                    if frame.expired or not frame.revalidate():
                        frame.rebind()
                        self.rebinds += 1
                    frame.connection.execute("SELECT count(*) FROM payload").fetchone()
        finally:
            for frame in frames:
                frame.close()

    async def _run(self, tmp_path: Path) -> None:
        coordinator = DaemonWriteCoordinator()
        stop = threading.Event()
        reader = threading.Thread(target=self._interactive_reads, args=(stop,), daemon=True)
        reader.start()
        try:
            for _ in range(_INGEST_BATCHES):
                started = perf_counter()
                await coordinator.run_sync("watcher.live_ingest", self._publish_batch)
                self.publication_hold_s += perf_counter() - started
            started = perf_counter()
            await coordinator.run_sync("maintenance.candidate", self._build_candidate, tmp_path / "candidate.db")
            self.publication_hold_s += perf_counter() - started

            started = perf_counter()
            self.observation = await coordinator.run_sync(
                "maintenance.wal_checkpoint",
                checkpoint_wal,
                self.db,
                reason="mixed-load",
                escalation="recurring",
                warn_bytes=1,
                collect_blockers=True,
            )
            self.checkpoint_hold_s = perf_counter() - started
        finally:
            stop.set()
            reader.join(timeout=10.0)
        self.over_budget_holds = coordinator.snapshot().over_budget_holds


@pytest.mark.benchmark
def test_bench_wal_mixed_load_profile(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    """Interactive reads, incremental ingest and candidate construction, then one checkpoint."""
    db = tmp_path / "index.db"
    _seed(db)
    before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    wal_start = _wal_bytes(db)

    def run() -> _MixedLoad:
        load = _MixedLoad(db)
        with arm_recurring_checkpoint_owner():
            asyncio.run(load._run(tmp_path))
        return load

    load = benchmark_one_shot(benchmark, run)
    observation = load.observation
    assert observation is not None

    record_metrics(
        benchmark,
        wal_bytes_start=wal_start,
        wal_bytes_peak=load.wal_peak,
        wal_bytes_end=_wal_bytes(db),
        checkpoint_mode=observation.mode,
        checkpoint_escalation=observation.escalation,
        checkpoint_log_pages=observation.log_pages,
        checkpointed_pages=observation.checkpointed_pages,
        checkpoint_busy_pages=observation.busy_pages,
        checkpoint_elapsed_ms=round(observation.elapsed_s * 1000, 3),
        checkpoint_blockers=len(observation.blocking_processes),
        checkpoint_writer_hold_ms=round(load.checkpoint_hold_s * 1000, 3),
        publication_writer_hold_ms=round(load.publication_hold_s * 1000, 3),
        checkpoint_hold_budget_ms=round(CHECKPOINT_HOLD_BUDGET_S * 1000, 3),
        over_budget_holds=load.over_budget_holds,
        read_connections=_READ_FRAMES,
        read_frame_rebinds=load.rebinds,
        max_read_frame_age_s=round(load.max_frame_age_s, 6),
        peak_rss_kib=max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before_rss),
    )

    # The recurring owner never escalates past PASSIVE, and the checkpoint's
    # hold is measured on its own rather than inside the publication total.
    assert observation.escalation == "recurring"
    assert observation.mode in {"none", "passive"}
    assert load.checkpoint_hold_s < CHECKPOINT_HOLD_BUDGET_S
    # A writer made progress while the interactive frames were open: the reads
    # bound their own lifetime instead of pinning the WAL for the whole run.
    survivor = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = survivor.execute("SELECT count(*) FROM payload").fetchone()[0]
    finally:
        survivor.close()
    assert rows == _ROWS_PER_BATCH * (_INGEST_BATCHES + 1)

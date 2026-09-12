"""Per-run archive-construction and write-cost receipt for pytest sessions.

The suite's dominant cost is not test logic: a six-tier archive root is
materialised thousands of times, and each one writes megabytes. Neither the
pytest report nor the run receipt records that, so a change meant to reduce it
cannot be shown to have worked. This plugin records the two numbers that
decide it -- how each archive-tier initialization resolved (page-copy
prototype, fresh DDL, or full DDL reapply) and the process's write bytes --
per xdist worker, and :func:`aggregate_suite_cost` sums them into one receipt.

Inert unless ``POLYLOGUE_SUITE_COST_DIR`` names a directory.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import pytest

SUITE_COST_DIR_ENV: Final = "POLYLOGUE_SUITE_COST_DIR"
PLUGIN_NAME: Final = "devtools.pytest_suite_cost_plugin"

#: Name of the run-level receipt inside the receipt directory. Excluded from
#: the worker glob so a re-aggregation never counts a previous run's total.
RUN_RECEIPT_NAME: Final = "run.json"

#: ``/proc/self/io`` counters worth carrying. ``write_bytes`` is the one the
#: storage budget is stated in: bytes this process caused to be sent to the
#: block layer, which page-cache-only churn does not inflate.
_IO_FIELDS: Final = ("rchar", "wchar", "syscr", "syscw", "read_bytes", "write_bytes")

#: Walking the temp tree is O(files); at this cadence a 20k-test worker pays
#: it a few dozen times, which is noise against the run.
_SAMPLE_EVERY: Final = 250


def _read_io() -> dict[str, int]:
    counters: dict[str, int] = {}
    with contextlib.suppress(OSError, ValueError):
        for line in Path("/proc/self/io").read_text().splitlines():
            name, _, value = line.partition(": ")
            if name in _IO_FIELDS:
                counters[name] = int(value)
    return counters


def _tree_bytes(root: Path) -> tuple[int, int]:
    """Apparent and allocated bytes under ``root``; missing entries are skipped."""
    apparent = 0
    allocated = 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as scan:
                entries = list(scan)
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    continue
                status = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            apparent += status.st_size
            allocated += status.st_blocks * 512
    return apparent, allocated


class SuiteCostRecorder:
    """Accumulate one worker's archive-construction and write-cost facts."""

    def __init__(
        self,
        directory: Path,
        worker_id: str,
        basetemp: Path | Callable[[], Path | None] | None,
        *,
        role: str = "worker",
    ) -> None:
        self._directory = directory
        self._worker_id = worker_id
        self._role = role
        self._basetemp_source = basetemp
        self._started_at = time.monotonic()
        self._io_start = _read_io()
        self._tests = 0
        self._peak_apparent = 0
        self._peak_allocated = 0

    def _basetemp(self) -> Path | None:
        """Resolve the scratch root late: the temp-path plugin configures after this one."""
        source = self._basetemp_source
        if source is None or isinstance(source, Path):
            return source
        with contextlib.suppress(Exception):
            resolved = source()
            if resolved is not None:
                self._basetemp_source = Path(resolved)
                return self._basetemp_source
        return None

    def note_test(self) -> None:
        self._tests += 1
        if self._tests % _SAMPLE_EVERY == 0:
            self.sample_storage()

    def sample_storage(self) -> None:
        basetemp = self._basetemp()
        if basetemp is None:
            return
        apparent, allocated = _tree_bytes(basetemp)
        self._peak_apparent = max(self._peak_apparent, apparent)
        self._peak_allocated = max(self._peak_allocated, allocated)

    def payload(self) -> dict[str, Any]:
        from polylogue.storage.sqlite.archive_tiers.bootstrap import archive_tier_init_counts

        io_end = _read_io()
        io_delta = {name: io_end[name] - self._io_start[name] for name in io_end if name in self._io_start}
        return {
            "worker_id": self._worker_id,
            "role": self._role,
            "tests": self._tests,
            "duration_s": round(time.monotonic() - self._started_at, 3),
            "io": io_delta,
            "tier_init": archive_tier_init_counts(),
            "peak_scratch_apparent_bytes": self._peak_apparent,
            "peak_scratch_allocated_bytes": self._peak_allocated,
        }

    def write(self) -> Path:
        self.sample_storage()
        self._directory.mkdir(parents=True, exist_ok=True)
        destination = self._directory / f"{self._worker_id}.json"
        destination.write_text(json.dumps(self.payload(), indent=2, sort_keys=True) + "\n")
        return destination


_RECORDER: SuiteCostRecorder | None = None


def pytest_configure(config: pytest.Config) -> None:
    global _RECORDER
    directory = os.environ.get(SUITE_COST_DIR_ENV, "").strip()
    if not directory:
        return
    worker_id = getattr(config, "workerinput", {}).get("workerid", "master")
    # The controller's collection and worker warm-up are part of the run's
    # elapsed time. Record it separately rather than pretending its duration
    # is another worker's active time.
    role = "controller" if worker_id == "master" and getattr(config.option, "numprocesses", 0) else "worker"

    def resolve_basetemp() -> Path | None:
        factory = getattr(config, "_tmp_path_factory", None)
        return None if factory is None else Path(factory.getbasetemp())

    _RECORDER = SuiteCostRecorder(Path(directory), worker_id, resolve_basetemp, role=role)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if _RECORDER is not None and report.when == "call":
        _RECORDER.note_test()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del session, exitstatus
    global _RECORDER
    if _RECORDER is None:
        return
    with contextlib.suppress(OSError):
        _RECORDER.write()
    _RECORDER = None


def aggregate_suite_cost(directory: Path) -> dict[str, Any]:
    """Aggregate run cost without inventing overlap that was not measured."""
    records: list[dict[str, Any]] = []
    for path in sorted(Path(directory).glob("*.json")):
        if path.name == RUN_RECEIPT_NAME:
            continue
        with contextlib.suppress(OSError, ValueError):
            records.append(json.loads(path.read_text()))
    controllers = [record for record in records if record.get("role") == "controller"]
    workers = [record for record in records if record.get("role") != "controller"]
    tests = sum(int(worker.get("tests", 0)) for worker in workers)
    io_total: dict[str, int] = {}
    tier_total: dict[str, int] = {}
    for worker in records:
        for name, value in dict(worker.get("io", {})).items():
            io_total[name] = io_total.get(name, 0) + int(value)
        for name, value in dict(worker.get("tier_init", {})).items():
            tier_total[name] = tier_total.get(name, 0) + int(value)
    write_bytes = io_total.get("write_bytes", 0)
    return {
        "workers": len(workers),
        "tests": tests,
        # The controller spans collection, worker warmup and shutdown. Where
        # it is available that is the run elapsed time; old receipts fall back
        # to the longest worker. Never sum concurrent durations.
        "wall_clock_s": max(
            (float(controller.get("duration_s", 0.0)) for controller in controllers),
            default=max((float(worker.get("duration_s", 0.0)) for worker in workers), default=0.0),
        ),
        "controller_duration_s": max(
            (float(controller.get("duration_s", 0.0)) for controller in controllers), default=None
        ),
        "worker_active_s": sum(float(worker.get("duration_s", 0.0)) for worker in workers),
        "io": io_total,
        "write_bytes_per_test": round(write_bytes / tests, 1) if tests else 0.0,
        "tier_init": dict(sorted(tier_total.items())),
        "archive_tier_initializations": sum(tier_total.values()),
        # Per-process peaks have no shared sampling clock. A sum would claim a
        # simultaneous suite peak we did not observe, so expose the largest
        # worker peak and retain every individual measurement below.
        "peak_scratch_apparent_bytes": max((int(w.get("peak_scratch_apparent_bytes", 0)) for w in workers), default=0),
        "peak_scratch_allocated_bytes": max(
            (int(w.get("peak_scratch_allocated_bytes", 0)) for w in workers), default=0
        ),
        "per_worker": workers,
        "controller": controllers,
    }


def write_run_receipt(directory: Path | str | None = None) -> Path | None:
    """Sum this run's worker receipts into ``run.json``; ``None`` when unconfigured.

    Called after a managed pytest step so the per-worker files a run leaves
    behind become one comparable number without a second tool.
    """
    target = directory if directory is not None else os.environ.get(SUITE_COST_DIR_ENV, "").strip()
    if not target:
        return None
    root = Path(target)
    aggregate = aggregate_suite_cost(root)
    if not aggregate["workers"]:
        return None
    destination = root / RUN_RECEIPT_NAME
    with contextlib.suppress(OSError):
        destination.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
        return destination
    return None


__all__ = [
    "PLUGIN_NAME",
    "RUN_RECEIPT_NAME",
    "SUITE_COST_DIR_ENV",
    "SuiteCostRecorder",
    "aggregate_suite_cost",
    "write_run_receipt",
]

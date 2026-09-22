"""Per-run archive-construction and write-cost receipt for pytest sessions.

The suite's dominant cost is not test logic: a six-tier archive root is
materialised thousands of times, and each one writes megabytes. Neither the
pytest report nor the run receipt records that, so a change meant to reduce it
cannot be shown to have worked. This plugin records the two numbers that
decide it -- how each archive-tier initialization resolved (page-copy
prototype, fresh DDL, or full DDL reapply) and the process's write bytes --
per xdist worker, and :func:`aggregate_suite_cost` sums them into one receipt.

A worker's resident peak is the other half of that cost, and the half that
caps parallelism: the corpus width is memory-bound, so a worker's peak decides
how many workers fit the pytest slice. ``POLYLOGUE_SUITE_COST_RSS`` adds an
O(1) resident-memory trajectory labelled with the nodeid at each sample, which
attributes growth to a directory instead of to the run as a whole.

Inert unless ``POLYLOGUE_SUITE_COST_DIR`` names a directory.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
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

#: Opt-in for the scratch-tree peak. The tier-init tally and ``/proc/self/io``
#: are O(1) reads and are always recorded; walking the basetemp is not. pytest
#: keeps a directory per test under it, so the tree grows with the run and a
#: fixed-cadence walk costs O(tests^2/cadence) -- measured at ~2.5us per entry,
#: a 20k-test worker would spend minutes measuring itself. The peak is a
#: diagnostic for a scratch-size investigation, so it is asked for explicitly.
SUITE_COST_SCRATCH_ENV: Final = "POLYLOGUE_SUITE_COST_SCRATCH"

#: Walking the temp tree is O(files); at this cadence a 20k-test worker pays
#: it a few dozen times, which is noise against the run.
_SAMPLE_EVERY: Final = 250

#: Opt-in for the resident-memory trajectory. ``/proc/self/statm`` is an O(1)
#: read of a handful of integers -- the same cost class as ``/proc/self/io`` --
#: but the trajectory it builds is a diagnostic for one question: a pytest
#: worker's peak grows with tests *executed*, and neither the import floor nor
#: the tier-init tally locates that growth. Sampling resident pages beside the
#: nodeid that had just run attributes the growth to a directory, which is what
#: turns "the worker peaks at 2.2 GiB" into a fixable place.
#:
#: Opt-in rather than always-on because it is an investigation aid, not a
#: budget number: the receipt keys are absent on an unsampled run so a run that
#: was never asked cannot read as a measured flat trajectory.
SUITE_COST_RSS_ENV: Final = "POLYLOGUE_SUITE_COST_RSS"

#: Opt-in for the heap-retention census, which answers the question the RSS
#: trajectory cannot: a worker's ~4 GiB plateau is known to be *bounded*
#: retention rather than an unbounded leak, and nothing has named what holds
#: it. At each trajectory point this forces a full collection and records
#: resident pages on both sides of it, plus a census of live objects by type.
#:
#: The pair of readings is the decisive one. If resident pages fall across the
#: collection the plateau is uncollected cycles and the remedy is a collection
#: policy; if they do not, something reachable holds it and
#: ``heap_top`` names the type. Neither reading alone distinguishes those.
#:
#: Costly by construction -- a full ``gc.collect()`` and one pass over
#: ``gc.get_objects()`` -- so it is a separate opt-in from the O(1)
#: trajectory above and never runs on an ordinary managed run.
SUITE_COST_HEAP_ENV: Final = "POLYLOGUE_SUITE_COST_HEAP"

#: Types kept per census point. The tail of a type histogram is thousands of
#: singletons; the retention question is answered by the head.
_HEAP_TYPE_LIMIT: Final = 30

#: Resident pages and the process page size, for converting ``statm`` to KiB.
_STATM_PATH: Final = Path("/proc/self/statm")
_PAGE_KIB: Final = os.sysconf("SC_PAGE_SIZE") // 1024

#: Trajectory points kept. At ``_SAMPLE_EVERY`` a 20k-test worker produces 80
#: points, so this bounds a pathological run rather than a realistic one: the
#: list must not itself become the growth it is measuring.
_RSS_SAMPLE_LIMIT: Final = 512


def _heap_census(*, limit: int = _HEAP_TYPE_LIMIT) -> dict[str, Any]:
    """Live objects by type after a forced full collection.

    ``gc.collect()`` first, so what is counted is what something still
    reaches. Counting before it would report garbage the allocator has not
    yet handed back as retention, which is the confusion this census exists
    to settle.
    """
    collected = gc.collect()
    objects = gc.get_objects()
    counts: Counter[str] = Counter()
    for item in objects:
        kind = type(item)
        counts[f"{kind.__module__}.{kind.__qualname__}"] += 1
    total = len(objects)
    del objects
    return {"collected": collected, "objects": total, "top": dict(counts.most_common(limit))}


def _read_rss_kib() -> int | None:
    """Resident set size in KiB, or None when ``statm`` is unreadable.

    One read of one short procfs line: O(1) in both the run length and the
    scratch tree, unlike the basetemp walk above.
    """
    with contextlib.suppress(OSError, ValueError, IndexError):
        return int(_STATM_PATH.read_text().split()[1]) * _PAGE_KIB
    return None


#: Hard stop for one walk, so even an opted-in sample cannot become the run's
#: dominant cost. A truncated sample is reported as truncated, never as a peak.
_TREE_ENTRY_BUDGET: Final = 200_000


def _read_io() -> dict[str, int]:
    counters: dict[str, int] = {}
    with contextlib.suppress(OSError, ValueError):
        for line in Path("/proc/self/io").read_text().splitlines():
            name, _, value = line.partition(": ")
            if name in _IO_FIELDS:
                counters[name] = int(value)
    return counters


def _tree_bytes(root: Path, *, budget: int = _TREE_ENTRY_BUDGET) -> tuple[int, int, bool]:
    """Apparent and allocated bytes under ``root``, plus whether the budget cut it short.

    Missing entries are skipped. The budget bounds one walk: a partial total is
    returned with ``truncated`` set rather than paying an unbounded walk.
    """
    apparent = 0
    allocated = 0
    visited = 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as scan:
                entries = list(scan)
        except OSError:
            continue
        for entry in entries:
            # Per entry, not per directory: a pytest basetemp is one directory
            # holding a subdirectory per test, so a between-directories check
            # would leave the walk unbounded exactly where it grows.
            if visited >= budget:
                return apparent, allocated, True
            visited += 1
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    continue
                status = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            apparent += status.st_size
            allocated += status.st_blocks * 512
    return apparent, allocated, False


class SuiteCostRecorder:
    """Accumulate one worker's archive-construction and write-cost facts."""

    def __init__(
        self,
        directory: Path,
        worker_id: str,
        basetemp: Path | Callable[[], Path | None] | None,
        *,
        role: str = "worker",
        sample_scratch: bool = False,
        sample_rss: bool = False,
        sample_heap: bool = False,
    ) -> None:
        self._directory = directory
        self._sample_scratch = sample_scratch
        # The heap census is taken at the trajectory's sample points and its
        # readings are recorded on them, so asking for it asks for the
        # trajectory too rather than silently producing nothing.
        self._sample_rss = sample_rss or sample_heap
        self._sample_heap = sample_heap
        self._worker_id = worker_id
        self._role = role
        self._basetemp_source = basetemp
        self._started_at = time.monotonic()
        self._io_start = _read_io()
        self._tests = 0
        self._peak_apparent = 0
        self._peak_allocated = 0
        self._scratch_truncated = False
        self._rss_start_kib = _read_rss_kib() if sample_rss else None
        self._peak_rss_kib = self._rss_start_kib or 0
        self._rss_trajectory: list[dict[str, Any]] = []
        self._rss_truncated = False
        self._last_nodeid = ""

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

    def note_test(self, nodeid: str = "") -> None:
        self._tests += 1
        if nodeid:
            self._last_nodeid = nodeid
        if self._tests % _SAMPLE_EVERY == 0:
            if self._sample_scratch:
                self.sample_storage()
            self.sample_memory()

    def sample_memory(self) -> None:
        """Record one resident-memory point; a no-op unless the trajectory was asked for.

        The nodeid carried is the test that had just finished when the sample
        was taken, so a rising segment names the directory it rose in. It is
        the sample's label, not a claim that this one test allocated the step.
        """
        if not self._sample_rss:
            return
        rss = _read_rss_kib()
        if rss is None:
            return
        self._peak_rss_kib = max(self._peak_rss_kib, rss)
        if len(self._rss_trajectory) >= _RSS_SAMPLE_LIMIT:
            self._rss_truncated = True
            return
        point: dict[str, Any] = {"tests": self._tests, "rss_kib": rss, "nodeid": self._last_nodeid}
        if self._sample_heap:
            census = _heap_census()
            # Read AFTER the census: the difference against ``rss_kib`` above
            # is what a full collection released, which is the term that says
            # whether the plateau is retention or uncollected garbage.
            point["rss_after_gc_kib"] = _read_rss_kib()
            point["gc_collected"] = census["collected"]
            point["heap_objects"] = census["objects"]
            point["heap_top"] = census["top"]
        self._rss_trajectory.append(point)

    def sample_storage(self) -> None:
        """Record the scratch-tree peak; a no-op unless the walk was asked for."""
        if not self._sample_scratch:
            return
        basetemp = self._basetemp()
        if basetemp is None:
            return
        apparent, allocated, truncated = _tree_bytes(basetemp)
        self._peak_apparent = max(self._peak_apparent, apparent)
        self._peak_allocated = max(self._peak_allocated, allocated)
        self._scratch_truncated = self._scratch_truncated or truncated

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
            **(
                {
                    "peak_scratch_apparent_bytes": self._peak_apparent,
                    "peak_scratch_allocated_bytes": self._peak_allocated,
                    "peak_scratch_truncated": self._scratch_truncated,
                }
                if self._sample_scratch
                else {}
            ),
            **(
                {
                    "rss_start_kib": self._rss_start_kib,
                    "peak_rss_kib": self._peak_rss_kib,
                    "rss_growth_kib": max(0, self._peak_rss_kib - (self._rss_start_kib or 0)),
                    "rss_trajectory": self._rss_trajectory,
                    "rss_trajectory_truncated": self._rss_truncated,
                }
                if self._sample_rss
                else {}
            ),
        }

    def write(self) -> Path:
        self.sample_storage()
        self.sample_memory()
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

    _RECORDER = SuiteCostRecorder(
        Path(directory),
        worker_id,
        resolve_basetemp,
        role=role,
        sample_scratch=os.environ.get(SUITE_COST_SCRATCH_ENV, "").strip() not in ("", "0", "false", "no"),
        sample_rss=os.environ.get(SUITE_COST_RSS_ENV, "").strip() not in ("", "0", "false", "no"),
        sample_heap=os.environ.get(SUITE_COST_HEAP_ENV, "").strip() not in ("", "0", "false", "no"),
    )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if _RECORDER is not None and report.when == "call":
        _RECORDER.note_test(report.nodeid)


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
    sampled = [worker for worker in workers if "peak_scratch_apparent_bytes" in worker]
    rss_sampled = [worker for worker in workers if "peak_rss_kib" in worker]
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
        # worker peak and retain every individual measurement below. The keys
        # are absent -- never zero -- when no worker was asked to walk its
        # scratch tree, so an unsampled run cannot read as a measured zero.
        **(
            {
                "peak_scratch_apparent_bytes": max(int(w.get("peak_scratch_apparent_bytes", 0)) for w in sampled),
                "peak_scratch_allocated_bytes": max(int(w.get("peak_scratch_allocated_bytes", 0)) for w in sampled),
                "peak_scratch_truncated": any(bool(w.get("peak_scratch_truncated")) for w in sampled),
            }
            if sampled
            else {}
        ),
        # Resident peaks, like scratch peaks, have no shared sampling clock:
        # report the largest single worker rather than a sum that would claim a
        # simultaneous suite peak nobody observed. Absent -- never zero -- when
        # no worker was asked, so an unsampled run cannot read as measured.
        **(
            {
                "peak_rss_kib": max(int(w.get("peak_rss_kib", 0)) for w in rss_sampled),
                "rss_growth_kib": max(int(w.get("rss_growth_kib", 0)) for w in rss_sampled),
                "rss_trajectory_truncated": any(bool(w.get("rss_trajectory_truncated")) for w in rss_sampled),
            }
            if rss_sampled
            else {}
        ),
        "per_worker": workers,
        "controller": controllers,
    }


#: The fields carried into the run receipt beside ``pytest_aggregate``. The
#: full per-worker detail stays in the step's ``suite-cost/run.json``; the
#: receipt carries what a before/after comparison is stated in.
_SUMMARY_FIELDS: Final = (
    "workers",
    "tests",
    "wall_clock_s",
    "tier_init",
    "archive_tier_initializations",
    "write_bytes_per_test",
)


def suite_cost_summary(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """Project one :func:`aggregate_suite_cost` result to its receipt fields."""
    io = dict(aggregate.get("io", {}))
    summary = {field: aggregate[field] for field in _SUMMARY_FIELDS if field in aggregate}
    summary["write_bytes"] = int(io.get("write_bytes", 0))
    summary["read_bytes"] = int(io.get("read_bytes", 0))
    return summary


def summarize_step_receipts(paths: Iterable[Path | str]) -> dict[str, Any] | None:
    """Combine the pytest steps of one run into a single receipt summary.

    Steps run one after another, so their counters and their wall clock add.
    This combines *steps*; summing one step's concurrent xdist workers stays
    with :func:`aggregate_suite_cost`, which is the only owner of that sum.
    """
    summaries: list[dict[str, Any]] = []
    for path in paths:
        with contextlib.suppress(OSError, ValueError):
            summaries.append(suite_cost_summary(json.loads(Path(path).read_text())))
    if not summaries:
        return None
    if len(summaries) == 1:
        return summaries[0]
    combined: dict[str, Any] = {"steps": len(summaries)}
    for field in ("workers", "tests", "wall_clock_s", "archive_tier_initializations", "write_bytes", "read_bytes"):
        combined[field] = sum(summary.get(field, 0) for summary in summaries)
    tier_init: dict[str, int] = {}
    for summary in summaries:
        for name, count in dict(summary.get("tier_init", {})).items():
            tier_init[name] = tier_init.get(name, 0) + int(count)
    combined["tier_init"] = dict(sorted(tier_init.items()))
    tests = combined["tests"]
    combined["write_bytes_per_test"] = round(combined["write_bytes"] / tests, 1) if tests else 0.0
    return combined


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
    "SUITE_COST_RSS_ENV",
    "SUITE_COST_SCRATCH_ENV",
    "SuiteCostRecorder",
    "aggregate_suite_cost",
    "suite_cost_summary",
    "summarize_step_receipts",
    "write_run_receipt",
]

"""Attribute a pytest worker's accumulated anonymous memory to what retains it.

A complete-corpus worker peaks near ``devtools.worker_memory.WORKER_PEAK_ANON_MIB``
while collection alone costs ~583 MiB, so ~87% of the footprint accumulates
while tests run.  That number sizes ``CORPUS_MAX_WORKERS`` and therefore the
wall time of every full run, and nothing measured *what* holds it.

This probe answers three separable questions in one pass, because the answers
imply different remedies:

* how much anonymous memory each test file leaves behind (``files``);
* how much of the accumulated total a ``gc.collect()`` returns -- reference
  cycles the collector had not reached yet;
* how much ``malloc_trim(0)`` returns afterwards -- memory the process already
  freed that glibc is holding at the top of its arenas rather than returning
  to the kernel.  SQLite page caches, parser buffers and payload bytes all
  allocate through libc malloc, not through CPython's object allocator, so
  this term is not visible to ``tracemalloc``.

Whatever neither call returns is genuinely live, and the closing object census
names the types holding it.

Enabled only by ``POLYLOGUE_TEST_RETENTION_PROBE=<report-path>``; when unset
nothing here is imported and no hook runs.
"""

from __future__ import annotations

import ctypes
import gc
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["RetentionProbe", "read_memory_kib"]

#: ``/proc/self/status`` fields worth recording.  ``RssAnon`` is the term the
#: worker-memory profile charges; ``VmHWM`` is the process high-water mark the
#: cgroup sampler would have seen.
_STATUS_FIELDS = ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmHWM")

#: How many types the closing census names.  A census that grows with the
#: corpus is not a receipt.
_CENSUS_WIDTH = 30


def read_memory_kib() -> dict[str, int]:
    """This process's memory totals in KiB, from ``/proc/self/status``."""
    values = dict.fromkeys(_STATUS_FIELDS, 0)
    try:
        text = Path("/proc/self/status").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if name in values:
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[name] = int(parts[0])
    return values


def _malloc_trim() -> bool:
    """Ask glibc to return arena tops to the kernel; False where unavailable."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        trim = libc.malloc_trim
    except (OSError, AttributeError):
        return False
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    trim(0)
    return True


def _object_census() -> list[dict[str, Any]]:
    """The widest retained types, by count and by shallow size."""
    counts: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    for obj in gc.get_objects():
        try:
            name = f"{type(obj).__module__}.{type(obj).__qualname__}"
        except Exception:  # pragma: no cover - exotic proxies
            continue
        counts[name] += 1
        try:
            sizes[name] += sys.getsizeof(obj)
        except Exception:  # pragma: no cover - objects refusing getsizeof
            continue
    named = {name for name, _ in counts.most_common(_CENSUS_WIDTH)}
    named |= {name for name, _ in sizes.most_common(_CENSUS_WIDTH)}
    return sorted(
        ({"type": name, "count": counts[name], "shallow_bytes": sizes[name]} for name in named),
        key=lambda row: row["shallow_bytes"],
        reverse=True,
    )


@dataclass
class _FileCost:
    tests: int = 0
    anon_delta_kib: int = 0
    hwm_raise_kib: int = 0


@dataclass
class RetentionProbe:
    """A pytest plugin that records per-file anonymous-memory growth."""

    report_path: Path
    baseline: dict[str, int] = field(default_factory=dict)
    post_collection: dict[str, int] = field(default_factory=dict)
    previous_anon_kib: int = 0
    previous_hwm_kib: int = 0
    tests_seen: int = 0
    files: dict[str, _FileCost] = field(default_factory=dict)
    peaks: list[tuple[int, str]] = field(default_factory=list)
    trajectory: list[dict[str, int]] = field(default_factory=list)

    #: How often a trajectory point is recorded.  The curve's shape (linear,
    #: plateauing, stepped) is what separates a leak from bounded retention,
    #: and it needs far fewer points than tests.
    sample_every: int = 100

    def pytest_sessionstart(self, session: Any) -> None:
        del session
        self.baseline = read_memory_kib()
        self.previous_anon_kib = self.baseline["RssAnon"]

    def pytest_collection_finish(self, session: Any) -> None:
        """Re-baseline once imports are done.

        Collection imports every selected module, and that floor is already
        measured by ``devtools bench collection``.  Charging it to whichever
        test happened to run first would hide the term this probe exists to
        find -- what a worker accumulates *while running*.
        """
        del session
        self.post_collection = read_memory_kib()
        self.previous_anon_kib = self.post_collection["RssAnon"]
        self.previous_hwm_kib = self.post_collection["VmHWM"]

    def pytest_runtest_logfinish(self, nodeid: str, location: tuple[str, int | None, str]) -> None:
        del location
        current = read_memory_kib()
        path = nodeid.partition("::")[0]
        cost = self.files.setdefault(path, _FileCost())
        cost.tests += 1
        cost.anon_delta_kib += current["RssAnon"] - self.previous_anon_kib
        # A raise in the process high-water mark is the term a cgroup ceiling
        # is actually sized against: the plateau a worker settles at is the
        # sum of whatever individual tests pushed the mark up, not an average
        # per-test drift.  Attributing it per test names the ones to bound.
        raised = current["VmHWM"] - self.previous_hwm_kib
        if raised > 0:
            cost.hwm_raise_kib += raised
            self.peaks.append((raised, nodeid))
            self.peaks.sort(reverse=True)
            del self.peaks[_CENSUS_WIDTH:]
        self.previous_anon_kib = current["RssAnon"]
        self.previous_hwm_kib = current["VmHWM"]
        self.tests_seen += 1
        if self.tests_seen % self.sample_every == 0:
            self.trajectory.append(
                {"tests": self.tests_seen, "anon_kib": current["RssAnon"], "hwm_kib": current["VmHWM"]}
            )

    def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
        del session, exitstatus
        before = read_memory_kib()
        collected = gc.collect()
        after_gc = read_memory_kib()
        trimmed = _malloc_trim()
        after_trim = read_memory_kib()
        ranked = sorted(self.files.items(), key=lambda row: row[1].hwm_raise_kib, reverse=True)
        payload = {
            "worker": os.environ.get("PYTEST_XDIST_WORKER", "main"),
            "pid": os.getpid(),
            "tests_executed": self.tests_seen,
            "baseline_kib": self.baseline,
            "post_collection_kib": self.post_collection,
            "final_kib": before,
            "after_gc_collect_kib": after_gc,
            "after_malloc_trim_kib": after_trim,
            "malloc_trim_available": trimmed,
            "gc_collected_objects": collected,
            "recovery_kib": {
                "import_floor": self.post_collection.get("RssAnon", 0) - self.baseline["RssAnon"],
                "accumulated_while_running": before["RssAnon"] - self.post_collection.get("RssAnon", 0),
                "returned_by_gc": before["RssAnon"] - after_gc["RssAnon"],
                "returned_by_malloc_trim": after_gc["RssAnon"] - after_trim["RssAnon"],
                "still_resident_above_collection": after_trim["RssAnon"] - self.post_collection.get("RssAnon", 0),
            },
            "trajectory": self.trajectory,
            "files": [
                {
                    "path": path,
                    "tests": cost.tests,
                    "anon_delta_kib": cost.anon_delta_kib,
                    "hwm_raise_kib": cost.hwm_raise_kib,
                }
                for path, cost in ranked[:_CENSUS_WIDTH]
            ],
            "peak_raising_tests": [{"hwm_raise_kib": raised, "nodeid": nodeid} for raised, nodeid in self.peaks],
            "object_census": _object_census(),
        }
        destination = self.report_path
        if destination.is_dir() or destination.suffix != ".json":
            destination.mkdir(parents=True, exist_ok=True)
            destination = destination / f"retention-{payload['worker']}-{payload['pid']}.json"
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

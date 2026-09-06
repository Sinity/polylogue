"""What a managed pytest run took, attributed to the processes that took it.

A killed run reports only that it died. Attribution is what says whether the
controller, one worker or their sum exceeded the budget the width was chosen
against, and therefore whether the next run should be narrower or the workload
lighter.

The unit of attribution is the process group the run owns: the managed pytest
controller leads its own group and xdist's workers inherit it, so membership
needs no bookkeeping the run could get wrong. ``smaps_rollup`` is read rather
than ``statm`` because workers share the controller's pages: RSS counts every
copy, and only PSS sums across processes to something the ceiling can be
compared against.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Final

__all__ = [
    "MAX_ATTRIBUTED_PROCESSES",
    "SAMPLE_INTERVAL_S",
    "ProcessGroupMemorySampler",
]

#: How often the group is measured. Half a second resolves the allocation
#: spikes that get a run killed; a coarser interval reports the plateau the
#: kill did not happen at.
SAMPLE_INTERVAL_S: Final = 0.5
#: How many processes the receipt names, worst first. A corpus run forks
#: thousands of short-lived children over hours, and a receipt that grows with
#: them is not a receipt.
MAX_ATTRIBUTED_PROCESSES: Final = 32

#: ``smaps_rollup`` keys that carry a whole-process total, and the peak each
#: one is recorded under. Private memory is what a process would still cost if
#: every sharing peer went away.
_ROLLUP_FIELDS: Final[dict[str, str]] = {
    "Rss": "rss_kib",
    "Pss": "pss_kib",
    "Private_Clean": "private_kib",
    "Private_Dirty": "private_kib",
    "Swap": "swap_kib",
}
_MEASURES: Final[tuple[str, ...]] = ("rss_kib", "pss_kib", "private_kib", "swap_kib")


def _process_group_members(pgid: int, *, proc: Path) -> list[int]:
    """Every live pid in ``pgid``, this reader's own process excluded by nature."""
    members: list[int] = []
    try:
        entries = list(proc.iterdir())
    except OSError:
        return members
    for entry in entries:
        name = entry.name
        if not name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="utf-8", errors="replace").rpartition(")")[2].split()
            if int(fields[2]) == pgid:
                members.append(int(name))
        except (OSError, IndexError, ValueError):
            continue
    return members


def _rollup(pid: int, *, proc: Path) -> dict[str, int] | None:
    """One process's memory totals in KiB, or None once it is gone."""
    try:
        text = (proc / str(pid) / "smaps_rollup").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    totals = dict.fromkeys(_MEASURES, 0)
    seen = False
    for line in text.splitlines():
        key, _, value = line.partition(":")
        measure = _ROLLUP_FIELDS.get(key)
        if measure is None:
            continue
        try:
            totals[measure] += int(value.split()[0])
        except (IndexError, ValueError):
            continue
        seen = True
    return totals if seen else None


def _command(pid: int, *, proc: Path) -> str:
    try:
        return (proc / str(pid) / "comm").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "?"


def _mem_available_mib(meminfo: Path) -> int | None:
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key == "MemAvailable":
                return int(value.split()[0]) // 1024
    except (OSError, IndexError, ValueError):
        return None
    return None


class ProcessGroupMemorySampler:
    """Peak memory of one process group, per process and in aggregate.

    Sampling runs in a thread so the caller keeps waiting on its child.
    :meth:`snapshot` is safe to read at any time, which is what lets a run
    that is being killed still file what it had taken.
    """

    def __init__(
        self,
        pgid: int,
        *,
        interval_s: float = SAMPLE_INTERVAL_S,
        proc: Path = Path("/proc"),
        meminfo: Path = Path("/proc/meminfo"),
    ) -> None:
        self._pgid = pgid
        self._interval_s = interval_s
        self._proc = proc
        self._meminfo = meminfo
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples = 0
        self._observed = 0
        self._aggregate_peak: dict[str, int] = dict.fromkeys(_MEASURES, 0)
        self._peak_processes = 0
        self._peak_at_s: float | None = None
        self._per_pid: dict[int, dict[str, Any]] = {}
        self._processes_seen = 0
        self._available_at_start = _mem_available_mib(meminfo)
        self._available_minimum = self._available_at_start
        self._started = time.monotonic()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="pytest-memory-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        """End sampling and return the run's attribution."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=self._interval_s * 4)
        return self.snapshot()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self._interval_s)

    def sample(self) -> None:
        """Take one measurement of the group. The sampling thread calls this."""
        elapsed = time.monotonic() - self._started
        totals = dict.fromkeys(_MEASURES, 0)
        readings: list[tuple[int, dict[str, int]]] = []
        for pid in _process_group_members(self._pgid, proc=self._proc):
            rollup = _rollup(pid, proc=self._proc)
            if rollup is None:
                continue
            readings.append((pid, rollup))
            for measure in _MEASURES:
                totals[measure] += rollup[measure]
        available = _mem_available_mib(self._meminfo)
        with self._lock:
            self._samples += 1
            if not readings:
                return
            self._observed += 1
            if totals["pss_kib"] > self._aggregate_peak["pss_kib"]:
                self._aggregate_peak = totals
                self._peak_processes = len(readings)
                self._peak_at_s = round(elapsed, 1)
            for pid, rollup in readings:
                entry = self._per_pid.get(pid)
                if entry is None:
                    self._processes_seen += 1
                    entry = {"pid": pid, "command": _command(pid, proc=self._proc)}
                    entry.update({f"peak_{measure}": 0 for measure in _MEASURES})
                    self._per_pid[pid] = entry
                for measure in _MEASURES:
                    if rollup[measure] > entry[f"peak_{measure}"]:
                        entry[f"peak_{measure}"] = rollup[measure]
            if available is not None and (self._available_minimum is None or available < self._available_minimum):
                self._available_minimum = available

    def snapshot(self) -> dict[str, Any]:
        """The attribution as it stands, whether or not sampling has ended."""
        with self._lock:
            processes = sorted(self._per_pid.values(), key=lambda entry: -int(entry["peak_pss_kib"]))
            document: dict[str, Any] = {
                "kind": "polylogue.pytest-memory",
                "process_group": self._pgid,
                "interval_s": self._interval_s,
                "samples": self._samples,
                "observed_samples": self._observed,
                "peak": {
                    **{measure: self._aggregate_peak[measure] for measure in _MEASURES},
                    "processes": self._peak_processes,
                    "at_s": self._peak_at_s,
                },
                "processes_seen": self._processes_seen,
                "processes": processes[:MAX_ATTRIBUTED_PROCESSES],
                "host_mem_available_mib": {
                    "at_start": self._available_at_start,
                    "minimum": self._available_minimum,
                },
            }
            if self._observed == 0:
                # A run too short to sample, or a procfs this process may not
                # read: either way the receipt says so rather than reporting a
                # peak of zero as a measurement.
                document["unmeasured"] = "no sample observed the process group"
            return document

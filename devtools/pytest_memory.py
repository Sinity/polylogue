"""What a managed pytest run took, attributed to the processes that took it.

A killed run reports only that it died. Attribution is what says whether the
controller, one worker or their sum exceeded the budget the width was chosen
against, and therefore whether the next run should be narrower or the workload
lighter.

Attribution follows the actual launched process group and its inherited
child-only custody marker. Shared cgroup membership is never ownership. The
marker retains detached children, including those reparented before a sample. ``smaps_rollup`` is read rather
than ``statm`` because workers share the controller's pages: RSS counts every
copy, and only PSS sums across processes to something the ceiling can be
compared against.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

__all__ = [
    "CUSTODY_ENV",
    "MAX_ATTRIBUTED_PROCESSES",
    "SAMPLE_INTERVAL_S",
    "ProcessGroupMemorySampler",
]

#: How often the group is measured. Half a second resolves the allocation
#: spikes that get a run killed; a coarser interval reports the plateau the
#: kill did not happen at.
SAMPLE_INTERVAL_S: Final = 0.5
#: A fresh actual-launch identity, inherited only by its children.
CUSTODY_ENV: Final = "POLYLOGUE_PYTEST_CUSTODY"
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


@dataclass(frozen=True, slots=True)
class _Identity:
    start_ticks: int
    group: int


def _identity(pid: int, *, proc: Path) -> _Identity | None:
    try:
        fields = (proc / str(pid) / "stat").read_text(encoding="utf-8", errors="replace").rpartition(")")[2].split()
        return _Identity(int(fields[19]), int(fields[2]))
    except (OSError, IndexError, ValueError):
        return None


def _marker_matches(pid: int, marker: str, *, proc: Path) -> bool | None:
    """Compare one environment field without retaining arbitrary environment values."""
    expected = (CUSTODY_ENV + "=" + marker).encode()
    pending = b""
    oversized = False
    try:
        with (proc / str(pid) / "environ").open("rb") as handle:
            while chunk := handle.read(65536):
                parts = chunk.split(b"\0")
                for index, part in enumerate(parts):
                    if not oversized:
                        if len(pending) + len(part) > len(expected):
                            pending, oversized = b"", True
                        else:
                            pending += part
                    if index < len(parts) - 1:
                        if not oversized and pending == expected:
                            return True
                        pending, oversized = b"", False
            return not oversized and pending == expected
    except OSError:
        return None


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
        custody_marker: str | None = None,
        meminfo: Path = Path("/proc/meminfo"),
        snapshot_path: Path | None = None,
        snapshot_context: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self._pgid = pgid
        self._interval_s = interval_s
        self._proc = proc
        self._custody_marker = custody_marker
        leader = _identity(pgid, proc=proc)
        self._leader_start = leader.start_ticks if leader is not None else None
        self._known_births: dict[int, tuple[int, bool]] = {}
        self._sample_lock = threading.Lock()
        self._incomplete = False
        self._meminfo = meminfo
        self._snapshot_path = snapshot_path
        self._snapshot_context = snapshot_context
        self._context_lock = threading.Lock()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples = 0
        self._observed = 0
        self._aggregate_peak: dict[str, int] = dict.fromkeys(_MEASURES, 0)
        self._peak_processes = 0
        self._peak_at_s: float | None = None
        self._per_pid: dict[tuple[int, int], dict[str, Any]] = {}
        self._processes_seen = 0
        self._available_at_start = _mem_available_mib(meminfo)
        self._available_minimum = self._available_at_start
        self._started = time.monotonic()

    def start(self) -> None:
        if self._thread is not None:
            return
        self.persist()
        self._thread = threading.Thread(target=self._loop, name="pytest-memory-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        """End sampling and return the run's attribution."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=self._interval_s * 4)
        return self.persist()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self._interval_s)

    def sample(self) -> None:
        """Take one identity-bound measurement, serialized with other samples."""
        with self._sample_lock:
            self._sample()

    def _sample(self) -> None:
        elapsed = time.monotonic() - self._started
        totals = dict.fromkeys(_MEASURES, 0)
        readings: list[tuple[int, int, str, dict[str, int]]] = []
        with self._lock:
            pgid, leader_start = self._pgid, self._leader_start
        leader = _identity(pgid, proc=self._proc)
        group_proven = leader_start is not None and leader is not None and leader.start_ticks == leader_start
        known: dict[int, tuple[int, bool]] = {}
        incomplete = False
        try:
            entries = self._proc.iterdir()
            for process_path in entries:
                if not process_path.name.isdigit():
                    continue
                pid = int(process_path.name)
                identity = _identity(pid, proc=self._proc)
                if identity is None:
                    if process_path.exists():
                        incomplete = True
                        if pid in self._known_births:
                            known[pid] = self._known_births[pid]
                    continue
                previous = self._known_births.get(pid)
                previously_proven = previous is not None and previous[0] == identity.start_ticks
                group_owned = group_proven and identity.group == pgid
                matched: bool | None = False
                if self._custody_marker is not None:
                    matched = _marker_matches(pid, self._custody_marker, proc=self._proc)
                    incomplete |= (
                        matched is None and not previously_proven and not group_owned and process_path.exists()
                    )
                owned = previously_proven or group_owned or matched is True
                if not owned:
                    continue
                known[pid] = (
                    identity.start_ticks,
                    previous[1] if previously_proven and previous is not None else False,
                )
                rollup = _rollup(pid, proc=self._proc)
                command = _command(pid, proc=self._proc)
                after = _identity(pid, proc=self._proc)
                if after is None or after.start_ticks != identity.start_ticks:
                    # Exit and PID reuse cannot attach bytes to the old identity.
                    incomplete |= after is not None or process_path.exists()
                    known.pop(pid, None)
                    continue
                if group_owned and not previously_proven and matched is not True:
                    current_leader = _identity(pgid, proc=self._proc)
                    if current_leader is None or current_leader.start_ticks != leader_start:
                        incomplete = True
                        known.pop(pid, None)
                        continue
                if rollup is None:
                    incomplete = True
                    continue
                readings.append((pid, identity.start_ticks, command, rollup))
                for measure in _MEASURES:
                    totals[measure] += rollup[measure]
        except OSError:
            incomplete = True
        # Enumeration and permission faults are not proof that a previously
        # owned process exited. Retire only an absent proc entry or new birth.
        for pid, proof in self._known_births.items():
            if pid in known:
                continue
            current = _identity(pid, proc=self._proc)
            if current is not None:
                if current.start_ticks == proof[0]:
                    known[pid] = proof
                    incomplete = True
                continue
            try:
                (self._proc / str(pid) / "stat").stat()
            except FileNotFoundError:
                continue
            except OSError:
                pass
            known[pid] = proof
            incomplete = True
        available = _mem_available_mib(self._meminfo)
        with self._lock:
            self._incomplete |= incomplete
            self._samples += 1
            self._observed += bool(readings)
            previous_pss = self._aggregate_peak["pss_kib"]
            # Keep each metric's peak independently. PSS determines which
            # moment names the group peak, but RSS/private/swap can spike at
            # a different sample and still belong in the receipt.
            for measure in _MEASURES:
                if totals[measure] > self._aggregate_peak[measure]:
                    self._aggregate_peak[measure] = totals[measure]
            if totals["pss_kib"] > previous_pss:
                self._peak_processes = len(readings)
                self._peak_at_s = round(elapsed, 1)
            for pid, start_ticks, command, rollup in readings:
                key = (pid, start_ticks)
                if not known[pid][1]:
                    self._processes_seen += 1
                    known[pid] = (start_ticks, True)
                entry = self._per_pid.get(key)
                if entry is None:
                    entry = {"pid": pid, "start_ticks": start_ticks, "command": command}
                    entry.update({f"peak_{measure}": 0 for measure in _MEASURES})
                    self._per_pid[key] = entry
                for measure in _MEASURES:
                    if rollup[measure] > entry[f"peak_{measure}"]:
                        entry[f"peak_{measure}"] = rollup[measure]
                if len(self._per_pid) > MAX_ATTRIBUTED_PROCESSES:
                    worst = min(self._per_pid, key=lambda member: int(self._per_pid[member]["peak_pss_kib"]))
                    self._per_pid.pop(worst)
            self._known_births = known
            if available is not None and (self._available_minimum is None or available < self._available_minimum):
                self._available_minimum = available
        self.persist()

    def snapshot(self) -> dict[str, Any]:
        """The attribution as it stands, whether or not sampling has ended."""
        with self._lock:
            processes = sorted(self._per_pid.values(), key=lambda entry: -int(entry["peak_pss_kib"]))
            document: dict[str, Any] = {
                "kind": "polylogue.pytest-memory",
                "process_group": self._pgid,
                "interval_s": self._interval_s,
                "attribution_scope": "launched_group_and_inherited_marker"
                if self._custody_marker
                else "launched_group",
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
            if self._incomplete:
                document["incomplete"] = "some process identity, custody, or memory evidence was unreadable or changed"
            if self._observed == 0:
                # A run too short to sample, or a procfs this process may not
                # read: either way the receipt says so rather than reporting a
                # peak of zero as a measurement.
                document["unmeasured"] = "no sample observed the process group"
            return document

    def persist(self) -> dict[str, Any]:
        """Publish a kill-survivable sidecar, when one was requested."""
        memory = self.snapshot()
        if self._snapshot_path is None:
            return memory
        context: Mapping[str, Any] = {}
        if self._snapshot_context is not None:
            with self._context_lock:
                try:
                    candidate = self._snapshot_context()
                except Exception as exc:  # pragma: no cover - defensive telemetry path
                    candidate = {"telemetry_error": f"{type(exc).__name__}: {exc}"}
                if isinstance(candidate, Mapping):
                    context = candidate
        document: dict[str, Any] = {
            "schema_version": 1,
            "kind": "polylogue.pytest-slot-telemetry",
            "memory": memory,
            **dict(context),
        }
        path = self._snapshot_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{self._pgid}.tmp")
            temporary.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
            os.replace(temporary, path)
        except (OSError, TypeError, ValueError):
            # Telemetry must never change the result of the verification run.
            pass
        return memory

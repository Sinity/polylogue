"""Size an xdist run from the pytest slot it owns.

The pytest pool admits work under host-pressure control before its command
starts.  Once admitted, the width belongs to ``agentctl-pytest.slice``: using
host-wide ``MemAvailable`` (or a shared ancestor slice's current use) lets
unrelated agents silently shrink an already-admitted corpus.  The local
cgroup ceiling remains a hard bound; unavailable host memory is retained in
the receipt as an observation, not misused as a second scheduler.

The reading is taken inside the pytest slot rather than when the command was
built: a run can sit in the queue for hours, and its own remaining cgroup
budget is what matters when workers start.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Final

__all__ = [
    "CGROUP_PROCESS_PATH",
    "CGROUP_ROOT",
    "CONTROLLER_PEAK_MIB",
    "CORPUS_MAX_WORKERS",
    "MEMORY_HEADROOM_FRACTION",
    "PYTEST_SLICE_MEMORY_HIGH_MIB",
    "WORKER_PEAK_MIB",
    "available_memory_mib",
    "cgroup_available_mib",
    "memory_bounded_worker_cap",
    "pytest_slot_available_mib",
    "resize_worker_argument",
    "width_within",
]

#: What one worker and the controller cost at peak, in MiB: worker PSS takes
#: the upper end of the measured spread so the estimate errs toward fewer
#: workers. These are properties of the workload; pressure enters as the live
#: readings, never as these constants.
#:
#: Measured 2026-09-14 by two independent methods that agree within 1%, after a
#: corpus run was OOM-killed at 93%: the kill-time arithmetic on that run gives
#: (5600 - 1075) / 2 = 2263 MiB per worker, and a slot receipt for a managed
#: pytest process running 4,293 tests shows 2290 MiB private / 2352 MiB PSS.
#: The previous 686 was low by ~3.3x, which is why a width derived from it
#: produced a sustained-pressure kill rather than the throttling this file
#: anticipates. Only ~420 MiB of the peak is the import floor (58 MiB base plus
#: 362 MiB for 1,256 test modules); the rest grows with tests executed, so
#: trimming imports does not recover width.
WORKER_PEAK_MIB = 2263
CONTROLLER_PEAK_MIB = 1075
#: Memory left unclaimed so the run stays clear of the out-of-memory daemon's
#: pressure threshold rather than approaching it.
MEMORY_HEADROOM_FRACTION = 0.2
#: The pytest pool's soft ceiling: ``agentctl-pytest.slice`` MemoryHigh, with a
#: MemoryMax above it and no swap. Above the soft ceiling the kernel does not
#: kill the run, it throttles every allocation, and the slice asks systemd-oomd
#: to kill on the sustained pressure that throttling produces.
#:
#: MIRROR of Sinnix's ``agentctl-pytest.slice`` ``MemoryHigh``, declared in
#: ``flake/data/runtime-defaults.nix``. It is a hand-kept copy, so it drifts
#: silently when that budget changes -- which it did on 2026-09-14, when the
#: slice went 6G -> 12G after a corpus run was OOM-killed at 93%. The binding
#: ceiling is the PARENT ``agentctl-pytest.slice``, not the per-pool leaves:
#: heavy and quick are its children and share one budget, so a run in the quick
#: pool charges pressure to the same parent a corpus run is accounted against.
#: ``pytest_slot_available_mib()`` reads the live cgroup and is the authority at
#: runtime; this constant only sizes the default before a slot is held.
PYTEST_SLICE_MEMORY_HIGH_MIB: Final = 12 * 1024


def width_within(budget_mib: float) -> int:
    """The widest run whose peak fits ``budget_mib`` with the headroom kept back.

    Never zero: a slow run beats a run that does not start.
    """
    budget = budget_mib * (1.0 - MEMORY_HEADROOM_FRACTION) - CONTROLLER_PEAK_MIB
    return max(1, int(budget // WORKER_PEAK_MIB))


#: The corpus width, and the ceiling any configured width is reduced to. It is
#: what the pytest slice holds at the peaks above rather than a number declared
#: beside them, so an idle slice yields exactly this many workers and the live
#: bounds below narrow only a slice that is already occupied.
CORPUS_MAX_WORKERS = width_within(PYTEST_SLICE_MEMORY_HIGH_MIB)

#: This process's cgroup v2 membership, and where that hierarchy is mounted.
CGROUP_PROCESS_PATH: Final = Path("/proc/self/cgroup")
CGROUP_ROOT: Final = Path("/sys/fs/cgroup")
#: The cgroup v2 files that bound a level. ``memory.high`` bounds as firmly as
#: ``memory.max`` here: the kernel does not kill for it, but sustained
#: allocation above it is reclaim thrash, and the runtime's slices ask
#: systemd-oomd to kill on exactly the memory pressure that produces.
_CGROUP_LIMIT_FILES: Final = ("memory.max", "memory.high")
_MIB: Final = 1024 * 1024


def available_memory_mib(*, meminfo: Path = Path("/proc/meminfo")) -> int | None:
    """``MemAvailable``, the kernel's own estimate of what a new workload may take."""
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key == "MemAvailable":
                return int(value.split()[0]) // 1024
    except (OSError, IndexError, ValueError):
        return None
    return None


def _cgroup_bytes(path: Path) -> int | None:
    """One cgroup v2 memory value in bytes; None for ``max``, absent or malformed."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value >= 0 else None


def _cgroup_usage_mib(directory: Path) -> int:
    """What this level holds that a new allocation cannot simply displace, in MiB.

    ``memory.current`` charges the level's page cache too, and a slice that has
    just run the corpus carries gigabytes of it. Counting that as spent would
    size the next run down to a single worker, which misses the slot timeout as
    surely as being killed misses the results. ``inactive_file`` is the cache
    the kernel reclaims first and is subtracted; active cache stays counted, so
    the estimate still errs toward fewer workers.
    """
    current = _cgroup_bytes(directory / "memory.current")
    if current is None:
        return 0
    try:
        stat = (directory / "memory.stat").read_text(encoding="utf-8")
    except OSError:
        return current // _MIB
    for line in stat.splitlines():
        key, _, value = line.partition(" ")
        if key == "inactive_file":
            try:
                return max(0, current - int(value)) // _MIB
            except ValueError:
                break
    return current // _MIB


def _cgroup_directories(process_cgroup: Path, root: Path) -> list[Path]:
    """This process's cgroup directory and every ancestor of it, leaf first.

    Every ancestor's limit applies to this process too, so all of them are
    read. An empty list means the membership was unreadable or names a path
    that is not under this mount -- a cgroup namespace, typically.
    """
    try:
        lines = process_cgroup.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        _hierarchy, separator, path = line.partition("::")
        if not separator:
            continue
        parts = PurePosixPath(path.strip()).parts[1:]
        directory = root.joinpath(*parts)
        if not directory.is_dir():
            return []
        return [root.joinpath(*parts[:depth]) for depth in range(len(parts), -1, -1)]
    return []


def cgroup_available_mib(*, process_cgroup: Path = CGROUP_PROCESS_PATH, root: Path = CGROUP_ROOT) -> int | None:
    """What this cgroup still allows before its tightest level bounds it, in MiB.

    None when no level carries a finite limit, which is the same answer as an
    unreadable hierarchy: the cgroup constrains nothing this run must respect.
    A level whose usage cannot be read constrains by its ceiling alone, which
    is never wider than ignoring the level.
    """
    budgets = []
    for directory in _cgroup_directories(process_cgroup, root):
        limits = [value for name in _CGROUP_LIMIT_FILES if (value := _cgroup_bytes(directory / name)) is not None]
        if not limits:
            continue
        budgets.append(max(0, min(limits) // _MIB - _cgroup_usage_mib(directory)))
    return min(budgets) if budgets else None


def pytest_slot_available_mib(*, process_cgroup: Path = CGROUP_PROCESS_PATH, root: Path = CGROUP_ROOT) -> int | None:
    """Return the local pytest-pool budget, excluding shared parent slices.

    ``agentctl.slice`` also contains agent workers.  Its *current* use is a
    host-admission concern, not capacity already consumed by this pytest
    command.  Stop at the named pytest pool when present.  A different runtime
    layout has no such proof, so fall back to every applicable cgroup limit.
    """
    budgets: list[int] = []
    found_pytest_pool = False
    for directory in _cgroup_directories(process_cgroup, root):
        limits = [value for name in _CGROUP_LIMIT_FILES if (value := _cgroup_bytes(directory / name)) is not None]
        if limits:
            budgets.append(max(0, min(limits) // _MIB - _cgroup_usage_mib(directory)))
        if directory.name == "agentctl-pytest.slice":
            found_pytest_pool = True
            break
    if found_pytest_pool:
        return min(budgets) if budgets else None
    return cgroup_available_mib(process_cgroup=process_cgroup, root=root)


def memory_bounded_worker_cap(
    *,
    requested: int = CORPUS_MAX_WORKERS,
    meminfo: Path = Path("/proc/meminfo"),
    process_cgroup: Path = CGROUP_PROCESS_PATH,
    cgroup_root: Path = CGROUP_ROOT,
) -> tuple[int, dict[str, Any]]:
    """The widest run this job's pytest cgroup may hold right now.

    The pytest cgroup is the only bound that narrows the width.  Host
    ``MemAvailable`` is an instantaneous reading of a resource every other
    agent on the machine is also taking from: letting it narrow an
    already-admitted run means an unrelated agent allocating for a second
    decides how wide the corpus runs, and this suite is SQLite-IO-bound, so
    that width is not recovered later.  Admission under host pressure is the
    pytest pool's decision and is already made before this runs; the host
    reading is kept in the receipt as an observation of the machine at launch.

    The reading is live at launch, so a job that waited in the queue is sized
    against the budget it actually has.  The derived ``CORPUS_MAX_WORKERS``
    cap remains the upper bound even when the cgroup is roomy.
    """
    host = available_memory_mib(meminfo=meminfo)
    cgroup = pytest_slot_available_mib(process_cgroup=process_cgroup, root=cgroup_root)
    if cgroup is None:
        return requested, {
            "basis": "unmeasured",
            "host_available_mib": host,
            "cgroup_available_mib": None,
            "workers": requested,
            "requested_workers": requested,
            "narrowed": False,
        }
    workers = max(1, min(requested, width_within(cgroup)))
    return workers, {
        "basis": "cgroup_budget",
        "available_mib": cgroup,
        "host_available_mib": host,
        "cgroup_available_mib": cgroup,
        "headroom_fraction": MEMORY_HEADROOM_FRACTION,
        "controller_peak_mib": CONTROLLER_PEAK_MIB,
        "worker_peak_mib": WORKER_PEAK_MIB,
        "workers": workers,
        "requested_workers": requested,
        "narrowed": workers < requested,
    }


def resize_worker_argument(
    argv: list[str],
    *,
    meminfo: Path = Path("/proc/meminfo"),
    process_cgroup: Path = CGROUP_PROCESS_PATH,
    cgroup_root: Path = CGROUP_ROOT,
) -> tuple[list[str], dict[str, Any] | None]:
    """Narrow an ``-n <count>`` xdist argument to what memory allows.

    Returns the command unchanged, and no basis, when it names no worker count
    or when the count already fits. A run with ``-n 0`` asked for no xdist at
    all and is left alone.
    """
    index: int | None = None
    requested_text: str | None = None
    inline = False
    for candidate, argument in enumerate(argv):
        if argument in {"-n", "--numprocesses"} and candidate + 1 < len(argv):
            index, requested_text = candidate + 1, argv[candidate + 1]
            break
        if argument.startswith("--numprocesses="):
            index, requested_text, inline = candidate, argument.split("=", 1)[1], True
            break
        if argument.startswith("-n") and len(argument) > 2:
            index, requested_text, inline = candidate, argument[2:].removeprefix("="), True
            break
    try:
        requested = int(requested_text) if requested_text is not None else None
    except ValueError:
        return argv, None
    if requested is None or index is None:
        return argv, None
    if requested <= 1:
        return argv, None
    workers, basis = memory_bounded_worker_cap(
        requested=requested, meminfo=meminfo, process_cgroup=process_cgroup, cgroup_root=cgroup_root
    )
    if not basis.get("narrowed"):
        return argv, basis
    resized = list(argv)
    resized[index] = (
        f"{argv[index].split('=', 1)[0]}={workers}"
        if inline and argv[index].startswith("--numprocesses=")
        else f"-n{workers}"
        if inline
        else str(workers)
    )
    return resized, basis

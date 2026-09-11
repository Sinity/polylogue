"""Size an xdist run to the memory the run may actually take when it starts.

Two bounds apply at once and the tighter one decides. The host's
``MemAvailable`` is what the kernel will hand any new workload; the job's
cgroup is what this run in particular may take before it is throttled or
killed, and it can be far tighter than the host while the host looks idle.

The reading is taken inside the pytest slot rather than when the command was
built: a run can sit in the queue for hours, and the memory that mattered is
the memory present when its workers start.
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
    "resize_worker_argument",
    "width_within",
]

#: What one worker and the controller cost at peak, in MiB: worker PSS takes
#: the upper end of the measured spread so the estimate errs toward fewer
#: workers. These are properties of the workload; pressure enters as the live
#: readings, never as these constants.
WORKER_PEAK_MIB = 686
CONTROLLER_PEAK_MIB = 1075
#: Memory left unclaimed so the run stays clear of the out-of-memory daemon's
#: pressure threshold rather than approaching it.
MEMORY_HEADROOM_FRACTION = 0.2
#: The pytest pool's soft ceiling: ``agentctl-pytest.slice`` MemoryHigh, 6 GiB
#: (MemoryMax 8 GiB, no swap). Above the soft ceiling the kernel does not kill
#: the run, it throttles every allocation, and the slice asks systemd-oomd to
#: kill on the sustained pressure that throttling produces.
PYTEST_SLICE_MEMORY_HIGH_MIB: Final = 6 * 1024


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


def memory_bounded_worker_cap(
    *,
    requested: int = CORPUS_MAX_WORKERS,
    meminfo: Path = Path("/proc/meminfo"),
    process_cgroup: Path = CGROUP_PROCESS_PATH,
    cgroup_root: Path = CGROUP_ROOT,
) -> tuple[int, dict[str, Any]]:
    """The widest run this job may hold right now, and the basis for it."""
    host = available_memory_mib(meminfo=meminfo)
    cgroup = cgroup_available_mib(process_cgroup=process_cgroup, root=cgroup_root)
    measured = [value for value in (host, cgroup) if value is not None]
    if not measured:
        return requested, {"basis": "unmeasured", "workers": requested, "requested_workers": requested}
    available = min(measured)
    workers = max(1, min(requested, width_within(available)))
    return workers, {
        "basis": "cgroup_budget" if cgroup is not None and cgroup == available else "mem_available",
        "available_mib": available,
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

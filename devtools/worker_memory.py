"""Size an xdist run to the memory the host has when the run actually starts.

Two full corpus runs were killed by the out-of-memory daemon at about 6.2 GB
peak while the operator's desktop and other lanes were resident. Width was
fixed regardless of what was free, so the run assumed memory that was not
there; a killed run measures nothing, and a narrower run that finishes measures
everything.

The reading is taken inside the pytest slot rather than when the command was
built: a run can sit in the queue for hours, and the memory that mattered is
the memory present when its workers start.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "CONTROLLER_PEAK_MIB",
    "CORPUS_MAX_WORKERS",
    "MEMORY_HEADROOM_FRACTION",
    "WORKER_PEAK_MIB",
    "available_memory_mib",
    "memory_bounded_worker_cap",
    "resize_worker_argument",
]

#: The widest the corpus and the runner's affected tier ever run, sized to the
#: pytest pool's 12 GiB cgroup ceiling rather than host cores. Measured
#: 2026-09-03 uncontended: 47 minutes for 20,860 tests at eight workers; at two
#: the same run takes about seven hours and the required check cannot finish
#: inside its slot timeout.
CORPUS_MAX_WORKERS = 8
#: What one worker and the controller cost at peak, in MiB. Measured on the
#: 8-worker seed: workers 0.53-0.67 GiB PSS, controller about 1.05 GiB. The
#: worker figure takes the upper end so the estimate errs toward fewer workers.
#: These are properties of the workload; host pressure enters as the live
#: reading, not as these constants.
WORKER_PEAK_MIB = 686
CONTROLLER_PEAK_MIB = 1075
#: Memory left unclaimed so the run stays clear of the out-of-memory daemon's
#: pressure threshold rather than approaching it.
MEMORY_HEADROOM_FRACTION = 0.2


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


def memory_bounded_worker_cap(
    *, requested: int = CORPUS_MAX_WORKERS, meminfo: Path = Path("/proc/meminfo")
) -> tuple[int, dict[str, Any]]:
    """The widest run this host can hold right now, and the basis for it."""
    available = available_memory_mib(meminfo=meminfo)
    if available is None:
        return requested, {"basis": "unmeasured", "workers": requested}
    budget = available * (1.0 - MEMORY_HEADROOM_FRACTION) - CONTROLLER_PEAK_MIB
    fits = int(budget // WORKER_PEAK_MIB)
    workers = max(1, min(requested, fits))
    return workers, {
        "basis": "mem_available",
        "available_mib": available,
        "headroom_fraction": MEMORY_HEADROOM_FRACTION,
        "controller_peak_mib": CONTROLLER_PEAK_MIB,
        "worker_peak_mib": WORKER_PEAK_MIB,
        "workers": workers,
        "requested_workers": requested,
        "narrowed": workers < requested,
    }


def resize_worker_argument(
    argv: list[str], *, meminfo: Path = Path("/proc/meminfo")
) -> tuple[list[str], dict[str, Any] | None]:
    """Narrow an ``-n <count>`` xdist argument to what memory allows.

    Returns the command unchanged, and no basis, when it names no worker count
    or when the count already fits. A run with ``-n 0`` asked for no xdist at
    all and is left alone.
    """
    try:
        index = argv.index("-n")
        requested = int(argv[index + 1])
    except (ValueError, IndexError):
        return argv, None
    if requested <= 1:
        return argv, None
    workers, basis = memory_bounded_worker_cap(requested=requested, meminfo=meminfo)
    if not basis.get("narrowed"):
        return argv, basis
    resized = list(argv)
    resized[index + 1] = str(workers)
    return resized, basis

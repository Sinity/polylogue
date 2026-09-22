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

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

__all__ = [
    "CGROUP_PROCESS_PATH",
    "CGROUP_ROOT",
    "CONTROLLER_PEAK_MIB",
    "CORPUS_MAX_WORKERS",
    "MEASURED_CHARGE",
    "PYTEST_SLICE_MEMORY_HIGH_MIB",
    "WORKER_PEAK_ANON_MIB",
    "WORKER_PEAK_CACHE_MIB",
    "ChargeProfile",
    "available_memory_mib",
    "cgroup_available_mib",
    "corroborate_profile",
    "memory_bounded_worker_cap",
    "pytest_slot_available_mib",
    "resize_worker_argument",
    "width_within",
]

#: A worker's own allocations at peak, in MiB -- ANONYMOUS memory only, which
#: is what a process-level RSS/PSS sampler reports.
#:
#: Measured 2026-09-21 by reading the slot sampler back against the complete
#: corpus run it sized (receipt
#: ``.cache/verify/runs/20260921T010003Z-all-2490183-814ed21d``, width 2,
#: 23,526 collected, 9,783 s, no pressure kill). The two xdist workers peaked
#: at ``peak_private_kib`` 4,724 MiB (15,416 tests executed) and 4,481 MiB
#: (8,109 tests), with ``peak_rss_kib`` only 15-20 MiB above each -- so a
#: worker's charge is anonymous memory almost entirely.
#:
#: This SUPERSEDES 700, which was the whole-corpus COLLECTION floor mistaken
#: for the peak. That floor is real and still measurable -- ``devtools bench
#: collection`` reports 583.5 MiB RSS for 23,528 collected at 0b1e99d69 -- but
#: it is 12% of what a worker reaches once it runs its share. The superseded
#: comment here claimed the trajectory PLATEAUS near 582 MiB and that "the
#: import floor therefore DOMINATES the peak"; that plateau was an artifact of
#: measuring one directory (3,318 node IDs). Over 15,416 executed tests the
#: worker climbs 4.1 GiB above the floor, so import trimming is NOT the lever
#: on width -- what a worker accumulates while running tests is.
#:
#: Taken at the widest per-worker observation rather than the mean. The peak
#: is not width-independent (a worker runs corpus/width tests, and this term
#: grows with tests executed), so charging every width the width-2 peak
#: overestimates at width 3+ -- in the safe direction, which is the direction
#: ``width_within`` must err.
WORKER_PEAK_ANON_MIB = 4750
#: What the same worker charges the cgroup BESIDES its anonymous memory, in
#: MiB: mapped page cache and slab. It stays in the model because
#: ``memory.high`` charges it -- see :class:`ChargeProfile` -- but it is a
#: small term, not the dominant one.
#:
#: Measured 2026-09-21 from the same receipt: ``peak_rss_kib -
#: peak_private_kib`` is 15 MiB and 20 MiB for the two workers, against 98 GiB
#: of scratch SQLite written during the run (suite-cost receipt
#: ``write_bytes`` 105,337,847,808). The writes do not accumulate as charged
#: cache for this process group.
#:
#: This SUPERSEDES 2850, which was never measured: it was back-solved as the
#: residual of ``11,776 - 1,075 - 3 * 700`` from the 2026-09-17 width-3 run
#: and therefore absorbed the anonymous-memory underestimate above. With the
#: anon term corrected the residual closes without it: that run's ~11,776 MiB
#: less an 869 MiB controller is ~3,636 MiB per worker at 7,842 tests each,
#: consistent with 4,724 MiB at 15,416 tests. Nothing is left over for
#: gigabytes of page cache.
#:
#: Consequently the remedy the superseded comment named -- capping the scratch
#: page cache with ``fadvise(DONTNEED)``/``sync_file_range`` on discarded
#: basetemps, or a ``memory.low`` split -- is REJECTED: it attacks ~20 MiB per
#: worker, not 2,850. The term to attack, if width is wanted, is what a worker
#: retains across 15,000 executed tests.
#:
#: LIMIT of this measurement, stated rather than implied: ``smaps_rollup``
#: sees only cache a process still maps. Cache the cgroup charges for files no
#: process maps is invisible to it, so this is a floor for the cgroup's file
#: charge, not a proof of its total. It is the residual arithmetic above, not
#: the sampler alone, that rules out a large hidden term.
#: :func:`corroborate_profile` re-runs this comparison on every future run so
#: the constant stops being a single observation nothing reads back.
WORKER_PEAK_CACHE_MIB = 25
#: The xdist controller at peak, in MiB. It collects the corpus but runs no
#: tests and writes no scratch databases, so it is carried as anon alone.
#:
#: Measured 869 MiB RSS / 852 MiB private in the 2026-09-21 receipt. Left at
#: 1075: it is already conservative against measurement, and narrowing it
#: would widen admission on the strength of one observation.
CONTROLLER_PEAK_MIB = 1075
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
#:
#: COUPLING: this value and the Sinnix declaration are one budget kept in two
#: places, and nothing gates the pair. Changing either without the other is the
#: drift that already happened once. When ``flake/data/runtime-defaults.nix``
#: moves, move this line in the same landing and say so in both messages; 12G
#: there is a closed budget (12G pytest + 8G agent = the plane's own 20G
#: MemoryHigh, derived from 31G host minus app/session/desktop reservations),
#: not free headroom to raise unilaterally from this side.
#:
#: HEADROOM OWNER: the Sinnix value, not this module. Sinnix picked 12G by
#: taking an intended width's peak and applying 1.2x -- under the anonymous-RSS
#: model this file used to carry, which understated the charge (see
#: :class:`ChargeProfile`). The ceiling already carries its safety margin, so
#: ``width_within`` does not discount it a second time. Raising or lowering the
#: margin is a Sinnix change to ``flake/data/runtime-defaults.nix``, not an
#: arithmetic change here.
PYTEST_SLICE_MEMORY_HIGH_MIB: Final = 12 * 1024


@dataclass(frozen=True)
class ChargeProfile:
    """What a run CHARGES its cgroup at peak, in MiB, per worker and in total.

    The quantity matters more than the numbers. ``memory.high`` accounts anon
    + page cache + slab; a per-process RSS/PSS sampler reports anon alone.
    Dividing the first by the second is a category error, and it is what was
    OOM-killing the corpus: a live managed worker measured 2026-09-20 held
    anon 544 MiB, file 457 MiB (nearly all ``inactive_file``) and slab 44 MiB
    for ``memory.current`` 1050 MiB -- 1.93x its anonymous footprint, with
    ~44% of the charge being page cache from the suite's own scratch writes.
    The width the anon model chose therefore overran the ceiling by roughly
    the same factor, which systemd-oomd observed at the 2026-09-19 kills.

    Every field here is a charge against the same ceiling, so the comparison
    in :func:`width_within` is between compatible quantities. Anon is kept as
    its own field rather than folded in because it is the term a sampler can
    re-measure directly, so drift in either component stays detectable.
    """

    #: One worker's anonymous peak.
    worker_anon_mib: float
    #: The page cache and slab charged alongside that worker.
    worker_cache_mib: float
    #: The controller's whole charge.
    controller_mib: float

    @property
    def worker_charge_mib(self) -> float:
        """One worker's whole charge against ``memory.high``."""
        return self.worker_anon_mib + self.worker_cache_mib

    def charge_mib(self, workers: int) -> float:
        """What a run of ``workers`` charges the slice at peak, controller included."""
        return self.controller_mib + workers * self.worker_charge_mib

    def admission_estimate(self, workers: int, budget_mib: float) -> dict[str, float]:
        """The predicted peak charge and what it leaves under ``budget_mib``.

        polylogue-k1o3t asks for the estimate a run is admitted on and the
        margin it carries, recorded rather than recomputed. Both numbers are
        derived from the same profile the width was chosen with, so a run's
        measured cgroup peak can be compared against what was predicted for it
        without reconstructing the arithmetic from the components.

        The margin is worth reading, not just recording: ``width_within``
        deliberately holds nothing back beyond the controller, so the chosen
        width fills the ceiling. Under the superseded 2026-09-20 constants a
        12 GiB ``memory.high`` predicted 11,725 MiB at width 3 -- a 4.6%
        margin -- and admitted it; the 2026-09-17 run at that width stalled
        pinned at ``memory.high``, throttled into continuous reclaim. The
        2026-09-21 constants predict 15,400 MiB for the same width and refuse
        it, which is the correction. Widening the margin is a Sinnix change to
        the slice budget, not an arithmetic change here.
        """
        predicted = self.charge_mib(workers)
        return {
            "predicted_charge_mib": round(predicted, 1),
            "budget_mib": round(float(budget_mib), 1),
            "margin_mib": round(budget_mib - predicted, 1),
            "margin_fraction": round((budget_mib - predicted) / budget_mib, 4) if budget_mib else 0.0,
        }


#: The profile every default width is derived from.
MEASURED_CHARGE: Final = ChargeProfile(
    worker_anon_mib=WORKER_PEAK_ANON_MIB,
    worker_cache_mib=WORKER_PEAK_CACHE_MIB,
    controller_mib=CONTROLLER_PEAK_MIB,
)


def width_within(budget_mib: float, *, profile: ChargeProfile = MEASURED_CHARGE) -> int:
    """The widest run whose peak CHARGE fits ``budget_mib``.

    ``budget_mib`` is a cgroup memory ceiling -- the slice's ``memory.high`` as
    its owner sized it, or what remains of it -- so what is divided into it is
    the charge a worker makes against that same accounting, not the anonymous
    memory a sampler sees. The ceiling already carries its own headroom, so
    nothing is held back here beyond the controller's own charge.

    Never zero: a slow run beats a run that does not start.
    """
    return max(1, int((budget_mib - profile.controller_mib) // profile.worker_charge_mib))


#: The corpus width, and the ceiling any configured width is reduced to. It is
#: what the pytest slice holds at the peaks above rather than a number declared
#: beside them, so an idle slice yields exactly this many workers and the live
#: bounds below narrow only a slice that is already occupied.
CORPUS_MAX_WORKERS = width_within(PYTEST_SLICE_MEMORY_HIGH_MIB)


def corroborate_profile(
    memory: Mapping[str, Any] | None,
    sizing: Mapping[str, Any] | None,
    *,
    profile: ChargeProfile = MEASURED_CHARGE,
) -> dict[str, Any] | None:
    """Read the run's own sampler back against the profile that sized it.

    Every constant above is a measurement taken once. Until this existed
    nothing compared them to what the next run actually took, so
    :data:`WORKER_PEAK_CACHE_MIB` sat at a value back-solved from a single
    2026-09-17 residual for four days while real workers charged the slice
    differently -- and the receipt that would have shown it was written and
    never read. This turns every managed run into a falsification of the
    profile it was admitted under, recorded beside the width it chose.

    The comparison is deliberately between a *predicted charge* and what the
    sampler saw, with the sampler's own limit named in the result rather than
    silently absorbed: ``smaps_rollup`` reports only cache a process still
    maps, so ``observed_worker_file_mib`` is a floor for the cgroup's file
    charge, not its total. ``observed_worker_anon_mib`` has no such caveat --
    it is exactly the quantity :attr:`ChargeProfile.worker_anon_mib` claims to
    bound, taken at the largest single process so one over-large worker cannot
    be averaged away.

    ``None`` when there is nothing to compare: no sampler document, a run too
    short to observe the group, or no width on record.
    """
    if not memory or not sizing or memory.get("unmeasured"):
        return None
    processes = [entry for entry in memory.get("processes") or [] if entry.get("peak_private_kib")]
    peak = memory.get("peak") or {}
    if not processes or not peak.get("rss_kib"):
        return None
    try:
        workers = int(sizing["workers"])
    except (KeyError, TypeError, ValueError):
        return None

    heaviest = max(processes, key=lambda entry: int(entry["peak_private_kib"]))
    worker_anon_mib = round(int(heaviest["peak_private_kib"]) / 1024, 1)
    worker_file_mib = round(max(0, int(heaviest["peak_rss_kib"]) - int(heaviest["peak_private_kib"])) / 1024, 1)
    group_peak_mib = round(int(peak["rss_kib"]) / 1024, 1)
    predicted_mib = round(profile.charge_mib(workers), 1)

    understated = worker_anon_mib > profile.worker_anon_mib or group_peak_mib > predicted_mib
    return {
        "verdict": "understated" if understated else "corroborated",
        "workers": workers,
        "heaviest_pid": heaviest.get("pid"),
        "observed_worker_anon_mib": worker_anon_mib,
        "observed_worker_file_mib": worker_file_mib,
        "observed_group_peak_mib": group_peak_mib,
        "declared_worker_anon_mib": float(profile.worker_anon_mib),
        "declared_worker_cache_mib": float(profile.worker_cache_mib),
        "predicted_charge_mib": predicted_mib,
        "worker_anon_headroom_mib": round(profile.worker_anon_mib - worker_anon_mib, 1),
        "group_headroom_mib": round(predicted_mib - group_peak_mib, 1),
        "file_term_is_a_mapped_floor": True,
    }


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

    When the cgroup carries no readable limit the fallback is the *declared*
    slice budget, not the host: ``requested`` is whatever ``-n`` the command
    named, so returning it unnarrowed would let ``-n 8`` run at 8 wherever the
    pool's layout is absent -- a cgroup namespace, a foreign runtime, a process
    outside the pool.  ``PYTEST_SLICE_MEMORY_HIGH_MIB`` is what the slice is
    declared to allow, so falling back to it keeps the same budget answering
    the question when its live enforcement cannot be read.
    """
    host = available_memory_mib(meminfo=meminfo)
    cgroup = pytest_slot_available_mib(process_cgroup=process_cgroup, root=cgroup_root)
    if cgroup is None:
        workers = max(1, min(requested, CORPUS_MAX_WORKERS))
        return workers, {
            "basis": "declared_budget",
            "available_mib": PYTEST_SLICE_MEMORY_HIGH_MIB,
            "host_available_mib": host,
            "cgroup_available_mib": None,
            "headroom_owner": "sinnix agentctl-pytest.slice MemoryHigh",
            "controller_peak_mib": CONTROLLER_PEAK_MIB,
            "worker_peak_anon_mib": WORKER_PEAK_ANON_MIB,
            "worker_peak_cache_mib": WORKER_PEAK_CACHE_MIB,
            "worker_peak_charge_mib": MEASURED_CHARGE.worker_charge_mib,
            "workers": workers,
            "requested_workers": requested,
            "narrowed": workers < requested,
            **MEASURED_CHARGE.admission_estimate(workers, PYTEST_SLICE_MEMORY_HIGH_MIB),
        }
    workers = max(1, min(requested, width_within(cgroup)))
    return workers, {
        "basis": "cgroup_budget",
        "available_mib": cgroup,
        "host_available_mib": host,
        "cgroup_available_mib": cgroup,
        "headroom_owner": "sinnix agentctl-pytest.slice MemoryHigh",
        "controller_peak_mib": CONTROLLER_PEAK_MIB,
        "worker_peak_anon_mib": WORKER_PEAK_ANON_MIB,
        "worker_peak_cache_mib": WORKER_PEAK_CACHE_MIB,
        "worker_peak_charge_mib": MEASURED_CHARGE.worker_charge_mib,
        "workers": workers,
        "requested_workers": requested,
        "narrowed": workers < requested,
        **MEASURED_CHARGE.admission_estimate(workers, cgroup),
    }


def resize_worker_argument(
    argv: list[str],
    *,
    meminfo: Path = Path("/proc/meminfo"),
    process_cgroup: Path = CGROUP_PROCESS_PATH,
    cgroup_root: Path = CGROUP_ROOT,
) -> tuple[list[str], dict[str, Any] | None]:
    """Narrow an ``-n <count>`` xdist argument to what memory allows.

    Always returns the observed budget as the basis -- selected width,
    requested width, and which budget source answered -- even when the
    command names no worker count or the requested count already fits. A
    receipt built from ``None`` here cannot say what this run was admitted
    on; only a malformed ``-n`` value returns no basis, because there is
    nothing to report. A missing flag, ``-n 0``, or ``-n 1`` are all read as
    a request for one worker, the width they already run at, and never
    rewrite ``argv``.
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
    effective_requested = requested if requested is not None and requested > 1 else 1
    workers, basis = memory_bounded_worker_cap(
        requested=effective_requested, meminfo=meminfo, process_cgroup=process_cgroup, cgroup_root=cgroup_root
    )
    if index is None or effective_requested <= 1 or not basis.get("narrowed"):
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

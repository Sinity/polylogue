"""A managed pytest run is sized to its pytest cgroup, never to host free RAM.

The pytest pool's remaining cgroup budget is the one live launch bound.  The
host's ``MemAvailable`` is read only to be recorded: AgentCTL backpressure
already decided admission under host pressure, and this suite is SQLite-IO
bound, so a width surrendered to an unrelated agent's momentary allocation is
not won back later in the run.

Where the pool's cgroup layout is absent entirely the fallback is the declared
slice budget, still never the host.  Measured on sinnix-prime 2026-09-15: the
live ``agentctl-pytest.slice`` is a *user*-manager unit with
``MemoryHigh=12G`` / ``MemoryMax=14G``, and a process inside it reads 12078 MiB
(heavy) or 4932 MiB (quick) remaining -- three workers and one.  Querying the
system manager for that unit reports ``infinity`` and describes nothing.

Anti-vacuity:
- replace ``pytest_slot_available_mib`` with the generic ancestor walk and
  ``test_the_pytest_slice_ignores_shared_agent_slice_usage`` goes red -- a
  busy agent slice once stole the pytest pool's independent budget;
- read only ``memory.max`` and ``test_a_soft_ceiling_bounds_as_firmly_as_a_hard_one``
  goes red, which is the production shape: the pytest slice declares a
  ``memory.high`` below its ``memory.max`` (the fixtures derive both from
  ``PYTEST_SLICE_MEMORY_HIGH_MIB`` rather than restating the host's numbers),
  and systemd-oomd kills on the pressure that running above the soft ceiling
  produces;
- stop walking ancestors and ``test_an_ancestor_slice_bounds_its_children``
  goes red -- an enclosing slice's limit binds this run just as its own does;
- charge the whole of ``memory.current`` and
  ``test_reclaimable_page_cache_is_not_spent_memory`` goes red -- the slice a
  corpus run just left is full of cache, and reading that as spent sizes the
  next run to one worker, which misses the slot timeout;
- drop the ``resize_worker_argument`` call from ``devtools.pytest_slot.main``
  and ``test_the_slot_resizes_the_queued_command`` goes red, which is the case
  that matters: a queued run can wait hours, so a width chosen when the command
  was built describes memory that is no longer there;
- ignore the cgroup reading and size from host ``MemAvailable`` alone, and
  ``test_the_pytest_slice_bounds_a_host_with_memory_to_spare``,
  ``test_the_pytest_slice_ignores_shared_agent_slice_usage`` and
  ``test_the_slot_records_which_bound_narrowed_the_run`` go red -- every case
  there pairs a slice that is the bound with a host reading derived to be
  roomier than the slice can ever hand out (``HOST_NOT_THE_BOUND_MIB``);
- return a bare ``requested`` when the cgroup is unreadable and
  ``test_an_unreadable_cgroup_falls_back_to_the_declared_budget`` goes red --
  ``requested`` is whatever ``-n`` the command named, so a host without the
  pool's slice layout would run ``-n 8`` at eight;
- restore host ``MemAvailable`` as a second narrowing input -- ``min(host,
  cgroup)`` -- and ``test_a_loaded_host_does_not_narrow_an_admitted_run`` and
  ``test_a_host_narrower_than_the_slice_does_not_decide`` go red: both pair a
  host derived to hold one worker fewer than the slice does
  (``HOST_NARROWER_THAN_THE_SLICE_MIB``) with a cgroup that is either unbounded
  or idle, so any host-side narrowing shows up as a width one short;
- restate any of these fixtures as a literal instead of deriving it from
  ``_budget_for_width`` / ``_occupancy_for_width`` and the next budget move
  silently reverses which bound a case tests rather than failing -- which is
  exactly what the 6 GiB -> 12 GiB slice change did to the literals these
  helpers replaced;
- drop it from ``devtools.pytest_slot._run_held`` and
  ``test_a_run_that_already_holds_the_slot_is_narrowed_too`` goes red -- the
  declared corpus and affected operations run inside the pytest pool, so they
  hold the slot and never reach the queued path at all.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

import pytest

from devtools.worker_memory import (
    CORPUS_MAX_WORKERS,
    MEASURED_CHARGE,
    PYTEST_SLICE_MEMORY_HIGH_MIB,
    ChargeProfile,
    available_memory_mib,
    cgroup_available_mib,
    memory_bounded_worker_cap,
    resize_worker_argument,
    width_within,
)


class CgroupPaths(TypedDict):
    """The two paths every sizing entry point reads its cgroup budget from."""

    process_cgroup: Path
    cgroup_root: Path


MIB = 1024 * 1024
#: The pytest pool's slice as the host declares it, in MiB: a soft ceiling
#: below the hard one, and an enclosing slice that is generous but finite.
#:
#: Derived from the production constant rather than restated. A hand-kept second
#: copy here drifted silently when the host slice went 6G -> 12G on 2026-09-14,
#: and these tests then asserted a width the host could no longer produce --
#: the same failure mode as any other mirrored budget. The hard ceiling sits a
#: declared step above the soft one, matching how the slice is declared.
PYTEST_SLICE_HIGH_MIB = PYTEST_SLICE_MEMORY_HIGH_MIB
PYTEST_SLICE_MAX_MIB = PYTEST_SLICE_HIGH_MIB + 2 * 1024
AGENTCTL_SLICE_HIGH_MIB = 20 * 1024


def _fixture_dir(tmp_path: Path, kind: str) -> Path:
    """A directory of its own for every fixture a test builds.

    A test that contrasts two readings builds both, and one writing over the
    other's files would answer the contrast with itself.
    """
    directory = tmp_path / f"{kind}-{len(list(tmp_path.glob(f'{kind}-*')))}"
    directory.mkdir()
    return directory


def _meminfo(tmp_path: Path, available_mib: int) -> Path:
    path = _fixture_dir(tmp_path, "meminfo") / "meminfo"
    path.write_text(
        f"MemTotal:       32689696 kB\nMemFree:         1000000 kB\nMemAvailable:   {available_mib * 1024} kB\n",
        encoding="utf-8",
    )
    return path


def _cgroup(
    tmp_path: Path, levels: Sequence[tuple[str, Mapping[str, str]]], *, membership: str | None = None
) -> CgroupPaths:
    """A cgroup v2 hierarchy: nested directories, each with the files it declares.

    Returns the paths ``memory_bounded_worker_cap`` takes, so no test reads the
    machine it runs on.
    """
    base = _fixture_dir(tmp_path, "cgroup")
    root = base / "root"
    root.mkdir()
    directory = root
    parts: list[str] = []
    for name, files in levels:
        parts.append(name)
        directory = directory / name
        directory.mkdir(parents=True, exist_ok=True)
        for filename, content in files.items():
            (directory / filename).write_text(content, encoding="utf-8")
    process_cgroup = base / "self-cgroup"
    declared = membership if membership is not None else "/" + "/".join(parts)
    process_cgroup.write_text(f"0::{declared}\n", encoding="utf-8")
    return CgroupPaths(process_cgroup=process_cgroup, cgroup_root=root)


def _bytes(mib: int) -> str:
    return str(mib * MIB)


def _pytest_slice(tmp_path: Path, *, current_mib: int) -> CgroupPaths:
    """The production shape: the pytest slice inside the runtime's slice."""
    return _cgroup(
        tmp_path,
        [
            ("user.slice", {}),
            ("user-1000.slice", {"memory.max": "max", "memory.high": "max", "memory.current": _bytes(9000)}),
            (
                "agentctl.slice",
                {
                    "memory.max": "max",
                    "memory.high": _bytes(AGENTCTL_SLICE_HIGH_MIB),
                    "memory.current": _bytes(4800),
                },
            ),
            (
                "agentctl-pytest.slice",
                {
                    "memory.max": _bytes(PYTEST_SLICE_MAX_MIB),
                    "memory.high": _bytes(PYTEST_SLICE_HIGH_MIB),
                    "memory.current": _bytes(current_mib),
                },
            ),
        ],
    )


def _unbounded_cgroup(tmp_path: Path) -> CgroupPaths:
    """A hierarchy that constrains nothing, so a case can isolate host memory."""
    return _cgroup(tmp_path, [("nolimit.slice", {"memory.max": "max", "memory.current": _bytes(100)})])


def _peak_mib(workers: int) -> int:
    """What a run of ``workers`` CHARGES its slice at peak, controller included.

    The charge, not the anonymous footprint: ``memory.high`` accounts page
    cache and slab too, and every budget these fixtures build is a cgroup
    ceiling, so the two sides of every comparison below are the same quantity.
    """
    return int(MEASURED_CHARGE.charge_mib(workers))


def _budget_for_width(workers: int) -> int:
    """The smallest budget, in MiB, that ``width_within`` answers with ``workers``.

    Derived rather than chosen: the peak is the closed form and the loop settles
    the flooring, so the result is exactly the boundary -- one MiB less holds
    fewer workers. Every fixture that wants "a bound that holds exactly N
    workers" is built from this, so moving a peak moves the fixtures with it
    instead of leaving them asserting a width the production constants can no
    longer produce.
    """
    budget = _peak_mib(workers)
    while width_within(budget) < workers:
        budget += 1
    return budget


def _occupancy_for_width(workers: int) -> int:
    """How much the pytest slice must already hold to yield exactly ``workers``.

    The complement of :func:`_budget_for_width` against the slice's own soft
    ceiling, so a slice budget change moves the occupancy a test needs rather
    than silently making its premise false.
    """
    occupied = PYTEST_SLICE_HIGH_MIB - _budget_for_width(workers)
    assert occupied >= 0, "the slice cannot hold fewer workers than its own ceiling allows"
    return occupied


#: A host reading that is deliberately *not* the tighter bound: roomier than
#: anything the pytest slice can hand out, so every case built on it is decided
#: by the cgroup. Restating a literal here is what broke when the slice went
#: 6 GiB -> 12 GiB: a host of 10 GiB stopped being the roomier of the two and
#: the cases silently changed which bound they were testing.
HOST_NOT_THE_BOUND_MIB = PYTEST_SLICE_MAX_MIB + 2 * 1024
#: A budget too small even for one worker, so only the never-zero floor answers.
STARVED_CGROUP_MIB = _budget_for_width(1) - 1
#: A host that holds one worker fewer than the slice does, so the host decides.
HOST_NARROWER_THAN_THE_SLICE_MIB = _budget_for_width(CORPUS_MAX_WORKERS - 1)


def test_an_idle_host_runs_the_full_width(tmp_path: Path) -> None:
    workers, basis = memory_bounded_worker_cap(
        meminfo=_meminfo(tmp_path, HOST_NOT_THE_BOUND_MIB), **_unbounded_cgroup(tmp_path)
    )
    assert workers == CORPUS_MAX_WORKERS
    assert basis["narrowed"] is False
    assert basis["cgroup_available_mib"] is None


def test_a_loaded_host_does_not_narrow_an_admitted_run(tmp_path: Path) -> None:
    """A loaded host is observed, not obeyed, once the run holds its slot.

    The host reading is sized to hold exactly one worker fewer than the
    declared width, so a re-introduced host bound would show as a width one
    short rather than as an arbitrary number that happens to match today. It
    is still recorded: the receipt says what the machine looked like at launch.
    """
    host_mib = HOST_NARROWER_THAN_THE_SLICE_MIB
    assert width_within(host_mib) == CORPUS_MAX_WORKERS - 1
    workers, basis = memory_bounded_worker_cap(meminfo=_meminfo(tmp_path, host_mib), **_unbounded_cgroup(tmp_path))
    assert workers == CORPUS_MAX_WORKERS
    assert basis["narrowed"] is False
    assert basis["basis"] == "declared_budget"
    assert basis["host_available_mib"] == host_mib


def test_a_starved_cgroup_still_runs_one_worker(tmp_path: Path) -> None:
    """Headroom never reduces the launch to zero workers."""
    paths = _cgroup(tmp_path, [("job.slice", {"memory.max": _bytes(STARVED_CGROUP_MIB), "memory.current": "0"})])
    workers, basis = memory_bounded_worker_cap(
        meminfo=_meminfo(tmp_path, HOST_NOT_THE_BOUND_MIB),
        process_cgroup=paths["process_cgroup"],
        cgroup_root=paths["cgroup_root"],
    )
    assert width_within(STARVED_CGROUP_MIB) == 1
    assert workers == 1
    assert basis["basis"] == "cgroup_budget"


def test_an_unbounded_cgroup_falls_back_to_the_declared_budget(tmp_path: Path) -> None:
    """With no readable cgroup limit the declared slice budget answers, not the host.

    A starved host reading must not become the answer here: an unreadable
    cgroup means the live bound could not be measured, not that memory is
    short. The width is what the slice is declared to allow.
    """
    workers, basis = memory_bounded_worker_cap(
        meminfo=_meminfo(tmp_path, STARVED_CGROUP_MIB), **_unbounded_cgroup(tmp_path)
    )
    assert workers == CORPUS_MAX_WORKERS
    assert basis["basis"] == "declared_budget"
    assert basis["available_mib"] == PYTEST_SLICE_MEMORY_HIGH_MIB
    assert basis["narrowed"] is False


def test_an_unreadable_cgroup_falls_back_to_the_declared_budget(tmp_path: Path) -> None:
    """A command asking for more than the slice is declared to hold is still cut down.

    ``requested`` is whatever ``-n`` the argv named, not the corpus default, so
    an unmeasured cgroup returning it unchanged would run a hand-written
    ``-n 8`` at eight on any host lacking the pool's slice layout -- a cgroup
    namespace, a foreign runtime, a process outside the pool. That is the
    width class m3018's originating incident was killed at.
    """
    over_wide = CORPUS_MAX_WORKERS + 5
    workers, basis = memory_bounded_worker_cap(
        requested=over_wide, meminfo=_meminfo(tmp_path, HOST_NOT_THE_BOUND_MIB), **_unbounded_cgroup(tmp_path)
    )
    assert workers == CORPUS_MAX_WORKERS
    assert basis["basis"] == "declared_budget"
    assert basis["requested_workers"] == over_wide
    assert basis["narrowed"] is True
    # Still not the host: a roomy host does not buy back the declared ceiling.
    assert basis["host_available_mib"] == HOST_NOT_THE_BOUND_MIB


def test_an_unreadable_meminfo_is_only_an_absent_observation(tmp_path: Path) -> None:
    """The host reading is a receipt field; losing it changes no width."""
    paths = _pytest_slice(tmp_path, current_mib=0)
    workers, basis = memory_bounded_worker_cap(meminfo=tmp_path / "absent", **paths)
    assert workers == CORPUS_MAX_WORKERS
    assert basis["basis"] == "cgroup_budget"
    assert basis["host_available_mib"] is None


@pytest.mark.parametrize("content", ["", "MemTotal: 100 kB\n", "MemAvailable: not-a-number kB\n"])
def test_a_malformed_meminfo_is_unmeasured(tmp_path: Path, content: str) -> None:
    path = tmp_path / "meminfo"
    path.write_text(content, encoding="utf-8")
    assert available_memory_mib(meminfo=path) is None


def test_a_finite_cgroup_limit_is_what_is_left_under_it(tmp_path: Path) -> None:
    paths = _cgroup(tmp_path, [("job.slice", {"memory.max": _bytes(8192), "memory.current": _bytes(1192)})])
    assert cgroup_available_mib(process_cgroup=paths["process_cgroup"], root=paths["cgroup_root"]) == 7000


def test_a_cgroup_already_over_its_limit_has_nothing_left(tmp_path: Path) -> None:
    """Usage above the ceiling reads as no budget, never as a negative one."""
    paths = _cgroup(tmp_path, [("job.slice", {"memory.max": _bytes(2048), "memory.current": _bytes(3000)})])
    assert cgroup_available_mib(process_cgroup=paths["process_cgroup"], root=paths["cgroup_root"]) == 0


def test_a_soft_ceiling_bounds_as_firmly_as_a_hard_one(tmp_path: Path) -> None:
    """``memory.high`` decides when it is the tighter of the two.

    The kernel does not kill for it, but the slice asks systemd-oomd to kill on
    the memory pressure that sustained allocation above it produces.
    """
    paths = _cgroup(
        tmp_path,
        [
            (
                "job.slice",
                {
                    "memory.max": _bytes(PYTEST_SLICE_MAX_MIB),
                    "memory.high": _bytes(PYTEST_SLICE_HIGH_MIB),
                    "memory.current": _bytes(1024),
                },
            )
        ],
    )
    assert (
        cgroup_available_mib(process_cgroup=paths["process_cgroup"], root=paths["cgroup_root"])
        == PYTEST_SLICE_HIGH_MIB - 1024
    )


def test_an_ancestor_slice_bounds_its_children(tmp_path: Path) -> None:
    """An enclosing slice's remaining budget binds this run as its own does."""
    paths = _cgroup(
        tmp_path,
        [
            ("outer.slice", {"memory.max": _bytes(4096), "memory.current": _bytes(3000)}),
            ("inner.slice", {"memory.max": _bytes(8192), "memory.current": _bytes(500)}),
        ],
    )
    assert cgroup_available_mib(process_cgroup=paths["process_cgroup"], root=paths["cgroup_root"]) == 1096


def test_an_unlimited_cgroup_constrains_nothing(tmp_path: Path) -> None:
    paths = _cgroup(
        tmp_path, [("job.slice", {"memory.max": "max", "memory.high": "max", "memory.current": _bytes(500)})]
    )
    assert cgroup_available_mib(process_cgroup=paths["process_cgroup"], root=paths["cgroup_root"]) is None


def test_reclaimable_page_cache_is_not_spent_memory(tmp_path: Path) -> None:
    """The slice a corpus run just left is full of cache, not of workload.

    Counting it as spent would size the next run to a single worker, which
    misses the slot timeout as surely as being killed misses the results.
    """
    paths = _cgroup(
        tmp_path,
        [
            (
                "job.slice",
                {
                    "memory.max": _bytes(6144),
                    "memory.current": _bytes(5000),
                    "memory.stat": f"anon {_bytes(600)}\ninactive_file {_bytes(4400)}\nslab 12345\n",
                },
            )
        ],
    )
    assert cgroup_available_mib(process_cgroup=paths["process_cgroup"], root=paths["cgroup_root"]) == 5544


def test_active_page_cache_stays_counted(tmp_path: Path) -> None:
    """Only the cache the kernel drops first is discounted; the rest errs low."""
    paths = _cgroup(
        tmp_path,
        [
            (
                "job.slice",
                {
                    "memory.max": _bytes(6144),
                    "memory.current": _bytes(5000),
                    "memory.stat": f"active_file {_bytes(4400)}\ninactive_file 0\n",
                },
            )
        ],
    )
    assert cgroup_available_mib(process_cgroup=paths["process_cgroup"], root=paths["cgroup_root"]) == 1144


@pytest.mark.parametrize("stat", ["", "anon 100\n", "inactive_file not-a-number\n"])
def test_usage_without_a_readable_breakdown_is_the_whole_charge(tmp_path: Path, stat: str) -> None:
    paths = _cgroup(
        tmp_path,
        [("job.slice", {"memory.max": _bytes(4096), "memory.current": _bytes(1096), "memory.stat": stat})],
    )
    assert cgroup_available_mib(process_cgroup=paths["process_cgroup"], root=paths["cgroup_root"]) == 3000


def test_a_level_whose_usage_is_unreadable_still_bounds_by_its_ceiling(tmp_path: Path) -> None:
    paths = _cgroup(tmp_path, [("job.slice", {"memory.max": _bytes(3072)})])
    assert cgroup_available_mib(process_cgroup=paths["process_cgroup"], root=paths["cgroup_root"]) == 3072


@pytest.mark.parametrize("limit", ["", "not-a-number", "-1", "  "])
def test_a_malformed_cgroup_limit_constrains_nothing(tmp_path: Path, limit: str) -> None:
    paths = _cgroup(tmp_path, [("job.slice", {"memory.max": limit, "memory.current": _bytes(100)})])
    assert cgroup_available_mib(process_cgroup=paths["process_cgroup"], root=paths["cgroup_root"]) is None


def test_absent_cgroup_files_constrain_nothing(tmp_path: Path) -> None:
    paths = _cgroup(tmp_path, [("job.slice", {})])
    assert cgroup_available_mib(process_cgroup=paths["process_cgroup"], root=paths["cgroup_root"]) is None


def test_an_absent_membership_file_constrains_nothing(tmp_path: Path) -> None:
    assert cgroup_available_mib(process_cgroup=tmp_path / "absent", root=tmp_path / "cgroup") is None


@pytest.mark.parametrize("membership", ["/outside.slice/elsewhere.scope", "1:name=systemd:/legacy", ""])
def test_a_membership_this_mount_cannot_resolve_constrains_nothing(tmp_path: Path, membership: str) -> None:
    """A cgroup namespace names a path this mount does not carry; guessing is worse."""
    paths = _cgroup(
        tmp_path,
        [("job.slice", {"memory.max": _bytes(1024), "memory.current": _bytes(0)})],
        membership=membership,
    )
    assert cgroup_available_mib(process_cgroup=paths["process_cgroup"], root=paths["cgroup_root"]) is None


def test_an_idle_pytest_slice_runs_the_declared_corpus_width(tmp_path: Path) -> None:
    """The declared width is what the slice holds, so an idle slice yields all of it.

    The width is derived from the slice's own ceiling rather than declared
    beside it, which is what keeps the two from disagreeing: a corpus that
    asks for more than its sizing rule allows is narrowed on every start and
    never once runs at the width it declares.

    Anti-vacuity: declare ``CORPUS_MAX_WORKERS`` as a literal either side of
    ``width_within(PYTEST_SLICE_MEMORY_HIGH_MIB)`` and this goes red -- wider
    overruns the soft ceiling the slice kills for, narrower leaves the slot
    holding memory it never spends.
    """
    paths = _pytest_slice(tmp_path, current_mib=0)
    workers, basis = memory_bounded_worker_cap(
        requested=CORPUS_MAX_WORKERS, meminfo=_meminfo(tmp_path, HOST_NOT_THE_BOUND_MIB), **paths
    )

    assert workers == CORPUS_MAX_WORKERS
    assert basis["narrowed"] is False
    assert basis["basis"] == "cgroup_budget"
    # The declared width is the widest whose peak fits the ceiling as its owner
    # sized it -- no second discount applied to an already-headroomed figure.
    assert _peak_mib(CORPUS_MAX_WORKERS) <= PYTEST_SLICE_HIGH_MIB
    assert _peak_mib(CORPUS_MAX_WORKERS + 1) > PYTEST_SLICE_HIGH_MIB


def test_the_pytest_slice_bounds_a_host_with_memory_to_spare(tmp_path: Path) -> None:
    """The killed condition of job 1836: the host was idle, the cgroup was not.

    The host reading is roomy enough to hold the full declared width; the pytest
    slice is occupied enough to hold one worker fewer, and it is the slice that
    kills. Both sides are derived -- the occupancy from the width it must force,
    the host from the slice it must not be tighter than -- so a budget change
    moves the fixture instead of quietly reversing which bound is under test.
    """
    occupied_mib = _occupancy_for_width(CORPUS_MAX_WORKERS - 1)
    paths = _pytest_slice(tmp_path, current_mib=occupied_mib)
    workers, basis = memory_bounded_worker_cap(meminfo=_meminfo(tmp_path, HOST_NOT_THE_BOUND_MIB), **paths)
    assert basis["basis"] == "cgroup_budget"
    assert basis["cgroup_available_mib"] == PYTEST_SLICE_HIGH_MIB - occupied_mib
    assert basis["host_available_mib"] == HOST_NOT_THE_BOUND_MIB
    assert workers == CORPUS_MAX_WORKERS - 1
    assert workers < CORPUS_MAX_WORKERS
    # The host alone would have chosen the width that was killed.
    host_only, _ = memory_bounded_worker_cap(
        meminfo=_meminfo(tmp_path, HOST_NOT_THE_BOUND_MIB), **_unbounded_cgroup(tmp_path)
    )
    assert host_only == CORPUS_MAX_WORKERS


def test_the_pytest_slice_ignores_shared_agent_slice_usage(tmp_path: Path) -> None:
    """The local pytest pool, rather than host or shared parent use, decides.

    The slice is occupied right up to the boundary that still holds the full
    width, so the shared ``agentctl.slice`` usage beside it and the roomier host
    are both visibly ignored: counting either would answer with fewer workers.
    """
    occupied_mib = _occupancy_for_width(CORPUS_MAX_WORKERS)
    paths = _pytest_slice(tmp_path, current_mib=occupied_mib)
    workers, basis = memory_bounded_worker_cap(meminfo=_meminfo(tmp_path, HOST_NOT_THE_BOUND_MIB), **paths)
    assert basis["basis"] == "cgroup_budget"
    assert basis["available_mib"] == PYTEST_SLICE_HIGH_MIB - occupied_mib
    assert basis["host_available_mib"] == HOST_NOT_THE_BOUND_MIB
    assert workers == CORPUS_MAX_WORKERS


def test_a_host_narrower_than_the_slice_does_not_decide(tmp_path: Path) -> None:
    """An idle pytest slice runs its full width while the host is busy elsewhere.

    The host here holds one worker fewer than the slice does, so ``min(host,
    cgroup)`` would answer ``CORPUS_MAX_WORKERS - 1``. The slice was admitted
    with this budget and keeps it.
    """
    host_mib = HOST_NARROWER_THAN_THE_SLICE_MIB
    paths = _pytest_slice(tmp_path, current_mib=0)
    workers, basis = memory_bounded_worker_cap(meminfo=_meminfo(tmp_path, host_mib), **paths)
    assert basis["basis"] == "cgroup_budget"
    assert basis["available_mib"] == PYTEST_SLICE_HIGH_MIB
    assert basis["host_available_mib"] == host_mib
    assert host_mib < basis["cgroup_available_mib"]
    assert workers == CORPUS_MAX_WORKERS


def test_the_slice_ceiling_is_not_discounted_a_second_time(tmp_path: Path) -> None:
    """The slice's ceiling already carries its headroom; sizing does not re-apply it.

    Sinnix picks ``MemoryHigh`` by taking the intended width's peak and
    multiplying by 1.2, so the number this module reads is a headroomed figure
    already. Discounting it again here cost a worker: a 12 GiB slice justified
    for four derived three, roughly a quarter of corpus throughput, with no
    correctness gain. One owner keeps the margin, and it is the Sinnix value.

    Anti-vacuity: restore any fractional discount inside ``width_within`` --
    ``budget * 0.8`` was the shipped form -- and the slice provisioned for its
    declared peak stops yielding the width that peak fits.
    """
    provisioned = _peak_mib(CORPUS_MAX_WORKERS)
    assert provisioned <= PYTEST_SLICE_HIGH_MIB
    assert width_within(provisioned) == CORPUS_MAX_WORKERS

    paths = _pytest_slice(tmp_path, current_mib=0)
    workers, basis = memory_bounded_worker_cap(
        requested=CORPUS_MAX_WORKERS, meminfo=_meminfo(tmp_path, HOST_NOT_THE_BOUND_MIB), **paths
    )
    assert workers == CORPUS_MAX_WORKERS
    # The margin's owner is named in the receipt, not a fraction applied here.
    assert basis["headroom_owner"] == "sinnix agentctl-pytest.slice MemoryHigh"
    assert "headroom_fraction" not in basis


def test_a_hosted_verify_launch_stays_inside_the_pytest_slice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The production launch: the runner's corpus command inside the pytest slice.

    The width it selects has to fit under every limit the slice hierarchy
    carries, with the run's own usage already counted.
    """
    from devtools import verify

    monkeypatch.delenv("POLYLOGUE_PYTEST_WORKERS", raising=False)
    current_mib = _occupancy_for_width(CORPUS_MAX_WORKERS - 1)
    argv = ["python", "-m", "pytest", *verify._pytest_worker_args(maximum=CORPUS_MAX_WORKERS), "tests"]
    assert argv[argv.index("-n") + 1] == str(CORPUS_MAX_WORKERS)

    resized, basis = resize_worker_argument(
        argv,
        meminfo=_meminfo(tmp_path, HOST_NOT_THE_BOUND_MIB),
        **_pytest_slice(tmp_path, current_mib=current_mib),
    )
    assert basis is not None and basis["narrowed"] is True
    workers = int(resized[resized.index("-n") + 1])
    assert workers == basis["workers"]
    # Inside every ceiling the slice hierarchy declares, usage included.
    assert current_mib + _peak_mib(workers) <= PYTEST_SLICE_HIGH_MIB
    assert current_mib + _peak_mib(workers) <= PYTEST_SLICE_MAX_MIB
    assert 4800 + _peak_mib(workers) <= AGENTCTL_SLICE_HIGH_MIB
    # It is the slice, not the host, that narrowed it.
    assert basis["basis"] == "cgroup_budget"
    # A slice already holding memory yields less than it holds when idle.
    assert workers < CORPUS_MAX_WORKERS


def test_resize_narrows_the_worker_argument_in_place(tmp_path: Path) -> None:
    argv = ["python", "-m", "pytest", "--dist=loadgroup", "-n", "8", "tests"]
    resized, basis = resize_worker_argument(
        argv,
        meminfo=_meminfo(tmp_path, HOST_NOT_THE_BOUND_MIB),
        **_pytest_slice(tmp_path, current_mib=_occupancy_for_width(CORPUS_MAX_WORKERS - 1)),
    )
    assert basis is not None and basis["narrowed"] is True
    assert resized[resized.index("-n") + 1] == str(basis["workers"])
    # Only the count changes; the rest of the command is untouched.
    assert resized[: resized.index("-n")] == argv[: argv.index("-n")]
    assert resized[resized.index("-n") + 2 :] == argv[argv.index("-n") + 2 :]


def test_resize_leaves_a_run_that_already_fits(tmp_path: Path) -> None:
    argv = ["python", "-m", "pytest", "-n", "2", "tests"]
    resized, _basis = resize_worker_argument(
        argv, meminfo=_meminfo(tmp_path, HOST_NOT_THE_BOUND_MIB), **_unbounded_cgroup(tmp_path)
    )
    assert resized == argv


@pytest.mark.parametrize(
    "argv", [["pytest", "tests"], ["pytest", "-n", "0"], ["pytest", "-n", "auto"], ["pytest", "-n"]]
)
def test_resize_leaves_commands_it_does_not_understand(argv: list[str], tmp_path: Path) -> None:
    """No xdist, an explicit single process, or a form this does not parse."""
    resized, _basis = resize_worker_argument(
        list(argv),
        meminfo=_meminfo(tmp_path, STARVED_CGROUP_MIB),
        **_pytest_slice(tmp_path, current_mib=PYTEST_SLICE_HIGH_MIB),
    )
    assert resized == argv


def test_the_slot_resizes_the_queued_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The width is decided when the queued run starts, not when it was built.

    A run can sit in the single-slot pytest queue for hours; the memory that
    matters is what the job may take when its workers start.
    """
    import subprocess

    import devtools.pytest_slot as slot

    launched: dict[str, list[str]] = {}

    class _Child:
        pid = 4321

        def poll(self) -> int | None:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    def _popen(command: list[str], **_kwargs: object) -> _Child:
        launched["command"] = command
        return _Child()

    launch = tmp_path / "launch.json"
    log = tmp_path / "run.log"
    launch.write_text(
        '{"argv": ["python", "-m", "pytest", "-n", "8", "tests"], "environment": {}, '
        f'"working_directory": "{tmp_path}", "log_path": "{log}"}}'
    )
    monkeypatch.setattr(subprocess, "Popen", _popen)
    monkeypatch.setattr(slot, "resize_worker_argument", lambda argv: (argv[:-3] + ["-n", "3", "tests"], None))
    assert slot.main([str(launch)]) == 0
    assert launched["command"][launched["command"].index("-n") + 1] == "3"


def test_the_slot_records_which_bound_narrowed_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The log says why the run is narrower than it asked to be."""
    import subprocess

    import devtools.pytest_slot as slot

    class _Child:
        pid = 4321

        def poll(self) -> int | None:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    launch = tmp_path / "launch.json"
    log = tmp_path / "run.log"
    launch.write_text(
        '{"argv": ["python", "-m", "pytest", "-n", "8", "tests"], "environment": {}, '
        f'"working_directory": "{tmp_path}", "log_path": "{log}"}}'
    )
    monkeypatch.setattr(subprocess, "Popen", lambda command, **_kwargs: _Child())
    # The cgroup must be the tighter bound for the log to name it, so the host
    # reading is the one that is deliberately never the bound.
    paths = _pytest_slice(tmp_path, current_mib=_occupancy_for_width(CORPUS_MAX_WORKERS - 1))
    monkeypatch.setattr(
        slot,
        "resize_worker_argument",
        lambda argv: resize_worker_argument(argv, meminfo=_meminfo(tmp_path, HOST_NOT_THE_BOUND_MIB), **paths),
    )
    assert slot.main([str(launch)]) == 0
    assert "from the job cgroup" in log.read_text(encoding="utf-8")


@pytest.mark.uses_real_clock("runs a real child process group to its end")
def test_a_run_that_already_holds_the_slot_is_narrowed_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The width is decided where the run starts, and a pool job starts here.

    ``verify_all`` and ``verify_affected`` execute inside the pytest pool: they
    hold the slot and run the command in place, so a narrowing applied only to
    queued commands leaves the corpus at the width the memory ceiling was never
    measured for.
    """
    import sys

    from devtools import pytest_slot

    recorder = tmp_path / "argv.json"
    command = [
        sys.executable,
        "-c",
        f"import json, sys; open({str(recorder)!r}, 'w').write(json.dumps(sys.argv[1:]))",
        "--dist=loadgroup",
        "-n",
        str(CORPUS_MAX_WORKERS),
    ]
    current_mib = _occupancy_for_width(CORPUS_MAX_WORKERS - 1)
    paths = _pytest_slice(tmp_path, current_mib=current_mib)
    monkeypatch.setattr(
        pytest_slot,
        "resize_worker_argument",
        lambda argv: resize_worker_argument(argv, meminfo=_meminfo(tmp_path, HOST_NOT_THE_BOUND_MIB), **paths),
    )

    outcome = pytest_slot.run_pytest(
        command,
        cwd=str(tmp_path),
        env={"PATH": os.environ["PATH"], "POLYLOGUE_PYTEST_SLOT": "held"},
        root=tmp_path,
    )

    assert outcome.returncode == 0
    executed = json.loads(recorder.read_text(encoding="utf-8"))
    workers = int(executed[executed.index("-n") + 1])
    assert workers < CORPUS_MAX_WORKERS
    assert outcome.receipt is not None
    assert outcome.receipt["sizing"]["workers"] == workers
    # The width it ran at fits the slice that would otherwise have killed it.
    assert current_mib + _peak_mib(workers) <= PYTEST_SLICE_HIGH_MIB


def test_the_width_fits_the_whole_charge_not_only_anonymous_memory() -> None:
    """``memory.high`` charges page cache and slab; the width must respect that.

    A per-process sampler sees anonymous memory. The ceiling being divided is a
    cgroup ceiling, which also accounts the page cache the suite's own scratch
    SQLite writes fill and the slab behind them -- 1.93x the anonymous
    footprint on the worker measured 2026-09-20 (anon 544 MiB, file 457 MiB,
    slab 44 MiB, ``memory.current`` 1050 MiB). Sizing from anon alone chose a
    width whose real charge overran the slice, which is what systemd-oomd
    killed.

    Anti-vacuity: divide the ceiling by the anonymous term alone -- the shipped
    form, ``(budget - controller) // WORKER_PEAK_MIB`` over an anon constant --
    and this profile answers one worker wider than its charge fits, so the
    charge assertion below goes red.
    """
    profile = ChargeProfile(worker_anon_mib=700.0, worker_cache_mib=2850.0, controller_mib=1075.0)
    budget = PYTEST_SLICE_HIGH_MIB

    workers = width_within(budget, profile=profile)

    assert workers == 3
    assert profile.charge_mib(workers) <= budget
    assert profile.charge_mib(workers + 1) > budget

    # The anonymous-only model, in both of the forms it shipped in: the
    # measured per-worker anon peak, and the 2263 MiB constant that stood in
    # this module until 2026-09-20. Each picks a width whose real charge
    # against the same ceiling is an overrun, and the wider it is the worse.
    for anon_peak_mib in (profile.worker_anon_mib, 2263.0):
        anon_only = max(1, int((budget - profile.controller_mib) // anon_peak_mib))
        assert anon_only > workers
        assert profile.charge_mib(anon_only) > budget


def test_the_shipped_profile_is_the_charge_the_slice_accounts() -> None:
    """The default width leaves margin under the declared ceiling, at the charge.

    Anti-vacuity: drop ``worker_cache_mib`` from ``MEASURED_CHARGE`` (or set it
    to zero) and the declared width rises to a number whose charge exceeds
    ``PYTEST_SLICE_MEMORY_HIGH_MIB``, which the second assertion catches.
    """
    assert MEASURED_CHARGE.worker_cache_mib > 0, "page cache is part of what memory.high accounts"
    assert MEASURED_CHARGE.charge_mib(CORPUS_MAX_WORKERS) <= PYTEST_SLICE_MEMORY_HIGH_MIB
    assert MEASURED_CHARGE.charge_mib(CORPUS_MAX_WORKERS + 1) > PYTEST_SLICE_MEMORY_HIGH_MIB

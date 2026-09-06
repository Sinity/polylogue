"""A managed pytest run is sized to what its job may take, not what the host has.

Two bounds apply at once. The host's ``MemAvailable`` was the only one read,
so jobs 1793, 1804 and 1836 were killed at eight workers inside the pytest
pool's cgroup while roughly 10 GiB was free on the host -- 1836 while it was
the only pytest task. A killed run measures nothing; a narrower run that
finishes measures everything.

Anti-vacuity:
- make ``cgroup_available_mib`` return ``None`` unconditionally and both
  ``test_the_pytest_slice_bounds_a_host_with_memory_to_spare`` and
  ``test_a_hosted_verify_launch_stays_inside_the_pytest_slice`` go red -- the
  width returns to the host-only one the OOM daemon killed;
- read only ``memory.max`` and ``test_a_soft_ceiling_bounds_as_firmly_as_a_hard_one``
  goes red, which is the production shape: the pytest slice's ``memory.max`` is
  8 GiB and its ``memory.high`` 6 GiB, and systemd-oomd kills on the pressure
  that running above the soft ceiling produces;
- stop walking ancestors and ``test_an_ancestor_slice_bounds_its_children``
  goes red -- an enclosing slice's limit binds this run just as its own does;
- charge the whole of ``memory.current`` and
  ``test_reclaimable_page_cache_is_not_spent_memory`` goes red -- the slice a
  corpus run just left is full of cache, and reading that as spent sizes the
  next run to one worker, which misses the slot timeout;
- drop the ``resize_worker_argument`` call from ``devtools.pytest_slot.main``
  and ``test_the_slot_resizes_the_queued_command`` goes red, which is the case
  that matters: a queued run can wait hours, so a width chosen when the command
  was built describes memory that is no longer there.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

import pytest

from devtools.worker_memory import (
    CONTROLLER_PEAK_MIB,
    CORPUS_MAX_WORKERS,
    MEMORY_HEADROOM_FRACTION,
    WORKER_PEAK_MIB,
    available_memory_mib,
    cgroup_available_mib,
    memory_bounded_worker_cap,
    resize_worker_argument,
)


class CgroupPaths(TypedDict):
    """The two paths every sizing entry point reads its cgroup budget from."""

    process_cgroup: Path
    cgroup_root: Path


MIB = 1024 * 1024
#: The pytest pool's slice as the host declares it, in MiB: a soft ceiling
#: below the hard one, and an enclosing slice that is generous but finite.
PYTEST_SLICE_HIGH_MIB = 6 * 1024
PYTEST_SLICE_MAX_MIB = 8 * 1024
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
    """What a run of ``workers`` costs at peak, controller included."""
    return workers * WORKER_PEAK_MIB + CONTROLLER_PEAK_MIB


def test_an_idle_host_runs_the_full_width(tmp_path: Path) -> None:
    workers, basis = memory_bounded_worker_cap(meminfo=_meminfo(tmp_path, 28000), **_unbounded_cgroup(tmp_path))
    assert workers == CORPUS_MAX_WORKERS
    assert basis["narrowed"] is False
    assert basis["cgroup_available_mib"] is None


def test_a_loaded_host_runs_narrower(tmp_path: Path) -> None:
    """The measured condition of the first killed runs: about 4.7 GiB available."""
    workers, basis = memory_bounded_worker_cap(meminfo=_meminfo(tmp_path, 4707), **_unbounded_cgroup(tmp_path))
    assert workers < CORPUS_MAX_WORKERS
    # The chosen width fits the headroom-adjusted budget; the fixed one does not.
    assert _peak_mib(workers) <= 4707 * 0.8
    assert _peak_mib(CORPUS_MAX_WORKERS) > 4707 * 0.8
    assert basis["basis"] == "mem_available"
    assert basis["narrowed"] is True
    assert basis["available_mib"] == 4707


def test_a_starved_host_still_runs_one_worker(tmp_path: Path) -> None:
    """A slow run beats a killed run; it never resolves to zero workers."""
    workers, _basis = memory_bounded_worker_cap(meminfo=_meminfo(tmp_path, 200), **_unbounded_cgroup(tmp_path))
    assert workers == 1


def test_an_unreadable_meminfo_does_not_narrow_silently(tmp_path: Path) -> None:
    workers, basis = memory_bounded_worker_cap(meminfo=tmp_path / "absent", **_unbounded_cgroup(tmp_path))
    assert workers == CORPUS_MAX_WORKERS
    assert basis["basis"] == "unmeasured"


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


def test_the_pytest_slice_bounds_a_host_with_memory_to_spare(tmp_path: Path) -> None:
    """The killed condition of job 1836: the host was idle, the cgroup was not.

    About 10 GiB available on the host would hold the full width; the pytest
    slice's own budget holds fewer, and it is the slice that kills.
    """
    paths = _pytest_slice(tmp_path, current_mib=350)
    workers, basis = memory_bounded_worker_cap(meminfo=_meminfo(tmp_path, 10000), **paths)
    assert basis["basis"] == "cgroup_budget"
    assert basis["cgroup_available_mib"] == PYTEST_SLICE_HIGH_MIB - 350
    assert basis["host_available_mib"] == 10000
    assert workers < CORPUS_MAX_WORKERS
    # The host alone would have chosen the width that was killed.
    host_only, _ = memory_bounded_worker_cap(meminfo=_meminfo(tmp_path, 10000), **_unbounded_cgroup(tmp_path))
    assert host_only == CORPUS_MAX_WORKERS


def test_the_tighter_bound_decides_when_the_host_is_the_tighter_one(tmp_path: Path) -> None:
    """A roomy cgroup does not license a run the host cannot hold."""
    paths = _pytest_slice(tmp_path, current_mib=350)
    workers, basis = memory_bounded_worker_cap(meminfo=_meminfo(tmp_path, 2500), **paths)
    assert basis["basis"] == "mem_available"
    assert basis["available_mib"] == 2500
    assert _peak_mib(workers) <= 2500 * (1.0 - MEMORY_HEADROOM_FRACTION)


def test_a_hosted_verify_launch_stays_inside_the_pytest_slice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The production launch: the runner's corpus command inside the pytest slice.

    The width it selects has to fit under every limit the slice hierarchy
    carries, with the run's own usage already counted.
    """
    from devtools import verify

    monkeypatch.delenv("POLYLOGUE_PYTEST_WORKERS", raising=False)
    current_mib = 350
    argv = ["python", "-m", "pytest", *verify._pytest_worker_args(maximum=CORPUS_MAX_WORKERS), "tests"]
    assert argv[argv.index("-n") + 1] == str(CORPUS_MAX_WORKERS)

    resized, basis = resize_worker_argument(
        argv, meminfo=_meminfo(tmp_path, 10000), **_pytest_slice(tmp_path, current_mib=current_mib)
    )
    assert basis is not None and basis["narrowed"] is True
    workers = int(resized[resized.index("-n") + 1])
    assert workers == basis["workers"]
    # Inside every ceiling the slice hierarchy declares, usage included.
    assert current_mib + _peak_mib(workers) <= PYTEST_SLICE_HIGH_MIB
    assert current_mib + _peak_mib(workers) <= PYTEST_SLICE_MAX_MIB
    assert 4800 + _peak_mib(workers) <= AGENTCTL_SLICE_HIGH_MIB
    # The width that was killed does not fit the ceiling that killed it.
    assert current_mib + _peak_mib(CORPUS_MAX_WORKERS) > PYTEST_SLICE_HIGH_MIB


def test_resize_narrows_the_worker_argument_in_place(tmp_path: Path) -> None:
    argv = ["python", "-m", "pytest", "--dist=loadgroup", "-n", "8", "tests"]
    resized, basis = resize_worker_argument(argv, meminfo=_meminfo(tmp_path, 4707), **_unbounded_cgroup(tmp_path))
    assert basis is not None and basis["narrowed"] is True
    assert resized[resized.index("-n") + 1] == str(basis["workers"])
    # Only the count changes; the rest of the command is untouched.
    assert resized[: resized.index("-n")] == argv[: argv.index("-n")]
    assert resized[resized.index("-n") + 2 :] == argv[argv.index("-n") + 2 :]


def test_resize_leaves_a_run_that_already_fits(tmp_path: Path) -> None:
    argv = ["python", "-m", "pytest", "-n", "2", "tests"]
    resized, _basis = resize_worker_argument(argv, meminfo=_meminfo(tmp_path, 28000), **_unbounded_cgroup(tmp_path))
    assert resized == argv


@pytest.mark.parametrize(
    "argv", [["pytest", "tests"], ["pytest", "-n", "0"], ["pytest", "-n", "auto"], ["pytest", "-n"]]
)
def test_resize_leaves_commands_it_does_not_understand(argv: list[str], tmp_path: Path) -> None:
    """No xdist, an explicit single process, or a form this does not parse."""
    resized, _basis = resize_worker_argument(
        list(argv), meminfo=_meminfo(tmp_path, 200), **_pytest_slice(tmp_path, current_mib=5000)
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
    paths = _pytest_slice(tmp_path, current_mib=350)
    monkeypatch.setattr(
        slot,
        "resize_worker_argument",
        lambda argv: resize_worker_argument(argv, meminfo=_meminfo(tmp_path, 10000), **paths),
    )
    assert slot.main([str(launch)]) == 0
    assert "from the job cgroup" in log.read_text(encoding="utf-8")

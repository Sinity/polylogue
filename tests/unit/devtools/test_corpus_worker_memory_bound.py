"""The corpus runs at a width the host can actually hold.

Two full corpus runs were killed by the out-of-memory daemon at about 6.2 GB
peak while the operator's desktop and other lanes were resident. A killed run
measures nothing; a narrower run that finishes measures everything.

Anti-vacuity:
- make ``memory_bounded_worker_cap`` return ``requested`` unconditionally and
  ``test_a_loaded_host_runs_narrower`` goes red -- the width returns to the
  fixed one that was killed at the memory that case describes;
- drop the ``resize_worker_argument`` call from ``devtools.pytest_slot.main``
  and ``test_the_slot_resizes_the_queued_command`` goes red, which is the case
  that matters: a queued run can wait hours, so a width chosen when the command
  was built describes memory that is no longer there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools.worker_memory import (
    CONTROLLER_PEAK_MIB,
    CORPUS_MAX_WORKERS,
    WORKER_PEAK_MIB,
    available_memory_mib,
    memory_bounded_worker_cap,
    resize_worker_argument,
)


def _meminfo(tmp_path: Path, available_mib: int) -> Path:
    path = tmp_path / "meminfo"
    path.write_text(
        f"MemTotal:       32689696 kB\nMemFree:         1000000 kB\nMemAvailable:   {available_mib * 1024} kB\n",
        encoding="utf-8",
    )
    return path


def test_an_idle_host_runs_the_full_width(tmp_path: Path) -> None:
    workers, basis = memory_bounded_worker_cap(meminfo=_meminfo(tmp_path, 28000))
    assert workers == CORPUS_MAX_WORKERS
    assert basis["narrowed"] is False


def test_a_loaded_host_runs_narrower(tmp_path: Path) -> None:
    """The measured condition of the killed runs: about 4.7 GiB available."""
    workers, basis = memory_bounded_worker_cap(meminfo=_meminfo(tmp_path, 4707))
    assert workers < CORPUS_MAX_WORKERS
    # The chosen width fits the headroom-adjusted budget; the fixed one does not.
    assert workers * WORKER_PEAK_MIB + CONTROLLER_PEAK_MIB <= 4707 * 0.8
    assert CORPUS_MAX_WORKERS * WORKER_PEAK_MIB + CONTROLLER_PEAK_MIB > 4707 * 0.8
    assert basis["narrowed"] is True
    assert basis["available_mib"] == 4707


def test_a_starved_host_still_runs_one_worker(tmp_path: Path) -> None:
    """A slow run beats a killed run; it never resolves to zero workers."""
    workers, _basis = memory_bounded_worker_cap(meminfo=_meminfo(tmp_path, 200))
    assert workers == 1


def test_an_unreadable_meminfo_does_not_narrow_silently(tmp_path: Path) -> None:
    workers, basis = memory_bounded_worker_cap(meminfo=tmp_path / "absent")
    assert workers == CORPUS_MAX_WORKERS
    assert basis["basis"] == "unmeasured"


@pytest.mark.parametrize("content", ["", "MemTotal: 100 kB\n", "MemAvailable: not-a-number kB\n"])
def test_a_malformed_meminfo_is_unmeasured(tmp_path: Path, content: str) -> None:
    path = tmp_path / "meminfo"
    path.write_text(content, encoding="utf-8")
    assert available_memory_mib(meminfo=path) is None


def test_resize_narrows_the_worker_argument_in_place(tmp_path: Path) -> None:
    argv = ["python", "-m", "pytest", "--dist=loadgroup", "-n", "8", "tests"]
    resized, basis = resize_worker_argument(argv, meminfo=_meminfo(tmp_path, 4707))
    assert basis is not None and basis["narrowed"] is True
    assert resized[resized.index("-n") + 1] == str(basis["workers"])
    # Only the count changes; the rest of the command is untouched.
    assert resized[: resized.index("-n")] == argv[: argv.index("-n")]
    assert resized[resized.index("-n") + 2 :] == argv[argv.index("-n") + 2 :]


def test_resize_leaves_a_run_that_already_fits(tmp_path: Path) -> None:
    argv = ["python", "-m", "pytest", "-n", "2", "tests"]
    resized, _basis = resize_worker_argument(argv, meminfo=_meminfo(tmp_path, 28000))
    assert resized == argv


@pytest.mark.parametrize(
    "argv", [["pytest", "tests"], ["pytest", "-n", "0"], ["pytest", "-n", "auto"], ["pytest", "-n"]]
)
def test_resize_leaves_commands_it_does_not_understand(argv: list[str], tmp_path: Path) -> None:
    """No xdist, an explicit single process, or a form this does not parse."""
    resized, _basis = resize_worker_argument(list(argv), meminfo=_meminfo(tmp_path, 200))
    assert resized == argv


def test_the_slot_resizes_the_queued_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The width is decided when the queued run starts, not when it was built.

    A run can sit in the single-slot pytest queue for hours; the memory that
    matters is the memory present when its workers start.
    """
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
    monkeypatch.setattr(slot.subprocess, "Popen", _popen)
    monkeypatch.setattr(slot, "resize_worker_argument", lambda argv: (argv[:-3] + ["-n", "3", "tests"], None))
    assert slot.main([str(launch)]) == 0
    assert launched["command"][launched["command"].index("-n") + 1] == "3"

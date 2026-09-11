"""A pytest-pool job must not enqueue into the slot it already holds.

Anti-vacuity: drop the ``declared_pool_of_enclosing_job`` leg from
``holds_pytest_slot`` and ``test_a_pytest_pool_job_holds_the_slot_it_runs_in``
goes red -- that is the exact shape that deadlocked ``verify_all`` on
2026-09-05, where the job waited for a single-slot group only it could drain.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from devtools.pytest_slot import declared_pool_of_enclosing_job, holds_pytest_slot


def _fake_proc(root: Path, chain: list[tuple[int, int, list[str]]]) -> Path:
    """Build a /proc-shaped tree: (pid, ppid, argv) innermost first."""
    proc = root / "proc"
    for pid, ppid, argv in chain:
        entry = proc / str(pid)
        entry.mkdir(parents=True)
        (entry / "stat").write_text(f"{pid} (python3.14) S {ppid} 0 0", encoding="utf-8")
        (entry / "cmdline").write_bytes(("\0".join(argv) + "\0").encode("utf-8"))
    return proc


def _launch(path: Path, *, pool: str) -> Path:
    path.write_text(json.dumps({"job_id": "polylogue-verify_all-1", "operation": "verify_all", "pool": pool}))
    return path


def _runner_argv(document: Path) -> list[str]:
    # The installed runner is a wrapper script; its basename is not the bare name.
    return ["/nix/store/x-python3/bin/python3.14", "/nix/store/y/bin/.agentctl-run-wrapped", str(document)]


def test_a_pytest_pool_job_holds_the_slot_it_runs_in(tmp_path: Path) -> None:
    document = _launch(tmp_path / "launch.json", pool="pytest")
    proc = _fake_proc(tmp_path, [(os.getpid(), 42, ["devtools", "verify", "--all"]), (42, 1, _runner_argv(document))])
    # No environment marker at all: this is the process shape the deadlocked
    # job actually had -- no AGENTCTL_/SINNIXD_ variables, no pytest slice.
    assert holds_pytest_slot({}, cgroup_reader=lambda: "", proc=proc) is True


def test_a_normal_pool_job_still_queues(tmp_path: Path) -> None:
    document = _launch(tmp_path / "launch.json", pool="normal")
    proc = _fake_proc(tmp_path, [(os.getpid(), 42, ["devtools", "verify"]), (42, 1, _runner_argv(document))])
    assert holds_pytest_slot({}, cgroup_reader=lambda: "", proc=proc) is False


def test_a_session_shell_outside_any_job_queues(tmp_path: Path) -> None:
    proc = _fake_proc(tmp_path, [(os.getpid(), 42, ["devtools", "verify"]), (42, 1, ["/bin/zsh"])])
    assert holds_pytest_slot({}, cgroup_reader=lambda: "", proc=proc) is False
    assert declared_pool_of_enclosing_job({}, proc=proc) is None


def test_the_explicit_held_marker_still_wins(tmp_path: Path) -> None:
    proc = _fake_proc(tmp_path, [(os.getpid(), 42, ["pytest"]), (42, 1, ["/bin/zsh"])])
    assert holds_pytest_slot({"POLYLOGUE_PYTEST_SLOT": "held"}, cgroup_reader=lambda: "", proc=proc) is True


def test_an_unreadable_or_malformed_launch_document_is_not_a_claim(tmp_path: Path) -> None:
    document = tmp_path / "launch.json"
    document.write_text("{not json")
    proc = _fake_proc(tmp_path, [(os.getpid(), 42, ["devtools", "verify"]), (42, 1, _runner_argv(document))])
    assert holds_pytest_slot({}, cgroup_reader=lambda: "", proc=proc) is False


def test_the_pool_is_read_through_intermediate_processes(tmp_path: Path) -> None:
    """The runner is an ancestor, not necessarily the parent."""
    document = _launch(tmp_path / "launch.json", pool="pytest")
    proc = _fake_proc(
        tmp_path,
        [
            (os.getpid(), 42, ["devtools", "verify", "--all"]),
            (42, 43, ["nix", "develop", "--command", "devtools"]),
            (43, 1, _runner_argv(document)),
        ],
    )
    assert holds_pytest_slot({}, cgroup_reader=lambda: "", proc=proc) is True

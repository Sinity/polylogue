"""Quick verification must not create a persistent type daemon."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from devtools import gate, mypy_gate


def test_quick_mypy_uses_a_foreground_checkout_local_process() -> None:
    """The checker stays in the managed task's process tree and exits with it.

    Anti-vacuity: changing this to ``dmypy`` makes the assertion red. A daemon
    is reparented outside the managed task and one accumulates per checkout.
    """
    assert gate.mypy_command() == [str(gate.ROOT / ".venv/bin/python"), "-m", "devtools.mypy_gate"]


def test_mypy_command_isolated_by_checkout(tmp_path: Path) -> None:
    """Each lane resolves its own environment through the checkout interpreter."""
    assert gate.mypy_command(root=tmp_path) == [str(tmp_path / ".venv/bin/python"), "-m", "devtools.mypy_gate"]


def test_shared_gate_uses_one_cache_and_returns_checker_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checker = tmp_path / ".venv/bin/mypy"
    checker.parent.mkdir(parents=True)
    checker.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
    checker.chmod(0o755)
    common = tmp_path / ".git"
    common.mkdir()
    calls: list[list[str]] = []

    monkeypatch.setattr(mypy_gate, "_git_common_dir", lambda _root: common)

    def fake_run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(list(argv))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr("devtools.mypy_gate.subprocess.run", fake_run)

    assert mypy_gate.main(["--root", str(tmp_path)]) == 7
    assert calls == [[str(checker), "--cache-dir", str(common / "polylogue-mypy/cache")]]
    assert (common / "polylogue-mypy/lock").is_file()


@pytest.mark.load_sensitive
def test_sibling_worktrees_serialize_on_one_lock_and_share_one_cache(tmp_path: Path) -> None:
    """Two lanes of one checkout never type-check at the same time.

    This is the shape polylogue-r2lud reported: batch workers in sibling
    worktrees each started a cold mypy scan, so the same analysis ran N times
    under I/O contention. The gate takes an exclusive lock on the *common*
    git dir and points every lane at one cache directory, so the second lane
    waits and then reuses the first lane's module results.

    The checker here is a stub that claims exclusive entry with ``mkdir`` and
    holds it briefly, so a lane that runs concurrently exits 9 rather than 0.

    Anti-vacuity: removing the ``fcntl.flock`` from ``devtools/mypy_gate.py``
    lets both stubs run inside the same window and the second exits 9;
    replacing ``--cache-dir`` with a per-worktree path makes the two recorded
    cache directories differ. Either mutation turns this red.
    """
    primary = tmp_path / "primary"
    primary.mkdir()

    def git(cwd: Path, *args: str) -> None:
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)

    git(primary, "init")
    git(primary, "config", "user.name", "Fixture")
    git(primary, "config", "user.email", "fixture@example.test")
    (primary / "seed.txt").write_text("seed\n", encoding="utf-8")
    git(primary, "add", "seed.txt")
    git(primary, "commit", "-m", "Seed")
    secondary = tmp_path / "secondary"
    git(primary, "worktree", "add", str(secondary), "-b", "lane")

    observed = tmp_path / "observed"
    observed.mkdir()
    for lane in (primary, secondary):
        checker = lane / ".venv" / "bin" / "mypy"
        checker.parent.mkdir(parents=True)
        checker.write_text(
            "#!/bin/sh\n"
            f'if ! mkdir "{observed}/busy" 2>/dev/null; then exit 9; fi\n'
            f'echo "$2" >> "{observed}/cache-dirs"\n'
            "sleep 0.5\n"
            f'rmdir "{observed}/busy"\n'
            "exit 0\n",
            encoding="utf-8",
        )
        checker.chmod(0o755)

    repository_root = Path(mypy_gate.__file__).parents[1]
    environment = {**os.environ, "PYTHONPATH": str(repository_root)}
    lanes = [
        subprocess.Popen(
            [sys.executable, "-m", "devtools.mypy_gate", "--root", str(lane)],
            cwd=str(lane),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for lane in (primary, secondary)
    ]
    statuses = [process.wait(timeout=120) for process in lanes]

    assert statuses == [0, 0], [process.communicate() for process in lanes]
    recorded = (observed / "cache-dirs").read_text(encoding="utf-8").split()
    assert len(recorded) == 2
    assert recorded[0] == recorded[1]
    assert Path(recorded[0]) == (primary / ".git" / "polylogue-mypy" / "cache")

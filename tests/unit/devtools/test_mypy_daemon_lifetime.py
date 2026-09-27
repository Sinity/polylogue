"""Quick verification must not create a persistent type daemon."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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


def _stub_checker(lane: Path, body: str) -> None:
    checker = lane / ".venv" / "bin" / "mypy"
    checker.parent.mkdir(parents=True, exist_ok=True)
    checker.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    checker.chmod(0o755)


def test_gate_checks_on_its_checkout_cache_and_publishes_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The checker's status is the gate's; a complete cache becomes the shared seed.

    Anti-vacuity: point ``--cache-dir`` at the shared cache and the recorded
    directory is not the checkout's; drop ``_publish`` and the shared seed
    never receives the entry the checker wrote.
    """
    common = tmp_path / ".git"
    common.mkdir()
    monkeypatch.setattr(mypy_gate, "_git_common_dir", lambda _root: common)
    shared = common / "polylogue-mypy" / "cache"
    shared.mkdir(parents=True)
    (shared / "seed.db").write_text("seed", encoding="utf-8")
    # mypy exits 1 on type errors with a complete cache; that cache is published.
    _stub_checker(tmp_path, 'echo "$2" > "$2/checked-by"\nexit 1\n')

    assert mypy_gate.main(["--root", str(tmp_path)]) == 1

    local = tmp_path / ".cache" / "mypy"
    assert (local / "seed.db").read_text(encoding="utf-8") == "seed"
    assert (local / "checked-by").read_text(encoding="utf-8").strip() == str(local)
    assert (shared / "checked-by").is_file()


def test_a_crashed_check_does_not_publish_its_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: publish on every exit and the partial entry reaches the seed."""
    common = tmp_path / ".git"
    common.mkdir()
    monkeypatch.setattr(mypy_gate, "_git_common_dir", lambda _root: common)
    shared = common / "polylogue-mypy" / "cache"
    shared.mkdir(parents=True)
    (shared / "seed.db").write_text("seed", encoding="utf-8")
    _stub_checker(tmp_path, 'echo partial > "$2/partial"\nexit 2\n')

    assert mypy_gate.main(["--root", str(tmp_path)]) == 2

    assert not (shared / "partial").exists()
    assert (shared / "seed.db").is_file()


def _worktrees(tmp_path: Path, count: int) -> list[Path]:
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
    lanes = [primary]
    for index in range(1, count):
        lane = tmp_path / f"lane{index}"
        git(primary, "worktree", "add", str(lane), "-b", f"lane{index}")
        lanes.append(lane)
    return lanes


def _run_lanes(lanes: list[Path]) -> list[int]:
    repository_root = Path(mypy_gate.__file__).parents[1]
    environment = {**os.environ, "PYTHONPATH": str(repository_root)}
    processes = [
        subprocess.Popen(
            [sys.executable, "-m", "devtools.mypy_gate", "--root", str(lane)],
            cwd=str(lane),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for lane in lanes
    ]
    statuses = [process.wait(timeout=120) for process in processes]
    assert all(status == 0 for status in statuses), [process.communicate() for process in processes]
    return statuses


@pytest.mark.load_sensitive
def test_warm_sibling_worktrees_check_side_by_side(tmp_path: Path) -> None:
    """Seeded siblings do not queue behind one another.

    Each stub records entry and waits until every sibling has entered, so a
    gate that serializes the checks never lets the first one finish and the
    lanes time out.

    Anti-vacuity: hold the lock across the warm check and no lane sees its
    siblings enter; the stubs give up and exit 9.
    """
    lanes = _worktrees(tmp_path, 3)
    shared = lanes[0] / ".git" / "polylogue-mypy" / "cache"
    shared.mkdir(parents=True)
    (shared / "seed.db").write_text("seed", encoding="utf-8")
    observed = tmp_path / "observed"
    observed.mkdir()
    for index, lane in enumerate(lanes):
        _stub_checker(
            lane,
            f'touch "{observed}/entered-{index}"\n'
            "i=0\n"
            f'while [ "$(ls "{observed}" | wc -l)" -lt {len(lanes)} ]; do\n'
            "  i=$((i+1)); [ $i -gt 200 ] && exit 9\n"
            "  sleep 0.05\n"
            "done\n"
            "exit 0\n",
        )

    _run_lanes(lanes)

    for lane in lanes:
        assert (lane / ".cache" / "mypy" / "seed.db").is_file()


@pytest.mark.load_sensitive
def test_cold_siblings_run_one_cold_scan_and_seed_from_it(tmp_path: Path) -> None:
    """With no shared cache, exactly one sibling scans cold; the rest seed from it.

    This is the polylogue-r2lud shape: several full cold analyses at once. The
    stub counts a cold scan as a check whose cache directory lacks the entry a
    completed scan writes.

    Anti-vacuity: run the cold scan outside the lock and the siblings scan
    cold together, so the recorded cold count exceeds one.
    """
    lanes = _worktrees(tmp_path, 3)
    observed = tmp_path / "observed"
    observed.mkdir()
    for lane in lanes:
        _stub_checker(
            lane,
            f'if [ ! -f "$2/scanned.db" ]; then echo cold >> "{observed}/cold"; sleep 0.5; fi\n'
            'echo done > "$2/scanned.db"\n'
            "exit 0\n",
        )

    _run_lanes(lanes)

    assert (observed / "cold").read_text(encoding="utf-8").split() == ["cold"]
    for lane in lanes:
        assert (lane / ".cache" / "mypy" / "scanned.db").is_file()

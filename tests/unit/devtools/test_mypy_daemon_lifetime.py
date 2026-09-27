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


def _stamped_seed(shared: Path, root: Path) -> None:
    """A complete shared cache for *root*'s default inputs."""
    shared.mkdir(parents=True)
    (shared / "seed.db").write_text("seed", encoding="utf-8")
    (shared / mypy_gate._STAMP).write_text(mypy_gate._input_key(root, []), encoding="utf-8")


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
    _stamped_seed(shared, tmp_path)
    # mypy exits 1 on type errors with a complete cache; that cache is published.
    _stub_checker(tmp_path, 'echo "$2" > "$2/checked-by"\nexit 1\n')

    assert mypy_gate.main(["--root", str(tmp_path)]) == 1

    local = tmp_path / ".cache" / "mypy"
    assert (local / "seed.db").read_text(encoding="utf-8") == "seed"
    assert (local / "checked-by").read_text(encoding="utf-8").strip() == str(local)
    assert (shared / "checked-by").is_file()
    assert mypy_gate._is_complete(local, mypy_gate._input_key(tmp_path, []))


def test_a_crashed_check_does_not_publish_its_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: publish on every exit and the partial entry reaches the seed."""
    common = tmp_path / ".git"
    common.mkdir()
    monkeypatch.setattr(mypy_gate, "_git_common_dir", lambda _root: common)
    shared = common / "polylogue-mypy" / "cache"
    _stamped_seed(shared, tmp_path)
    _stub_checker(tmp_path, 'echo partial > "$2/partial"\nexit 2\n')

    assert mypy_gate.main(["--root", str(tmp_path)]) == 2

    assert not (shared / "partial").exists()
    assert (shared / "seed.db").is_file()
    # The crashed checkout's own cache is not complete either: the next run re-checks.
    assert not mypy_gate._is_complete(tmp_path / ".cache" / "mypy", mypy_gate._input_key(tmp_path, []))


def test_an_unstamped_shared_cache_is_not_a_seed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A partial or differently configured cache is cold, not warm.

    Anti-vacuity: treat any non-empty directory as warm and the checkout seeds
    from the partial cache, so ``partial.db`` reaches the checkout.
    """
    common = tmp_path / ".git"
    common.mkdir()
    monkeypatch.setattr(mypy_gate, "_git_common_dir", lambda _root: common)
    shared = common / "polylogue-mypy" / "cache"
    shared.mkdir(parents=True)
    (shared / "partial.db").write_text("interrupted", encoding="utf-8")
    (shared / mypy_gate._STAMP).write_text(
        mypy_gate._input_key(tmp_path, ["--python-version", "3.12"]), encoding="utf-8"
    )
    _stub_checker(tmp_path, "exit 0\n")

    assert mypy_gate.main(["--root", str(tmp_path)]) == 0

    assert not (tmp_path / ".cache" / "mypy" / "partial.db").exists()


def test_seeding_preserves_cache_file_timestamps(tmp_path: Path) -> None:
    """mypy's filesystem cache rejects a data file whose mtime moved.

    Anti-vacuity: drop ``--preserve=timestamps`` from the copy and the seeded
    file carries the copy time.
    """
    shared = tmp_path / "shared"
    shared.mkdir()
    data = shared / "module.data.json"
    data.write_text("{}", encoding="utf-8")
    os.utime(data, ns=(1_000_000_000_000_000_000, 1_000_000_000_000_000_000))
    local = tmp_path / "checkout" / ".cache" / "mypy"

    mypy_gate._seed(local, shared)

    assert (local / "module.data.json").stat().st_mtime_ns == 1_000_000_000_000_000_000


def test_two_gates_in_one_checkout_do_not_share_a_running_cache(tmp_path: Path) -> None:
    """Gates in the same checkout take turns on its cache.

    Anti-vacuity: drop the checkout-local lock and both stubs hold the cache at
    once, so one exits 9.
    """
    lane = _worktrees(tmp_path, 1)[0]
    _stamped_seed(lane / ".git" / "polylogue-mypy" / "cache", lane)
    _stub_checker(
        lane,
        'if ! mkdir "$2/busy" 2>/dev/null; then exit 9; fi\nsleep 0.5\nrmdir "$2/busy"\nexit 0\n',
    )

    _run_lanes([lane, lane])


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
    _stamped_seed(shared, lanes[0])
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


def test_arguments_that_bypass_the_cache_run_unmanaged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``--no-incremental`` produces no reusable cache, so nothing is stamped or published.

    Anti-vacuity: manage such a run like any other and its empty cache is
    stamped complete and published as the shared seed.
    """
    common = tmp_path / ".git"
    common.mkdir()
    monkeypatch.setattr(mypy_gate, "_git_common_dir", lambda _root: common)
    _stub_checker(tmp_path, 'echo "$@" > "$(dirname "$0")/argv"\nexit 0\n')

    assert mypy_gate.main(["--root", str(tmp_path), "--no-incremental"]) == 0

    assert (tmp_path / ".venv" / "bin" / "argv").read_text(encoding="utf-8").split() == ["--no-incremental"]
    assert not (tmp_path / ".cache" / "mypy").exists()
    assert not (common / "polylogue-mypy" / "cache").exists()


def test_publishing_reclaims_abandoned_staging_copies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A gate killed mid-copy leaves a staging directory the next publisher removes.

    Anti-vacuity: remove only this process's own staging path and the
    abandoned copy survives.
    """
    common = tmp_path / ".git"
    common.mkdir()
    monkeypatch.setattr(mypy_gate, "_git_common_dir", lambda _root: common)
    shared = common / "polylogue-mypy" / "cache"
    _stamped_seed(shared, tmp_path)
    abandoned = common / "polylogue-mypy" / "cache.publish-1"
    abandoned.mkdir()
    (abandoned / "partial.db").write_text("partial", encoding="utf-8")
    _stub_checker(tmp_path, "exit 0\n")

    assert mypy_gate.main(["--root", str(tmp_path)]) == 0

    assert not abandoned.exists()


def test_the_mypy_configuration_is_part_of_the_cache_key(tmp_path: Path) -> None:
    """A sibling that narrowed ``[tool.mypy].files`` does not seed a full checkout.

    Anti-vacuity: key only the version and argv and the two keys are equal.
    """
    narrowed = tmp_path / "narrowed"
    full = tmp_path / "full"
    for root, files in ((narrowed, '["polylogue/core"]'), (full, '["polylogue"]')):
        root.mkdir()
        (root / "pyproject.toml").write_text(f"[tool.mypy]\nfiles = {files}\n", encoding="utf-8")

    assert mypy_gate._input_key(narrowed, []) != mypy_gate._input_key(full, [])


def test_an_explicit_config_file_is_part_of_the_cache_key(tmp_path: Path) -> None:
    """Two checkouts passing the same ``--config-file`` name with different contents differ.

    Anti-vacuity: key only the default ``pyproject.toml`` table and the argv,
    and the two keys are equal.
    """
    narrowed = tmp_path / "narrowed"
    full = tmp_path / "full"
    for root, files in ((narrowed, "polylogue/core"), (full, "polylogue")):
        root.mkdir()
        (root / "mypy-ci.ini").write_text(f"[mypy]\nfiles = {files}\n", encoding="utf-8")
    args = ["--config-file", "mypy-ci.ini"]

    assert mypy_gate._input_key(narrowed, args) != mypy_gate._input_key(full, args)


def test_a_swap_interrupted_between_renames_restores_the_shared_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A publisher killed after retiring the old cache leaves it recoverable, not cold.

    Anti-vacuity: decide coldness without restoring the retired copy and the
    checkout scans cold, so ``seed.db`` never reaches it.
    """
    common = tmp_path / ".git"
    common.mkdir()
    monkeypatch.setattr(mypy_gate, "_git_common_dir", lambda _root: common)
    retired = common / "polylogue-mypy" / "cache.retired-4242"
    _stamped_seed(retired, tmp_path)
    _stub_checker(tmp_path, "exit 0\n")

    assert mypy_gate.main(["--root", str(tmp_path)]) == 0

    assert (tmp_path / ".cache" / "mypy" / "seed.db").read_text(encoding="utf-8") == "seed"

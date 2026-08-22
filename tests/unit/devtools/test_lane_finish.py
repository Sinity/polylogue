"""Tests for ``devtools workspace lane-finish``."""

from __future__ import annotations

import subprocess
from pathlib import Path

from devtools.lane_finish import finish_lane


def _git(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True)


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "base.txt").write_text("base\n")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    lane = tmp_path / "lane"
    _git(repo, "worktree", "add", "-b", "worktree-agent-test", str(lane), "master")
    return repo, lane


def test_finish_packages_exact_delta_and_unlocks_lane(tmp_path: Path) -> None:
    repo, lane = _repo(tmp_path)
    (lane / "change.txt").write_text("change\n")
    _git(lane, "add", "change.txt")
    _git(lane, "commit", "-m", "change")
    _git(repo, "worktree", "lock", "--reason", "active test lane", str(lane))

    report = finish_lane(lane, base="master")

    assert report.status == "ready-for-assimilation"
    assert report.branch == "worktree-agent-test"
    assert report.head == _git(lane, "rev-parse", "HEAD").stdout.strip()
    assert report.commits == [report.head]
    assert report.changed_paths == ["change.txt"]
    assert report.unlocked is True
    porcelain = _git(repo, "worktree", "list", "--porcelain").stdout
    lane_block = next(block for block in porcelain.split("\n\n") if f"worktree {lane}" in block)
    assert "locked" not in lane_block


def test_finish_refuses_dirty_lane_and_preserves_lock(tmp_path: Path) -> None:
    repo, lane = _repo(tmp_path)
    (lane / "dirty.txt").write_text("not committed\n")
    _git(repo, "worktree", "lock", "--reason", "active test lane", str(lane))

    report = finish_lane(lane, base="master")

    assert report.status == "blocked"
    assert report.error is not None and "uncommitted changes" in report.error
    assert report.unlocked is False
    porcelain = _git(repo, "worktree", "list", "--porcelain").stdout
    lane_block = next(block for block in porcelain.split("\n\n") if f"worktree {lane}" in block)
    assert "locked active test lane" in lane_block


def test_finish_reports_clean_read_only_lane(tmp_path: Path) -> None:
    _repo_path, lane = _repo(tmp_path)

    report = finish_lane(lane, base="master")

    assert report.status == "no-changes"
    assert report.commits == []
    assert report.changed_paths == []

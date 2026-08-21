"""Focused tests for the explicit workspace lane integration boundary."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from devtools.workspace_integrate import integrate, main


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _commit(cwd: Path, message: str) -> str:
    _git(cwd, "commit", "-m", message)
    return _git(cwd, "rev-parse", "HEAD")


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "base.txt").write_text("base\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    _git(repo, "branch", "lane-a")
    _git(repo, "branch", "lane-b")
    target = tmp_path / "target"
    _git(repo, "worktree", "add", "-b", "integration", str(target), "master")
    return repo, target


def test_integrate_derives_ordered_ranges_and_reports_applied_shas(tmp_path: Path) -> None:
    repo, target = _repo(tmp_path)
    _git(repo, "switch", "lane-a")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", ".")
    a = _commit(repo, "lane a")
    _git(repo, "switch", "lane-b")
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", ".")
    b = _commit(repo, "lane b")

    report = integrate(target, ["lane-a", "lane-b"])

    assert report.status == "applied"
    assert report.applied_commits == [a, b]
    assert (target / "a.txt").read_text() == "a\n"
    assert (target / "b.txt").read_text() == "b\n"


def test_ambiguous_source_requires_explicit_commit(tmp_path: Path) -> None:
    repo, target = _repo(tmp_path)
    _git(repo, "switch", "lane-a")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", ".")
    a = _commit(repo, "lane a")
    (target / "target.txt").write_text("target\n")
    _git(target, "add", ".")
    _git(target, "commit", "-m", "target divergence")

    report = integrate(target, ["lane-a"])
    assert report.status == "blocked"
    assert report.error is not None and "ambiguous" in report.error
    assert integrate(target, explicit_commits=[a]).status == "applied"


def test_conflict_stops_in_place_and_json_reports_state(tmp_path: Path, capsys: object) -> None:
    repo, target = _repo(tmp_path)
    _git(repo, "switch", "lane-a")
    (repo / "base.txt").write_text("lane\n")
    _git(repo, "add", ".")
    commit = _commit(repo, "conflicting lane")
    (target / "base.txt").write_text("target\n")
    _git(target, "add", ".")
    _git(target, "commit", "-m", "conflicting target")

    assert main(["--target", str(target), "--commit", commit, "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "conflict"
    assert payload["conflict"] is True
    assert payload["applied_commits"] == []
    assert _git(target, "rev-parse", "--verify", "CHERRY_PICK_HEAD")

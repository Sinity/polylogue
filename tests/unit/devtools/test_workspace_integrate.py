"""Focused tests for the explicit workspace lane integration boundary."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import devtools.workspace_integrate as integration_module
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


def test_conflict_stops_in_place_and_json_reports_state(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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


def test_target_must_be_exact_linked_worktree_root(tmp_path: Path) -> None:
    repo, target = _repo(tmp_path)
    nested = target / "nested"
    nested.mkdir()
    report = integrate(nested, explicit_commits=[_git(repo, "rev-parse", "HEAD")])
    assert report.status == "blocked"
    assert report.error is not None and "top-level" in report.error


def test_overlapping_parent_and_child_ranges_are_deduplicated(tmp_path: Path) -> None:
    repo, target = _repo(tmp_path)
    _git(repo, "switch", "lane-a")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", ".")
    a = _commit(repo, "lane a")
    _git(repo, "branch", "lane-child")
    _git(repo, "switch", "lane-child")
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", ".")
    b = _commit(repo, "lane b")

    report = integrate(target, ["lane-a", "lane-child"])

    assert report.status == "applied"
    assert report.planned_commits == [a, b]
    assert report.applied_commits == [a, b]


@pytest.mark.parametrize("marker", ["CHERRY_PICK_HEAD", "MERGE_HEAD", "REVERT_HEAD", "REBASE_HEAD"])
def test_active_git_operation_is_rejected_even_with_clean_status(tmp_path: Path, marker: str) -> None:
    repo, target = _repo(tmp_path)
    marker_path = Path(_git(target, "rev-parse", "--git-path", marker))
    if not marker_path.is_absolute():
        marker_path = target / marker_path
    marker_path = marker_path.resolve()
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    if marker == "REBASE_HEAD":
        marker_path.write_text("active\n")
    else:
        marker_path.write_text(_git(repo, "rev-parse", "HEAD") + "\n")
    report = integrate(target, explicit_commits=[_git(repo, "rev-parse", "HEAD")])
    assert report.status == "blocked"
    assert report.error is not None and "active Git operation" in report.error


def test_sequencer_operation_is_rejected(tmp_path: Path) -> None:
    repo, target = _repo(tmp_path)
    sequencer = Path(_git(target, "rev-parse", "--git-path", "sequencer"))
    if not sequencer.is_absolute():
        sequencer = target / sequencer
    sequencer = sequencer.resolve()
    sequencer.mkdir(parents=True)
    (sequencer / "todo").write_text("pick active\n")
    report = integrate(target, explicit_commits=[_git(repo, "rev-parse", "HEAD")])
    assert report.status == "blocked"
    assert report.error is not None and "active Git operation" in report.error


def test_mixing_positional_and_source_options_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo, target = _repo(tmp_path)
    assert main(["--target", str(target), "lane-a", "--source", "lane-b", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["error"] is not None and "cannot mix" in payload["error"]


def test_empty_pick_is_not_reported_as_content_conflict(tmp_path: Path) -> None:
    repo, target = _repo(tmp_path)
    commit = _git(repo, "rev-parse", "HEAD")
    report = integrate(target, explicit_commits=[commit])
    assert report.status == "empty"
    assert report.empty_pick is True
    assert report.conflict is False
    assert report.error is not None and "no changes" in report.error


def test_git_timeout_reports_failure_and_inspects_operation_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, target = _repo(tmp_path)
    commit = _git(repo, "rev-parse", "HEAD")
    original_git = integration_module.__dict__["_git"]

    def timeout_on_cherry_pick(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
        if args[:1] == ("cherry-pick",):
            marker = Path(original_git(path, "rev-parse", "--git-path", "CHERRY_PICK_HEAD").stdout.strip())
            if not marker.is_absolute():
                marker = path / marker
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(commit + "\n")
            raise subprocess.TimeoutExpired(["git", *args], timeout=30)
        return original_git(path, *args)

    monkeypatch.setattr(integration_module, "_git", timeout_on_cherry_pick)
    report = integration_module.integrate(target, explicit_commits=[commit])
    assert report.status == "timeout"
    assert report.conflict is False
    assert report.error is not None and "timed out" in report.error
    assert report.conflict_head is None
    assert report.active_operation == "CHERRY_PICK_HEAD"

"""verify-worktree must catch the isolation-escape and stale-beads failure modes.

The production dependency exercised is real git: tests build an actual main
checkout plus a `git worktree add` linked worktree, then assert the check
distinguishes them. Mutation that would fail these tests: dropping the
git-dir vs git-common-dir comparison (main checkout would report ok),
dropping the branch comparison, or dropping the jsonl updated_at comparison.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from devtools.verify_worktree import _max_updated_at, inspect_worktree, main

_GIT_ENV_ARGS = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *_GIT_ENV_ARGS, "-C", str(cwd), *args], check=True, capture_output=True, text=True)


@pytest.fixture
def repo_pair(tmp_path: Path) -> tuple[Path, Path]:
    """A real main checkout and a linked worktree on branch lane-branch."""
    main_repo = tmp_path / "main"
    main_repo.mkdir()
    _git(main_repo, "init", "-b", "master")
    (main_repo / "file.txt").write_text("hello\n")
    _git(main_repo, "add", "file.txt")
    _git(main_repo, "commit", "-m", "initial", "--no-gpg-sign")
    worktree = tmp_path / "lane"
    _git(main_repo, "worktree", "add", "-b", "lane-branch", str(worktree))
    return main_repo, worktree


def test_linked_worktree_passes_hard_checks(repo_pair: tuple[Path, Path]) -> None:
    _, worktree = repo_pair
    report = inspect_worktree(worktree, expect_branch="lane-branch")
    assert not report.hard_failures()
    assert report.branch == "lane-branch"


def test_main_checkout_fails_isolation_check(repo_pair: tuple[Path, Path]) -> None:
    main_repo, _ = repo_pair
    report = inspect_worktree(main_repo)
    names = {c.name for c in report.hard_failures()}
    assert "linked_worktree" in names
    assert "distinct_toplevel" in names


def test_wrong_branch_is_a_hard_failure(repo_pair: tuple[Path, Path]) -> None:
    _, worktree = repo_pair
    report = inspect_worktree(worktree, expect_branch="feature/other")
    assert any(c.name == "branch_matches" for c in report.hard_failures())


def test_missing_path_fails(tmp_path: Path) -> None:
    report = inspect_worktree(tmp_path / "does-not-exist")
    assert any(c.name == "path_exists" for c in report.hard_failures())


def test_non_git_directory_fails(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    report = inspect_worktree(plain)
    assert any(c.name == "inside_git_repo" for c in report.hard_failures())


def test_dirty_worktree_is_advisory_not_hard(repo_pair: tuple[Path, Path]) -> None:
    _, worktree = repo_pair
    (worktree / "uncommitted.txt").write_text("wip\n")
    report = inspect_worktree(worktree, expect_branch="lane-branch")
    assert not report.hard_failures()
    assert any(c.name == "committed_state" for c in report.advisory_failures())


def test_stale_beads_jsonl_is_flagged(repo_pair: tuple[Path, Path]) -> None:
    main_repo, worktree = repo_pair
    (main_repo / ".beads").mkdir()
    (worktree / ".beads").mkdir()
    (main_repo / ".beads" / "issues.jsonl").write_text(
        json.dumps({"id": "x-1", "updated_at": "2026-08-01T10:00:00Z"}) + "\n"
    )
    (worktree / ".beads" / "issues.jsonl").write_text(
        json.dumps({"id": "x-1", "updated_at": "2026-07-30T10:00:00Z"}) + "\n"
    )
    report = inspect_worktree(worktree)
    stale = [c for c in report.advisory_failures() if c.name == "beads_freshness"]
    assert stale and "STALE" in stale[0].detail


def test_fresh_beads_jsonl_passes(repo_pair: tuple[Path, Path]) -> None:
    main_repo, worktree = repo_pair
    (main_repo / ".beads").mkdir()
    (worktree / ".beads").mkdir()
    same = json.dumps({"id": "x-1", "updated_at": "2026-08-01T10:00:00Z"}) + "\n"
    (main_repo / ".beads" / "issues.jsonl").write_text(same)
    (worktree / ".beads" / "issues.jsonl").write_text(same)
    report = inspect_worktree(worktree)
    assert not [c for c in report.advisory_failures() if c.name == "beads_freshness"]


def test_max_updated_at_skips_garbage_lines(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    path.write_text(
        "not json\n"
        + json.dumps({"id": "a", "updated_at": "2026-01-01T00:00:00Z"})
        + "\n"
        + json.dumps({"id": "b", "updated_at": "2026-06-01T00:00:00Z"})
        + "\n"
        + json.dumps({"id": "c"})
        + "\n"
    )
    assert _max_updated_at(path) == "2026-06-01T00:00:00Z"


def test_main_exit_codes_and_json(repo_pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    main_repo, worktree = repo_pair
    assert main([str(worktree), "--expect-branch", "lane-branch", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["branch"] == "lane-branch"

    assert main([str(main_repo), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False


def test_strict_promotes_advisory_to_failure(repo_pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    _, worktree = repo_pair
    (worktree / "uncommitted.txt").write_text("wip\n")
    assert main([str(worktree)]) == 0
    capsys.readouterr()
    assert main([str(worktree), "--strict"]) == 1

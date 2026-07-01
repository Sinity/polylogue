"""Tests for ``devtools worktree-gc``."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from devtools.worktree_gc import (
    GcCandidate,
    WorktreeEntry,
    _build_payload,
    _resolve_target,
    apply_removals,
    check_dirty,
    classify_candidates,
    collect_candidates,
    main,
    parse_worktree_list,
)

# ── porcelain parsing ──────────────────────────────────────────────

PORCELAIN_SAMPLE = """\
worktree /home/user/repo
HEAD abc123def456789
branch refs/heads/master
bare

worktree /home/user/repo-feat
HEAD def456789abc123
branch refs/heads/feature/foo

worktree /home/user/repo-detached
HEAD fff111222333444
detached

worktree /home/user/repo-locked
HEAD aaa111bbb222ccc
branch refs/heads/feature/locked
locked
"""


def test_parse_worktree_list() -> None:
    entries = parse_worktree_list(PORCELAIN_SAMPLE)

    assert len(entries) == 4

    master = entries[0]
    assert master.path == Path("/home/user/repo")
    assert master.branch == "refs/heads/master"
    assert master.bare is True

    feat = entries[1]
    assert feat.path == Path("/home/user/repo-feat")
    assert feat.branch == "refs/heads/feature/foo"
    assert feat.bare is False

    detached = entries[2]
    assert detached.path == Path("/home/user/repo-detached")
    assert detached.detached is True
    assert detached.branch is None

    locked = entries[3]
    assert locked.locked is True


def test_parse_worktree_list_empty() -> None:
    assert parse_worktree_list("") == []


# ── dirty check ────────────────────────────────────────────────────


def test_check_dirty_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """check_dirty returns False for a clean git repo."""
    repo = _make_repo(tmp_path / "repo")
    # Worktree entry pointing at the repo itself won't be dirty
    # because status --porcelain returns empty for clean repo.

    # Simulate by checking the repo itself — it's clean after init+commit.
    assert check_dirty(repo) is False


def test_check_dirty_true(tmp_path: Path) -> None:
    """check_dirty returns True when there are uncommitted changes."""
    repo = _make_repo(tmp_path / "repo")
    (repo / "new-file.txt").write_text("hello")
    assert check_dirty(repo) is True


def test_check_dirty_missing_path(tmp_path: Path) -> None:
    """check_dirty returns False for non-existent paths (prunable)."""
    assert check_dirty(tmp_path / "nonexistent") is False


def test_default_target_prefers_origin_master(tmp_path: Path) -> None:
    """Repo workflows integrate through origin/master, not stale local master."""
    repo = _make_repo(tmp_path / "repo")
    remote = tmp_path / "remote.git"
    _run_git(["init", "--bare", str(remote)], cwd=tmp_path)
    _run_git(["remote", "add", "origin", str(remote)], cwd=repo)
    _run_git(["push", "-u", "origin", "master"], cwd=repo)

    assert _resolve_target(repo, None) == "origin/master"
    assert _resolve_target(repo, "master") == "master"


# ── classification ─────────────────────────────────────────────────


def _entry(**kw: Any) -> WorktreeEntry:
    defaults: dict[str, Any] = {
        "path": Path("/tmp/worktrees/feat"),
        "head": "abc123",
        "branch": "refs/heads/feature/foo",
    }
    defaults.update(kw)
    return WorktreeEntry(**defaults)


def test_classify_merged_safe() -> None:
    c = classify_candidates(
        [_entry(branch="refs/heads/feature/merged")],
        repo_root=Path("/tmp/repo"),
        merged={"refs/heads/feature/merged", "refs/heads/master"},
        existing={"refs/heads/feature/merged", "refs/heads/master"},
    )
    assert len(c) == 1
    assert c[0].reason == "merged"
    assert c[0].safe is True
    assert c[0].action == "remove"


def test_classify_unmerged_blocked() -> None:
    c = classify_candidates(
        [_entry(branch="refs/heads/feature/active")],
        repo_root=Path("/tmp/repo"),
        merged={"refs/heads/master"},
        existing={"refs/heads/feature/active", "refs/heads/master"},
    )
    assert c[0].reason == "unmerged"
    assert c[0].safe is False
    assert c[0].blocked_reason == "branch-not-merged"


def test_classify_branch_deleted_safe(tmp_path: Path) -> None:
    """Worktree whose branch ref no longer exists = safe to remove."""
    repo = _make_repo(tmp_path / "repo")
    c = classify_candidates(
        [_entry(branch="refs/heads/gone", path=repo)],
        repo_root=Path("/tmp/repo"),
        merged=set(),
        existing=set(),
    )
    assert c[0].reason == "branch-deleted"
    assert c[0].safe is True
    assert c[0].action == "remove"


def test_classify_detached_requires_force() -> None:
    c = classify_candidates(
        [_entry(branch=None, detached=True)],
        repo_root=Path("/tmp/repo"),
        merged=set(),
        existing=set(),
    )
    assert c[0].reason == "detached"
    assert c[0].safe is False
    assert c[0].action == "remove-force"
    assert c[0].blocked_reason == "requires-force"


def test_classify_no_branch_ref_blocked() -> None:
    c = classify_candidates(
        [_entry(branch=None, detached=False)],
        repo_root=Path("/tmp/repo"),
        merged=set(),
        existing=set(),
    )
    assert c[0].safe is False
    assert c[0].blocked_reason == "no-branch-ref"


def test_classify_bare_skipped() -> None:
    """Main (bare) worktree is never a candidate."""
    entries: list[WorktreeEntry] = [
        _entry(path=Path("/tmp/repo"), branch="refs/heads/master", bare=True),
        _entry(path=Path("/tmp/repo-feat"), branch="refs/heads/feature/merged"),
    ]
    c = classify_candidates(
        entries,
        repo_root=Path("/tmp/repo"),
        merged={"refs/heads/feature/merged", "refs/heads/master"},
        existing={"refs/heads/feature/merged", "refs/heads/master"},
    )
    assert len(c) == 1
    assert c[0].entry.branch == "refs/heads/feature/merged"


def test_classify_repo_root_skipped(tmp_path: Path) -> None:
    """Worktree at repo root path is skipped."""
    repo = _make_repo(tmp_path / "repo-root")
    entries: list[WorktreeEntry] = [
        _entry(path=repo, branch="refs/heads/feature/x"),
    ]
    c = classify_candidates(
        entries,
        repo_root=repo,
        merged={"refs/heads/feature/x"},
        existing={"refs/heads/feature/x"},
    )
    assert len(c) == 0


def test_classify_locked_merged_blocked(tmp_path: Path) -> None:
    """Locked worktree with merged branch is blocked, not safe."""
    repo = _make_repo(tmp_path / "repo")
    c = classify_candidates(
        [_entry(branch="refs/heads/feature/merged", locked=True, path=repo)],
        repo_root=Path("/tmp/repo"),
        merged={"refs/heads/feature/merged", "refs/heads/master"},
        existing={"refs/heads/feature/merged", "refs/heads/master"},
    )
    assert c[0].reason == "merged"
    assert c[0].safe is False
    assert c[0].blocked_reason == "locked"


def test_classify_dirty_never_safe(tmp_path: Path) -> None:
    """Dirty check is deferred in classify (no fs access). Test via apply."""
    repo = _make_repo(tmp_path / "repo-dirty")
    (repo / "unstaged").write_text("x")
    entry_dirty = _entry(branch="refs/heads/feature/merged", path=repo)

    c = classify_candidates(
        [entry_dirty],
        repo_root=Path("/tmp/repo"),
        merged={"refs/heads/feature/merged"},
        existing={"refs/heads/feature/merged"},
    )
    assert c[0].safe is False
    assert c[0].blocked_reason == "dirty"


# ── apply ──────────────────────────────────────────────────────────


def test_apply_removes_clean_merged(tmp_path: Path) -> None:
    """apply_removals removes safe merged worktrees."""
    repo = _make_repo(tmp_path / "main")
    wt_path = _make_worktree(repo, tmp_path / "wt-merged", "feature/merged")

    candidates: list[GcCandidate] = [
        GcCandidate(
            entry=WorktreeEntry(path=wt_path, head="abc", branch="refs/heads/feature/merged"),
            reason="merged",
            safe=True,
            action="remove",
        )
    ]
    results = apply_removals(candidates, repo_root=repo)
    assert results[0].get("removed") is True
    # prune should have been called
    assert any(r.get("prune") for r in results)


def test_apply_skips_blocked(tmp_path: Path) -> None:
    """Blocked candidates are not removed."""
    repo = _make_repo(tmp_path / "repo")
    candidates: list[GcCandidate] = [
        GcCandidate(
            entry=WorktreeEntry(path=tmp_path / "nonexistent-wt", head="abc", branch="refs/heads/feature/unmerged"),
            reason="unmerged",
            safe=False,
            blocked_reason="branch-not-merged",
        )
    ]
    results = apply_removals(candidates, repo_root=repo)
    assert results[0].get("removed") is False


def test_apply_force_removes_detached_clean(tmp_path: Path) -> None:
    """With --force, clean detached worktrees are removed."""
    repo = _make_repo(tmp_path / "main")
    wt_path = _make_worktree(repo, tmp_path / "wt-detached", "feature/detached")
    # Detach HEAD
    subprocess.run(["git", "-C", str(wt_path), "checkout", "--detach"], capture_output=True)

    candidates: list[GcCandidate] = [
        GcCandidate(
            entry=WorktreeEntry(path=wt_path, head="abc", branch=None, detached=True),
            reason="detached",
            safe=False,
            action="remove-force",
            blocked_reason="requires-force",
        )
    ]
    results = apply_removals(candidates, repo_root=repo, force=True)
    assert results[0].get("removed") is True


def test_apply_prunable_skipped(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    candidates: list[GcCandidate] = [
        GcCandidate(
            entry=WorktreeEntry(path=tmp_path / "gone", head="abc", prunable=True),
            reason="prunable",
            safe=False,
            action="prune",
            blocked_reason="requires-prune",
        )
    ]
    results = apply_removals(candidates, repo_root=repo)
    assert results[0].get("removed") is False
    assert results[0].get("reason") == "prunable-skipped"


# ── payload / JSON ─────────────────────────────────────────────────


def test_build_payload_shape() -> None:
    candidates: list[GcCandidate] = [
        GcCandidate(
            entry=WorktreeEntry(
                path=Path("/tmp/wt"),
                head="abc123",
                branch="refs/heads/feature/merged",
                locked=False,
            ),
            reason="merged",
            safe=True,
            action="remove",
        ),
    ]
    payload = _build_payload(candidates, target="origin/master")
    assert payload["safe_count"] == 1
    assert payload["blocked_count"] == 0
    assert payload["total_count"] == 1
    assert payload["target"] == "origin/master"
    entries = payload["worktrees"]
    assert isinstance(entries, list)
    assert entries[0]["path"] == "/tmp/wt"
    assert entries[0]["reason"] == "merged"
    assert entries[0]["safe"] is True


def test_main_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main --json emits valid JSON with expected keys."""
    repo = _make_repo(tmp_path / "repo")
    # Run from within the repo
    import os

    orig_cwd = os.getcwd()
    try:
        os.chdir(repo)
        exit_code = main(["--json"])
    finally:
        os.chdir(orig_cwd)

    assert exit_code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "worktrees" in payload
    assert "safe_count" in payload
    assert "total_count" in payload


# ── end-to-end with synthetic repo ─────────────────────────────────


def test_collect_end_to_end(tmp_path: Path) -> None:
    """Full pipeline on a synthetic repo with merged + unmerged worktrees."""
    repo = _make_repo(tmp_path / "repo")

    # Create a branch, merge it, then create a worktree for it
    _run_git(["checkout", "-b", "feature/merged"], cwd=repo)
    (repo / "merged-file.txt").write_text("merged work")
    _run_git(["add", "merged-file.txt"], cwd=repo)
    _run_git(["commit", "-m", "merged commit"], cwd=repo)
    _run_git(["checkout", "master"], cwd=repo)
    _run_git(["merge", "feature/merged"], cwd=repo)

    merged_wt = _make_worktree(repo, tmp_path / "wt-merged", "feature/merged")

    # Create an unmerged branch worktree
    _run_git(["checkout", "-b", "feature/active"], cwd=repo)
    (repo / "active-file.txt").write_text("active work")
    _run_git(["add", "active-file.txt"], cwd=repo)
    _run_git(["commit", "-m", "active commit"], cwd=repo)
    _run_git(["checkout", "master"], cwd=repo)

    active_wt = _make_worktree(repo, tmp_path / "wt-active", "feature/active")

    candidates, entries = collect_candidates(repo)

    merged_c = next((c for c in candidates if c.entry.path == merged_wt), None)
    active_c = next((c for c in candidates if c.entry.path == active_wt), None)

    assert merged_c is not None
    assert merged_c.reason == "merged"
    assert merged_c.safe is True
    assert merged_c.action == "remove"

    assert active_c is not None
    assert active_c.reason == "unmerged"
    assert active_c.safe is False
    assert active_c.blocked_reason == "branch-not-merged"


def test_collect_marks_clean_squash_equivalent_worktree_safe(tmp_path: Path) -> None:
    """A squash-merged branch is safe when all commits are patch-equivalent."""
    repo = _make_repo(tmp_path / "repo")

    _run_git(["checkout", "-b", "feature/squashed"], cwd=repo)
    (repo / "squashed-file.txt").write_text("squashed work")
    _run_git(["add", "squashed-file.txt"], cwd=repo)
    _run_git(["commit", "-m", "squashed work"], cwd=repo)
    _run_git(["checkout", "master"], cwd=repo)
    _run_git(["merge", "--squash", "feature/squashed"], cwd=repo)
    _run_git(["commit", "-m", "squash feature"], cwd=repo)

    wt = _make_worktree(repo, tmp_path / "wt-squashed", "feature/squashed")

    candidates, _entries = collect_candidates(repo)
    candidate = next((c for c in candidates if c.entry.path == wt), None)

    assert candidate is not None
    assert candidate.reason == "squash-equivalent"
    assert candidate.safe is True
    assert candidate.action == "remove"
    assert candidate.evidence == {
        "patch_equivalent": True,
        "equivalent_commits": 1,
        "unique_commits": 0,
        "unknown_commits": 0,
    }


def test_collect_keeps_partly_unique_squash_branch_blocked(tmp_path: Path) -> None:
    """A branch with any unique commit remains blocked after a squash merge."""
    repo = _make_repo(tmp_path / "repo")

    _run_git(["checkout", "-b", "feature/partial"], cwd=repo)
    (repo / "landed.txt").write_text("landed")
    _run_git(["add", "landed.txt"], cwd=repo)
    _run_git(["commit", "-m", "landed"], cwd=repo)
    _run_git(["checkout", "master"], cwd=repo)
    _run_git(["merge", "--squash", "feature/partial"], cwd=repo)
    _run_git(["commit", "-m", "squash partial"], cwd=repo)
    _run_git(["checkout", "feature/partial"], cwd=repo)
    (repo / "unique.txt").write_text("not landed")
    _run_git(["add", "unique.txt"], cwd=repo)
    _run_git(["commit", "-m", "unique"], cwd=repo)
    _run_git(["checkout", "master"], cwd=repo)

    wt = _make_worktree(repo, tmp_path / "wt-partial", "feature/partial")

    candidates, _entries = collect_candidates(repo)
    candidate = next((c for c in candidates if c.entry.path == wt), None)

    assert candidate is not None
    assert candidate.reason == "unmerged"
    assert candidate.safe is False
    assert candidate.blocked_reason == "branch-not-merged"
    assert candidate.evidence is not None
    assert candidate.evidence["patch_equivalent"] is False
    assert candidate.evidence["equivalent_commits"] == 1
    assert candidate.evidence["unique_commits"] == 1


def test_dirty_blocks_in_collect(tmp_path: Path) -> None:
    """collect_candidates marks dirty merged worktrees as blocked."""
    repo = _make_repo(tmp_path / "repo")

    _run_git(["checkout", "-b", "feature/done"], cwd=repo)
    (repo / "f.txt").write_text("x")
    _run_git(["add", "f.txt"], cwd=repo)
    _run_git(["commit", "-m", "done"], cwd=repo)
    _run_git(["checkout", "master"], cwd=repo)
    _run_git(["merge", "feature/done"], cwd=repo)

    dirty_wt = _make_worktree(repo, tmp_path / "wt-dirty", "feature/done")
    (dirty_wt / "unstaged.txt").write_text("oops")

    candidates, _entries = collect_candidates(repo)
    dc = next((c for c in candidates if c.entry.path == dirty_wt), None)
    assert dc is not None
    assert dc.safe is False
    assert dc.blocked_reason == "dirty"


# ── helpers ────────────────────────────────────────────────────────


def _run_git(args: list[str], *, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


def _make_repo(path: Path) -> Path:
    """Create a git repo with an initial commit on master."""
    path.mkdir(parents=True, exist_ok=True)
    _run_git(["init", "-b", "master"], cwd=path)
    _run_git(["config", "user.email", "test@test"], cwd=path)
    _run_git(["config", "user.name", "Test"], cwd=path)
    (path / "README.md").write_text("# test")
    _run_git(["add", "README.md"], cwd=path)
    _run_git(["commit", "-m", "initial"], cwd=path)
    return path


def _make_worktree(repo: Path, worktree_path: Path, branch: str) -> Path:
    """Create a linked worktree for *branch* and return its path.

    If *branch* does not exist locally it is created first.
    """
    result = subprocess.run(
        ["git", "branch", "--list", branch.removeprefix("refs/heads/")],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        _run_git(["worktree", "add", str(worktree_path), branch], cwd=repo)
    else:
        _run_git(["worktree", "add", "-b", branch, str(worktree_path)], cwd=repo)
    return worktree_path

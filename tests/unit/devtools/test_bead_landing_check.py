"""Tests for ``devtools workspace bead-landing-check``."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from devtools.bead_landing_check import (
    CommitChecker,
    Evidence,
    PrChecker,
    PrResult,
    extract_evidence,
    load_beads_from_jsonl,
    remove_worktree,
    verdict_for_bead,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    result.check_returncode()
    return result


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _run_git(["init", "-b", "master"], cwd=path)
    _run_git(["config", "user.email", "test@test"], cwd=path)
    _run_git(["config", "user.name", "Test"], cwd=path)
    (path / "file.txt").write_text("A\n")
    _run_git(["add", "file.txt"], cwd=path)
    _run_git(["commit", "-m", "initial"], cwd=path)
    return path


def _bead(
    bead_id: str = "polylogue-x",
    *,
    status: str = "open",
    priority: int = 2,
    title: str = "untitled",
    description: str = "",
    design: str = "",
    acceptance_criteria: str = "",
    notes: str = "",
    close_reason: str = "",
    comments: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "id": bead_id,
        "status": status,
        "priority": priority,
        "title": title,
        "description": description,
        "design": design,
        "acceptance_criteria": acceptance_criteria,
        "notes": notes,
        "close_reason": close_reason,
        "comments": comments or [],
    }


# ---------------------------------------------------------------------------
# evidence extraction
# ---------------------------------------------------------------------------


def test_extract_evidence_finds_pr_numbers() -> None:
    bead = _bead(description="Fixed via PR #3414 (branch feature/x, commit e9e7a7245).")
    ev = extract_evidence(bead)
    assert ev.pr_numbers == [3414]


def test_extract_evidence_ignores_pull_slash_url_to_avoid_sample_payload_false_positives() -> None:
    # A bead quoting a *sample payload* containing an unrelated PR URL must not
    # be read as citing that PR as its own resolving evidence (this was a real
    # false positive against polylogue-pbuh, which quotes a pr-link sample
    # payload referencing https://github.com/Sinity/polylogue/pull/3126).
    bead = _bead(
        description='Sample payload: {"prUrl": "https://github.com/Sinity/polylogue/pull/3126"}',
    )
    ev = extract_evidence(bead)
    assert ev.pr_numbers == []


def test_extract_evidence_finds_commit_hashes_but_not_plain_numbers() -> None:
    bead = _bead(description="commit e9e7a7245 fixed this. Also saw 850678 records and port 4000.")
    ev = extract_evidence(bead)
    assert "e9e7a7245" in ev.commit_candidates
    assert "850678" not in ev.commit_candidates
    assert "4000" not in ev.commit_candidates


def test_extract_evidence_reads_comments_too() -> None:
    bead = _bead(comments=[{"text": "Landed in #3390."}])
    ev = extract_evidence(bead)
    assert ev.pr_numbers == [3390]


def test_extract_evidence_finds_file_paths() -> None:
    bead = _bead(description="See polylogue/sources/parsers/claude/code_parser.py:87 for the skip list.")
    ev = extract_evidence(bead)
    assert "polylogue/sources/parsers/claude/code_parser.py" in ev.file_paths


def test_extract_evidence_ignores_snapshot_anchor_citations() -> None:
    # "Generated from master @ <hash>" is this repo's prework-packet snapshot
    # header (185+ occurrences in the live backlog) -- it records what master
    # looked like when the note was written, not that the hash is the bead's
    # own resolving commit. Confirmed false positive against polylogue-2qx.
    bead = _bead(
        description=(
            "Generated from master @ 8a975a40 2026-07-06 -- verify source anchors before coding; "
            "line numbers are snapshot-relative."
        )
    )
    ev = extract_evidence(bead)
    assert ev.commit_candidates == []


def test_extract_evidence_ignores_verification_pass_master_anchor() -> None:
    # Confirmed false positive against polylogue-lkrc: the note explicitly
    # says the named gaps are STILL open as of this master snapshot.
    bead = _bead(
        description=(
            "2026-07-14 code-verification pass: re-checked the residual gaps against "
            "current master (031d8d183) source. All 3 named residual gaps are still open."
        )
    )
    ev = extract_evidence(bead)
    assert ev.commit_candidates == []


def test_extract_evidence_still_finds_genuine_self_citation_commit() -> None:
    bead = _bead(description="Foundation phase merged via PR #2915 as d6501ac4615efa30cb0e2413c97614a4bf44b253.")
    ev = extract_evidence(bead)
    assert ev.commit_candidates == ["d6501ac4615efa30cb0e2413c97614a4bf44b253"]
    assert ev.pr_numbers == [2915]


def test_extract_evidence_empty_when_no_signal() -> None:
    ev = extract_evidence(_bead(description="This is a plain description with no citations."))
    assert ev == Evidence()


# ---------------------------------------------------------------------------
# verdict logic (pure function, no subprocess)
# ---------------------------------------------------------------------------


def test_verdict_empty_diff_commit_is_likely_stale_strong() -> None:
    bead = _bead("polylogue-a")
    ev = extract_evidence(_bead(description="commit abc1234ff already did this"))
    v = verdict_for_bead(bead, ev, {"abc1234ff": "empty-diff"}, {})
    assert v.verdict == "LIKELY-STALE"
    assert v.confidence == "strong"


def test_verdict_already_on_master_is_likely_stale_strong() -> None:
    bead = _bead("polylogue-a")
    ev = Evidence(commit_candidates=["abc1234ff"])
    v = verdict_for_bead(bead, ev, {"abc1234ff": "already-on-master"}, {})
    assert v.verdict == "LIKELY-STALE"
    assert v.confidence == "strong"


def test_verdict_content_equivalent_is_likely_stale_strong() -> None:
    bead = _bead("polylogue-a")
    ev = Evidence(commit_candidates=["abc1234ff"])
    v = verdict_for_bead(bead, ev, {"abc1234ff": "content-equivalent"}, {})
    assert v.verdict == "LIKELY-STALE"
    assert v.confidence == "strong"


def test_verdict_non_empty_diff_is_likely_live_strong() -> None:
    bead = _bead("polylogue-a")
    ev = Evidence(commit_candidates=["abc1234ff"])
    v = verdict_for_bead(bead, ev, {"abc1234ff": "non-empty-diff"}, {})
    assert v.verdict == "LIKELY-LIVE"
    assert v.confidence == "strong"


def test_verdict_merged_pr_only_is_likely_stale_weak() -> None:
    bead = _bead("polylogue-a")
    ev = Evidence(pr_numbers=[3390])
    pr_results = {3390: PrResult(number=3390, found=True, state="MERGED")}
    v = verdict_for_bead(bead, ev, {}, pr_results)
    assert v.verdict == "LIKELY-STALE"
    assert v.confidence == "weak"


def test_verdict_open_pr_is_likely_live_weak() -> None:
    bead = _bead("polylogue-a")
    ev = Evidence(pr_numbers=[3390])
    pr_results = {3390: PrResult(number=3390, found=True, state="OPEN")}
    v = verdict_for_bead(bead, ev, {}, pr_results)
    assert v.verdict == "LIKELY-LIVE"
    assert v.confidence == "weak"


def test_verdict_no_evidence_is_undetermined() -> None:
    bead = _bead("polylogue-a")
    v = verdict_for_bead(bead, Evidence(), {}, {})
    assert v.verdict == "UNDETERMINED"
    assert v.confidence == "none"
    assert "not verifiable" in v.reasons[0]


def test_verdict_unresolvable_commit_is_undetermined_not_guessed() -> None:
    # A bogus hex token (e.g. a worktree/session id) must never be silently
    # treated as proof of anything -- this is the core honesty requirement.
    bead = _bead("polylogue-a")
    ev = Evidence(commit_candidates=["ad682bc849a1cd0f0"])
    v = verdict_for_bead(bead, ev, {"ad682bc849a1cd0f0": "unknown-revision"}, {})
    assert v.verdict == "UNDETERMINED"


def test_verdict_conflicted_commit_is_undetermined_not_guessed() -> None:
    bead = _bead("polylogue-a")
    ev = Evidence(commit_candidates=["abc1234ff"])
    v = verdict_for_bead(bead, ev, {"abc1234ff": "does-not-apply"}, {})
    assert v.verdict == "UNDETERMINED"


def test_verdict_pr_not_found_is_undetermined_not_guessed() -> None:
    bead = _bead("polylogue-a")
    ev = Evidence(pr_numbers=[99999])
    pr_results = {99999: PrResult(number=99999, found=False, error="not found")}
    v = verdict_for_bead(bead, ev, {}, pr_results)
    assert v.verdict == "UNDETERMINED"


def test_verdict_mixed_empty_and_non_empty_notes_partial_landing() -> None:
    bead = _bead("polylogue-a")
    ev = Evidence(commit_candidates=["aaa1111ff", "bbb2222ff"])
    v = verdict_for_bead(bead, ev, {"aaa1111ff": "empty-diff", "bbb2222ff": "non-empty-diff"}, {})
    assert v.verdict == "LIKELY-STALE"
    assert any("partial" in r.lower() for r in v.reasons)


# ---------------------------------------------------------------------------
# CommitChecker against a real throwaway repo
# ---------------------------------------------------------------------------


def test_commit_checker_unknown_revision(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    checker = CommitChecker(repo, tmp_path / "wt", base_ref="master")
    assert checker.check("deadbeef00") == "unknown-revision"


def test_commit_checker_already_on_master(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    tip = _run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()
    checker = CommitChecker(repo, tmp_path / "wt", base_ref="master")
    assert checker.check(tip) == "already-on-master"


def test_commit_checker_non_empty_diff_for_unmerged_commit(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    _run_git(["checkout", "-b", "feature"], cwd=repo)
    (repo / "other.txt").write_text("new file\n")
    _run_git(["add", "other.txt"], cwd=repo)
    _run_git(["commit", "-m", "add other.txt"], cwd=repo)
    feature_sha = _run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()
    _run_git(["checkout", "master"], cwd=repo)

    checker = CommitChecker(repo, tmp_path / "wt", base_ref="master")
    assert checker.check(feature_sha) == "non-empty-diff"


def test_commit_checker_empty_diff_when_change_already_present_via_different_history(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    # master: file.txt A -> B
    (repo / "file.txt").write_text("B\n")
    _run_git(["add", "file.txt"], cwd=repo)
    _run_git(["commit", "-m", "A to B on master"], cwd=repo)

    # A divergent branch from the *original* commit that makes the exact same
    # A -> B change independently (simulating a squash-merged equivalent).
    initial_sha = _run_git(["rev-list", "--max-parents=0", "HEAD"], cwd=repo).stdout.strip()
    _run_git(["checkout", "-b", "other-lane", initial_sha], cwd=repo)
    (repo / "file.txt").write_text("B\n")
    _run_git(["add", "file.txt"], cwd=repo)
    _run_git(["commit", "-m", "same A to B change, different lineage"], cwd=repo)
    other_sha = _run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()
    _run_git(["checkout", "master"], cwd=repo)

    checker = CommitChecker(repo, tmp_path / "wt", base_ref="master")
    assert checker.check(other_sha) == "empty-diff"


def test_commit_checker_reuses_one_worktree_across_checks(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    wt_dir = tmp_path / "wt"
    checker = CommitChecker(repo, wt_dir, base_ref="master")

    checker.check("deadbeef00")
    assert not wt_dir.exists()  # unknown-revision never needs the worktree

    tip = _run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()
    checker.check(tip)
    assert not wt_dir.exists()  # already-on-master short-circuits too

    _run_git(["checkout", "-b", "feature"], cwd=repo)
    (repo / "other.txt").write_text("x\n")
    _run_git(["add", "other.txt"], cwd=repo)
    _run_git(["commit", "-m", "add"], cwd=repo)
    feature_sha = _run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()
    _run_git(["checkout", "master"], cwd=repo)

    checker.check(feature_sha)
    assert wt_dir.exists()
    marker_mtime = (wt_dir / ".git").stat().st_mtime

    # A second, distinct commit must reuse the same worktree directory rather
    # than creating a fresh one.
    (repo / "third.txt").write_text("y\n")
    _run_git(["add", "third.txt"], cwd=repo)
    _run_git(["commit", "-m", "add third"], cwd=repo)
    third_sha = _run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()
    _run_git(["checkout", "master~1"], cwd=repo)  # detach so third_sha isn't already master's tip
    _run_git(["checkout", "master"], cwd=repo)

    checker.check(third_sha)
    assert wt_dir.exists()
    assert (wt_dir / ".git").stat().st_mtime == marker_mtime


def test_commit_checker_rejects_worktree_dir_equal_to_repo_root(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    with pytest.raises(ValueError):
        CommitChecker(repo, repo, base_ref="master")


# ---------------------------------------------------------------------------
# worktree removal safety
# ---------------------------------------------------------------------------


def test_remove_worktree_not_present(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    assert remove_worktree(repo, tmp_path / "does-not-exist") == "not-present"


def test_remove_worktree_blocks_on_live_process(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    _run_git(["worktree", "add", "--detach", str(wt), "master"], cwd=repo)

    proc = subprocess.Popen(["sleep", "5"], cwd=wt)
    try:
        # Give the OS a moment to publish /proc/<pid>/cwd.
        for _ in range(50):
            if Path(f"/proc/{proc.pid}/cwd").exists():
                break
            time.sleep(0.05)
        assert remove_worktree(repo, wt) == "blocked-live-process"
        assert wt.exists()  # never removed while occupied
    finally:
        proc.kill()
        proc.wait()


def test_remove_worktree_removes_clean_unoccupied_worktree(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    wt = tmp_path / "wt"
    _run_git(["worktree", "add", "--detach", str(wt), "master"], cwd=repo)
    assert remove_worktree(repo, wt) == "removed"
    assert not wt.exists()


# ---------------------------------------------------------------------------
# PrChecker caching (subprocess mocked out)
# ---------------------------------------------------------------------------


def test_pr_checker_caches_merged_result_permanently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["n"] += 1
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"state": "MERGED", "mergedAt": "t", "mergeCommit": {"oid": "abc"}, "title": "x"}),
            stderr="",
        )

    monkeypatch.setattr("devtools.bead_landing_check._run", fake_run)
    cache_path = tmp_path / "cache.json"

    checker = PrChecker("Sinity/polylogue", cache_path, ttl_days=7)
    r1 = checker.check(100)
    assert r1.state == "MERGED"
    assert calls["n"] == 1
    checker.flush()

    # Fresh PrChecker instance reading the persisted cache must not re-fetch.
    checker2 = PrChecker("Sinity/polylogue", cache_path, ttl_days=7)
    r2 = checker2.check(100)
    assert r2.state == "MERGED"
    assert calls["n"] == 1


def test_pr_checker_offline_uses_cache_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("gh must not be called in offline mode")

    monkeypatch.setattr("devtools.bead_landing_check._run", fake_run)
    cache_path = tmp_path / "cache.json"
    checker = PrChecker("Sinity/polylogue", cache_path, ttl_days=7, offline=True)
    r = checker.check(200)
    assert r.found is False
    assert r.error == "offline: not cached"


def test_pr_checker_refresh_ignores_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["n"] += 1
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps({"state": "OPEN", "mergedAt": None, "mergeCommit": None, "title": "x"}), stderr=""
        )

    monkeypatch.setattr("devtools.bead_landing_check._run", fake_run)
    cache_path = tmp_path / "cache.json"
    checker = PrChecker("Sinity/polylogue", cache_path, ttl_days=7)
    checker.check(300)
    assert calls["n"] == 1
    checker.check(300)
    assert calls["n"] == 1  # OPEN is TTL-cached, reused within TTL

    checker_refresh = PrChecker("Sinity/polylogue", cache_path, ttl_days=7, refresh=True)
    checker_refresh.check(300)
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# jsonl loading
# ---------------------------------------------------------------------------


def test_load_beads_from_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "issues.jsonl"
    p.write_text('{"id": "a"}\n\n{"id": "b"}\n')
    beads = load_beads_from_jsonl(p)
    assert [b["id"] for b in beads] == ["a", "b"]

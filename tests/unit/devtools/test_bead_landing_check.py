"""Tests for ``devtools workspace bead-landing-check``."""

from __future__ import annotations

import json
import os
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
    build_open_parent_child_dependents_index,
    extract_evidence,
    find_suppression_signal,
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


def test_verdict_empty_diff_commit_is_likely_stale_strong_when_consumed() -> None:
    bead = _bead("polylogue-a")
    ev = extract_evidence(_bead(description="commit abc1234ff already did this"))
    v = verdict_for_bead(bead, ev, {"abc1234ff": "empty-diff"}, {}, commit_consumer={"abc1234ff": True})
    assert v.verdict == "LIKELY-STALE"
    assert v.confidence == "strong"


def test_verdict_already_on_master_is_likely_stale_strong_when_consumed() -> None:
    bead = _bead("polylogue-a")
    ev = Evidence(commit_candidates=["abc1234ff"])
    v = verdict_for_bead(bead, ev, {"abc1234ff": "already-on-master"}, {}, commit_consumer={"abc1234ff": True})
    assert v.verdict == "LIKELY-STALE"
    assert v.confidence == "strong"


def test_verdict_content_equivalent_is_likely_stale_strong_when_consumed() -> None:
    bead = _bead("polylogue-a")
    ev = Evidence(commit_candidates=["abc1234ff"])
    v = verdict_for_bead(bead, ev, {"abc1234ff": "content-equivalent"}, {}, commit_consumer={"abc1234ff": True})
    assert v.verdict == "LIKELY-STALE"
    assert v.confidence == "strong"


# ---------------------------------------------------------------------------
# fix 1 (2026-07-31): live-consumer check -- landed without a live caller
# must NOT be LIKELY-STALE. Measured against 114 human-verified beads: ~95%
# of "cited commit exists on master" verdicts were false positives, and the
# dominant cause was exactly this -- shipped code with zero production
# callers (e.g. polylogue-rxdo.9.6's blind_items(), polylogue-yp0's EventBus).
# ---------------------------------------------------------------------------


def test_verdict_landed_but_unconsumed_commit_is_undetermined_not_stale() -> None:
    bead = _bead("polylogue-a")
    ev = Evidence(commit_candidates=["abc1234ff"])
    v = verdict_for_bead(bead, ev, {"abc1234ff": "empty-diff"}, {}, commit_consumer={"abc1234ff": False})
    assert v.verdict == "UNDETERMINED"
    assert any("no live production consumer" in r.lower() for r in v.reasons)


def test_verdict_landed_but_consumer_unknown_is_undetermined_not_stale() -> None:
    # No symbols extracted (e.g. a non-Python change) -- inconclusive, and
    # inconclusive must never default to LIKELY-STALE.
    bead = _bead("polylogue-a")
    ev = Evidence(commit_candidates=["abc1234ff"])
    v = verdict_for_bead(bead, ev, {"abc1234ff": "already-on-master"}, {}, commit_consumer={"abc1234ff": None})
    assert v.verdict == "UNDETERMINED"
    assert any("could not be checked" in r for r in v.reasons)


def test_verdict_missing_commit_consumer_entry_defaults_to_unknown_not_stale() -> None:
    # If the caller forgets to populate commit_consumer for a landed commit,
    # the safe default is "unknown", not "assume consumed".
    bead = _bead("polylogue-a")
    ev = Evidence(commit_candidates=["abc1234ff"])
    v = verdict_for_bead(bead, ev, {"abc1234ff": "empty-diff"}, {}, commit_consumer={})
    assert v.verdict == "UNDETERMINED"


# ---------------------------------------------------------------------------
# fix 2 (2026-07-31): open parent-child dependents suppress a stale verdict
# ---------------------------------------------------------------------------


def test_verdict_suppressed_by_open_parent_child_dependents() -> None:
    bead = _bead("polylogue-epic")
    ev = Evidence(commit_candidates=["abc1234ff"])
    open_dependents = [{"id": "polylogue-epic.1", "status": "open"}, {"id": "polylogue-epic.2", "status": "open"}]
    v = verdict_for_bead(
        bead,
        ev,
        {"abc1234ff": "empty-diff"},
        {},
        commit_consumer={"abc1234ff": True},
        open_dependents=open_dependents,
    )
    assert v.verdict == "UNDETERMINED"
    assert any("open parent-child dependent" in r for r in v.reasons)


def test_verdict_open_dependents_do_not_suppress_a_live_verdict() -> None:
    # The downgrade only applies to LIKELY-STALE -- it must not mask genuine
    # LIKELY-LIVE evidence.
    bead = _bead("polylogue-epic")
    ev = Evidence(commit_candidates=["abc1234ff"])
    open_dependents = [{"id": "polylogue-epic.1", "status": "open"}]
    v = verdict_for_bead(bead, ev, {"abc1234ff": "non-empty-diff"}, {}, open_dependents=open_dependents)
    assert v.verdict == "LIKELY-LIVE"


def test_build_open_parent_child_dependents_index() -> None:
    beads = [
        _bead("polylogue-parent"),
        {
            **_bead("polylogue-parent.1"),
            "status": "open",
            "dependencies": [{"depends_on_id": "polylogue-parent", "type": "parent-child"}],
        },
        {
            **_bead("polylogue-parent.2"),
            "status": "closed",
            "dependencies": [{"depends_on_id": "polylogue-parent", "type": "parent-child"}],
        },
        {
            **_bead("polylogue-parent.3"),
            "status": "open",
            "dependencies": [{"depends_on_id": "polylogue-parent", "type": "related"}],
        },
    ]
    index = build_open_parent_child_dependents_index(beads)
    assert [d["id"] for d in index["polylogue-parent"]] == ["polylogue-parent.1"]


# ---------------------------------------------------------------------------
# fix 3 (2026-07-31): the bead's own words suppress a stale verdict
# ---------------------------------------------------------------------------


def test_verdict_suppressed_by_bead_own_not_done_note() -> None:
    bead = _bead(
        "polylogue-a",
        notes="Landed the core in commit abc1234ff. AC4 is NOT satisfied: the DSL still lacks float literals.",
    )
    ev = Evidence(commit_candidates=["abc1234ff"])
    v = verdict_for_bead(
        bead,
        ev,
        {"abc1234ff": "empty-diff"},
        {},
        commit_consumer={"abc1234ff": True},
        suppression_signal=find_suppression_signal(bead),
    )
    assert v.verdict == "UNDETERMINED"
    assert any("contradicts a stale verdict" in r for r in v.reasons)


def test_find_suppression_signal_detects_common_phrasings() -> None:
    assert (
        find_suppression_signal(_bead(notes="EventBus core landed. NOT wired to a live producer/consumer.")) is not None
    )
    assert find_suppression_signal(_bead(notes="14 SELECT methods / 503 accessors untouched, still open.")) is not None
    assert find_suppression_signal(_bead(notes="Everything here is done and fully wired end to end.")) is None


def test_find_suppression_signal_catches_deferred_and_xfail() -> None:
    # Confirmed false positive against polylogue-hg97: the PR merged, but the
    # bead's own text says the follow-on work was "Explicitly deferred this
    # session" and the regression test is "Marked xfail" pending it.
    bead = _bead(
        notes=(
            "Explicitly deferred this session (2026-07-18) per operator scoping choice. "
            "Marked xfail (strict=False, reason references this bead) rather than deleted."
        )
    )
    assert find_suppression_signal(bead) is not None


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
    v = verdict_for_bead(
        bead,
        ev,
        {"aaa1111ff": "empty-diff", "bbb2222ff": "non-empty-diff"},
        {},
        commit_consumer={"aaa1111ff": True},
    )
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
# fix 1 (2026-07-31): live-consumer check against a real throwaway repo
# ---------------------------------------------------------------------------


def test_added_symbols_extracts_new_function_and_class_names(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    (repo / "module.py").write_text("def existing():\n    pass\n")
    _run_git(["add", "module.py"], cwd=repo)
    _run_git(["commit", "-m", "add module"], cwd=repo)

    (repo / "module.py").write_text(
        "def existing():\n    pass\n\n\ndef new_helper():\n    pass\n\n\nclass NewThing:\n    pass\n"
    )
    _run_git(["add", "module.py"], cwd=repo)
    _run_git(["commit", "-m", "add new_helper and NewThing"], cwd=repo)
    sha = _run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()

    checker = CommitChecker(repo, tmp_path / "wt", base_ref="master")
    symbols = checker.added_symbols(sha)
    assert "new_helper" in symbols
    assert "NewThing" in symbols
    assert "existing" not in symbols  # only NEW definitions, not pre-existing ones


def test_has_live_consumer_true_when_symbol_used_elsewhere(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    (repo / "lib.py").write_text("def blind_items():\n    return []\n")
    _run_git(["add", "lib.py"], cwd=repo)
    _run_git(["commit", "-m", "add lib"], cwd=repo)
    sha = _run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()

    # The caller lands in a LATER, separate commit -- has_live_consumer must
    # find it via base_ref, not just the cited commit's own touched files.
    (repo / "caller.py").write_text("from lib import blind_items\n\nblind_items()\n")
    _run_git(["add", "caller.py"], cwd=repo)
    _run_git(["commit", "-m", "wire up the real caller"], cwd=repo)

    checker = CommitChecker(repo, tmp_path / "wt", base_ref="master")
    assert checker.has_live_consumer(sha, ["blind_items"]) is True


def test_has_live_consumer_false_when_symbol_unreferenced(tmp_path: Path) -> None:
    # Models polylogue-rxdo.9.6: blind_items() shipped, nothing outside its
    # own file (and no test) calls it.
    repo = _make_repo(tmp_path / "repo")
    (repo / "lib.py").write_text("def blind_items():\n    return []\n")
    _run_git(["add", "lib.py"], cwd=repo)
    _run_git(["commit", "-m", "add lib, unconsumed"], cwd=repo)
    sha = _run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()

    checker = CommitChecker(repo, tmp_path / "wt", base_ref="master")
    assert checker.has_live_consumer(sha, ["blind_items"]) is False


def test_has_live_consumer_ignores_test_only_references(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    (repo / "lib.py").write_text("def blind_items():\n    return []\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_lib.py").write_text("from lib import blind_items\n\ndef test_x():\n    blind_items()\n")
    _run_git(["add", "lib.py", "tests/test_lib.py"], cwd=repo)
    _run_git(["commit", "-m", "add lib, only referenced from its own test"], cwd=repo)
    sha = _run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()

    checker = CommitChecker(repo, tmp_path / "wt", base_ref="master")
    # lib.py and tests/test_lib.py were BOTH touched by this commit, so the
    # only reference is inside the commit's own touched files -- no external
    # production consumer.
    assert checker.has_live_consumer(sha, ["blind_items"]) is False


def test_has_live_consumer_none_when_no_symbols(tmp_path: Path) -> None:
    # No extractable symbols (e.g. a non-Python change) is inconclusive, not
    # a "no consumer" verdict -- the two must stay distinguishable so the
    # verdict layer can report them with different reasons.
    repo = _make_repo(tmp_path / "repo")
    checker = CommitChecker(repo, tmp_path / "wt", base_ref="master")
    assert checker.has_live_consumer("deadbeef", []) is None


def test_consumer_check_is_cached(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    (repo / "lib.py").write_text("def blind_items():\n    return []\n")
    _run_git(["add", "lib.py"], cwd=repo)
    _run_git(["commit", "-m", "add lib"], cwd=repo)
    sha = _run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()

    (repo / "caller.py").write_text("from lib import blind_items\n\nblind_items()\n")
    _run_git(["add", "caller.py"], cwd=repo)
    _run_git(["commit", "-m", "wire up the real caller"], cwd=repo)

    checker = CommitChecker(repo, tmp_path / "wt", base_ref="master")
    first = checker.consumer_check(sha)
    assert first is True
    assert sha in checker._consumer_cache
    # Second call must reuse the cache rather than re-run git show/grep.
    assert checker.consumer_check(sha) is True


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


# ---------------------------------------------------------------------------
# Labelled-evaluation regression: five independent human reviewers checked
# all 190 LIKELY-STALE verdicts from the pre-fix sweep against 114 real
# beads (2026-07-31) and found ~95% were false positives. These tests pin
# the fixes against the ACTUAL live archive and repo history so precision
# cannot silently regress -- no synthetic fixtures, real .beads/issues.jsonl
# and real git objects. Network-free: PR merge state is not queried here, so
# only the commit-consumer and text-based checks are exercised.
# ---------------------------------------------------------------------------


def _test_worktree_dir(repo_root: Path) -> Path:
    # Worker-scoped so `-n auto` distribution can't race two xdist workers
    # onto the same throwaway worktree path.
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    return repo_root / ".cache" / "bead-landing-check" / f"test-worktree-{worker}"


def _repo_root() -> Path:
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
    return Path(result.stdout.strip())


@pytest.fixture(scope="module", autouse=True)
def _cleanup_live_repo_test_worktree() -> Any:
    """These regression tests exercise CommitChecker against the real repo
    (not tmp_path) so they see real production evidence -- but that means
    they create a real, gitignored throwaway worktree under the checkout's
    own .cache/. Remove it after this module's tests finish so it doesn't
    linger as a stray artifact (same safety contract as --remove-worktree:
    only ever removes a path this fixture created, via remove_worktree()).
    """
    yield
    repo_root = _repo_root()
    test_worktree = _test_worktree_dir(repo_root)
    if test_worktree.exists():
        remove_worktree(repo_root, test_worktree)


def _live_bead(bead_id: str) -> dict[str, Any] | None:
    repo_root = _repo_root()
    beads_file = repo_root / ".beads" / "issues.jsonl"
    if not beads_file.exists():
        return None
    for bead in load_beads_from_jsonl(beads_file):
        if bead["id"] == bead_id:
            return bead
    return None


@pytest.mark.parametrize(
    "bead_id",
    [
        "polylogue-rxdo.9.6",  # blind_items() has zero callers outside its own test
        "polylogue-rxdo.6",  # ReferenceQueryPipeline has zero CLI/MCP/daemon references, still hard-errors
        "polylogue-rxdo.9.7",  # ClaimWithControls has zero callers outside its own test
        "polylogue-dcz5",  # 3.14t live in prod, but daemon_parse_stage_split is still False
        "polylogue-hg97",  # cost_outlook absent from polylogue/mcp/, contract test still xfail
        "polylogue-yp0",  # EventBus core landed but notes say "NOT wired to a live producer/consumer"
    ],
)
def test_known_false_positive_no_longer_verifies_as_strong_likely_stale(bead_id: str) -> None:
    """Five independent human reviewers checked all 190 LIKELY-STALE verdicts
    from the pre-fix sweep against 114 real beads (2026-07-31) and found
    roughly 95% were false positives -- these six were named explicitly. Run
    the REAL pipeline (evidence extraction, real git history for commit/
    consumer checks, real .beads/issues.jsonl for dependents/suppression)
    against each one and require the fixed verdict logic to no longer call
    it a strong-confidence LIKELY-STALE. PR-merge state is not queried here
    (network-free); PR numbers this bead cites are treated as unverified,
    which only ever pushes a verdict TOWARD UNDETERMINED, never masks a
    regression back to LIKELY-STALE.
    """
    bead = _live_bead(bead_id)
    if bead is None or bead.get("status") not in ("open", "in_progress"):
        pytest.skip(f"{bead_id} no longer open in the live backlog -- regression target moved on")

    repo_root = _repo_root()
    beads_file = repo_root / ".beads" / "issues.jsonl"
    all_beads = load_beads_from_jsonl(beads_file)
    dependents_index = build_open_parent_child_dependents_index(all_beads)

    evidence = extract_evidence(bead)
    checker = CommitChecker(repo_root, _test_worktree_dir(repo_root))
    commit_results = {c: checker.check(c) for c in evidence.commit_candidates}
    landed_states = {"empty-diff", "already-on-master", "content-equivalent"}
    commit_consumer = {c: checker.consumer_check(c) for c, s in commit_results.items() if s in landed_states}

    verdict = verdict_for_bead(
        bead,
        evidence,
        commit_results,
        {},  # PR state unverified (network-free) -- can only push toward UNDETERMINED, never mask a regression
        commit_consumer=commit_consumer,
        open_dependents=dependents_index.get(bead_id, []),
        suppression_signal=find_suppression_signal(bead),
    )
    assert verdict.verdict != "LIKELY-STALE" or verdict.confidence != "strong", (
        f"{bead_id} regressed back to a strong-confidence LIKELY-STALE verdict: {verdict.reasons}"
    )

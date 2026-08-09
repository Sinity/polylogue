from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from devtools import merge_boundary, merge_gate, pr_scope

_SCOPE_BEAD = {
    "_type": "issue",
    "id": "polylogue-test-scope",
    "title": "test scope",
    "description": "test record",
    "acceptance_criteria": "Opaque acceptance prose.",
    "status": "open",
}


@pytest.fixture(autouse=True)
def _scope_bead_record(tmp_path: Path) -> None:
    beads_dir = tmp_path / ".beads"
    beads_dir.mkdir()
    (beads_dir / "issues.jsonl").write_text(json.dumps(_SCOPE_BEAD) + "\n")


def _scope_body(head_sha: str) -> str:
    carrier = {
        "version": 1,
        "head_sha": head_sha,
        "assigned_beads": ["polylogue-test-scope"],
        "beads_digest": pr_scope.canonical_beads_digest({_SCOPE_BEAD["id"]: _SCOPE_BEAD}, ["polylogue-test-scope"]),
        "dispositions": [
            {
                "bead_id": "polylogue-test-scope",
                "disposition": "satisfied",
                "evidence": [{"kind": "test", "ref": "tests/unit/devtools/test_merge_boundary.py"}],
                "successors": [],
            }
        ],
    }
    carrier["scope_digest"] = pr_scope.carrier_digest(carrier)
    return pr_scope.render_carrier(carrier)


def _base_pr_view(head_sha: str = "abc123", title: str = "fix: thing (#42)", state: str = "OPEN") -> dict[str, object]:
    return {
        "headRefOid": head_sha,
        "headRefName": "feature/x",
        "title": title,
        "state": state,
        "mergeStateStatus": "CLEAN",
        "commits": [{"oid": head_sha, "committedDate": "2026-08-01T12:00:00Z"}],
        "body": _scope_body(head_sha),
        "isDraft": False,
    }


def _fake_run(
    pr_view: dict[str, object],
    comments: list[dict[str, object]] | None = None,
    *,
    local_exit: int = 0,
    local_head_sha: str | None = None,
    merge_exit: int = 0,
) -> Callable[..., MagicMock]:
    comments = comments or []
    local_head_sha = local_head_sha if local_head_sha is not None else str(pr_view["headRefOid"])

    def _run(cmd: list[str], **kwargs: Any) -> MagicMock:
        joined = " ".join(cmd)
        if cmd[:3] == ["gh", "pr", "view"]:
            return MagicMock(returncode=0, stdout=json.dumps(pr_view), stderr="")
        if cmd[:3] == ["gh", "pr", "merge"]:
            return MagicMock(returncode=merge_exit, stdout="merged\n", stderr="" if merge_exit == 0 else "merge failed")
        if "/issues/" in joined and "/comments" in joined:
            return MagicMock(returncode=0, stdout=json.dumps([[]]), stderr="")
        if "/pulls/" in joined and "/reviews" in joined:
            return MagicMock(returncode=0, stdout=json.dumps([[]]), stderr="")
        if "/pulls/" in joined and "/comments" in joined:
            return MagicMock(returncode=0, stdout=json.dumps([comments]), stderr="")
        if cmd[:2] == ["git", "rev-parse"]:
            return MagicMock(returncode=0, stdout=local_head_sha + "\n", stderr="")
        if cmd[:2] == ["git", "status"]:
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(
            returncode=local_exit,
            stdout=json.dumps({"verification_scope": "affected", "release_baseline_allowed": False}),
            stderr="",
        )

    return _run


# ---------------------------------------------------------------------------
# clean_merge_title
# ---------------------------------------------------------------------------


def test_clean_merge_title_collapses_doubled_suffix() -> None:
    assert merge_boundary.clean_merge_title("fix: thing (#42) (#42)", 42) == "fix: thing (#42)"


def test_clean_merge_title_leaves_correct_title_untouched() -> None:
    assert merge_boundary.clean_merge_title("fix: thing (#42)", 42) == "fix: thing (#42)"


def test_clean_merge_title_appends_missing_suffix() -> None:
    assert merge_boundary.clean_merge_title("fix: thing", 42) == "fix: thing (#42)"


# ---------------------------------------------------------------------------
# cmd_merge
# ---------------------------------------------------------------------------


def test_merge_auto_records_when_no_fresh_receipt_then_merges(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view))

    exit_code = merge_boundary.cmd_merge(
        42,
        command="devtools test x",
        max_age_s=3600,
        poll_rounds=1,
        poll_interval_s=0,
        dry_run=False,
        with_verify=False,
        verify_command="devtools verify --all",
    )

    assert exit_code == 0
    assert merge_gate._receipt_path(42).exists()
    ledger = json.loads(merge_boundary._LEDGER_PATH.read_text())
    assert ledger["merges"][0]["pr"] == 42
    assert ledger["merges"][0]["title"] == "fix: thing (#42)"


def test_merge_strips_doubled_pr_suffix_before_merging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view(title="fix: thing (#42) (#42)")
    captured: dict[str, list[str]] = {}

    base_fake = _fake_run(pr_view)

    def _run(cmd: list[str], **kwargs: Any) -> MagicMock:
        if cmd[:3] == ["gh", "pr", "merge"]:
            captured["cmd"] = cmd
        return base_fake(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", _run)

    exit_code = merge_boundary.cmd_merge(
        42,
        command="devtools test x",
        max_age_s=3600,
        poll_rounds=1,
        poll_interval_s=0,
        dry_run=False,
        with_verify=False,
        verify_command="devtools verify --all",
    )

    assert exit_code == 0
    subject_index = captured["cmd"].index("--subject") + 1
    assert captured["cmd"][subject_index] == "fix: thing (#42)"
    match_index = captured["cmd"].index("--match-head-commit") + 1
    assert captured["cmd"][match_index] == "abc123"


def test_merge_refuses_when_pr_not_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view(state="MERGED")
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view))

    exit_code = merge_boundary.cmd_merge(
        42,
        command="devtools test x",
        max_age_s=3600,
        poll_rounds=1,
        poll_interval_s=0,
        dry_run=False,
        with_verify=False,
        verify_command="devtools verify --all",
    )

    assert exit_code == 1


def test_merge_refuses_when_pr_scope_carrier_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    pr_view["body"] = "## Summary\n\nNo carrier."
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view))

    exit_code = merge_boundary.cmd_merge(
        42,
        command="devtools test x",
        max_age_s=3600,
        poll_rounds=1,
        poll_interval_s=0,
        dry_run=False,
        with_verify=False,
        verify_command="devtools verify --all",
    )

    assert exit_code == 2
    assert merge_boundary._read_ledger()["merges"] == []


def test_merge_refuses_when_late_unacked_review_comment_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    late_comment = {
        "id": 999,
        "path": "polylogue/foo.py",
        "line": 1,
        "created_at": "2026-08-01T12:05:00Z",
        "body": "real finding",
    }
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, [late_comment]))

    exit_code = merge_boundary.cmd_merge(
        42,
        command="devtools test x",
        max_age_s=3600,
        poll_rounds=1,
        poll_interval_s=0,
        dry_run=False,
        with_verify=False,
        verify_command="devtools verify --all",
    )

    assert exit_code != 0
    ledger = merge_boundary._read_ledger()
    assert ledger["merges"] == []


def test_merge_refuses_when_local_verify_command_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, local_exit=1))

    exit_code = merge_boundary.cmd_merge(
        42,
        command="devtools test x",
        max_age_s=3600,
        poll_rounds=1,
        poll_interval_s=0,
        dry_run=False,
        with_verify=False,
        verify_command="devtools verify --all",
    )

    assert exit_code == 1
    assert not (Path("cache") / "does-not-exist").exists()  # sanity: no merge attempted
    ledger = merge_boundary._read_ledger()
    assert ledger["merges"] == []


def test_merge_dry_run_never_calls_gh_pr_merge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    merge_calls: list[list[str]] = []

    base_fake = _fake_run(pr_view)

    def _run(cmd: list[str], **kwargs: Any) -> MagicMock:
        if cmd[:3] == ["gh", "pr", "merge"]:
            merge_calls.append(cmd)
        return base_fake(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", _run)

    exit_code = merge_boundary.cmd_merge(
        42,
        command="devtools test x",
        max_age_s=3600,
        poll_rounds=1,
        poll_interval_s=0,
        dry_run=True,
        with_verify=False,
        verify_command="devtools verify --all",
    )

    assert exit_code == 0
    assert merge_calls == []
    ledger = merge_boundary._read_ledger()
    assert ledger["merges"] == []


def test_merge_propagates_gh_pr_merge_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, merge_exit=3))

    exit_code = merge_boundary.cmd_merge(
        42,
        command="devtools test x",
        max_age_s=3600,
        poll_rounds=1,
        poll_interval_s=0,
        dry_run=False,
        with_verify=False,
        verify_command="devtools verify --all",
    )

    assert exit_code == 3
    ledger = merge_boundary._read_ledger()
    assert ledger["merges"] == []


def test_merge_with_verify_records_terminal_full_verify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    base = _fake_run(pr_view)

    def run(cmd: list[str], **kwargs: Any) -> MagicMock:
        if cmd[:3] == ["devtools", "verify", "--all"]:
            return MagicMock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "git_head": "merged-master",
                        "verification_scope": "release-baseline",
                        "release_baseline_allowed": True,
                    }
                ),
                stderr="",
            )
        return base(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(merge_boundary, "_fetched_merged_default_branch_sha", lambda _pr: "merged-master")
    monkeypatch.setattr(
        merge_boundary,
        "_run_post_merge_terminal_verify",
        lambda command, target: merge_boundary.cmd_record_full_verify(command, target_sha=target),
    )

    exit_code = merge_boundary.cmd_merge(
        42,
        command="devtools test x",
        max_age_s=3600,
        poll_rounds=1,
        poll_interval_s=0,
        dry_run=False,
        with_verify=True,
        verify_command="devtools verify --all",
    )

    assert exit_code == 0
    ledger = merge_boundary._read_ledger()
    assert ledger["last_full_verify"]["command"] == "devtools verify --all"
    assert ledger["last_full_verify"]["exit_code"] == 0
    # train-status should now report clean.
    assert merge_boundary.cmd_train_status(as_json=False) == 0


def test_merge_with_verify_returns_nonzero_when_terminal_authority_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view))
    monkeypatch.setattr(merge_boundary, "_fetched_merged_default_branch_sha", lambda _pr: "merged-master")
    monkeypatch.setattr(merge_boundary, "_run_post_merge_terminal_verify", lambda _command, _target: 1)

    assert (
        merge_boundary.cmd_merge(
            42,
            command="devtools test x",
            max_age_s=3600,
            poll_rounds=1,
            poll_interval_s=0,
            dry_run=False,
            with_verify=True,
            verify_command="devtools verify --all",
        )
        == 1
    )
    assert merge_boundary._read_ledger()["merges"]


def test_post_merge_terminal_verify_rejects_stale_feature_head(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda _cmd, **_kwargs: MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "git_head": "feature-head",
                    "verification_scope": "release-baseline",
                    "release_baseline_allowed": True,
                }
            ),
            stderr="",
        ),
    )

    assert merge_boundary.cmd_record_full_verify("devtools verify --all", target_sha="merged-master") == 1
    assert merge_boundary._read_ledger()["last_full_verify"]["accepted"] is False


def test_fetched_default_branch_must_include_squash_merge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: Any) -> MagicMock:
        calls.append(cmd)
        if cmd[:3] == ["gh", "repo", "view"]:
            return MagicMock(returncode=0, stdout=json.dumps({"defaultBranchRef": {"name": "master"}}), stderr="")
        if cmd[:3] == ["gh", "pr", "view"]:
            return MagicMock(
                returncode=0,
                stdout=json.dumps({"state": "MERGED", "mergeCommit": {"oid": "squash-sha"}}),
                stderr="",
            )
        if cmd[:3] == ["git", "fetch", "origin"]:
            return MagicMock(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["git", "rev-parse", "FETCH_HEAD"]:
            return MagicMock(returncode=0, stdout="stale-feature-sha\n", stderr="")
        if cmd[:3] == ["git", "merge-base", "--is-ancestor"]:
            return MagicMock(returncode=1, stdout="", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(subprocess, "run", run)

    assert merge_boundary._fetched_merged_default_branch_sha(42) is None
    assert ["git", "fetch", "origin", "master"] in calls


def test_fetched_default_branch_sha_is_the_verified_terminal_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    def run(cmd: list[str], **_kwargs: Any) -> MagicMock:
        if cmd[:3] == ["gh", "repo", "view"]:
            return MagicMock(returncode=0, stdout=json.dumps({"defaultBranchRef": {"name": "master"}}), stderr="")
        if cmd[:3] == ["gh", "pr", "view"]:
            return MagicMock(
                returncode=0,
                stdout=json.dumps({"state": "MERGED", "mergeCommit": {"oid": "squash-sha"}}),
                stderr="",
            )
        if cmd[:3] == ["git", "fetch", "origin"]:
            return MagicMock(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["git", "rev-parse", "FETCH_HEAD"]:
            return MagicMock(returncode=0, stdout="merged-master-sha\n", stderr="")
        if cmd[:3] == ["git", "merge-base", "--is-ancestor"]:
            return MagicMock(returncode=0, stdout="", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(subprocess, "run", run)

    assert merge_boundary._fetched_merged_default_branch_sha(42) == "merged-master-sha"


# ---------------------------------------------------------------------------
# train-status / record-full-verify
# ---------------------------------------------------------------------------


def test_train_status_ok_with_empty_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert merge_boundary.cmd_train_status(as_json=False) == 0


def test_train_status_blocks_when_pr_merged_after_last_full_verify(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    merge_boundary._write_ledger(
        {"merges": [], "last_full_verify": {"at": 1000.0, "command": "devtools verify --all", "exit_code": 0}}
    )
    merge_boundary._append_merge_entry(1, "sha1", "some title")

    # Manually push merged_at ahead of the recorded verify.
    ledger = merge_boundary._read_ledger()
    ledger["merges"][0]["merged_at"] = 2000.0
    merge_boundary._write_ledger(ledger)

    assert merge_boundary.cmd_train_status(as_json=False) == 1


def test_train_status_rejects_untyped_accepted_terminal_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    merge_boundary._write_ledger(
        {
            "merges": [],
            "last_full_verify": {
                "at": 1000.0,
                "command": "devtools verify --all",
                "exit_code": 0,
                "release_baseline_allowed": True,
                "accepted": True,
            },
        }
    )
    merge_boundary._append_merge_entry(1, "sha1", "some title")

    assert merge_boundary.cmd_train_status(as_json=False) == 1


def test_record_full_verify_clears_pending_prs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    merge_boundary._append_merge_entry(1, "sha1", "some title")

    def _run(cmd: list[str], **kwargs: Any) -> MagicMock:
        return MagicMock(
            returncode=0,
            stdout=json.dumps({"verification_scope": "release-baseline", "release_baseline_allowed": True}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _run)

    exit_code = merge_boundary.cmd_record_full_verify("devtools verify --all")

    assert exit_code == 0
    assert merge_boundary.cmd_train_status(as_json=False) == 0


def test_record_full_verify_propagates_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    merge_boundary._append_merge_entry(1, "sha1", "some title")

    def _run(cmd: list[str], **kwargs: Any) -> MagicMock:
        return MagicMock(returncode=1, stdout="", stderr="broke")

    monkeypatch.setattr(subprocess, "run", _run)

    exit_code = merge_boundary.cmd_record_full_verify("devtools verify --all")

    assert exit_code == 1
    ledger = merge_boundary._read_ledger()
    assert ledger["last_full_verify"]["exit_code"] == 1
    assert ledger["last_full_verify"]["accepted"] is False
    assert merge_boundary.cmd_train_status(as_json=False) == 1


def test_record_full_verify_rejects_success_without_structured_release_permission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    merge_boundary._append_merge_entry(1, "sha1", "some title")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda _cmd, **_kwargs: MagicMock(returncode=0, stdout="all good\n", stderr=""),
    )

    assert merge_boundary.cmd_record_full_verify("devtools verify --all") == 1
    ledger = merge_boundary._read_ledger()
    assert ledger["last_full_verify"]["accepted"] is False
    assert merge_boundary.cmd_train_status(as_json=False) == 1


def test_record_full_verify_rejects_skip_slow_without_typed_authorization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    merge_boundary._append_merge_entry(1, "sha1", "some title")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda _cmd, **_kwargs: MagicMock(
            returncode=0,
            stdout=json.dumps({"verification_scope": "narrow-terminal", "release_baseline_allowed": False}),
            stderr="",
        ),
    )

    assert merge_boundary.cmd_record_full_verify("devtools verify --all --skip-slow") == 1
    assert merge_boundary._read_ledger()["last_full_verify"]["accepted"] is False
    assert merge_boundary.cmd_train_status(as_json=False) == 1


def test_record_full_verify_rejects_explicit_typed_narrow_terminal_authorization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    merge_boundary._append_merge_entry(1, "sha1", "some title")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda _cmd, **_kwargs: MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "verification_scope": "narrow-terminal",
                    "terminal_authorization": "narrow-terminal",
                    "release_baseline_allowed": True,
                }
            ),
            stderr="",
        ),
    )

    assert merge_boundary.cmd_record_full_verify("devtools verify --all --skip-slow") == 1
    assert merge_boundary._read_ledger()["last_full_verify"]["accepted"] is False
    assert merge_boundary.cmd_train_status(as_json=False) == 1


def test_record_full_verify_rejects_untyped_scope_even_when_permission_is_true(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    merge_boundary._append_merge_entry(1, "sha1", "some title")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda _cmd, **_kwargs: MagicMock(
            returncode=0,
            stdout=json.dumps({"release_baseline_allowed": True}),
            stderr="",
        ),
    )

    assert merge_boundary.cmd_record_full_verify("devtools verify --all") == 1
    assert merge_boundary._read_ledger()["last_full_verify"]["accepted"] is False
    assert merge_boundary.cmd_train_status(as_json=False) == 1

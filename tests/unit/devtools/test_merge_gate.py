from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from devtools import merge_gate


def _fake_run(pr_view: dict[str, object], comments: list[dict[str, object]], local_exit: int = 0) -> object:
    def _run(cmd: list[str], **kwargs: object) -> MagicMock:
        if cmd[:3] == ["gh", "pr", "view"]:
            return MagicMock(returncode=0, stdout=json.dumps(pr_view), stderr="")
        if cmd[:2] == ["gh", "api"]:
            return MagicMock(returncode=0, stdout=json.dumps(comments), stderr="")
        return MagicMock(returncode=local_exit, stdout="ok\n", stderr="")

    return _run


def test_record_persists_receipt_keyed_to_current_head_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run({"headRefOid": "abc123", "headRefName": "feature/x"}, []),
    )

    exit_code = merge_gate.cmd_record(42, "true")

    assert exit_code == 0
    receipt = json.loads(merge_gate._receipt_path(42).read_text())
    assert receipt["head_sha"] == "abc123"
    assert receipt["exit_code"] == 0


def test_record_captures_nonzero_local_command_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run({"headRefOid": "abc123", "headRefName": "feature/x"}, [], local_exit=1),
    )

    exit_code = merge_gate.cmd_record(42, "false")

    assert exit_code == 1
    receipt = json.loads(merge_gate._receipt_path(42).read_text())
    assert receipt["exit_code"] == 1


def _base_pr_view(head_sha: str = "abc123", committed_date: str = "2026-08-01T12:00:00Z") -> dict[str, object]:
    return {
        "headRefOid": head_sha,
        "headRefName": "feature/x",
        "state": "OPEN",
        "mergeStateStatus": "CLEAN",
        "commits": [{"oid": head_sha, "committedDate": committed_date}],
    }


def test_check_blocks_when_no_receipt_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", _fake_run(_base_pr_view(), []))

    exit_code = merge_gate.cmd_check(42, max_age_s=3600, as_json=False)

    assert exit_code == 1


def test_check_ok_when_receipt_fresh_and_matches_head_with_no_late_comments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", _fake_run(_base_pr_view(), []))
    merge_gate.cmd_record(42, "true")

    exit_code = merge_gate.cmd_check(42, max_age_s=3600, as_json=False)

    assert exit_code == 0


def test_check_blocks_when_receipt_is_for_a_stale_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", _fake_run(_base_pr_view(head_sha="abc123"), []))
    merge_gate.cmd_record(42, "true")

    # A new commit landed after the receipt was recorded.
    monkeypatch.setattr(subprocess, "run", _fake_run(_base_pr_view(head_sha="def456"), []))
    exit_code = merge_gate.cmd_check(42, max_age_s=3600, as_json=False)

    assert exit_code == 1


def test_check_blocks_on_review_comment_newer_than_head_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view(committed_date="2026-08-01T12:00:00Z")
    late_comment = [
        {
            "path": "polylogue/foo.py",
            "line": 10,
            "created_at": "2026-08-01T12:05:00Z",
            "body": "this is a real finding",
        }
    ]
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, []))
    merge_gate.cmd_record(42, "true")

    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, late_comment))
    exit_code = merge_gate.cmd_check(42, max_age_s=3600, as_json=False)

    assert exit_code == 1


def test_check_ignores_comment_older_than_head_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view(committed_date="2026-08-01T12:00:00Z")
    stale_comment = [
        {
            "path": "polylogue/foo.py",
            "line": 10,
            "created_at": "2026-08-01T11:55:00Z",
            "body": "already addressed by the fix commit",
        }
    ]
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, []))
    merge_gate.cmd_record(42, "true")

    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, stale_comment))
    exit_code = merge_gate.cmd_check(42, max_age_s=3600, as_json=False)

    assert exit_code == 0


def test_check_blocks_when_pr_is_not_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    pr_view["state"] = "MERGED"
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, []))

    exit_code = merge_gate.cmd_check(42, max_age_s=3600, as_json=False)

    assert exit_code == 1


def test_check_blocks_when_receipt_older_than_max_age(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", _fake_run(_base_pr_view(), []))
    merge_gate.cmd_record(42, "true")

    exit_code = merge_gate.cmd_check(42, max_age_s=-1, as_json=False)

    assert exit_code == 1

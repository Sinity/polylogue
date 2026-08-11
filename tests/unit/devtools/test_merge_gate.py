from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from devtools import merge_gate, pr_scope
from tests.infra.frozen_clock import FrozenClock

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
                "evidence": [{"kind": "test", "ref": "tests/unit/devtools/test_merge_gate.py"}],
                "successors": [],
            }
        ],
    }
    carrier["scope_digest"] = pr_scope.carrier_digest(carrier)
    return pr_scope.render_carrier(carrier)


def _fake_run(
    pr_view: dict[str, object],
    comments: list[dict[str, object]],
    local_exit: int = 0,
    local_head_sha: str = "abc123",
    dirty: bool = False,
    poll_rounds: list[list[dict[str, object]]] | None = None,
) -> object:
    """``comments`` (inline review comments) is returned on every poll round
    unless ``poll_rounds`` gives an explicit per-round sequence (for testing
    the multi-round poll itself). Issue comments and review bodies are always
    empty here -- covered separately in the normalization tests below."""
    comment_rounds: list[list[dict[str, object]]] = poll_rounds if poll_rounds is not None else [comments]
    call_count = {"round": 0}
    pr_view.setdefault("body", _scope_body(str(pr_view["headRefOid"])))
    pr_view.setdefault("isDraft", False)

    def _run(cmd: list[str], **kwargs: object) -> MagicMock:
        joined = " ".join(cmd)
        if cmd[:3] == ["gh", "pr", "view"]:
            return MagicMock(returncode=0, stdout=json.dumps(pr_view), stderr="")
        if "/issues/" in joined and "/comments" in joined:
            return MagicMock(returncode=0, stdout=json.dumps([[]]), stderr="")
        if "/pulls/" in joined and "/reviews" in joined:
            return MagicMock(returncode=0, stdout=json.dumps([[]]), stderr="")
        if "/pulls/" in joined and "/comments" in joined:
            round_index = min(call_count["round"], len(comment_rounds) - 1)
            call_count["round"] += 1
            return MagicMock(returncode=0, stdout=json.dumps([comment_rounds[round_index]]), stderr="")
        if cmd[:2] == ["git", "rev-parse"]:
            if "--show-toplevel" in cmd:
                return MagicMock(returncode=0, stdout=str(Path.cwd()) + "\n", stderr="")
            return MagicMock(returncode=0, stdout=local_head_sha + "\n", stderr="")
        if cmd[:2] == ["git", "status"]:
            return MagicMock(returncode=0, stdout=" M dirty.py\n" if dirty else "", stderr="")
        env = kwargs.get("env")
        if isinstance(env, dict) and merge_gate.VERIFICATION_RECEIPT_PATH_ENV in env:
            Path(env[merge_gate.VERIFICATION_RECEIPT_PATH_ENV]).write_text(
                json.dumps(
                    {
                        "invocation_id": env[merge_gate.VERIFICATION_INVOCATION_ID_ENV],
                        "git_head": local_head_sha,
                        "checkout_root": str(Path.cwd()),
                        "exit_code": local_exit,
                        "verification_scope": "affected",
                        "release_baseline_allowed": False,
                        "terminal_authorization": None,
                    }
                )
            )
        return MagicMock(
            returncode=local_exit,
            stdout=json.dumps({"verification_scope": "affected", "release_baseline_allowed": False}),
            stderr="",
        )

    return _run


def test_record_persists_receipt_keyed_to_current_head_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run({"headRefOid": "abc123", "headRefName": "feature/x"}, [], local_head_sha="abc123"),
    )

    exit_code = merge_gate.cmd_record(42, "devtools test tests/unit/foo.py")

    assert exit_code == 0
    receipt = json.loads(merge_gate._receipt_path(42).read_text())
    assert receipt["head_sha"] == "abc123"
    assert receipt["pr_scope_digest"]
    assert receipt["exit_code"] == 0


@pytest.mark.parametrize("command", ["devtools verify", "devtools verify --lab", "devtools verify --json --skip-slow"])
def test_check_accepts_affected_receipt_without_release_baseline_permission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: str
) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    _record(monkeypatch, pr_view, command=command)
    receipt_path = merge_gate._receipt_path(42)
    receipt = json.loads(receipt_path.read_text())
    receipt["release_baseline_allowed"] = False
    receipt_path.write_text(json.dumps(receipt))

    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, []))
    assert merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False) == 0


@pytest.mark.parametrize(
    "command", ["devtools verify --all", "devtools verify --full", "devtools verify --seed-testmon"]
)
def test_check_blocks_full_receipt_without_release_baseline_permission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: str
) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    _record(monkeypatch, pr_view, command=command)
    receipt_path = merge_gate._receipt_path(42)
    receipt = json.loads(receipt_path.read_text())
    receipt["verification_scope"] = "release-baseline"
    receipt["release_baseline_allowed"] = False
    receipt_path.write_text(json.dumps(receipt))

    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, []))
    assert merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False) == 1


def test_record_rejects_stale_plausible_stdout_without_bound_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view: dict[str, object] = {"headRefOid": "abc123", "headRefName": "feature/x"}
    base = cast(Callable[..., MagicMock], _fake_run(pr_view, [], local_head_sha="abc123"))

    def _run(cmd: list[str], **kwargs: object) -> MagicMock:
        if cmd[:2] in (["git", "rev-parse"], ["git", "status"]):
            return base(cmd, **kwargs)
        if cmd[:3] == ["gh", "pr", "view"]:
            return base(cmd, **kwargs)
        return MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "invocation_id": "stale-invocation",
                    "git_head": "abc123",
                    "checkout_root": str(tmp_path),
                    "exit_code": 0,
                    "verification_scope": "release-baseline",
                    "release_baseline_allowed": True,
                    "terminal_authorization": None,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _run)
    assert merge_gate.cmd_record(42, "devtools verify") == 0
    receipt = json.loads(merge_gate._receipt_path(42).read_text())
    assert receipt["verification_scope"] is None
    assert receipt["release_baseline_allowed"] is None


def test_record_consumes_receipt_after_streamed_verifier_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view: dict[str, object] = {"headRefOid": "abc123", "headRefName": "feature/x"}
    base = cast(Callable[..., MagicMock], _fake_run(pr_view, [], local_head_sha="abc123"))
    payload = {
        "verification_scope": "narrow-terminal",
        "release_baseline_allowed": False,
        "terminal_authorization": "narrow-terminal",
    }

    def _run(cmd: list[str], **kwargs: object) -> MagicMock:
        if cmd[:2] in (["git", "rev-parse"], ["git", "status"]):
            return base(cmd, **kwargs)
        if cmd[:3] == ["gh", "pr", "view"]:
            return base(cmd, **kwargs)
        env = cast(dict[str, str], kwargs["env"])
        Path(env[merge_gate.VERIFICATION_RECEIPT_PATH_ENV]).write_text(
            json.dumps(
                {
                    **payload,
                    "invocation_id": env[merge_gate.VERIFICATION_INVOCATION_ID_ENV],
                    "git_head": "abc123",
                    "checkout_root": str(tmp_path),
                    "exit_code": 0,
                }
            )
        )
        return MagicMock(returncode=0, stdout=f"pytest progress\n{json.dumps(payload, indent=2)}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)
    assert merge_gate.cmd_record(42, "devtools test tests/unit/foo.py") == 0
    receipt = json.loads(merge_gate._receipt_path(42).read_text())
    assert receipt["verification_scope"] == "narrow-terminal"
    assert receipt["release_baseline_allowed"] is False
    assert receipt["terminal_authorization"] == "narrow-terminal"


def test_record_consumes_exact_invocation_receipt_when_verifier_writes_only_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view: dict[str, object] = {"headRefOid": "abc123", "headRefName": "feature/x"}
    base = cast(Callable[..., MagicMock], _fake_run(pr_view, [], local_head_sha="abc123"))

    def _run(cmd: list[str], **kwargs: object) -> MagicMock:
        if cmd[:2] in (["git", "rev-parse"], ["git", "status"]):
            return base(cmd, **kwargs)
        if cmd[:3] == ["gh", "pr", "view"]:
            return base(cmd, **kwargs)
        env = cast(dict[str, str], kwargs["env"])
        receipt_path = Path(env[merge_gate.VERIFICATION_RECEIPT_PATH_ENV])
        receipt_path.write_text(
            json.dumps(
                {
                    "run_id": "new",
                    "invocation_id": env[merge_gate.VERIFICATION_INVOCATION_ID_ENV],
                    "git_head": "abc123",
                    "checkout_root": str(tmp_path),
                    "exit_code": 0,
                    "verification_scope": "affected",
                    "release_baseline_allowed": False,
                    "terminal_authorization": None,
                }
            )
        )
        return MagicMock(returncode=0, stdout="", stderr="pytest progress")

    monkeypatch.setattr(subprocess, "run", _run)
    assert merge_gate.cmd_record(42, "devtools test tests/unit/foo.py") == 0
    receipt = json.loads(merge_gate._receipt_path(42).read_text())
    assert receipt["verification_scope"] == "affected"
    assert receipt["release_baseline_allowed"] is False


def test_record_rejects_invocation_receipt_from_another_head(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view: dict[str, object] = {"headRefOid": "abc123", "headRefName": "feature/x"}
    base = cast(Callable[..., MagicMock], _fake_run(pr_view, [], local_head_sha="abc123"))

    def _run(cmd: list[str], **kwargs: object) -> MagicMock:
        if cmd[:2] in (["git", "rev-parse"], ["git", "status"]):
            return base(cmd, **kwargs)
        if cmd[:3] == ["gh", "pr", "view"]:
            return base(cmd, **kwargs)
        env = cast(dict[str, str], kwargs["env"])
        Path(env[merge_gate.VERIFICATION_RECEIPT_PATH_ENV]).write_text(
            json.dumps(
                {
                    "invocation_id": env[merge_gate.VERIFICATION_INVOCATION_ID_ENV],
                    "git_head": "different",
                    "checkout_root": str(tmp_path),
                    "exit_code": 0,
                    "verification_scope": "affected",
                    "release_baseline_allowed": False,
                }
            )
        )
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)
    assert merge_gate.cmd_record(42, "devtools test tests/unit/foo.py") == 0
    receipt = json.loads(merge_gate._receipt_path(42).read_text())
    assert receipt["verification_scope"] is None
    assert receipt["release_baseline_allowed"] is None


def test_record_rejects_unrelated_invocation_receipt_with_same_head_and_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view: dict[str, object] = {"headRefOid": "abc123", "headRefName": "feature/x"}
    base = cast(Callable[..., MagicMock], _fake_run(pr_view, [], local_head_sha="abc123"))

    def _run(cmd: list[str], **kwargs: object) -> MagicMock:
        if cmd[:2] in (["git", "rev-parse"], ["git", "status"]):
            return base(cmd, **kwargs)
        if cmd[:3] == ["gh", "pr", "view"]:
            return base(cmd, **kwargs)
        env = cast(dict[str, str], kwargs["env"])
        Path(env[merge_gate.VERIFICATION_RECEIPT_PATH_ENV]).write_text(
            json.dumps(
                {
                    "invocation_id": "another-command",
                    "git_head": "abc123",
                    "checkout_root": str(tmp_path),
                    "exit_code": 0,
                    "verification_scope": "affected",
                    "release_baseline_allowed": False,
                }
            )
        )
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)
    assert merge_gate.cmd_record(42, "devtools test tests/unit/foo.py") == 0
    receipt = json.loads(merge_gate._receipt_path(42).read_text())
    assert receipt["verification_scope"] is None
    assert receipt["release_baseline_allowed"] is None


def test_record_anchors_invocation_to_repository_root_from_subdirectory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    subdirectory = tmp_path / "devtools"
    subdirectory.mkdir()
    monkeypatch.chdir(subdirectory)
    pr_view: dict[str, object] = {"headRefOid": "abc123", "headRefName": "feature/x"}
    base = cast(Callable[..., MagicMock], _fake_run(pr_view, [], local_head_sha="abc123"))

    def _run(cmd: list[str], **kwargs: object) -> MagicMock:
        if cmd == ["git", "rev-parse", "--show-toplevel"]:
            return MagicMock(returncode=0, stdout=str(tmp_path) + "\n", stderr="")
        if cmd[:2] in (["git", "rev-parse"], ["git", "status"]):
            return base(cmd, **kwargs)
        if cmd[:3] == ["gh", "pr", "view"]:
            return base(cmd, **kwargs)
        assert kwargs["cwd"] == tmp_path
        env = cast(dict[str, str], kwargs["env"])
        Path(env[merge_gate.VERIFICATION_RECEIPT_PATH_ENV]).write_text(
            json.dumps(
                {
                    "invocation_id": env[merge_gate.VERIFICATION_INVOCATION_ID_ENV],
                    "git_head": "abc123",
                    "checkout_root": str(tmp_path),
                    "exit_code": 0,
                    "verification_scope": "affected",
                    "release_baseline_allowed": False,
                }
            )
        )
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)
    assert merge_gate.cmd_record(42, "devtools test tests/unit/foo.py") == 0
    receipt = json.loads(merge_gate._receipt_path(42).read_text())
    assert receipt["verification_scope"] == "affected"


def test_record_accepts_authoritative_dependabot_dependency_only_pr_without_carrier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view: dict[str, object] = {
        "headRefOid": "abc123",
        "headRefName": "dependabot/uv/hypothesis",
        "body": "",
        "isDraft": False,
        "author": {"login": "app/dependabot", "is_bot": True},
        "files": [{"path": "pyproject.toml"}, {"path": "uv.lock"}],
    }
    base = cast(Callable[..., MagicMock], _fake_run(pr_view, [], local_head_sha="abc123"))

    def _run(cmd: list[str], **kwargs: object) -> MagicMock:
        if cmd[:2] in (["git", "rev-parse"], ["git", "status"]):
            return base(cmd, **kwargs)
        if cmd[:3] == ["gh", "pr", "view"]:
            return base(cmd, **kwargs)
        return MagicMock(
            returncode=0,
            stdout=json.dumps({"verification_scope": "affected", "release_baseline_allowed": False}),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _run)
    assert merge_gate.cmd_record(42, "devtools verify") == 0


def test_check_accepts_authoritative_dependabot_dependency_only_pr_without_carrier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    pr_view.update(
        {
            "body": "",
            "author": {"login": "app/dependabot", "is_bot": True},
            "files": [{"path": "pyproject.toml"}, {"path": "uv.lock"}],
        }
    )
    _record(monkeypatch, pr_view)
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, []))
    assert merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False) == 0


def test_record_rejects_draft_dependabot_dependency_only_pr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = {
        "headRefOid": "abc123",
        "headRefName": "dependabot/uv/hypothesis",
        "body": "",
        "isDraft": True,
        "author": {"login": "app/dependabot", "is_bot": True},
        "files": [{"path": "pyproject.toml"}, {"path": "uv.lock"}],
    }
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, [], local_head_sha="abc123"))
    assert merge_gate.cmd_record(42, "devtools verify") == 2
    assert not merge_gate._receipt_path(42).exists()


def test_record_captures_nonzero_local_command_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run({"headRefOid": "abc123", "headRefName": "feature/x"}, [], local_exit=1, local_head_sha="abc123"),
    )

    exit_code = merge_gate.cmd_record(42, "devtools test somefile")

    assert exit_code == 1
    receipt = json.loads(merge_gate._receipt_path(42).read_text())
    assert receipt["exit_code"] == 1


def test_record_refuses_when_local_checkout_does_not_match_pr_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run({"headRefOid": "abc123", "headRefName": "feature/x"}, [], local_head_sha="deadbeef"),
    )

    exit_code = merge_gate.cmd_record(42, "devtools test somefile")

    assert exit_code == 2
    assert not merge_gate._receipt_path(42).exists()


def test_record_refuses_when_checkout_is_dirty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run({"headRefOid": "abc123", "headRefName": "feature/x"}, [], local_head_sha="abc123", dirty=True),
    )

    exit_code = merge_gate.cmd_record(42, "devtools test somefile")

    assert exit_code == 2
    assert not merge_gate._receipt_path(42).exists()


def _base_pr_view(head_sha: str = "abc123", committed_date: str = "2026-08-01T12:00:00Z") -> dict[str, object]:
    return {
        "headRefOid": head_sha,
        "headRefName": "feature/x",
        "state": "OPEN",
        "mergeStateStatus": "CLEAN",
        "commits": [{"oid": head_sha, "committedDate": committed_date}],
    }


def _record(monkeypatch: pytest.MonkeyPatch, pr_view: dict[str, object], command: str = "devtools test x") -> None:
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, [], local_head_sha=str(pr_view["headRefOid"])))
    merge_gate.cmd_record(42, command)


def test_check_blocks_when_no_receipt_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", _fake_run(_base_pr_view(), []))

    exit_code = merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False)

    assert exit_code == 1


def test_check_ok_when_receipt_fresh_and_matches_head_with_no_late_comments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    _record(monkeypatch, _base_pr_view())

    monkeypatch.setattr(subprocess, "run", _fake_run(_base_pr_view(), []))
    exit_code = merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False)

    assert exit_code == 0


def test_check_refuses_an_unrelated_local_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    _record(monkeypatch, pr_view)
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, [], local_head_sha="other-head"))

    assert merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False) == 1
    assert "does not match PR #42's head" in capsys.readouterr().out


def test_check_refuses_a_dirty_local_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    _record(monkeypatch, pr_view)
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, [], dirty=True))

    assert merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False) == 1
    assert "has uncommitted changes" in capsys.readouterr().out


def test_check_rejects_command_text_without_typed_scope_or_permission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    _record(monkeypatch, pr_view, command="devtools verify --all")
    receipt_path = merge_gate._receipt_path(42)
    receipt = json.loads(receipt_path.read_text())
    receipt["verification_scope"] = None
    receipt["release_baseline_allowed"] = True
    receipt_path.write_text(json.dumps(receipt))

    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, []))
    assert merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False) == 1


@pytest.mark.parametrize(
    ("receipt_field", "mutated_value", "reason"),
    [
        ("pr_scope_digest", "changed-body-digest", "pr_scope_digest"),
        ("pr_scope_beads_digest", "stale", "pr_scope_beads_digest"),
        ("pr_scope_assigned_beads", ["polylogue-other"], "pr_scope_assigned_beads"),
        ("pr_scope_mutated_beads", ["polylogue-unlisted"], "pr_scope_mutated_beads"),
        ("pr_scope_attestation_digest", "stale-attestation", "pr_scope_attestation_digest"),
    ],
)
def test_check_blocks_when_receipt_scope_components_are_mutated(
    receipt_field: str,
    mutated_value: object,
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    _record(monkeypatch, pr_view)
    receipt_path = merge_gate._receipt_path(42)
    receipt = json.loads(receipt_path.read_text())
    receipt[receipt_field] = mutated_value
    receipt_path.write_text(json.dumps(receipt))

    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, []))
    exit_code = merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False)

    assert exit_code == 1
    assert reason in capsys.readouterr().out


def test_check_blocks_when_receipt_is_for_a_stale_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _record(monkeypatch, _base_pr_view(head_sha="abc123"))

    # A new commit landed after the receipt was recorded.
    monkeypatch.setattr(subprocess, "run", _fake_run(_base_pr_view(head_sha="def456"), []))
    exit_code = merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False)

    assert exit_code == 1


def test_check_blocks_on_review_comment_newer_than_head_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view(committed_date="2026-08-01T12:00:00Z")
    _record(monkeypatch, pr_view)

    late_comment = [
        {
            "id": 111,
            "path": "polylogue/foo.py",
            "line": 10,
            "created_at": "2026-08-01T12:05:00Z",
            "body": "this is a real finding",
        }
    ]
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, late_comment))
    exit_code = merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False)

    assert exit_code == 1


def test_check_catches_a_comment_that_arrives_only_on_a_later_poll_round(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The PR #3502 incident: CodeRabbit posts 30-60s after the first snapshot."""
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view(committed_date="2026-08-01T12:00:00Z")
    _record(monkeypatch, pr_view)

    late_comment = {
        "id": 222,
        "path": "polylogue/foo.py",
        "line": 10,
        "created_at": "2026-08-01T12:05:00Z",
        "body": "arrived late",
    }
    # Round 1: empty. Round 2: the comment has landed.
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, [], poll_rounds=[[], [late_comment]]))
    exit_code = merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=2, poll_interval_s=0, as_json=False)

    assert exit_code == 1


def test_check_ignores_comment_older_than_head_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view(committed_date="2026-08-01T12:00:00Z")
    _record(monkeypatch, pr_view)

    stale_comment = [
        {
            "id": 333,
            "path": "polylogue/foo.py",
            "line": 10,
            "created_at": "2026-08-01T11:55:00Z",
            "body": "already addressed by the fix commit",
        }
    ]
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, stale_comment))
    exit_code = merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False)

    assert exit_code == 0


def test_check_allows_an_acknowledged_late_comment_for_the_same_head_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view(committed_date="2026-08-01T12:00:00Z")
    _record(monkeypatch, pr_view)

    late_comment = [
        {
            "id": 444,
            "path": "polylogue/foo.py",
            "line": 10,
            "created_at": "2026-08-01T12:05:00Z",
            "body": "false positive, already fine",
        }
    ]
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, []))
    merge_gate.cmd_ack(42, 444, reason="false positive")

    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, late_comment))
    exit_code = merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False)

    assert exit_code == 0


def test_check_ignores_an_ack_recorded_for_a_different_head_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A new push must invalidate old acks -- otherwise a stale triage silently covers new code."""
    monkeypatch.chdir(tmp_path)
    old_pr_view = _base_pr_view(head_sha="abc123", committed_date="2026-08-01T12:00:00Z")
    monkeypatch.setattr(subprocess, "run", _fake_run(old_pr_view, []))
    merge_gate.cmd_ack(42, 555, reason="false positive on the old commit")

    new_pr_view = _base_pr_view(head_sha="def456", committed_date="2026-08-01T13:00:00Z")
    _record(monkeypatch, new_pr_view)
    late_comment = [
        {
            "id": 555,
            "path": "polylogue/foo.py",
            "line": 10,
            "created_at": "2026-08-01T13:05:00Z",
            "body": "same comment id, but this is a new push",
        }
    ]
    monkeypatch.setattr(subprocess, "run", _fake_run(new_pr_view, late_comment))
    exit_code = merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False)

    assert exit_code == 1


def test_check_blocks_when_pr_is_not_open(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    pr_view["state"] = "MERGED"
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, []))

    exit_code = merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False)

    assert exit_code == 1


def test_check_blocks_when_receipt_older_than_max_age(
    frozen_clock: FrozenClock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _record(monkeypatch, _base_pr_view())

    frozen_clock.advance(7200)
    monkeypatch.setattr(subprocess, "run", _fake_run(_base_pr_view(), []))
    exit_code = merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False)

    assert exit_code == 1


def test_check_blocks_when_receipt_exit_code_is_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, [], local_exit=1, local_head_sha="abc123"))
    merge_gate.cmd_record(42, "devtools test x")

    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, []))
    exit_code = merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False)

    assert exit_code == 1


def test_check_blocks_when_merge_state_status_is_dirty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    _record(monkeypatch, pr_view)

    pr_view["mergeStateStatus"] = "DIRTY"
    monkeypatch.setattr(subprocess, "run", _fake_run(pr_view, []))
    exit_code = merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False)

    assert exit_code == 1


def test_check_blocks_when_comment_polling_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    _record(monkeypatch, pr_view)

    def _run_with_broken_api(cmd: list[str], **kwargs: object) -> MagicMock:
        if cmd[:3] == ["gh", "pr", "view"]:
            return MagicMock(returncode=0, stdout=json.dumps(pr_view), stderr="")
        if cmd[:2] == ["gh", "api"]:
            return MagicMock(returncode=1, stdout="", stderr="rate limited")
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _run_with_broken_api)
    exit_code = merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False)

    assert exit_code == 1


def test_check_catches_a_late_review_body_not_just_inline_comments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Review summaries and issue-level comments carry findings too -- a gate
    that only watches inline diff comments misses them."""
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view(committed_date="2026-08-01T12:00:00Z")
    _record(monkeypatch, pr_view)

    def _run_with_late_review(cmd: list[str], **kwargs: object) -> MagicMock:
        joined = " ".join(cmd)
        if cmd[:3] == ["gh", "pr", "view"]:
            return MagicMock(returncode=0, stdout=json.dumps(pr_view), stderr="")
        if "/pulls/" in joined and "/reviews" in joined:
            late_review = [{"id": 999, "submitted_at": "2026-08-01T12:10:00Z", "body": "Request changes: real bug"}]
            return MagicMock(returncode=0, stdout=json.dumps([late_review]), stderr="")
        if "/issues/" in joined and "/comments" in joined:
            return MagicMock(returncode=0, stdout=json.dumps([[]]), stderr="")
        if "/pulls/" in joined and "/comments" in joined:
            return MagicMock(returncode=0, stdout=json.dumps([[]]), stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _run_with_late_review)
    exit_code = merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False)

    assert exit_code == 1


def _fake_run_with_status_capture(
    pr_view: dict[str, object],
    comments: list[dict[str, object]],
    *,
    status_calls: list[list[str]],
    status_exit: int = 0,
) -> object:
    """Like ``_fake_run`` but also captures ``gh api .../statuses/<sha>`` calls
    (the commit-status POST issued by ``--post-status``) into ``status_calls``,
    the same way other devtools tests intercept ``gh`` subprocess invocations
    -- by matching on the argv shape rather than hitting the network."""
    base: Callable[..., MagicMock] = _fake_run(pr_view, comments)  # type: ignore[assignment]

    def _run(cmd: list[str], **kwargs: object) -> MagicMock:
        if cmd[:2] == ["gh", "api"] and "/statuses/" in cmd[2]:
            status_calls.append(cmd)
            return MagicMock(returncode=status_exit, stdout="{}", stderr="" if status_exit == 0 else "boom")
        return base(cmd, **kwargs)

    return _run


def test_check_post_status_posts_success_for_ok_verdict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view(head_sha="abc123")
    _record(monkeypatch, pr_view)

    status_calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_run_with_status_capture(pr_view, [], status_calls=status_calls))

    exit_code = merge_gate.cmd_check(
        42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False, post_status=True
    )

    assert exit_code == 0
    assert len(status_calls) == 1
    call = status_calls[0]
    assert call[2] == "repos/{owner}/{repo}/statuses/abc123"
    assert "state=success" in call
    assert "context=merge-gate" in call


def test_check_post_status_posts_failure_with_block_reason_for_block_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view(head_sha="def456")
    # No receipt recorded at all -- guaranteed BLOCK.

    status_calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_run_with_status_capture(pr_view, [], status_calls=status_calls))
    monkeypatch.setattr(merge_gate, "_git_head_sha", lambda: "def456")

    exit_code = merge_gate.cmd_check(
        42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False, post_status=True
    )

    assert exit_code == 1
    assert len(status_calls) == 1
    call = status_calls[0]
    assert call[2] == "repos/{owner}/{repo}/statuses/def456"
    assert "state=failure" in call
    description_arg = next(arg for arg in call if arg.startswith("description="))
    assert "no local verification receipt" in description_arg


def test_check_post_status_targets_the_pr_current_head_sha_not_a_stale_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A receipt recorded against an old sha, with the PR now on a new head,
    must post the status against the *current* head sha -- posting against
    the stale sha would create a status GitHub never associates with the
    commit branch protection actually needs to gate on."""
    monkeypatch.chdir(tmp_path)
    _record(monkeypatch, _base_pr_view(head_sha="old111"))

    new_pr_view = _base_pr_view(head_sha="new222")
    status_calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_run_with_status_capture(new_pr_view, [], status_calls=status_calls))

    exit_code = merge_gate.cmd_check(
        42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False, post_status=True
    )

    assert exit_code == 1  # stale-receipt BLOCK
    assert len(status_calls) == 1
    assert status_calls[0][2] == "repos/{owner}/{repo}/statuses/new222"
    assert "state=failure" in status_calls[0]


def test_check_without_post_status_flag_never_calls_gh_api_statuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view()
    _record(monkeypatch, pr_view)

    status_calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_run_with_status_capture(pr_view, [], status_calls=status_calls))

    merge_gate.cmd_check(42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=False)

    assert status_calls == []


def test_check_post_status_failure_is_reported_but_does_not_change_verdict_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A `gh api` failure while posting the status (rate limit, auth) must not
    silently flip an otherwise-OK verdict, and must not raise -- it surfaces
    as an advisory in the JSON verdict for the caller to notice."""
    monkeypatch.chdir(tmp_path)
    pr_view = _base_pr_view(head_sha="abc123")
    _record(monkeypatch, pr_view)

    status_calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run_with_status_capture(pr_view, [], status_calls=status_calls, status_exit=1),
    )

    exit_code = merge_gate.cmd_check(
        42, max_age_s=3600, poll_rounds=1, poll_interval_s=0, as_json=True, post_status=True
    )

    assert exit_code == 0
    assert len(status_calls) == 1

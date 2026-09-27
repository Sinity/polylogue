"""``devtools test`` and ``devtools verify`` refuse to test the base by accident.

An agent's shell can reset to the primary checkout, which sits on the default
branch, between commands; a runner started there tested the base and reported
a green that said nothing about the change.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from devtools import run_tests, verify
from devtools.checkout_identity import (
    ON_DEFAULT_BRANCH_FLAG,
    REFUSAL_EXIT,
    checkout_identity,
    default_branch_refusal,
)
from devtools.verify_runs import VerifyRun

pytestmark = pytest.mark.real_checkout_identity


class _PastTheGuardError(Exception):
    """Raised by the first step after the branch guard: the run was admitted."""


def _repository(path: Path, branch: str) -> Path:
    path.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True)

    git("init", "--initial-branch", "master")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.test")
    (path / "seed.txt").write_text("seed\n", encoding="utf-8")
    git("add", "seed.txt")
    git("commit", "-m", "Seed")
    if branch != "master":
        git("switch", "-c", branch)
    return path


def _admitted(*_args: object, **_kwargs: object) -> None:
    raise _PastTheGuardError


def test_the_default_branch_is_refused_and_a_feature_branch_is_not(tmp_path: Path) -> None:
    """Anti-vacuity: compare against any branch but the default and the first
    assertion goes red; ignore ``allowed`` and the second does."""
    base = checkout_identity(_repository(tmp_path / "base", "master"))
    feature = checkout_identity(_repository(tmp_path / "feature", "claude/change"))

    assert default_branch_refusal(base, command="devtools test", allowed=False) is not None
    assert default_branch_refusal(base, command="devtools test", allowed=True) is None
    assert default_branch_refusal(feature, command="devtools test", allowed=False) is None


def test_verify_refuses_on_the_default_branch_before_any_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Anti-vacuity: drop the guard from ``verify._main`` and the run proceeds
    to its first step, raising ``_PastTheGuardError`` instead of returning."""
    monkeypatch.setattr(verify, "ROOT", _repository(tmp_path / "base", "master"))
    monkeypatch.setattr(verify, "reconcile_and_record_abandoned_verify_runs", _admitted)

    assert verify._main(["--quick", "--json"]) == REFUSAL_EXIT

    refusal = json.loads(capsys.readouterr().out)
    assert refusal["diagnosis"] == "default_branch_refused"
    assert "env -C" in refusal["message"]


def test_verify_runs_on_the_default_branch_when_asked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The opt-in admits the run, and its first line names what it tests."""
    root = _repository(tmp_path / "base", "master")
    monkeypatch.setattr(verify, "ROOT", root)
    monkeypatch.setattr(verify, "reconcile_and_record_abandoned_verify_runs", _admitted)

    with pytest.raises(_PastTheGuardError):
        verify._main(["--quick", ON_DEFAULT_BRANCH_FLAG])

    first = capsys.readouterr().err.splitlines()[0]
    assert first.startswith(f"verify: checkout={root.resolve()} branch=master head=")


def test_verify_runs_on_a_feature_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(verify, "ROOT", _repository(tmp_path / "feature", "claude/change"))
    monkeypatch.setattr(verify, "reconcile_and_record_abandoned_verify_runs", _admitted)

    with pytest.raises(_PastTheGuardError):
        verify._main(["--quick"])


def test_focused_tests_refuse_on_the_default_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Anti-vacuity: drop the guard from ``run_tests.main`` and the run reaches
    the checkout import check, raising ``_PastTheGuardError``."""
    root = _repository(tmp_path / "base", "master")
    monkeypatch.setattr(run_tests, "ROOT", root)
    monkeypatch.setattr(run_tests, "_anchor_test_paths", lambda: None)
    monkeypatch.setattr(run_tests, "assert_polylogue_matches_checkout", _admitted)

    assert run_tests.main(["tests/unit/example_test.py"]) == REFUSAL_EXIT
    assert "refused on the default branch `master`" in capsys.readouterr().err

    with pytest.raises(_PastTheGuardError):
        run_tests.main(["tests/unit/example_test.py", ON_DEFAULT_BRANCH_FLAG])


def test_a_receipt_names_the_branch_it_tested(tmp_path: Path) -> None:
    """Anti-vacuity: drop ``git_branch`` from the receipt and a cited receipt
    no longer says whether it tested the base or a change."""
    root = _repository(tmp_path / "feature", "claude/change")

    run = VerifyRun(tier="focused-test", argv=[], git_head=None, root=root, mirror_current=False)

    receipt = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
    assert receipt["git_branch"] == "claude/change"


def test_the_verify_verdict_names_the_tested_checkout(capsys: pytest.CaptureFixture[str]) -> None:
    """Anti-vacuity: drop the identity clause and the last line stops naming
    the branch and head the receipt proves."""
    payload = {
        "artifact_dir": ".cache/verify/runs/verify-quick-20260927",
        "exit_code": 0,
        "diagnosis": None,
        "git_head": "a" * 40,
        "git_branch": "claude/change",
    }

    verify._emit(payload, use_json=False, operation=None)

    final = capsys.readouterr().err.strip().splitlines()[-1]
    assert final.endswith(f"checkout={verify.ROOT.resolve()} branch=claude/change head={'a' * 12}")

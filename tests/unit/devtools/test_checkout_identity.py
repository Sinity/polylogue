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


def test_a_queued_run_rechecks_the_branch_when_its_slot_starts(tmp_path: Path) -> None:
    """A checkout that switched to the default branch while queued is refused at start.

    Anti-vacuity: capture execution provenance without the branch check and a
    run admitted on a feature branch executes on the base.
    """
    from devtools import pytest_slot
    from devtools.checkout_identity import ALLOW_DEFAULT_BRANCH_ENV

    root = _repository(tmp_path / "base", "master")
    environment = {"POLYLOGUE_FOCUSED_WORKTREE_PROVENANCE": "1"}

    with pytest.raises(pytest_slot.PytestSlotUnavailableError, match="default branch at slot start"):
        pytest_slot._focused_worktree_provenance(str(root), environment)

    provenance = pytest_slot._focused_worktree_provenance(str(root), {**environment, ALLOW_DEFAULT_BRANCH_ENV: "1"})
    assert provenance is not None and provenance["git_branch"] == "master"


def test_verify_pytest_steps_ask_the_slot_to_recheck_the_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A verifier pytest step, like a focused run, is identified when its slot starts.

    Anti-vacuity: drop the provenance request from ``verify._run`` and the
    slot never re-checks the branch, so a checkout that switched to the
    default branch while the run queued is tested.
    """
    from types import SimpleNamespace

    from devtools.pytest_slot import WORKTREE_PROVENANCE_ENV

    captured: dict[str, str] = {}

    def managed(*_args: object, env: dict[str, str], **_kwargs: object) -> SimpleNamespace:
        captured.update(env)
        return SimpleNamespace(returncode=0, slot="managed", receipt=None)

    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "_clear_pytest_report", lambda _command: None)
    monkeypatch.setattr(verify, "executable_gate_result", lambda *_args, **_kwargs: SimpleNamespace(ok=True))
    monkeypatch.setattr(verify, "run_pytest", managed)
    run = VerifyRun(tier="test", argv=[], git_head=None, root=tmp_path, mirror_current=False)

    verify._run("pytest selected", ["pytest"], run=run, runner="managed")

    assert captured[WORKTREE_PROVENANCE_ENV] == "1"


def test_a_checkout_that_moves_during_verification_voids_the_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Static gates have no slot to re-check the branch, so the run checks at its end.

    Anti-vacuity: drop the end-of-run identity comparison and the switch to
    ``master`` below leaves a passing quick run.
    """
    from devtools import verify as verify_module

    root = _repository(tmp_path / "feature", "claude/change")
    history: dict[str, object] = {}

    def gate_that_switches_branch(
        label: str, command: list[str], *, run: object, runner: str
    ) -> tuple[int, float, dict[str, object]]:
        del label, command, run, runner
        subprocess.run(["git", "switch", "-q", "master"], cwd=root, check=True)
        return 0, 0.0, {"diagnosis": "gate_passed"}

    monkeypatch.setattr(verify_module, "ROOT", root)
    monkeypatch.chdir(root)
    monkeypatch.setattr(verify_module, "assert_polylogue_matches_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr(verify_module, "build_verify_steps", lambda **_kwargs: [("gate only", ["true"])])
    monkeypatch.setattr(verify_module, "_run", gate_that_switches_branch)
    monkeypatch.setattr(verify_module, "append_verify_history", lambda payload: history.update(payload))

    assert verify_module._main(["--quick"]) == 1
    assert history["diagnosis"] == "checkout_moved_during_run"
    assert "the checkout moved during the run" in capsys.readouterr().err


def test_a_rerun_of_different_content_clears_no_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A flaky-rerun pass counts only for the content the failing run executed.

    Anti-vacuity: drop the provenance comparison in ``rerun_failed_once`` and
    the rerun's pass on edited content turns the failed test green.
    """
    from devtools.pytest_rerun import rerun_failed_once
    from devtools.pytest_slot import SlotOutcome

    report_path = tmp_path / "pytest-report.json"
    report_path.write_text(
        json.dumps({"exitcode": 1, "tests": [{"nodeid": "tests/test_a.py::test_x", "outcome": "failed"}]}),
        encoding="utf-8",
    )
    first = {"git_head": "a" * 40, "git_branch": "claude/change", "git_worktree_content_sha256": "before"}

    def rerun_on_edited_content(cmd: list[str], **_kwargs: object) -> SlotOutcome:
        rerun_report = Path(next(arg for arg in cmd if arg.startswith("--polylogue-report-file=")).split("=", 1)[1])
        rerun_report.write_text(
            json.dumps({"tests": [{"nodeid": "tests/test_a.py::test_x", "outcome": "passed"}]}), encoding="utf-8"
        )
        edited = {**first, "git_worktree_content_sha256": "after"}
        return SlotOutcome(returncode=0, slot="held", receipt={"worktree_provenance": edited})

    monkeypatch.setattr("devtools.pytest_rerun.venv_python", lambda root: "python")
    monkeypatch.setattr("devtools.pytest_rerun.run_pytest", rerun_on_edited_content)
    step_dir = tmp_path / "step"
    step_dir.mkdir()

    rerun = rerun_failed_once(report_path=report_path, step_dir=step_dir, env={}, root=tmp_path, first_provenance=first)

    assert rerun is not None
    assert rerun["still_failed"] == ["tests/test_a.py::test_x"]
    assert rerun["content_moved"] is True

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
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", _admitted)

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
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", _admitted)

    with pytest.raises(_PastTheGuardError):
        verify._main(["--quick", ON_DEFAULT_BRANCH_FLAG])

    first = capsys.readouterr().err.splitlines()[0]
    assert first.startswith(f"verify: checkout={root.resolve()} branch=master head=")


def test_verify_runs_on_a_feature_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(verify, "ROOT", _repository(tmp_path / "feature", "claude/change"))
    monkeypatch.setattr(verify, "assert_polylogue_matches_checkout", _admitted)

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
        return SimpleNamespace(returncode=0, slot="managed", receipt=None, termination=None)

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
    monkeypatch.setattr(verify_module, "append_verify_history", lambda payload, **_kwargs: history.update(payload))

    assert verify_module._main(["--quick"]) == 1
    assert history["diagnosis"] == "checkout_moved_during_run"
    assert "the checkout moved during the run" in capsys.readouterr().err


def test_a_content_edit_during_verification_voids_the_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Branch and HEAD alone do not identify the tree; an uncommitted edit mid-run voids it.

    Anti-vacuity: compare only branch and HEAD and this run passes.
    """
    from devtools import verify as verify_module

    root = _repository(tmp_path / "feature", "claude/change")
    history: dict[str, object] = {}

    def gate_that_edits(
        label: str, command: list[str], *, run: object, runner: str
    ) -> tuple[int, float, dict[str, object]]:
        del label, command, run, runner
        (root / "seed.txt").write_text("edited\n", encoding="utf-8")
        return 0, 0.0, {"diagnosis": "gate_passed"}

    monkeypatch.setattr(verify_module, "ROOT", root)
    monkeypatch.chdir(root)
    monkeypatch.setattr(verify_module, "assert_polylogue_matches_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr(verify_module, "build_verify_steps", lambda **_kwargs: [("gate only", ["true"])])
    monkeypatch.setattr(verify_module, "_run", gate_that_edits)
    monkeypatch.setattr(verify_module, "append_verify_history", lambda payload, **_kwargs: history.update(payload))

    assert verify_module._main(["--quick"]) == 1
    assert history["diagnosis"] == "checkout_moved_during_run"


def test_a_detached_head_at_the_default_tip_is_refused(tmp_path: Path) -> None:
    """Anti-vacuity: treat every detached HEAD as off the default branch and this is admitted."""
    root = _repository(tmp_path / "base", "master")
    subprocess.run(["git", "switch", "-q", "--detach", "master"], cwd=root, check=True)

    identity = checkout_identity(root)

    assert identity.branch is None
    assert default_branch_refusal(identity, command="devtools test", allowed=False) is not None


def test_slot_provenance_names_the_head_it_checked(tmp_path: Path) -> None:
    """The recorded HEAD, branch and content come from one stable capture."""
    from devtools import pytest_slot

    root = _repository(tmp_path / "feature", "claude/change")
    provenance = pytest_slot._focused_worktree_provenance(str(root), {"POLYLOGUE_FOCUSED_WORKTREE_PROVENANCE": "1"})

    assert provenance is not None
    assert provenance["git_head"] == checkout_identity(root).head
    assert provenance["git_branch"] == "claude/change"


def test_a_focused_run_whose_content_moves_during_pytest_is_void(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Anti-vacuity: publish the slot-start identity without an end-of-run
    comparison and this edit-during-pytest run reports PASSED."""
    from devtools import pytest_slot
    from devtools.pytest_slot import SlotOutcome

    root = _repository(tmp_path / "feature", "claude/change")
    (root / ".gitignore").write_text(".cache/\n", encoding="utf-8")

    def pytest_that_sees_an_edit(cmd: list[str], **kwargs: object) -> SlotOutcome:
        env = kwargs["env"]
        assert isinstance(env, dict)
        provenance = pytest_slot._focused_worktree_provenance(str(root), env)
        (root / "seed.txt").write_text("edited while pytest ran\n", encoding="utf-8")
        report = Path(next(arg for arg in cmd if arg.startswith("--polylogue-report-file=")).split("=", 1)[1])
        report.write_text(json.dumps({"tests": [{"nodeid": "seed_test.py", "outcome": "passed"}]}), encoding="utf-8")
        Path(env["POLYLOGUE_PYTEST_SELECTION_PATH"]).write_text(json.dumps({"selected_count": 1}), encoding="utf-8")
        Path(env["POLYLOGUE_PYTEST_SUMMARY_PATH"]).write_text(json.dumps({"exitstatus": 0}), encoding="utf-8")
        events = Path(env["POLYLOGUE_PYTEST_EVENTS_DIR"])
        events.mkdir()
        (events / "gw0.jsonl").write_text(
            json.dumps({"event": "collection_finished", "updated_at": "2026-01-01T00:00:00Z"}) + "\n", encoding="utf-8"
        )
        return SlotOutcome(returncode=0, slot="agentctl job 1", receipt={"worktree_provenance": provenance})

    # The selection names a module that exists: a missing one is refused
    # before the run is queued at all.
    (root / "seed_test.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(run_tests, "ROOT", root)
    monkeypatch.chdir(root)
    monkeypatch.setattr(run_tests, "assert_polylogue_matches_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr(run_tests, "_clear_pytest_report", lambda _path: None)
    monkeypatch.setattr(run_tests, "run_pytest", pytest_that_sees_an_edit)

    assert run_tests.main(["seed_test.py"]) == 1
    final = capsys.readouterr().err.strip().splitlines()[-1]
    assert "diagnosis=checkout_moved_during_run" in final


def test_an_inherited_default_branch_opt_in_does_not_reach_the_slot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Anti-vacuity: keep an ambient ``POLYLOGUE_ALLOW_DEFAULT_BRANCH=1`` and the
    slot would accept a run the invocation never authorized."""
    from devtools.checkout_identity import ALLOW_DEFAULT_BRANCH_ENV
    from devtools.pytest_slot import SlotOutcome

    root = _repository(tmp_path / "feature", "claude/change")
    seen: dict[str, str] = {}

    def capture(_cmd: list[str], **kwargs: object) -> SlotOutcome:
        env = kwargs["env"]
        assert isinstance(env, dict)
        seen.update(env)
        return SlotOutcome(returncode=2, slot="held")

    monkeypatch.setenv(ALLOW_DEFAULT_BRANCH_ENV, "1")
    monkeypatch.setattr(run_tests, "ROOT", root)
    monkeypatch.chdir(root)
    monkeypatch.setattr(run_tests, "assert_polylogue_matches_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr(run_tests, "_clear_pytest_report", lambda _path: None)
    monkeypatch.setattr(run_tests, "run_pytest", capture)

    run_tests.main(["seed_test.py"])

    assert ALLOW_DEFAULT_BRANCH_ENV not in seen


def test_pytest_on_another_head_with_identical_content_voids_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Anti-vacuity: compare only content digests and an empty commit's pytest
    provenance passes against the admitted HEAD."""
    from devtools import verify as verify_module
    from devtools.verify_runs import git_worktree_content_sha256

    root = _repository(tmp_path / "feature", "claude/change")
    history: dict[str, object] = {}
    digest = git_worktree_content_sha256(root)
    elsewhere = {"git_branch": "claude/change", "git_head": "b" * 40, "git_worktree_content_sha256": digest}

    def pytest_step(
        label: str, command: list[str], *, run: object, runner: str
    ) -> tuple[int, float, dict[str, object]]:
        del label, command, run, runner
        return 0, 0.0, {"diagnosis": "gate_passed", "pytest_slot_receipt": {"worktree_provenance": elsewhere}}

    monkeypatch.setattr(verify_module, "ROOT", root)
    monkeypatch.chdir(root)
    monkeypatch.setattr(verify_module, "assert_polylogue_matches_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr(verify_module, "build_verify_steps", lambda **_kwargs: [("gate only", ["true"])])
    monkeypatch.setattr(verify_module, "_run", pytest_step)
    monkeypatch.setattr(verify_module, "append_verify_history", lambda payload, **_kwargs: history.update(payload))

    assert verify_module._main(["--quick"]) == 1
    assert history["diagnosis"] == "checkout_moved_during_run"

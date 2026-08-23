"""Behavioral contracts for the pre-push verification selector."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from devtools import pre_push_gate
from devtools.verify_runs import worktree_fingerprint

ROOT = Path(__file__).resolve().parents[3]
HOOK_PATHS = (ROOT / ".githooks" / "pre-push",)
ZERO_SHA = "0" * 40


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path, path: str, content: str, message: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-b", "master")
    _git(tmp_path, "config", "user.email", "tests@example.invalid")
    _git(tmp_path, "config", "user.name", "Polylogue tests")
    # Mirror the real repo's .gitignore (/.cache/): a verify receipt written
    # under `.cache/verify/` must not itself make the fixture repo read as a
    # dirty worktree -- only a real production-relevant untracked/tracked
    # change should.
    _commit(tmp_path, ".gitignore", "/.cache/\n", "gitignore")
    _commit(tmp_path, "README.md", "base\n", "base")
    return tmp_path


def test_hook_files_exist_and_are_executable() -> None:
    for path in HOOK_PATHS:
        assert path.is_file()
        assert os.access(path, os.X_OK), f"{path} must be executable"


def test_hook_emits_devshell_hint_when_devtools_missing(tmp_path: Path) -> None:
    bash = shutil.which("bash") or "/bin/bash"
    coreutils_paths = [shutil.which(name) for name in ("cat", "echo", "mktemp", "rm")]
    coreutils_dirs = sorted({Path(path).parent for path in coreutils_paths if path is not None})
    env = {"PATH": ":".join(str(directory) for directory in coreutils_dirs), "HOME": str(tmp_path)}
    result = subprocess.run(
        [bash, str(HOOK_PATHS[0])],
        input="",
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
        check=False,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "devtools is not importable" in combined
    assert "nix develop" in combined
    assert "Traceback" not in combined


def test_parse_updates_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="expected four fields"):
        pre_push_gate.parse_updates("refs/heads/topic only-two-fields")


@pytest.mark.parametrize(
    ("local_sha", "remote_sha"),
    [("not-a-sha", ZERO_SHA), ("a" * 40, "not-a-sha")],
)
def test_parse_updates_rejects_malformed_local_or_remote_sha(local_sha: str, remote_sha: str) -> None:
    with pytest.raises(ValueError, match="40-character hexadecimal SHA"):
        pre_push_gate.parse_updates(f"refs/heads/topic {local_sha} refs/heads/topic {remote_sha}")


def test_parse_updates_accepts_full_sha_updates() -> None:
    update = pre_push_gate.parse_updates("refs/heads/topic " + "a" * 40 + " refs/heads/topic " + ZERO_SHA)
    assert update[0].local_sha == "a" * 40


def _code_update(repo: Path) -> pre_push_gate.PushUpdate:
    base = _git(repo, "rev-parse", "HEAD")
    tip = _commit(repo, "polylogue/example.py", "VALUE = 1\n", "code")
    return pre_push_gate.PushUpdate("refs/heads/topic", tip, "refs/heads/topic", base)


def _quick_receipt(
    repo: Path,
    *,
    head: str,
    environment: Mapping[str, object],
    worktree_fingerprint: str = "tree-fingerprint",
) -> None:
    path = repo / ".cache" / "verify" / "current-run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tier": "quick",
                "status": "success",
                "exit_code": 0,
                "git_head": head,
                "final_git_head": head,
                "checkout_root": str(repo.resolve()),
                "worktree_fingerprint": worktree_fingerprint,
                "final_worktree_fingerprint": worktree_fingerprint,
                "environment_fingerprint": environment,
            }
        ),
        encoding="utf-8",
    )


def test_foreign_pushed_update_refuses_receipt_reuse(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    update = _code_update(git_repo)
    head = _git(git_repo, "rev-parse", "HEAD")
    environment = {"python_executable": "/venv/bin/python", "artifacts": []}
    _quick_receipt(git_repo, head=head, environment=environment)
    foreign_update = pre_push_gate.PushUpdate(
        update.local_ref,
        _git(git_repo, "rev-parse", "HEAD^"),
        update.remote_ref,
        update.remote_sha,
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(pre_push_gate, "_current_provenance", lambda cwd: (environment, "tree-fingerprint"))
    monkeypatch.setattr(pre_push_gate, "_run", lambda command, cwd: commands.append(command))

    assert pre_push_gate.run_gate([foreign_update], cwd=git_repo) == "quick"
    assert commands == [[sys.executable, "-m", "devtools", "verify", "--quick"]]


def test_dirty_worktree_refuses_receipt_reuse(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    update = _code_update(git_repo)
    head = _git(git_repo, "rev-parse", "HEAD")
    environment = {"python_executable": "/venv/bin/python", "artifacts": []}
    _quick_receipt(git_repo, head=head, environment=environment)
    commands: list[list[str]] = []
    monkeypatch.setattr(pre_push_gate, "_current_provenance", lambda cwd: (environment, "tree-fingerprint"))
    monkeypatch.setattr(pre_push_gate, "_worktree_is_clean", lambda cwd: False)
    monkeypatch.setattr(pre_push_gate, "_run", lambda command, cwd: commands.append(command))

    assert pre_push_gate.run_gate([update], cwd=git_repo) == "quick"
    assert commands == [[sys.executable, "-m", "devtools", "verify", "--quick"]]


def test_worktree_is_clean_ignores_beads_only_uncommitted_change(git_repo: Path) -> None:
    """``.beads/`` is the shared receipt-cleanliness exclusion
    (`verify_runs.RECEIPT_EXCLUDED_PATHSPECS`, already applied by
    `worktree_fingerprint`): nothing at runtime reads it, so an uncommitted
    bead-bookkeeping edit must not read as a dirty worktree."""
    (git_repo / ".beads").mkdir(parents=True, exist_ok=True)
    (git_repo / ".beads" / "issues.jsonl").write_text('{"id": "uncommitted"}\n', encoding="utf-8")

    assert pre_push_gate._worktree_is_clean(git_repo) is True


def test_worktree_is_clean_refuses_on_tracked_code_change(git_repo: Path) -> None:
    """The `.beads` exclusion must not become a general dirty-tree bypass:
    an uncommitted change to a real tracked file still refuses."""
    _code_update(git_repo)
    (git_repo / "polylogue" / "example.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert pre_push_gate._worktree_is_clean(git_repo) is False


class _FakeCheckoutEnvironment:
    def __init__(self, payload: Mapping[str, object]) -> None:
        self._payload = payload

    def as_dict(self) -> Mapping[str, object]:
        return self._payload


def test_beads_only_uncommitted_change_permits_receipt_reuse(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: an uncommitted, *untracked* `.beads/` file alongside an
    otherwise matching exact-head receipt must still reuse, exercising the
    real (unmocked) `_worktree_is_clean` AND the real `worktree_fingerprint`
    through `_current_provenance` -- proving the .beads exclusion holds
    through the actual fingerprint value, not a fingerprint stubbed away to a
    constant."""
    update = _code_update(git_repo)
    head = _git(git_repo, "rev-parse", "HEAD")
    environment = {"python_executable": "/venv/bin/python", "artifacts": []}
    baseline_fingerprint = worktree_fingerprint(git_repo)
    _quick_receipt(git_repo, head=head, environment=environment, worktree_fingerprint=baseline_fingerprint)
    (git_repo / ".beads").mkdir(parents=True, exist_ok=True)
    (git_repo / ".beads" / "issues.jsonl").write_text('{"id": "uncommitted"}\n', encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(
        pre_push_gate, "assert_polylogue_matches_checkout", lambda cwd, context: _FakeCheckoutEnvironment(environment)
    )
    monkeypatch.setattr(pre_push_gate, "_run", lambda command, cwd: commands.append(command))

    assert pre_push_gate.run_gate([update], cwd=git_repo) == "reused"
    assert commands == []


def test_tracked_code_change_still_refuses_receipt_reuse(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end counterpart: a genuinely dirty tracked file (not
    `.beads/`) must still refuse reuse through the real, unmocked
    `_worktree_is_clean` and `worktree_fingerprint`."""
    update = _code_update(git_repo)
    head = _git(git_repo, "rev-parse", "HEAD")
    environment = {"python_executable": "/venv/bin/python", "artifacts": []}
    baseline_fingerprint = worktree_fingerprint(git_repo)
    _quick_receipt(git_repo, head=head, environment=environment, worktree_fingerprint=baseline_fingerprint)
    (git_repo / "polylogue" / "example.py").write_text("VALUE = 2\n", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(
        pre_push_gate, "assert_polylogue_matches_checkout", lambda cwd, context: _FakeCheckoutEnvironment(environment)
    )
    monkeypatch.setattr(pre_push_gate, "_run", lambda command, cwd: commands.append(command))

    assert pre_push_gate.run_gate([update], cwd=git_repo) == "quick"
    assert commands == [[sys.executable, "-m", "devtools", "verify", "--quick"]]


def test_untracked_nonbeads_file_still_refuses_receipt_reuse(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail-closed counterpart at the reuse-gate level: an untracked file
    outside `.beads/` must still refuse reuse through the real, unmocked
    `worktree_fingerprint` -- the .beads exclusion must not become a general
    untracked-content bypass."""
    update = _code_update(git_repo)
    head = _git(git_repo, "rev-parse", "HEAD")
    environment = {"python_executable": "/venv/bin/python", "artifacts": []}
    baseline_fingerprint = worktree_fingerprint(git_repo)
    _quick_receipt(git_repo, head=head, environment=environment, worktree_fingerprint=baseline_fingerprint)
    (git_repo / "untracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(
        pre_push_gate, "assert_polylogue_matches_checkout", lambda cwd, context: _FakeCheckoutEnvironment(environment)
    )
    monkeypatch.setattr(pre_push_gate, "_run", lambda command, cwd: commands.append(command))

    assert pre_push_gate.run_gate([update], cwd=git_repo) == "quick"
    assert commands == [[sys.executable, "-m", "devtools", "verify", "--quick"]]


def test_toolchain_drift_refuses_receipt_reuse(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    update = _code_update(git_repo)
    head = _git(git_repo, "rev-parse", "HEAD")
    recorded = {"python_executable": "/venv/bin/python", "quick_gate_toolchain": {"ruff": "old"}}
    current = {"python_executable": "/venv/bin/python", "quick_gate_toolchain": {"ruff": "new"}}
    _quick_receipt(git_repo, head=head, environment=recorded)
    commands: list[list[str]] = []
    monkeypatch.setattr(pre_push_gate, "_current_provenance", lambda cwd: (current, "tree-fingerprint"))
    monkeypatch.setattr(pre_push_gate, "_run", lambda command, cwd: commands.append(command))

    assert pre_push_gate.run_gate([update], cwd=git_repo) == "quick"
    assert commands == [[sys.executable, "-m", "devtools", "verify", "--quick"]]


def test_post_validation_provenance_drift_refuses_receipt_reuse(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    update = _code_update(git_repo)
    head = _git(git_repo, "rev-parse", "HEAD")
    environment = {"python_executable": "/venv/bin/python", "quick_gate_toolchain": {"ruff": "same"}}
    drifted = {"python_executable": "/venv/bin/python", "quick_gate_toolchain": {"ruff": "changed"}}
    _quick_receipt(git_repo, head=head, environment=environment)
    commands: list[list[str]] = []
    monkeypatch.setattr(pre_push_gate, "_current_provenance", lambda cwd: (environment, "tree-fingerprint"))
    monkeypatch.setattr(pre_push_gate, "_run", lambda command, cwd: commands.append(command))

    # The implementation must take a second live sample after receipt validation.
    samples = iter(((environment, "tree-fingerprint"), (drifted, "tree-fingerprint")))
    monkeypatch.setattr(pre_push_gate, "_current_provenance", lambda cwd: next(samples))

    assert pre_push_gate.run_gate([update], cwd=git_repo) == "quick"
    assert commands == [[sys.executable, "-m", "devtools", "verify", "--quick"]]


def test_head_transition_after_initial_sample_refuses_reuse(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A HEAD move between the initial sample and the reuse decision must refuse reuse.

    The worktree fingerprint alone cannot observe this: it diffs the tracked tree
    against HEAD, so a concurrent operation that moves HEAD to a different commit
    with an identical tree (amend, rebase --onto, another writer) leaves the
    fingerprint unchanged on both sides of the transition. Only re-sampling live
    HEAD right before committing to reuse, and requiring it still match the
    initially observed HEAD, can catch this.
    """
    update = _code_update(git_repo)
    head = _git(git_repo, "rev-parse", "HEAD")
    moved = "f" * 40
    environment = {"python_executable": "/venv/bin/python", "artifacts": []}
    _quick_receipt(git_repo, head=head, environment=environment)
    commands: list[list[str]] = []
    monkeypatch.setattr(pre_push_gate, "_worktree_is_clean", lambda cwd: True)
    monkeypatch.setattr(pre_push_gate, "_current_provenance", lambda cwd: (environment, "tree-fingerprint"))
    monkeypatch.setattr(pre_push_gate, "_run", lambda command, cwd: commands.append(command))

    real_git = pre_push_gate._git
    calls = {"rev_parse_head": 0}

    def fake_git(*args: str, cwd: Path) -> str:
        if args == ("rev-parse", "HEAD"):
            calls["rev_parse_head"] += 1
            # First sample matches the pushed SHA and the cached receipt; a
            # simulated concurrent operation then moves HEAD before the gate
            # commits to reuse.
            return head if calls["rev_parse_head"] == 1 else moved
        return real_git(*args, cwd=cwd)

    monkeypatch.setattr(pre_push_gate, "_git", fake_git)

    assert pre_push_gate.run_gate([update], cwd=git_repo) == "quick"
    assert commands == [[sys.executable, "-m", "devtools", "verify", "--quick"]]
    assert calls["rev_parse_head"] >= 2, "the gate must re-sample live HEAD before trusting reuse"


def test_matching_quick_receipt_is_reused(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    update = _code_update(git_repo)
    head = _git(git_repo, "rev-parse", "HEAD")
    environment = {"python_executable": "/venv/bin/python", "artifacts": []}
    _quick_receipt(git_repo, head=head, environment=environment)
    commands: list[list[str]] = []
    monkeypatch.setattr(pre_push_gate, "_worktree_is_clean", lambda cwd: True)
    monkeypatch.setattr(pre_push_gate, "_current_provenance", lambda cwd: (environment, "tree-fingerprint"))
    monkeypatch.setattr(pre_push_gate, "_run", lambda command, cwd: commands.append(command))

    assert pre_push_gate.run_gate([update], cwd=git_repo) == "reused"
    assert commands == []


@pytest.mark.parametrize(
    ("receipt_kwargs", "write_receipt"),
    [
        ({"head": "0" * 40}, True),
        ({"head": ""}, True),
        ({"head": ""}, False),
    ],
    ids=("stale-head", "foreign-environment", "missing-receipt"),
)
def test_incompatible_or_missing_quick_receipt_reruns_gate(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_kwargs: dict[str, str],
    write_receipt: bool,
) -> None:
    update = _code_update(git_repo)
    head = _git(git_repo, "rev-parse", "HEAD")
    environment = {"python_executable": "/venv/bin/python", "artifacts": []}
    if write_receipt:
        _quick_receipt(
            git_repo,
            head=receipt_kwargs.get("head") or head,
            environment=(
                {"python_executable": "/foreign/bin/python", "artifacts": []}
                if receipt_kwargs.get("head") == ""
                else environment
            ),
        )
    commands: list[list[str]] = []
    monkeypatch.setattr(pre_push_gate, "_current_provenance", lambda cwd: (environment, "tree-fingerprint"))
    monkeypatch.setattr(pre_push_gate, "_run", lambda command, cwd: commands.append(command))

    assert pre_push_gate.run_gate([update], cwd=git_repo) == "quick"
    assert commands == [[sys.executable, "-m", "devtools", "verify", "--quick"]]


def test_main_parses_updates_and_runs_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[list[pre_push_gate.PushUpdate]] = []

    def fake_run_gate(updates: list[pre_push_gate.PushUpdate], cwd: Path) -> str:
        observed.append(updates)
        return "reused"

    monkeypatch.setattr(pre_push_gate, "run_gate", fake_run_gate)

    updates_file = tmp_path / "updates"
    updates_file.write_text("refs/heads/topic " + "a" * 40 + " refs/heads/topic " + "b" * 40 + "\n", encoding="utf-8")

    assert pre_push_gate.main([str(updates_file)]) == 0
    assert len(observed) == 1
    assert observed[0][0].local_sha == "a" * 40

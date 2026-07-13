"""Behavioral contracts for the pre-push verification selector."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from devtools import pre_push_gate

ROOT = Path(__file__).resolve().parents[3]
HOOK_PATHS = (ROOT / ".githooks" / "pre-push", ROOT / ".beads-hooks" / "pre-push")
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


def test_beads_only_update_is_classified_without_code_gate(git_repo: Path) -> None:
    base = _git(git_repo, "rev-parse", "HEAD")
    tip = _commit(git_repo, ".beads/issues.jsonl", "{}\n", "beads")
    update = pre_push_gate.PushUpdate("refs/heads/topic", tip, "refs/heads/topic", base)

    paths = pre_push_gate.changed_paths([update], cwd=git_repo)

    assert paths == {".beads/issues.jsonl"}
    assert pre_push_gate.is_beads_only(paths)


def test_mixed_update_retains_quick_gate_classification(git_repo: Path) -> None:
    base = _git(git_repo, "rev-parse", "HEAD")
    _commit(git_repo, ".beads/issues.jsonl", "{}\n", "beads")
    tip = _commit(git_repo, "polylogue/example.py", "VALUE = 1\n", "code")
    update = pre_push_gate.PushUpdate("refs/heads/topic", tip, "refs/heads/topic", base)

    paths = pre_push_gate.changed_paths([update], cwd=git_repo)

    assert paths == {".beads/issues.jsonl", "polylogue/example.py"}
    assert not pre_push_gate.is_beads_only(paths)


def test_new_branch_uses_default_branch_merge_base(git_repo: Path) -> None:
    _git(git_repo, "switch", "-c", "topic")
    tip = _commit(git_repo, ".beads/issues.jsonl", "{}\n", "beads")
    update = pre_push_gate.PushUpdate("refs/heads/topic", tip, "refs/heads/topic", ZERO_SHA)

    assert pre_push_gate.changed_paths([update], cwd=git_repo) == {".beads/issues.jsonl"}


def test_parse_updates_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="expected four fields"):
        pre_push_gate.parse_updates("refs/heads/topic only-two-fields")

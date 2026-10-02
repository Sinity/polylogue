"""Entered-checkout hook configuration through the literal devshell route."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.uses_real_clock("Git config mtimes come from the filesystem")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _hook_entry() -> str:
    flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
    start = flake.index("          # Configure only the entered checkout.")
    end = flake.index("          # Clean stale __pycache__", start)
    return textwrap.dedent(flake[start:end])


def test_hook_entry_configures_only_current_worktree_and_warm_entry_does_not_write(tmp_path: Path) -> None:
    """A worktree loop changes the sibling; unconditional writes fail under locks."""
    common = tmp_path / "common"
    common.mkdir()
    _git(common, "init", "--initial-branch=main")
    _git(
        common,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "fixture",
    )
    entered = tmp_path / "entered"
    sibling = tmp_path / "sibling"
    _git(common, "worktree", "add", "-b", "entered", str(entered))
    _git(common, "worktree", "add", "-b", "sibling", str(sibling))
    _git(common, "config", "--local", "extensions.worktreeConfig", "true")
    _git(sibling, "config", "--worktree", "core.hooksPath", "/synthetic/sibling-hooks")
    sibling_config = Path(_git(sibling, "rev-parse", "--git-path", "config.worktree"))
    sibling_before = (sibling_config.read_bytes(), sibling_config.stat().st_mtime_ns)
    # The common checkout's helper must not decide the entered branch's behavior.
    common_script = common / "scripts/configure-git-hooks"
    common_script.parent.mkdir()
    common_script.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
    common_script.chmod(0o755)
    script = entered / "scripts/configure-git-hooks"
    script.parent.mkdir()
    script.write_bytes((REPO_ROOT / "scripts/configure-git-hooks").read_bytes())
    script.chmod(0o755)

    subprocess.run(["bash", "-e", "-c", _hook_entry()], cwd=entered, check=True, capture_output=True)
    expected = str(common / ".githooks")
    assert _git(entered, "config", "--worktree", "--get", "core.hooksPath") == expected
    assert _git(entered, "config", "--get", "core.hooksPath") == expected
    assert _git(common, "config", "--local", "--get", "core.hooksPath") == expected
    assert (sibling_config.read_bytes(), sibling_config.stat().st_mtime_ns) == sibling_before
    configs = [common / ".git/config", Path(_git(entered, "rev-parse", "--git-path", "config.worktree"))]
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in configs]
    for path in configs:
        path.with_suffix(path.suffix + ".lock").touch()
    subprocess.run(["bash", "-e", "-c", _hook_entry()], cwd=entered, check=True, capture_output=True)
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in configs] == before
    assert (sibling_config.read_bytes(), sibling_config.stat().st_mtime_ns) == sibling_before


def test_helper_repairs_current_override_and_handles_main_checkout(tmp_path: Path) -> None:
    """Effective config reads would hide the missing worktree anchor."""
    _git(tmp_path, "init", "--initial-branch=main")
    helper = str(REPO_ROOT / "scripts/configure-git-hooks")
    subprocess.run([helper], cwd=tmp_path, check=True, capture_output=True)
    expected = str(tmp_path / ".githooks")
    assert _git(tmp_path, "config", "--worktree", "--get", "core.hooksPath") == expected
    _git(tmp_path, "config", "--worktree", "core.hooksPath", "/synthetic/wrong-hooks")
    subprocess.run([helper], cwd=tmp_path, check=True, capture_output=True)
    assert _git(tmp_path, "config", "--get", "core.hooksPath") == expected


def test_helper_outside_git_has_no_side_effects(tmp_path: Path) -> None:
    subprocess.run([str(REPO_ROOT / "scripts/configure-git-hooks")], cwd=tmp_path, check=True, capture_output=True)
    assert not list(tmp_path.iterdir())

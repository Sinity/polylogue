"""Regression coverage for the linked-worktree Beads import guard.

This exercises the production post-checkout hook and the installed ``bd``
binary against a temporary git repository and temporary embedded Dolt store.
It never opens the repository's shared Beads database.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {command!r}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _bd_environment(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    cache = tmp_path / "cache"
    config = tmp_path / "config"
    data = tmp_path / "data"
    for path in (home, cache, config, data):
        path.mkdir()

    env = os.environ.copy()
    for key in ("BD_DB", "BEADS_DB", "BEADS_DIR", "BEADS_DOLT_SERVER_DATABASE", "BEADS_DOLT_SERVER_PORT"):
        env.pop(key, None)
    env.update(
        {
            "HOME": str(home),
            "XDG_CACHE_HOME": str(cache),
            "XDG_CONFIG_HOME": str(config),
            "XDG_DATA_HOME": str(data),
            "BEADS_DOLT_AUTO_START": "0",
            "BEADS_NO_DAEMON": "1",
            "BD_DISABLE_EVENT_FLUSH": "1",
            "BD_DISABLE_METRICS": "1",
            "NO_COLOR": "1",
        }
    )
    return env


def _json_from_output(output: str) -> object:
    decoder = json.JSONDecoder()
    for index, char in enumerate(output):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        return value
    raise AssertionError(f"bd did not emit JSON:\n{output}")


def _write_import_policy(repo: Path, import_auto: bool) -> None:
    config_path = repo / ".beads" / "config.yaml"
    production_config = (ROOT / ".beads" / "config.yaml").read_text()
    updated, replacements = re.subn(
        r"(?m)^import\.auto: false$",
        f"import.auto: {str(import_auto).lower()}",
        production_config,
    )
    assert replacements == 1, "the production config must declare exactly one import.auto policy"
    config_path.write_text(updated)


def test_stale_worktree_hook_then_read_preserves_coordinator_write(tmp_path: Path) -> None:
    """A stale branch cannot overwrite a coordinator close before its read.

    Production dependency exercised: ``.beads-hooks/post-checkout`` delegates
    to Beads' real ``bd hooks run post-checkout`` import gate. The wrapper only
    records the process boundary, then delegates unchanged to
    the installed binary. Changing the committed ``import.auto`` policy back to
    ``true`` is the pre-fix mutation: older Beads releases delegate to a blind
    ``bd import`` here, so the test's no-import assertion catches the unsafe
    path before the stale-worktree read.
    """
    bd = shutil.which("bd")
    if bd is None:
        pytest.skip("bd is not installed")

    env = _bd_environment(tmp_path)
    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    invocation_log = tmp_path / "bd-invocations.log"
    wrapper = wrapper_dir / "bd"
    wrapper.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$BD_GUARD_COMMAND_LOG"\nexec "$BD_GUARD_REAL_BD" "$@"\n')
    wrapper.chmod(0o755)
    env["BD_GUARD_COMMAND_LOG"] = str(invocation_log)
    env["BD_GUARD_REAL_BD"] = bd
    env["PATH"] = str(wrapper_dir) + os.pathsep + env["PATH"]
    repo = tmp_path / "coordinator"
    lane = tmp_path / "stale-lane"
    repo.mkdir()

    _run(["git", "init", "--initial-branch=main"], repo, env)
    _run(["git", "config", "user.email", "test@example.invalid"], repo, env)
    _run(["git", "config", "user.name", "Beads guard test"], repo, env)
    _run([bd, "init", "--prefix", "guard", "--quiet", "--non-interactive", "--skip-hooks", "--skip-agents"], repo, env)
    _write_import_policy(repo, False)
    configured_policy = _run([bd, "config", "get", "import.auto"], repo, env).stdout
    assert "false" in configured_policy.lower()

    hooks_dir = repo / ".githooks"
    hooks_dir.mkdir()
    hook_path = hooks_dir / "post-checkout"
    hook_path.write_text((ROOT / ".beads-hooks" / "post-checkout").read_text())
    hook_path.chmod(0o755)
    _run(["git", "config", "core.hooksPath", str(hooks_dir)], repo, env)

    _run([bd, "create", "coordinator-owned issue", "--type", "bug"], repo, env)
    _run([bd, "export"], repo, env)
    issue_rows = _json_from_output(_run([bd, "list", "--json"], repo, env).stdout)
    assert isinstance(issue_rows, list) and len(issue_rows) == 1
    issue_id = issue_rows[0]["id"]

    _run(["git", "add", ".beads", ".githooks"], repo, env)
    _run(["git", "commit", "-m", "baseline Beads snapshot"], repo, env)
    _run(["git", "branch", "stale-lane"], repo, env)
    _run(["git", "worktree", "add", str(lane), "stale-lane"], repo, env)

    _run([bd, "close", issue_id, "--reason", "coordinator close"], repo, env)
    invocation_log.write_text("")
    _run([str(hook_path), "old-head", "new-head", "1"], lane, env)

    shown = _json_from_output(_run([bd, "show", issue_id, "--json"], lane, env).stdout)
    assert isinstance(shown, list) and len(shown) == 1
    assert shown[0]["status"] == "closed"

    delegated_commands = invocation_log.read_text().splitlines()
    imported = any(command.startswith("import --quiet ") for command in delegated_commands)
    assert not imported

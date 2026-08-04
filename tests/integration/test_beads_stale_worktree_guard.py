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
    for key in (
        "BD_DB",
        "BD_IMPORT_AUTO",
        "BEADS_DB",
        "BEADS_DIR",
        "BEADS_DOLT_SERVER_DATABASE",
        "BEADS_DOLT_SERVER_PORT",
    ):
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
    updated, removed = re.subn(r'(?m)^sync\.remote: ".*"\n?', "", updated)
    assert removed == 1, "the fixture must not have a Dolt sync remote"
    config_path.write_text(updated)


def _set_export_timestamp(repo: Path, issue_id: str, updated_at: str) -> None:
    export_path = repo / ".beads" / "issues.jsonl"
    rows = [json.loads(line) for line in export_path.read_text().splitlines()]
    matching_rows = [row for row in rows if row.get("id") == issue_id]
    assert len(matching_rows) == 1
    matching_rows[0]["updated_at"] = updated_at
    export_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _setup_stale_worktree(
    tmp_path: Path,
    import_auto: bool,
    stale_snapshot_updated_at: str | None = None,
) -> tuple[str, dict[str, str], Path, Path, Path]:
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
    _write_import_policy(repo, import_auto)

    hooks_dir = repo / ".githooks"
    hooks_dir.mkdir()
    hook_path = hooks_dir / "post-checkout"
    hook_path.write_text((ROOT / ".beads-hooks" / "post-checkout").read_text())
    hook_path.chmod(0o755)
    _run(["git", "config", "core.hooksPath", str(hooks_dir)], repo, env)

    _run([bd, "create", "coordinator-owned issue", "--type", "bug"], repo, env)
    _run([bd, "export", "-o", ".beads/issues.jsonl"], repo, env)
    issue_rows = _json_from_output(_run([bd, "list", "--json"], repo, env).stdout)
    assert isinstance(issue_rows, list) and len(issue_rows) == 1
    issue_id = issue_rows[0]["id"]
    if stale_snapshot_updated_at is not None:
        _set_export_timestamp(repo, issue_id, stale_snapshot_updated_at)

    _run(["git", "add", ".beads", ".githooks"], repo, env)
    _run(["git", "commit", "-m", "baseline Beads snapshot"], repo, env)
    _run(["git", "branch", "stale-lane"], repo, env)
    _run(["git", "worktree", "add", str(lane), "stale-lane"], repo, env)
    _run([bd, "close", issue_id, "--reason", "coordinator close"], repo, env)
    invocation_log.write_text("")
    return issue_id, env, repo, lane, invocation_log


def test_stale_worktree_plain_read_preserves_coordinator_write_when_env_reenables_import(tmp_path: Path) -> None:
    """A stale branch cannot overwrite a coordinator close before its read.

    Production dependency exercised: Git invokes the committed
    ``.beads-hooks/post-checkout`` shim, which invokes Beads' real hook import
    gate. The fixture clears ambient ``BD_IMPORT_AUTO`` before setup, then sets
    it to ``true`` only for the stale lane. The production shim must clear that
    unsafe Viper override before it can re-enable imports.
    """
    issue_id, env, _repo, lane, invocation_log = _setup_stale_worktree(tmp_path, import_auto=False)
    env["BD_IMPORT_AUTO"] = "true"
    _run(["git", "switch", "--detach", "HEAD"], lane, env)

    shown = _json_from_output(_run([env["BD_GUARD_REAL_BD"], "show", issue_id, "--json"], lane, env).stdout)
    assert isinstance(shown, list) and len(shown) == 1
    assert shown[0]["status"] == "closed"

    delegated_commands = invocation_log.read_text().splitlines()
    imported = any(command.startswith("import --quiet ") for command in delegated_commands)
    assert not imported


def test_unsafe_import_auto_control_reverts_clock_skewed_stale_snapshot(tmp_path: Path) -> None:
    """The config mutation reaches the importer and reopens the stale issue.

    This is the causal negative control for the protected test. It removes the
    production ``import.auto: false`` mutation while retaining the same actual
    git checkout and installed Beads engine. A clock-skewed branch snapshot
    has a later ``updated_at`` despite containing the old open status, so the
    importer accepts it over the coordinator's close.
    """
    issue_id, env, _repo, lane, invocation_log = _setup_stale_worktree(
        tmp_path,
        import_auto=True,
        stale_snapshot_updated_at="2099-01-01T00:00:00Z",
    )
    _run(["git", "switch", "--detach", "HEAD"], lane, env)

    shown = _json_from_output(_run([env["BD_GUARD_REAL_BD"], "show", issue_id, "--json"], lane, env).stdout)
    assert isinstance(shown, list) and len(shown) == 1
    assert shown[0]["status"] == "open"

    delegated_commands = invocation_log.read_text().splitlines()
    assert any(command.startswith("import --quiet ") for command in delegated_commands)

"""Regression coverage for the linked-worktree Beads import guard.

The fixture has a deployed coordinator checkout and a linked lane created
before deployment. The lane therefore retains the historical relative
``.beads-hooks`` and ``.envrc`` paths while Git routes hooks through the
coordinator's shared common-directory installation. It never opens this
checkout's Beads database.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "a516b3caa791ef84a1a4133d217e39c20123651a"


def _installed_bd() -> str | None:
    """Resolve the installed binary without using this checkout's wrapper."""
    scripts_dir = (ROOT / "scripts").resolve()
    search_path = os.pathsep.join(
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if (Path(entry or ".").resolve() / "bd").resolve().parent.name != scripts_dir.name
    )
    return shutil.which("bd", path=search_path)


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
        "POLYLOGUE_BD_REAL",
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


def _set_export_row(repo: Path, issue_id: str, **updates: object) -> None:
    export_path = repo / ".beads" / "issues.jsonl"
    rows = [json.loads(line) for line in export_path.read_text().splitlines()]
    matching_rows = [row for row in rows if row.get("id") == issue_id]
    assert len(matching_rows) == 1
    matching_rows[0].update(updates)
    export_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _historical_file(relative_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{BASE_COMMIT}^:{relative_path}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _deploy_current_common_checkout(repo: Path) -> None:
    hooks = repo / ".beads-hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    for hook in ("post-checkout", "post-merge", "pre-commit", "pre-push", "prepare-commit-msg"):
        target = hooks / hook
        shutil.copy2(ROOT / ".beads-hooks" / hook, target)
        target.chmod(0o755)

    envrc = repo / ".envrc"
    shutil.copy2(ROOT / ".envrc", envrc)
    wrapper = repo / "scripts" / "bd"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts" / "bd", wrapper)
    wrapper.chmod(0o755)
    hook_configurator = repo / "scripts" / "configure-git-hooks"
    shutil.copy2(ROOT / "scripts" / "configure-git-hooks", hook_configurator)
    hook_configurator.chmod(0o755)
    guard = repo / "devtools" / "bd_reimport_guard.py"
    guard.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "devtools" / "bd_reimport_guard.py", guard)


def _execute_historical_envrc(lane: Path, env: dict[str, str], tmp_path: Path, installed_bd: str) -> None:
    """Source the retained envrc and execute its branch-point use-flake hook."""
    historical_use_flake = tmp_path / "historical-use-flake"
    historical_flake = _historical_file("flake.nix")
    start_marker = "          # Install repo git hooks"
    end_marker = "          # Clean stale __pycache__"
    start = historical_flake.index(start_marker)
    end = historical_flake.index(end_marker, start)
    historical_shell_hook = textwrap.dedent(historical_flake[start:end])
    historical_use_flake.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{historical_shell_hook}")
    historical_use_flake.chmod(0o755)

    marker = tmp_path / "historical-direct-bd-ran"
    probe_bin = tmp_path / "historical-bin"
    probe_bin.mkdir()
    direct_bd = probe_bin / "bd"
    direct_bd.write_text(f"#!/bin/sh\nprintf '%s\\n' ran > {marker!s}\n")
    direct_bd.chmod(0o755)

    historical_env = env.copy()
    historical_env["HISTORICAL_USE_FLAKE"] = str(historical_use_flake)
    # Keep the historical direct command-v bd route observable through the
    # probe while the deployed wrapper follows the valid installed binary.
    historical_env["POLYLOGUE_BD_REAL"] = installed_bd
    historical_env["PATH"] = os.pathsep.join((str(probe_bin), historical_env["PATH"]))
    _run(
        [
            "bash",
            "-c",
            'use() { "$HISTORICAL_USE_FLAKE" "$@"; }; source .envrc',
        ],
        lane,
        historical_env,
    )

    # The old shell hook wrote a relative value through git config --local.
    # The worktree-level common-dir pin must still select the deployed hook.
    effective_hooks_path = _run(["git", "config", "--get", "core.hooksPath"], lane, historical_env).stdout.strip()
    assert effective_hooks_path == str((lane.parent / "coordinator" / ".beads-hooks").resolve())
    effective_hooks_origin = _run(
        ["git", "config", "--show-origin", "--get", "core.hooksPath"], lane, historical_env
    ).stdout.strip()
    assert effective_hooks_origin.endswith("/config.worktree\t" + effective_hooks_path)

    _run(["git", "switch", "--detach", "HEAD"], lane, historical_env)
    assert not marker.exists(), "historical direct bd hook must not execute"


def _setup_stale_worktree(
    tmp_path: Path,
    import_auto: bool,
) -> tuple[str, dict[str, str], Path, Path]:
    bd = _installed_bd()
    if bd is None:
        pytest.skip("bd is not installed")

    env = _bd_environment(tmp_path)
    repo = tmp_path / "coordinator"
    lane = tmp_path / "stale-lane"
    repo.mkdir()

    _run(["git", "init", "--initial-branch=main"], repo, env)
    _run(["git", "config", "user.email", "test@example.invalid"], repo, env)
    _run(["git", "config", "user.name", "Beads guard test"], repo, env)
    _run([bd, "init", "--prefix", "guard", "--quiet", "--non-interactive", "--skip-hooks", "--skip-agents"], repo, env)
    _write_import_policy(repo, import_auto)

    # This is the branch-point checkout: it has only the historical relative
    # hook and envrc paths, with no current wrapper or guard files.
    historical_hooks = repo / ".beads-hooks"
    historical_hooks.mkdir()
    historical_hook = historical_hooks / "post-checkout"
    historical_hook.write_text(_historical_file(".beads-hooks/post-checkout"))
    historical_hook.chmod(0o755)
    (repo / ".envrc").write_text(_historical_file(".envrc"))

    setup_hooks = repo / ".setup-hooks"
    setup_hooks.mkdir()
    _run(["git", "config", "core.hooksPath", str(setup_hooks)], repo, env)
    _run([bd, "create", "coordinator-owned issue", "--type", "bug"], repo, env)
    _run([bd, "export", "-o", ".beads/issues.jsonl"], repo, env)
    issue_rows = _json_from_output(_run([bd, "list", "--json"], repo, env).stdout)
    assert isinstance(issue_rows, list) and len(issue_rows) == 1
    issue_id = issue_rows[0]["id"]

    _run(["git", "add", ".beads", ".beads-hooks", ".envrc"], repo, env)
    _run(["git", "commit", "-m", "historical Beads snapshot"], repo, env)
    _run(["git", "branch", "stale-lane"], repo, env)
    _run(["git", "worktree", "add", str(lane), "stale-lane"], repo, env)

    # Deployment happens only in the common coordinator checkout. The shared
    # absolute hook path is what protects the stale lane's historical files.
    _deploy_current_common_checkout(repo)
    _run(
        [
            "git",
            "add",
            ".beads-hooks",
            ".envrc",
            "scripts/bd",
            "scripts/configure-git-hooks",
            "devtools/bd_reimport_guard.py",
        ],
        repo,
        env,
    )
    _run(["git", "commit", "-m", "deploy Beads guard"], repo, env)

    _run([str(repo / "scripts" / "configure-git-hooks")], repo, env)
    _execute_historical_envrc(lane, env, tmp_path, bd)

    env["PATH"] = os.pathsep.join((str(repo / "scripts"), env["PATH"]))
    _run([bd, "close", issue_id, "--reason", "coordinator close"], repo, env)
    assert not (lane / "scripts" / "bd").exists()
    assert (lane / ".envrc").read_text() == _historical_file(".envrc")
    assert (lane / ".beads-hooks" / "post-checkout").read_text() == _historical_file(".beads-hooks/post-checkout")
    return issue_id, env, repo, lane


def test_stale_worktree_plain_read_preserves_coordinator_write(tmp_path: Path) -> None:
    """A stale branch's historical hook path cannot overwrite a close."""
    issue_id, env, _repo, lane = _setup_stale_worktree(tmp_path, import_auto=True)
    env["BD_IMPORT_AUTO"] = "true"

    _run(["git", "switch", "--detach", "HEAD"], lane, env)
    shown = _json_from_output(_run(["bd", "show", issue_id, "--json"], lane, env).stdout)
    assert isinstance(shown, list) and len(shown) == 1
    assert shown[0]["status"] == "closed"


def test_bd_wrapper_preserves_legitimate_newer_snapshot_import(tmp_path: Path) -> None:
    """The guard rejects stale rows without disabling newer rows."""
    issue_id, env, _repo, lane = _setup_stale_worktree(tmp_path, import_auto=True)
    _set_export_row(
        lane,
        issue_id,
        status="open",
        updated_at="2099-01-01T00:00:00Z",
        title="legitimate newer snapshot",
    )

    shown = _json_from_output(_run(["bd", "show", issue_id, "--json"], lane, env).stdout)
    assert isinstance(shown, list) and len(shown) == 1
    assert shown[0]["status"] == "open"
    assert shown[0]["title"] == "legitimate newer snapshot"


def test_unsafe_import_auto_control_reverts_clock_skewed_stale_snapshot(tmp_path: Path) -> None:
    """The installed Beads importer remains a causal unsafe control."""
    issue_id, env, _repo, lane = _setup_stale_worktree(tmp_path, import_auto=True)
    _set_export_row(lane, issue_id, status="open", updated_at="2099-01-01T00:00:00Z")
    env["BD_IMPORT_AUTO"] = "true"

    bd = _installed_bd()
    assert bd is not None
    unsafe_hooks = tmp_path / "unsafe-hooks"
    unsafe_hooks.mkdir()
    unsafe_hook = unsafe_hooks / "post-checkout"
    unsafe_hook.write_text(f"#!/bin/sh\nexec {bd} import .beads/issues.jsonl\n")
    unsafe_hook.chmod(0o755)
    _run(["git", "config", "core.hooksPath", str(unsafe_hooks)], lane, env)
    _run(["git", "switch", "--detach", "HEAD"], lane, env)
    shown = _json_from_output(_run([bd, "show", issue_id, "--json"], lane, env).stdout)
    assert isinstance(shown, list) and len(shown) == 1
    assert shown[0]["status"] == "open"


@pytest.mark.parametrize(
    "route",
    ["bare-wrapper", "symlink-wrapper", "script-recursion", "script-symlink-recursion", "interpreter-recursion"],
)
def test_bd_wrapper_rejects_recursive_real_binary_routes(tmp_path: Path, route: str) -> None:
    """Wrapper aliases and script recursion fail before the guard lock."""
    deployed = tmp_path / "deployed"
    scripts = deployed / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "bd", scripts / "bd")
    (scripts / "bd").chmod(0o755)
    guard = deployed / "devtools" / "bd_reimport_guard.py"
    guard.parent.mkdir()
    shutil.copy2(ROOT / "devtools" / "bd_reimport_guard.py", guard)
    wrapper = scripts / "bd"
    probe = tmp_path / "probe"
    probe.mkdir()

    env = os.environ.copy()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    env["XDG_RUNTIME_DIR"] = str(runtime)
    tool_dirs: list[str] = []
    for tool in ("sh", "dirname", "readlink", "python3"):
        executable = shutil.which(tool)
        assert executable is not None
        directory = str(Path(executable).parent)
        if directory not in tool_dirs:
            tool_dirs.append(directory)
    env["PATH"] = os.pathsep.join((str(scripts), *tool_dirs))
    if route == "bare-wrapper":
        env["POLYLOGUE_BD_REAL"] = "bd"
    elif route == "symlink-wrapper":
        alias = tmp_path / "bd-alias"
        alias.symlink_to(wrapper)
        env["POLYLOGUE_BD_REAL"] = str(alias)
    elif route == "script-recursion":
        recursive = tmp_path / "recursive-bd"
        recursive.write_text(f'#!/bin/sh\nexec {wrapper!s} "$@"\n')
        recursive.chmod(0o755)
        env["POLYLOGUE_BD_REAL"] = str(recursive)
    elif route == "script-symlink-recursion":
        alias = tmp_path / "bd-alias"
        alias.symlink_to(wrapper)
        recursive = tmp_path / "recursive-bd"
        recursive.write_text(f'#!/bin/sh\nexec {alias!s} "$@"\n')
        recursive.chmod(0o755)
        env["POLYLOGUE_BD_REAL"] = str(recursive)
    else:
        recursive = tmp_path / "interpreter-recursive-bd"
        recursive.write_text(f'#!/bin/sh\nexec /bin/sh -c \'exec {wrapper!s} "$@"\' sh "$@"\n')
        recursive.chmod(0o755)
        env["POLYLOGUE_BD_REAL"] = str(recursive)

    result = subprocess.run([str(wrapper), "--help"], cwd=probe, env=env, capture_output=True, text=True, timeout=10)
    expected_returncode = 126 if route != "bare-wrapper" else 127
    assert result.returncode == expected_returncode
    assert not list(runtime.iterdir())

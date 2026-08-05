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
import shlex
import shutil
import subprocess
import sys
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


def _historical_file(relative_path: str, *, commit: str = BASE_COMMIT) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
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


def _execute_historical_envrc(lane: Path, env: dict[str, str], tmp_path: Path, installed_bd: str) -> dict[str, str]:
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

    historical_env = env.copy()
    environment_dump = tmp_path / "historical-environment"
    historical_env["HISTORICAL_USE_FLAKE"] = str(historical_use_flake)
    historical_env["PATH"] = os.pathsep.join((str(Path(installed_bd).parent), historical_env["PATH"]))
    _run(
        [
            "bash",
            "-c",
            'PATH_add() { PATH="$1:$PATH"; export PATH; }; '
            'use() { "$HISTORICAL_USE_FLAKE" "$@"; }; '
            'source .envrc; env -0 > "$HISTORICAL_ENVIRONMENT_DUMP"',
        ],
        lane,
        historical_env | {"HISTORICAL_ENVIRONMENT_DUMP": str(environment_dump)},
    )
    for entry in environment_dump.read_bytes().split(b"\0"):
        if entry:
            key, _, value = entry.partition(b"=")
            historical_env[key.decode()] = value.decode()
    assert historical_env.get("POLYLOGUE_BD_REAL") == installed_bd

    # The old shell hook wrote a relative value through git config --local.
    # The worktree-level common-dir pin must still select the deployed hook.
    effective_hooks_path = _run(["git", "config", "--get", "core.hooksPath"], lane, historical_env).stdout.strip()
    assert effective_hooks_path == str((lane.parent / "coordinator" / ".beads-hooks").resolve())
    effective_hooks_origin = _run(
        ["git", "config", "--show-origin", "--get", "core.hooksPath"], lane, historical_env
    ).stdout.strip()
    assert effective_hooks_origin.endswith("/config.worktree\t" + effective_hooks_path)

    _run(["git", "switch", "--detach", "HEAD"], lane, historical_env)
    return historical_env


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

    # This is the branch-point checkout: it has the historical relative hook,
    # envrc, wrapper, and a deliberately stale guard implementation.
    historical_hooks = repo / ".beads-hooks"
    historical_hooks.mkdir()
    historical_hook = historical_hooks / "post-checkout"
    historical_hook.write_text(_historical_file(".beads-hooks/post-checkout", commit=BASE_COMMIT))
    historical_hook.chmod(0o755)
    (repo / ".envrc").write_text(_historical_file(".envrc", commit=BASE_COMMIT))
    historical_scripts = repo / "scripts"
    historical_scripts.mkdir()
    historical_wrapper = historical_scripts / "bd"
    historical_wrapper.write_text(_historical_file("scripts/bd", commit=BASE_COMMIT))
    historical_wrapper.chmod(0o755)
    stale_guard = repo / "devtools" / "bd_reimport_guard.py"
    stale_guard.parent.mkdir()
    stale_guard.write_text("raise SystemExit('stale guard loaded')\n")

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
    env = _execute_historical_envrc(lane, env, tmp_path, bd)
    lane_wrapper = lane / "scripts" / "bd"
    assert lane_wrapper.is_symlink()
    assert lane_wrapper.resolve() == (repo / "scripts" / "bd").resolve()

    _run([bd, "close", issue_id, "--reason", "coordinator close"], repo, env)
    assert (lane / "scripts" / "bd").is_symlink()
    assert (lane / ".envrc").read_text() == _historical_file(".envrc", commit=BASE_COMMIT)
    assert (lane / ".beads-hooks" / "post-checkout").read_text() == _historical_file(
        ".beads-hooks/post-checkout", commit=BASE_COMMIT
    )
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


def _setup_guard_linked_worktree(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    """Create a real Git common directory and linked worktree for route checks."""
    env = _bd_environment(tmp_path)
    repo = tmp_path / "coordinator"
    lane = tmp_path / "recursion-lane"
    repo.mkdir()
    _run(["git", "init", "--initial-branch=main"], repo, env)
    _run(["git", "config", "user.email", "test@example.invalid"], repo, env)
    _run(["git", "config", "user.name", "Beads guard test"], repo, env)
    (repo / "anchor").write_text("anchor\n")
    _run(["git", "add", "anchor"], repo, env)
    _run(["git", "commit", "-m", "anchor"], repo, env)
    _run(["git", "branch", "recursion-lane"], repo, env)
    _run(["git", "worktree", "add", str(lane), "recursion-lane"], repo, env)

    scripts = repo / "scripts"
    scripts.mkdir()
    wrapper = scripts / "bd"
    shutil.copy2(ROOT / "scripts" / "bd", wrapper)
    wrapper.chmod(0o755)
    guard = repo / "devtools" / "bd_reimport_guard.py"
    guard.parent.mkdir()
    shutil.copy2(ROOT / "devtools" / "bd_reimport_guard.py", guard)

    tool_dirs: list[str] = []
    for executable in ("git", "sh", "readlink", "python3"):
        resolved = shutil.which(executable)
        assert resolved is not None
        directory = str(Path(resolved).parent)
        if directory not in tool_dirs:
            tool_dirs.append(directory)
    env["PATH"] = os.pathsep.join((str(scripts), *tool_dirs))
    common_dir_text = _run(["git", "rev-parse", "--git-common-dir"], lane, env).stdout.strip()
    common_dir = Path(common_dir_text)
    if not common_dir.is_absolute():
        common_dir = lane / common_dir
    return env, repo, lane, common_dir.resolve()


@pytest.mark.parametrize(
    "route",
    [
        "bare-wrapper",
        "symlink-wrapper",
        "script-recursion",
        "script-symlink-recursion",
        "interpreter-recursion-versioned",
        "interpreter-recursion-renamed",
    ],
)
def test_bd_wrapper_rejects_recursive_real_binary_routes(tmp_path: Path, route: str) -> None:
    """Aliases and interpreter-mediated recursion fail before the common lock."""
    env, _repo, lane, common_dir = _setup_guard_linked_worktree(tmp_path)
    wrapper = lane.parent / "coordinator" / "scripts" / "bd"
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
        interpreter = tmp_path / ("python3.14" if route.endswith("versioned") else "renamed-interpreter")
        shutil.copy2(Path(sys.executable).resolve(), interpreter)
        interpreter.chmod(0o755)
        recursive = tmp_path / route
        python_code = f"import os; os.execv({str(wrapper)!r}, [{str(wrapper)!r}, *os.sys.argv[1:]])"
        recursive.write_text(f'#!/bin/sh\nexec {interpreter!s} -c {shlex.quote(python_code)} "$@"\n')
        recursive.chmod(0o755)
        env["POLYLOGUE_BD_REAL"] = str(recursive)

    result = subprocess.run([str(wrapper), "--help"], cwd=lane, env=env, capture_output=True, text=True, timeout=10)
    expected_returncode = 126 if route != "bare-wrapper" else 127
    assert result.returncode == expected_returncode
    assert not (common_dir / "polylogue-bd-guard.lock").exists()


def test_bd_wrapper_rejects_direct_python_target_before_guard_lock(tmp_path: Path) -> None:
    """POLYLOGUE_BD_REAL cannot turn the wrapper into a Python trampoline."""
    env, _repo, lane, common_dir = _setup_guard_linked_worktree(tmp_path)
    wrapper = lane.parent / "coordinator" / "scripts" / "bd"
    env["POLYLOGUE_BD_REAL"] = sys.executable

    result = subprocess.run([str(wrapper), "--help"], cwd=lane, env=env, capture_output=True, text=True, timeout=10)

    assert result.returncode == 126
    assert "interpreter" in result.stderr
    assert not (common_dir / "polylogue-bd-guard.lock").exists()


@pytest.mark.parametrize("interpreter_name", ["sh", "python3"])
def test_bd_wrapper_survives_interpreter_path_alias_to_wrapper(tmp_path: Path, interpreter_name: str) -> None:
    """The wrapper reaches the guard when PATH aliases an interpreter to itself."""
    env, repo, lane, common_dir = _setup_guard_linked_worktree(tmp_path)
    wrapper = repo / "scripts" / "bd"
    alias_dir = tmp_path / "interpreter-alias"
    alias_dir.mkdir()
    (alias_dir / interpreter_name).symlink_to(wrapper)
    env["PATH"] = os.pathsep.join((str(alias_dir), env["PATH"]))
    true_binary = shutil.which("true", path=env["PATH"])
    assert true_binary is not None
    env["POLYLOGUE_BD_REAL"] = true_binary

    result = subprocess.run([str(wrapper), "--help"], cwd=lane, env=env, capture_output=True, text=True, timeout=10)

    assert result.returncode == 0, result.stderr
    assert (common_dir / "polylogue-bd-guard.lock").exists()


def test_bd_wrapper_skips_indirect_python3_trampoline(tmp_path: Path) -> None:
    """A python3 launcher that invokes the wrapper cannot recurse indefinitely."""
    env, repo, lane, common_dir = _setup_guard_linked_worktree(tmp_path)
    wrapper = repo / "scripts" / "bd"
    alias_dir = tmp_path / "interpreter-trampoline"
    alias_dir.mkdir()
    trampoline = alias_dir / "python3"
    trampoline.write_text(f'#!/bin/sh\nexec {wrapper!s} "$@"\n')
    trampoline.chmod(0o755)
    env["PATH"] = os.pathsep.join((str(alias_dir), env["PATH"]))
    true_binary = shutil.which("true", path=env["PATH"])
    assert true_binary is not None
    env["POLYLOGUE_BD_REAL"] = true_binary

    result = subprocess.run([str(wrapper), "--help"], cwd=lane, env=env, capture_output=True, text=True, timeout=10)

    assert result.returncode == 0, result.stderr
    assert (common_dir / "polylogue-bd-guard.lock").exists()


@pytest.mark.parametrize(
    ("label", "candidate_payload"),
    [
        ("malformed", '{"id":"guard-a"}\nnot-json\n'),
        ("missing-id", '{"title":"guard-a"}\n'),
        ("duplicate", '{"id":"guard-a"}\n{"id":"guard-a"}\n'),
        ("non-finite", '{"id":"guard-a","priority":NaN}\n'),
        ("duplicate-key", '{"id":"guard-a","id":"guard-b"}\n'),
        ("invalid-type", '{"id":"guard-a","labels":"area:beads"}\n'),
        ("invalid-timestamp", '{"id":"guard-a","updated_at":"tomorrow"}\n'),
        ("invalid-id", '{"id":42}\n'),
    ],
)
def test_bd_wrapper_fails_closed_on_invalid_candidate_jsonl(tmp_path: Path, label: str, candidate_payload: str) -> None:
    """The real wrapper rejects malicious candidate JSONL before delegation."""
    env, repo, lane, _common_dir = _setup_guard_linked_worktree(tmp_path)
    wrapper = repo / "scripts" / "bd"
    bd = _installed_bd()
    if bd is None:
        pytest.skip("bd is not installed")
    _run([bd, "init", "--prefix", "guard", "--quiet", "--non-interactive", "--skip-hooks", "--skip-agents"], lane, env)
    (lane / ".beads").mkdir(exist_ok=True)
    (lane / ".beads" / "issues.jsonl").write_text(candidate_payload)
    env["POLYLOGUE_BD_REAL"] = bd

    result = subprocess.run([str(wrapper), "show"], cwd=lane, env=env, capture_output=True, text=True, timeout=30)

    assert result.returncode == 125, result.stderr
    assert "failed validation" in result.stderr


def test_bd_wrapper_does_not_resurrect_deleted_candidate_only_row(tmp_path: Path) -> None:
    """A row absent from live export is not treated as a newly-created row."""
    env, repo, lane, _common_dir = _setup_guard_linked_worktree(tmp_path)
    wrapper = repo / "scripts" / "bd"
    bd = _installed_bd()
    if bd is None:
        pytest.skip("bd is not installed")
    _run([bd, "init", "--prefix", "guard", "--quiet", "--non-interactive", "--skip-hooks", "--skip-agents"], lane, env)
    candidate_id = "guard-deleted"
    (lane / ".beads").mkdir(exist_ok=True)
    (lane / ".beads" / "issues.jsonl").write_text(
        json.dumps(
            {
                "_type": "issue",
                "id": candidate_id,
                "title": "deleted live row",
                "description": "",
                "status": "open",
                "priority": 1,
                "issue_type": "bug",
                "owner": "",
                "created_at": "2026-01-01T00:00:00Z",
                "created_by": "test",
                "updated_at": "2026-01-01T00:00:00Z",
                "labels": [],
                "comment_count": 0,
                "dependency_count": 0,
                "dependent_count": 0,
            }
        )
        + "\n"
    )
    env["POLYLOGUE_BD_REAL"] = bd

    result = subprocess.run(
        [str(wrapper), "import", ".beads/issues.jsonl"],
        cwd=lane,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 125
    assert "durable new-row proof" in result.stderr
    safe_env = env | {"BD_IMPORT_AUTO": "false"}
    snapshot = tmp_path / "live.jsonl"
    _run([bd, "export", "-o", str(snapshot)], lane, safe_env)
    assert candidate_id not in snapshot.read_text()

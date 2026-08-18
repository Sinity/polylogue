"""Linked worktrees keep tracked wrappers clean while routing through common authority.

The production path exercised here is the deployed shell configurator and
``.envrc``.  A real temporary Git repository gains a linked worktree on an
older branch, then the common checkout configures every worktree.  Before the
fix, the configurator replaced the lane's tracked ``scripts/bd`` file with a
symlink, making ``git status --porcelain`` report ``T scripts/bd``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {command!r}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _source_envrc_and_find_bd(lane: Path, env: dict[str, str]) -> str:
    """Use the real envrc routing logic without invoking the Beads command."""
    return _run(
        [
            "bash",
            "-c",
            'PATH_add() { PATH="$1:$PATH"; export PATH; }; use() { :; }; source .envrc; command -v bd',
        ],
        lane,
        env,
    ).stdout.strip()


def test_common_authority_keeps_stale_linked_worktree_clean(tmp_path: Path) -> None:
    """The common wrapper and hooks win without modifying a lane's tracked files."""
    common = tmp_path / "coordinator"
    lane = tmp_path / "stale-lane"
    binary_dir = tmp_path / "bin"
    setup_hooks = tmp_path / "setup-hooks"
    probe = tmp_path / "hook-authority"
    common.mkdir()
    binary_dir.mkdir()
    setup_hooks.mkdir()
    (binary_dir / "bd").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (binary_dir / "bd").chmod(0o755)

    env = os.environ.copy()
    for key in ("POLYLOGUE_BD_REAL", "VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
        env.pop(key, None)
    env["PATH"] = os.pathsep.join((str(binary_dir), env["PATH"]))
    env["WORKTREE_HOOK_PROBE"] = str(probe)

    _run(["git", "init", "--initial-branch=main"], common, env)
    _run(["git", "config", "user.email", "test@example.invalid"], common, env)
    _run(["git", "config", "user.name", "worktree authority test"], common, env)
    _run(["git", "config", "core.hooksPath", str(setup_hooks)], common, env)

    for directory in (common / "polylogue", common / "tests", common / "devtools"):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / ".envrc", common / ".envrc")
    scripts = common / "scripts"
    scripts.mkdir()
    stale_wrapper = "#!/bin/sh\nprintf '%s\\n' stale-wrapper\n"
    (scripts / "bd").write_text(stale_wrapper, encoding="utf-8")
    (scripts / "bd").chmod(0o755)
    shutil.copy2(ROOT / "scripts" / "configure-git-hooks", scripts / "configure-git-hooks")
    (scripts / "configure-git-hooks").chmod(0o755)
    # `.beads-hooks` is the only hook set the installer selects; the `.githooks`
    # fallback was unreachable (it is chosen only when `.beads-hooks` is absent,
    # and that directory is committed) and has been removed. This test is about
    # common-checkout hook authority, not the directory's name.
    hooks = common / ".beads-hooks"
    hooks.mkdir()
    (hooks / "post-checkout").write_text(
        "#!/bin/sh\nprintf '%s\\n' stale-hook > \"$WORKTREE_HOOK_PROBE\"\n", encoding="utf-8"
    )
    (hooks / "post-checkout").chmod(0o755)
    _run(["git", "add", ".envrc", "scripts", ".beads-hooks"], common, env)
    _run(["git", "commit", "-m", "historical lane"], common, env)
    _run(["git", "worktree", "add", "-b", "stale-lane", str(lane)], common, env)

    shutil.copy2(ROOT / "scripts" / "bd", scripts / "bd")
    (scripts / "bd").chmod(0o755)
    (hooks / "post-checkout").write_text(
        "#!/bin/sh\nprintf '%s\\n' common-hook > \"$WORKTREE_HOOK_PROBE\"\n", encoding="utf-8"
    )
    _run(["git", "add", "scripts/bd", ".beads-hooks/post-checkout"], common, env)
    _run(["git", "commit", "-m", "deploy common authority"], common, env)

    _run([str(scripts / "configure-git-hooks")], common, env)

    assert _run(["git", "status", "--porcelain"], lane, env).stdout == ""
    lane_wrapper = lane / "scripts" / "bd"
    assert not lane_wrapper.is_symlink()
    assert lane_wrapper.read_text(encoding="utf-8") == stale_wrapper
    assert _source_envrc_and_find_bd(lane, env) == str((common / "scripts" / "bd").resolve())
    assert _run(["git", "config", "--get", "core.hooksPath"], lane, env).stdout.strip() == str(hooks.resolve())

    _run(["git", "switch", "--detach", "HEAD"], lane, env)
    assert probe.read_text(encoding="utf-8") == "common-hook\n"

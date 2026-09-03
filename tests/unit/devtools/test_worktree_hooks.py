"""The worktree provisioning hooks are shell; a path with a space proves it.

Worktree paths come from the user's own configuration, so a hook that
interpolates one unquoted works until the day a checkout lives under a
directory with a space in its name and then silently provisions the wrong
interpreter or seeds nothing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]


def _hook(name: str) -> str:
    hooks = tomllib.loads((REPO_ROOT / ".config" / "wt.toml").read_text(encoding="utf-8"))["pre-start"]
    return str(next(hook[name] for hook in hooks if name in hook))


def _argv_under_sh(command: str, primary_worktree_path: Path, workspace: Path) -> dict[str, list[str]]:
    """Run *command* with the real binaries replaced by per-binary argv recorders."""
    rendered = command.replace("{{ primary_worktree_path }}", str(primary_worktree_path))
    recorder = workspace / "bin"
    recorder.mkdir(parents=True, exist_ok=True)
    for name in ("uv", "readlink"):
        script = recorder / name
        body = 'printf "%s\\n" "$@" >> ' + f'"{workspace}/{name}.argv"\n'
        if name == "readlink":
            # The venv hook consumes readlink's output inside a command
            # substitution, so the stub must still echo its operand.
            body += 'shift; printf "%s\\n" "$1"\n'
        script.write_text("#!/bin/sh\n" + body)
        script.chmod(0o755)
    subprocess.run(
        ["/bin/sh", "-c", rendered],
        cwd=workspace,
        env={"PATH": f"{recorder}:/usr/bin:/bin"},
        check=False,
        capture_output=True,
    )
    return {
        name: (log.read_text(encoding="utf-8").splitlines() if (log := workspace / f"{name}.argv").exists() else [])
        for name in ("uv", "readlink")
    }


def test_the_venv_hook_passes_a_spaced_worktree_path_as_one_argument(tmp_path: Path) -> None:
    """Anti-vacuity: dropping the inner quotes around the readlink operand splits
    the path at the space, so ``readlink`` is handed two operands and
    ``--python`` receives only the fragment before the space.
    """
    primary = tmp_path / "my checkouts" / "polylogue"
    primary.mkdir(parents=True)
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    expected = f"{primary}/.venv/bin/python"

    argv = _argv_under_sh(_hook("venv"), primary, workspace)

    assert argv["readlink"] == ["-f", expected]
    assert argv["uv"][argv["uv"].index("--python") + 1] == expected
    assert len(argv["uv"]) == argv["uv"].index("--python") + 2


def test_the_seed_hook_snapshots_rather_than_byte_copying() -> None:
    """Anti-vacuity: restoring ``cp -f`` makes this red.

    ``cp`` of a datafile a concurrent run is writing yields a torn file, and
    leaves the source's -wal behind.
    """
    seed = _hook("seed")
    assert "cp -f" not in seed
    assert "devtools.testmon_provision --seed" in seed
    assert '"{{ primary_worktree_path }}/.cache/testmon/testmondata"' in seed

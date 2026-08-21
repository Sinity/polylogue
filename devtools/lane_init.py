"""Provision an isolated worktree that executes repository tooling correctly.

The command creates or adopts a lane, gives it its own venv pinned to the
coordinator interpreter, proves the checkout import invariant, and verifies
that Git recognizes the expected linked worktree. It does not carry testmon
state, create a lane ledger, or configure an agent shell: selected verification
owns graph reuse, and the devtools entry point routes later commands through
the lane interpreter automatically.

Usage:
    devtools workspace lane-init /realm/worktrees/lane-cursors --branch feature/sources/cursor-catchup
    devtools workspace lane-init "$PWD"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from devtools import repo_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("worktree", help="lane worktree path (created if missing)")
    parser.add_argument(
        "--branch",
        help="lane branch (derived from an existing worktree; required when creating one)",
    )
    parser.add_argument("--base", default="origin/master", help="base ref for a new branch (default: origin/master)")
    return parser


def _run(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), cwd=cwd, env=env, text=True, capture_output=True, check=False)


def _branch_exists(root: Path, branch: str) -> bool:
    return _run(["git", "-C", str(root), "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"]).returncode == 0


def _ensure_worktree(root: Path, worktree: Path, branch: str, base: str) -> str | None:
    """Create the worktree/branch if missing. Returns an error string or None."""
    if worktree.exists():
        probe = _run(["git", "-C", str(worktree), "rev-parse", "--show-toplevel"])
        if probe.returncode != 0:
            return f"{worktree} exists but is not a git worktree"
        return None
    if _branch_exists(root, branch):
        result = _run(["git", "-C", str(root), "worktree", "add", str(worktree), branch])
    else:
        result = _run(["git", "-C", str(root), "worktree", "add", "-b", branch, str(worktree), base])
    if result.returncode != 0:
        return f"git worktree add failed: {result.stderr.strip()}"
    return None


def _checked_out_branch(worktree: Path) -> str | None:
    """Return an existing worktree's branch, never guessing detached state."""
    probe = _run(["git", "-C", str(worktree), "branch", "--show-current"])
    branch = probe.stdout.strip() if probe.returncode == 0 else ""
    return branch or None


def _base_executable_of(python: Path) -> Path | None:
    """Resolve the real interpreter behind a venv's ``python`` shim.

    ``sys._base_executable`` is the interpreter a venv was created FROM, which
    is what ``uv venv --python`` needs and what the testmon environment digest
    fingerprints. ``sys.executable`` inside a venv is the shim, not the build.
    """
    probe = _run([str(python), "-c", "import sys; print(sys._base_executable or sys.executable)"])
    if probe.returncode != 0:
        return None
    text = probe.stdout.strip()
    return Path(text) if text else None


def coordinator_base_interpreter(root: Path) -> Path | None:
    """The interpreter every lane must be built from: the coordinator's own.

    A lane provisioned with a DIFFERENT Python is not merely stylistically
    inconsistent, it is broken in two separate ways, both observed
    2026-08-19 (polylogue-l218h):

    1. ``uv`` left to its own resolution downloaded a free-threaded CPython
       3.14.5 for a lane whose coordinator runs Nix CPython 3.14.4. On that
       build ``import hypothesis`` raises ``AttributeError: 'installed_base'``
       from ``sysconfig``, so ``tests/conftest.py`` fails to import and EVERY
       ``devtools test`` invocation in the lane dies during collection. The
       lane looks provisioned and cannot run a single test.
    Derived from the coordinator's venv rather than hardcoded: the store path
    changes on every nixpkgs bump, and a pinned literal would rot silently
    into the same class of mismatch it exists to prevent.
    """
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.exists():
        resolved = _base_executable_of(venv_python)
        if resolved is not None and resolved.exists():
            return resolved
    fallback = getattr(sys, "_base_executable", None) or sys.executable
    candidate = Path(fallback)
    return candidate if candidate.exists() else None


def _interpreter_guard(worktree: Path, expected: Path) -> str | None:
    """Prove the provisioned lane venv was actually built from ``expected``.

    ``uv sync --python`` can silently fall back (an already-present ``.venv``
    built from another interpreter is reused rather than rebuilt), so pinning
    the request is not the same as verifying the result.
    """
    python = worktree / ".venv" / "bin" / "python"
    if not python.exists():
        return f"no venv python at {python}"
    actual = _base_executable_of(python)
    if actual is None:
        return f"could not resolve the lane interpreter behind {python}"
    if actual.resolve() != expected.resolve():
        return (
            "lane interpreter does not match the coordinator's:\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            "a lane built from a different Python cannot share the coordinator's "
            "testmon graph, and has been observed unable to import hypothesis at all"
        )
    return None


#: Devshell-exported variables that describe the DEVSHELL's interpreter and
#: silently reroute a different interpreter's ``sysconfig``. A harness-spawned
#: lane inherits these, and the lane venv's Python then resolves build
#: configuration belonging to another build entirely -- pytest dies before
#: collecting anything, with ``AttributeError: 'installed_base'`` or
#: ``ModuleNotFoundError: No module named '_sysconfigdata_...'`` depending on
#: which pair of builds collide. Observed 2026-08-19 in a lane and in the
#: coordinator checkout (polylogue-l218h).
_INTERPRETER_DESCRIBING_ENV = (
    "_PYTHON_SYSCONFIGDATA_NAME",
    "_PYTHON_HOST_PLATFORM",
    "PYTHONPYCACHEPREFIX",
)
_LANE_ENV_UNSETS = (
    "VIRTUAL_ENV",
    "PYTHONHOME",
    "PYTHONPATH",
    "UV_PROJECT",
    "UV_WORKING_DIR",
    *_INTERPRETER_DESCRIBING_ENV,
)


def _implementation_root() -> Path:
    """Return the checkout containing the lane-init implementation."""
    return Path(__file__).resolve().parents[1]


def _bootstrap_source_error(root: Path) -> str | None:
    """Refuse bootstrap if this command was imported from another checkout."""
    implementation_root = _implementation_root()
    if implementation_root == root.resolve():
        return None
    return (
        "lane-init implementation belongs to a different checkout:\n"
        f"  invoking checkout: {root.resolve()}\n"
        f"  implementation:    {implementation_root}"
    )


def lane_subprocess_env(worktree: Path) -> dict[str, str]:
    """Return an environment that cannot inherit another checkout's Python."""
    env = os.environ.copy()
    inherited_venv = env.get("VIRTUAL_ENV")
    if inherited_venv:
        inherited_bin = str(Path(inherited_venv) / "bin")
        path_entries = env.get("PATH", "").split(os.pathsep)
        env["PATH"] = os.pathsep.join(entry for entry in path_entries if entry != inherited_bin)
    for key in _LANE_ENV_UNSETS:
        env.pop(key, None)
    # Not merely hygiene: these describe a specific interpreter BUILD, so
    # inheriting them into a venv built from a different one is fatal, and
    # PYTHONPYCACHEPREFIX additionally points bytecode caching at the
    # coordinator's .cache.
    for key in _INTERPRETER_DESCRIBING_ENV:
        env.pop(key, None)
    # uv's project-routing environment variables override subprocess.cwd.
    # Leaving either behind can install the coordinator checkout into this
    # lane's otherwise correctly selected .venv.
    env.pop("UV_PROJECT", None)
    env.pop("UV_WORKING_DIR", None)
    env["UV_PROJECT_ENVIRONMENT"] = str(worktree / ".venv")
    return env


def lane_command_env(worktree: Path) -> dict[str, str]:
    """Return the activated, sanitized environment for an existing lane."""
    env = lane_subprocess_env(worktree)
    venv = worktree / ".venv"
    venv_bin = str(venv / "bin")
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = os.pathsep.join((venv_bin, env.get("PATH", "")))
    return env


def _provision_venv(worktree: Path, interpreter: Path | None = None) -> str | None:
    # ``dev`` is the canonical devshell selector in flake.nix; it expands to
    # the same dev-common+speed surface while keeping lane provisioning on the
    # exact extras contract used by the coordinator.
    cmd = ["uv", "sync", "--extra", "dev"]
    if interpreter is not None:
        # Without this uv picks an interpreter by its own resolution order and
        # will happily download one that does not match the coordinator.
        cmd.extend(["--python", str(interpreter)])
    if (worktree / "uv.lock").exists():
        cmd.append("--frozen")
    result = _run(cmd, cwd=worktree, env=lane_subprocess_env(worktree))
    if result.returncode != 0:
        return f"uv sync failed: {result.stderr.strip()[-800:]}"
    return None


def _guard_check(worktree: Path) -> str | None:
    python = worktree / ".venv" / "bin" / "python"
    if not python.exists():
        return f"no venv python at {python}"
    result = _run(
        [
            str(python),
            "-P",
            "-c",
            (
                "from pathlib import Path; "
                "from devtools.checkout_guard import assert_polylogue_matches_checkout; "
                "fingerprint = assert_polylogue_matches_checkout("
                "Path.cwd(), context='devtools workspace lane-init'); "
                "print(fingerprint.polylogue_import_path)"
            ),
        ],
        cwd=worktree,
        env=lane_subprocess_env(worktree),
    )
    if result.returncode != 0:
        return f"lane checkout guard failed: {result.stderr.strip()[-800:]}"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    if error := _bootstrap_source_error(root):
        print(f"lane-init: {error}", file=sys.stderr)
        return 125
    worktree = Path(args.worktree).resolve()
    branch = args.branch
    if branch is None:
        branch = _checked_out_branch(worktree) if worktree.exists() else None
        if branch is None:
            print(
                "lane-init: --branch is required when creating a worktree or initializing a detached checkout",
                file=sys.stderr,
            )
            return 2

    error = _ensure_worktree(root, worktree, branch, args.base)
    if error:
        print(f"lane-init: {error}", file=sys.stderr)
        return 1

    interpreter = coordinator_base_interpreter(root)
    error = _provision_venv(worktree, interpreter)
    if error:
        print(f"lane-init: {error}", file=sys.stderr)
        return 1
    if interpreter is not None:
        error = _interpreter_guard(worktree, interpreter)
        if error:
            print(f"lane-init: {error}", file=sys.stderr)
            return 1
    error = _guard_check(worktree)
    if error:
        print(f"lane-init: {error}", file=sys.stderr)
        return 1

    verify = _run(
        [
            str(worktree / ".venv" / "bin" / "python"),
            "-m",
            "devtools",
            "workspace",
            "verify-worktree",
            str(worktree),
            "--expect-branch",
            branch,
        ],
        cwd=worktree,
        env=lane_subprocess_env(worktree),
    )
    if verify.returncode != 0:
        sys.stderr.write(verify.stdout + verify.stderr)
        print("lane-init: verify-worktree failed", file=sys.stderr)
        return 1

    base_sha = _run(["git", "-C", str(worktree), "rev-parse", "--short=9", "HEAD"]).stdout.strip()
    print(f"lane ready: {worktree}")
    print(f"  branch: {branch} @ {base_sha}")
    print("  venv: provisioned (guard-verified)")
    if interpreter is not None:
        print(f"  interpreter: {interpreter} (verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Subprocess-backed CLI boundary for lab smoke scenarios."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from devtools import repo_root as _get_root
from polylogue.scenarios import ExecutionSpec, run_execution

_PROJECT_ROOT = _get_root()


def _project_polylogue_cli() -> str | None:
    """Return the CLI entrypoint for the current checkout, if present."""
    candidate = _PROJECT_ROOT / ".venv" / "bin" / "polylogue"
    if candidate.exists():
        return str(candidate)
    return None


@dataclass(frozen=True, slots=True)
class CliInvocationResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


def invoke_polylogue_cli(
    execution: ExecutionSpec,
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = 60.0,
) -> CliInvocationResult:
    """Run the real public CLI entrypoint for lab smoke checks."""
    cli_path = _project_polylogue_cli()
    if cli_path is None:
        return _invoke_python_polylogue_cli(execution, env=env, cwd=cwd or _PROJECT_ROOT, timeout=timeout)
    process = run_execution(
        execution,
        binary_overrides={"polylogue": cli_path},
        capture_output=True,
        cwd=cwd or _PROJECT_ROOT,
        env=env,
        timeout=timeout,
    )
    return CliInvocationResult(
        exit_code=process.exit_code,
        stdout=process.stdout,
        stderr=process.stderr,
    )


def _invoke_python_polylogue_cli(
    execution: ExecutionSpec,
    *,
    env: dict[str, str] | None,
    cwd: Path,
    timeout: float,
) -> CliInvocationResult:
    """Run the checkout's Click app when a worktree has no console script."""
    command = execution.command
    if command is None:
        raise ValueError(f"{execution.kind.value} execution has no direct command")
    head, *args = command
    if head != "polylogue":
        raise ValueError(f"CLI boundary only supports polylogue executions, got {head!r}")

    command_env = dict(os.environ)
    if env:
        command_env.update(env)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from polylogue.cli.click_app import cli; cli(prog_name='polylogue')",
            *args,
        ],
        capture_output=True,
        cwd=cwd,
        env=command_env,
        text=True,
        timeout=timeout,
        check=False,
    )
    return CliInvocationResult(
        exit_code=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


__all__ = ["CliInvocationResult", "invoke_polylogue_cli"]

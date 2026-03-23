"""Subprocess-backed CLI boundary for showcase exercises."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ShowcaseCliResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


def invoke_showcase_cli(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = 60.0,
) -> ShowcaseCliResult:
    """Run the real module entrypoint for showcase verification."""
    command = [sys.executable, "-m", "polylogue", *args]
    command_env = dict(os.environ)
    if env:
        command_env.update(env)
    process = subprocess.run(
        command,
        capture_output=True,
        cwd=cwd or _PROJECT_ROOT,
        env=command_env,
        text=True,
        timeout=timeout,
    )
    return ShowcaseCliResult(
        exit_code=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
    )


__all__ = ["ShowcaseCliResult", "invoke_showcase_cli"]

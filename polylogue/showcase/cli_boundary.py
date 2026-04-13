"""Subprocess-backed CLI boundary for showcase exercises."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from polylogue.scenarios import ExecutionSpec, run_execution

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
    execution: ExecutionSpec,
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = 60.0,
) -> ShowcaseCliResult:
    """Run the real public CLI entrypoint for showcase verification."""
    cli_path = shutil.which("polylogue")
    if cli_path is None:
        raise RuntimeError("showcase verification requires `polylogue` on PATH")
    process = run_execution(
        execution,
        binary_overrides={"polylogue": cli_path},
        capture_output=True,
        cwd=cwd or _PROJECT_ROOT,
        env=env,
        timeout=timeout,
    )
    return ShowcaseCliResult(
        exit_code=process.exit_code,
        stdout=process.stdout,
        stderr=process.stderr,
    )


__all__ = ["ShowcaseCliResult", "invoke_showcase_cli"]

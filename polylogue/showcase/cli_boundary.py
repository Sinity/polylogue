"""Subprocess-backed CLI boundary for showcase exercises."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from polylogue.scenarios import ExecutionSpec, run_execution

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_polylogue_cli() -> str | None:
    """Return the CLI entrypoint for the current checkout, if present."""
    candidate = _PROJECT_ROOT / ".venv" / "bin" / "polylogue"
    if candidate.exists():
        return str(candidate)
    return None


@dataclass(frozen=True, slots=True)
class ShowcaseCliResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


def invoke_showcase_cli(
    execution: ExecutionSpec,
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = 60.0,
) -> ShowcaseCliResult:
    """Run the real public CLI entrypoint for showcase verification."""
    cli_path = _project_polylogue_cli()
    if cli_path is None:
        raise RuntimeError("showcase verification requires `.venv/bin/polylogue` in the project root")
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

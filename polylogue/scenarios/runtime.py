"""Shared subprocess-backed runtime for authored execution specs."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .execution import ExecutionSpec


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Completed subprocess result for one authored execution spec."""

    execution: ExecutionSpec
    command: tuple[str, ...]
    exit_code: int
    stdout: str = ""
    stderr: str = ""

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


def resolve_execution_command(
    execution: ExecutionSpec,
    *,
    binary_overrides: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Resolve one authored execution spec into an argv tuple."""
    command = execution.command
    if command is None:
        raise ValueError(f"{execution.kind.value} execution has no direct command")
    if not binary_overrides:
        return command
    head, *tail = command
    replacement = binary_overrides.get(head)
    if replacement is None:
        return command
    return (replacement, *tail)


def run_execution(
    execution: ExecutionSpec,
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    capture_output: bool = False,
    binary_overrides: Mapping[str, str] | None = None,
) -> ExecutionResult:
    """Execute one authored execution spec with consistent subprocess semantics."""
    command = resolve_execution_command(execution, binary_overrides=binary_overrides)
    command_env = dict(os.environ)
    if env:
        command_env.update(env)
    completed = subprocess.run(
        list(command),
        capture_output=capture_output,
        cwd=cwd,
        env=command_env,
        text=True,
        timeout=timeout,
        check=False,
    )
    return ExecutionResult(
        execution=execution,
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


__all__ = ["ExecutionResult", "resolve_execution_command", "run_execution"]

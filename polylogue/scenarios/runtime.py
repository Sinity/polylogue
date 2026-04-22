"""Shared subprocess-backed runtime for authored execution specs."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, TypeVar

from .execution import ExecutionSpec

_RunnerT = TypeVar("_RunnerT")


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


ExecutionRunner: TypeAlias = Callable[..., _RunnerT | Awaitable[_RunnerT]]
ExecutionRunnerResolver: TypeAlias = Callable[[str], ExecutionRunner[_RunnerT]]


@dataclass(frozen=True, slots=True)
class RunnerInvocation:
    """Typed argument bundle for named in-process execution runners."""

    args: tuple[object, ...] = ()
    kwargs: Mapping[str, object] | None = None


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


def resolve_execution_runner(
    execution: ExecutionSpec,
    *,
    runner_resolver: ExecutionRunnerResolver[_RunnerT],
) -> ExecutionRunner[_RunnerT]:
    """Resolve one runner-backed execution spec into a callable."""
    if not execution.is_runner or not execution.runner:
        raise ValueError(f"{execution.kind.value} execution has no named runner")
    return runner_resolver(execution.runner)


async def dispatch_runner_execution(
    execution: ExecutionSpec,
    *,
    runner_resolver: ExecutionRunnerResolver[_RunnerT],
    invocation: RunnerInvocation | None = None,
) -> _RunnerT:
    """Dispatch one runner-backed execution with a typed result contract."""
    runner = resolve_execution_runner(execution, runner_resolver=runner_resolver)
    call = invocation or RunnerInvocation()
    dispatched = runner(*call.args, **dict(call.kwargs or {}))
    if isinstance(dispatched, Awaitable):
        return await dispatched
    return dispatched


async def dispatch_execution(
    execution: ExecutionSpec,
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    capture_output: bool = False,
    binary_overrides: Mapping[str, str] | None = None,
    runner_resolver: ExecutionRunnerResolver[object] | None = None,
    runner_args: Sequence[object] = (),
    runner_kwargs: Mapping[str, object] | None = None,
) -> object:
    """Dispatch one authored execution through either subprocess or runner runtime."""
    if execution.is_runner:
        if runner_resolver is None:
            raise ValueError("runner execution requires a runner_resolver")
        return await dispatch_runner_execution(
            execution,
            runner_resolver=runner_resolver,
            invocation=RunnerInvocation(args=tuple(runner_args), kwargs=runner_kwargs),
        )
    return run_execution(
        execution,
        cwd=cwd,
        env=env,
        timeout=timeout,
        capture_output=capture_output,
        binary_overrides=binary_overrides,
    )


__all__ = [
    "dispatch_execution",
    "dispatch_runner_execution",
    "ExecutionResult",
    "ExecutionRunner",
    "ExecutionRunnerResolver",
    "RunnerInvocation",
    "resolve_execution_command",
    "resolve_execution_runner",
    "run_execution",
]

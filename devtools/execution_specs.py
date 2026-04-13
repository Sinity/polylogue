"""Typed authored execution specs shared across control-plane scenario catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExecutionKind(str, Enum):
    """Authored execution substrate for control-plane scenario catalogs."""

    COMMAND = "command"
    PYTEST = "pytest"
    COMPOSITE = "composite"
    RUNNER = "runner"


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """One authored execution workload."""

    kind: ExecutionKind
    argv: tuple[str, ...] = ()
    members: tuple[str, ...] = ()
    runner: str = ""

    @property
    def is_composite(self) -> bool:
        return self.kind is ExecutionKind.COMPOSITE

    @property
    def is_runner(self) -> bool:
        return self.kind is ExecutionKind.RUNNER

    @property
    def command(self) -> tuple[str, ...] | None:
        if self.is_composite or self.is_runner:
            return None
        if self.kind is ExecutionKind.PYTEST:
            return ("pytest", *self.argv)
        return self.argv

    @property
    def pytest_targets(self) -> tuple[str, ...]:
        if self.kind is not ExecutionKind.PYTEST:
            return ()
        return self.argv

    def pytest_command(self, *prefix_args: str) -> tuple[str, ...]:
        if self.kind is not ExecutionKind.PYTEST:
            raise ValueError(f"{self.kind.value} execution cannot render a pytest command")
        return ("pytest", *prefix_args, *self.argv)


def command_execution(*argv: str) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.COMMAND, argv=tuple(argv))


def pytest_execution(*argv: str) -> ExecutionSpec:
    if argv and argv[0] == "pytest":
        argv = argv[1:]
    return ExecutionSpec(kind=ExecutionKind.PYTEST, argv=tuple(argv))


def composite_execution(*members: str) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.COMPOSITE, members=tuple(members))


def runner_execution(runner: str) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.RUNNER, runner=runner)


__all__ = [
    "command_execution",
    "composite_execution",
    "ExecutionKind",
    "ExecutionSpec",
    "pytest_execution",
    "runner_execution",
]

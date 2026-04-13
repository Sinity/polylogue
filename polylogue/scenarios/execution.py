"""Shared authored execution specs for scenario-bearing surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ExecutionKind(str, Enum):
    """Authored execution substrate for scenario-bearing catalogs."""

    COMMAND = "command"
    POLYLOGUE = "polylogue"
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
        if self.kind is ExecutionKind.POLYLOGUE:
            return ("polylogue", "--plain", *self.argv)
        if self.kind is ExecutionKind.PYTEST:
            return ("pytest", *self.argv)
        return self.argv

    @property
    def display_command(self) -> tuple[str, ...] | None:
        command = self.command
        if command is None:
            return None
        if self.kind is ExecutionKind.POLYLOGUE:
            return ("polylogue", *self.argv)
        return command

    @property
    def pytest_targets(self) -> tuple[str, ...]:
        if self.kind is not ExecutionKind.PYTEST:
            return ()
        return self.argv

    @property
    def polylogue_args(self) -> tuple[str, ...]:
        if self.kind is not ExecutionKind.POLYLOGUE:
            return ()
        return self.argv

    @property
    def polylogue_invoke_args(self) -> tuple[str, ...]:
        if self.kind is not ExecutionKind.POLYLOGUE:
            return ()
        return ("--plain", *self.argv)

    def pytest_command(self, *prefix_args: str) -> tuple[str, ...]:
        if self.kind is not ExecutionKind.PYTEST:
            raise ValueError(f"{self.kind.value} execution cannot render a pytest command")
        return ("pytest", *prefix_args, *self.argv)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind.value}
        if self.argv:
            payload["argv"] = list(self.argv)
        if self.members:
            payload["members"] = list(self.members)
        if self.runner:
            payload["runner"] = self.runner
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ExecutionSpec:
        kind = ExecutionKind(str(payload["kind"]))
        argv = tuple(str(item) for item in payload.get("argv", ()))
        members = tuple(str(item) for item in payload.get("members", ()))
        runner = str(payload.get("runner", "")) if payload.get("runner") is not None else ""
        return cls(kind=kind, argv=argv, members=members, runner=runner)


def command_execution(*argv: str) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.COMMAND, argv=tuple(argv))


def polylogue_execution(*argv: str) -> ExecutionSpec:
    if argv[:2] == ("polylogue", "--plain"):
        argv = argv[2:]
    elif argv[:1] == ("polylogue",):
        argv = argv[1:]
    return ExecutionSpec(kind=ExecutionKind.POLYLOGUE, argv=tuple(argv))


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
    "polylogue_execution",
    "pytest_execution",
    "runner_execution",
]

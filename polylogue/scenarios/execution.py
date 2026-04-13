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
    DEVTOOLS = "devtools"
    PYTEST = "pytest"
    MEMORY_BUDGET = "memory-budget"
    COMPOSITE = "composite"
    RUNNER = "runner"


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """One authored execution workload."""

    kind: ExecutionKind
    argv: tuple[str, ...] = ()
    members: tuple[str, ...] = ()
    runner: str = ""
    subcommand: str = ""
    max_rss_mb: int = 0
    wrapped: ExecutionSpec | None = None

    @property
    def is_composite(self) -> bool:
        return self.kind is ExecutionKind.COMPOSITE

    @property
    def is_runner(self) -> bool:
        return self.kind is ExecutionKind.RUNNER

    @property
    def is_devtools(self) -> bool:
        return self.kind is ExecutionKind.DEVTOOLS

    @property
    def is_memory_budget(self) -> bool:
        return self.kind is ExecutionKind.MEMORY_BUDGET

    @property
    def command(self) -> tuple[str, ...] | None:
        if self.is_composite or self.is_runner:
            return None
        if self.kind is ExecutionKind.POLYLOGUE:
            return ("polylogue", "--plain", *self.argv)
        if self.kind is ExecutionKind.DEVTOOLS:
            from devtools.command_catalog import control_plane_argv

            return tuple(control_plane_argv(self.subcommand, *self.argv))
        if self.kind is ExecutionKind.MEMORY_BUDGET:
            if self.wrapped is None or self.max_rss_mb <= 0:
                return None
            wrapped_command = self.wrapped.command
            if wrapped_command is None:
                return None
            from devtools.command_catalog import control_plane_argv

            return tuple(
                control_plane_argv(
                    "query-memory-budget",
                    "--max-rss-mb",
                    str(self.max_rss_mb),
                    "--",
                    *wrapped_command,
                )
            )
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
        if self.kind is ExecutionKind.MEMORY_BUDGET and self.wrapped is not None:
            wrapped_display = self.wrapped.display_command or self.wrapped.command
            if wrapped_display is None:
                return command
            return (
                "devtools",
                "query-memory-budget",
                "--max-rss-mb",
                str(self.max_rss_mb),
                "--",
                *wrapped_display,
            )
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
        if self.subcommand:
            payload["subcommand"] = self.subcommand
        if self.max_rss_mb > 0:
            payload["max_rss_mb"] = self.max_rss_mb
        if self.wrapped is not None:
            payload["wrapped"] = self.wrapped.to_payload()
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ExecutionSpec:
        kind = ExecutionKind(str(payload["kind"]))
        argv = tuple(str(item) for item in payload.get("argv", ()))
        members = tuple(str(item) for item in payload.get("members", ()))
        runner = str(payload.get("runner", "")) if payload.get("runner") is not None else ""
        subcommand = str(payload.get("subcommand", "")) if payload.get("subcommand") is not None else ""
        max_rss_mb = int(payload.get("max_rss_mb", 0) or 0)
        wrapped_payload = payload.get("wrapped")
        wrapped = cls.from_payload(wrapped_payload) if isinstance(wrapped_payload, Mapping) else None
        return cls(
            kind=kind,
            argv=argv,
            members=members,
            runner=runner,
            subcommand=subcommand,
            max_rss_mb=max_rss_mb,
            wrapped=wrapped,
        )


def command_execution(*argv: str) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.COMMAND, argv=tuple(argv))


def polylogue_execution(*argv: str) -> ExecutionSpec:
    if argv[:2] == ("polylogue", "--plain"):
        argv = argv[2:]
    elif argv[:1] == ("polylogue",):
        argv = argv[1:]
    return ExecutionSpec(kind=ExecutionKind.POLYLOGUE, argv=tuple(argv))


def devtools_execution(subcommand: str, *argv: str) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.DEVTOOLS, subcommand=subcommand, argv=tuple(argv))


def pytest_execution(*argv: str) -> ExecutionSpec:
    if argv and argv[0] == "pytest":
        argv = argv[1:]
    return ExecutionSpec(kind=ExecutionKind.PYTEST, argv=tuple(argv))


def composite_execution(*members: str) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.COMPOSITE, members=tuple(members))


def runner_execution(runner: str) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.RUNNER, runner=runner)


def memory_budget_execution(max_rss_mb: int, execution: ExecutionSpec) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.MEMORY_BUDGET, max_rss_mb=max_rss_mb, wrapped=execution)


__all__ = [
    "command_execution",
    "composite_execution",
    "devtools_execution",
    "ExecutionKind",
    "ExecutionSpec",
    "memory_budget_execution",
    "polylogue_execution",
    "pytest_execution",
    "runner_execution",
]

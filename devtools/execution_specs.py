"""Typed authored execution specs shared across control-plane scenario catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExecutionKind(str, Enum):
    """Authored execution substrate for control-plane scenario catalogs."""

    COMMAND = "command"
    PYTEST = "pytest"
    COMPOSITE = "composite"


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """One authored execution workload."""

    kind: ExecutionKind
    argv: tuple[str, ...] = ()
    members: tuple[str, ...] = ()

    @property
    def is_composite(self) -> bool:
        return self.kind is ExecutionKind.COMPOSITE

    @property
    def command(self) -> tuple[str, ...] | None:
        if self.is_composite:
            return None
        return self.argv

    @property
    def pytest_targets(self) -> tuple[str, ...]:
        if self.kind is not ExecutionKind.PYTEST:
            return ()
        return self.argv


def command_execution(*argv: str) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.COMMAND, argv=tuple(argv))


def pytest_execution(*argv: str) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.PYTEST, argv=tuple(argv))


def composite_execution(*members: str) -> ExecutionSpec:
    return ExecutionSpec(kind=ExecutionKind.COMPOSITE, members=tuple(members))


__all__ = [
    "command_execution",
    "composite_execution",
    "ExecutionKind",
    "ExecutionSpec",
    "pytest_execution",
]

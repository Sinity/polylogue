"""Shared executable scenario-source models."""

from __future__ import annotations

from dataclasses import dataclass

from .execution import ExecutionSpec
from .sources import NamedScenarioSource


@dataclass(frozen=True, kw_only=True)
class ExecutableScenario(NamedScenarioSource):
    """One authored scenario source with an attached execution workload."""

    execution: ExecutionSpec | None = None

    @property
    def is_composite(self) -> bool:
        return self.execution is not None and self.execution.is_composite

    @property
    def is_runner(self) -> bool:
        return self.execution is not None and self.execution.is_runner

    @property
    def command(self) -> list[str] | None:
        if self.execution is None:
            return None
        command = self.execution.command
        return list(command) if command is not None else None

    @property
    def members(self) -> tuple[str, ...]:
        if self.execution is None:
            return ()
        return self.execution.members

    @property
    def runner(self) -> str | None:
        if self.execution is None or not self.execution.is_runner:
            return None
        return self.execution.runner

    @property
    def tests(self) -> tuple[str, ...]:
        if self.execution is None:
            return ()
        return self.execution.pytest_targets


__all__ = ["ExecutableScenario"]

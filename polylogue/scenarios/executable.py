"""Shared executable scenario-source models."""

from __future__ import annotations

from dataclasses import dataclass, field

from polylogue.products.authored_payloads import PayloadDict

from .assertions import AssertionSpec
from .execution import ExecutionSpec
from .metadata import ScenarioMetadata
from .sources import NamedScenarioSource


@dataclass(frozen=True, kw_only=True)
class ExecutableScenario(NamedScenarioSource):
    """One authored scenario source with an attached execution workload."""

    execution: ExecutionSpec | None = None
    assertion: AssertionSpec = field(default_factory=AssertionSpec)

    def __post_init__(self) -> None:
        if self.execution is None:
            return
        merged = ScenarioMetadata.from_object(self).with_default_targets(self.execution.metadata)
        object.__setattr__(self, "path_targets", merged.path_targets)
        object.__setattr__(self, "artifact_targets", merged.artifact_targets)
        object.__setattr__(self, "operation_targets", merged.operation_targets)
        object.__setattr__(self, "tags", merged.tags)

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

    def projection_source_payload(self) -> PayloadDict:
        payload = self.scenario_payload()
        if self.execution is not None:
            payload["execution"] = self.execution.to_payload()
        assertion_payload = self.assertion.to_payload()
        if assertion_payload:
            payload["assertion"] = assertion_payload
        return payload


__all__ = ["ExecutableScenario"]

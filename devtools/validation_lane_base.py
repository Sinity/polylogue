"""Shared validation-lane config and command builders."""

from __future__ import annotations

from dataclasses import dataclass

from devtools.command_catalog import control_plane_argv
from devtools.execution_specs import ExecutionSpec, command_execution, composite_execution, pytest_execution
from polylogue.scenarios import ScenarioMetadata


@dataclass(frozen=True)
class LaneConfig(ScenarioMetadata):
    """Configuration for a validation lane."""

    name: str
    description: str
    timeout_s: int
    execution: ExecutionSpec

    @property
    def is_composite(self) -> bool:
        return self.execution.is_composite

    @property
    def command(self) -> list[str] | None:
        command = self.execution.command
        return list(command) if command is not None else None

    @property
    def sub_lanes(self) -> tuple[str, ...]:
        return self.execution.members


def cli_lane(
    name: str,
    description: str,
    timeout_s: int,
    executable: str,
    *args: str,
    origin: str = "authored.validation-lane",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneConfig:
    return LaneConfig(
        name=name,
        description=description,
        timeout_s=timeout_s,
        execution=command_execution(executable, *args),
        origin=origin,
        path_targets=path_targets,
        artifact_targets=artifact_targets,
        operation_targets=operation_targets,
        tags=tags,
    )


def pytest_lane(
    name: str,
    description: str,
    timeout_s: int,
    *args: str,
    origin: str = "authored.validation-lane",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneConfig:
    return cli_lane(
        name,
        description,
        timeout_s,
        *pytest_execution("pytest", *args).argv,
        origin=origin,
        path_targets=path_targets,
        artifact_targets=artifact_targets,
        operation_targets=operation_targets,
        tags=tags,
    )


def devtools_lane(
    name: str,
    description: str,
    timeout_s: int,
    subcommand: str,
    *args: str,
    origin: str = "authored.validation-lane",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneConfig:
    return LaneConfig(
        name=name,
        description=description,
        timeout_s=timeout_s,
        execution=command_execution(*control_plane_argv(subcommand, *args)),
        origin=origin,
        path_targets=path_targets,
        artifact_targets=artifact_targets,
        operation_targets=operation_targets,
        tags=tags,
    )


def polylogue_lane(
    name: str,
    description: str,
    timeout_s: int,
    *args: str,
    origin: str = "authored.validation-lane",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneConfig:
    return cli_lane(
        name,
        description,
        timeout_s,
        "polylogue",
        "--plain",
        *args,
        origin=origin,
        path_targets=path_targets,
        artifact_targets=artifact_targets,
        operation_targets=operation_targets,
        tags=tags,
    )


def memory_budget_lane(
    name: str,
    description: str,
    timeout_s: int,
    *,
    max_rss_mb: int,
    command: list[str],
    origin: str = "authored.validation-lane",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneConfig:
    return devtools_lane(
        name,
        description,
        timeout_s,
        "query-memory-budget",
        "--max-rss-mb",
        str(max_rss_mb),
        "--",
        *command,
        origin=origin,
        path_targets=path_targets,
        artifact_targets=artifact_targets,
        operation_targets=operation_targets,
        tags=tags,
    )


def composite_lane(
    name: str,
    description: str,
    timeout_s: int,
    *sub_lanes: str,
    origin: str = "authored.validation-lane.composite",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneConfig:
    return LaneConfig(
        name=name,
        description=description,
        timeout_s=timeout_s,
        execution=composite_execution(*sub_lanes),
        origin=origin,
        path_targets=path_targets,
        artifact_targets=artifact_targets,
        operation_targets=operation_targets,
        tags=tags,
    )


__all__ = [
    "LaneConfig",
    "cli_lane",
    "composite_lane",
    "devtools_lane",
    "memory_budget_lane",
    "polylogue_lane",
    "pytest_lane",
]

"""Shared validation-lane config and command builders."""

from __future__ import annotations

from devtools.lane_models import LaneEntry
from polylogue.scenarios import (
    ExecutionSpec,
    command_execution,
    composite_execution,
    devtools_execution,
    memory_budget_execution,
    polylogue_execution,
    pytest_execution,
)


def cli_lane(
    name: str,
    description: str,
    timeout_s: int,
    executable: str,
    *args: str,
    category: str,
    origin: str = "authored.validation-lane",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneEntry:
    return LaneEntry(
        name=name,
        description=description,
        timeout_s=timeout_s,
        category=category,
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
    category: str,
    origin: str = "authored.validation-lane",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneEntry:
    return LaneEntry(
        name=name,
        description=description,
        timeout_s=timeout_s,
        category=category,
        execution=pytest_execution(*args),
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
    category: str,
    origin: str = "authored.validation-lane",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneEntry:
    return LaneEntry(
        name=name,
        description=description,
        timeout_s=timeout_s,
        category=category,
        execution=devtools_execution(subcommand, *args),
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
    category: str,
    origin: str = "authored.validation-lane",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneEntry:
    return LaneEntry(
        name=name,
        description=description,
        timeout_s=timeout_s,
        category=category,
        execution=polylogue_execution(*args),
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
    category: str,
    origin: str = "authored.validation-lane",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneEntry:
    wrapped = _command_list_to_execution(command)
    return LaneEntry(
        name=name,
        description=description,
        timeout_s=timeout_s,
        category=category,
        execution=memory_budget_execution(max_rss_mb, wrapped),
        origin=origin,
        path_targets=path_targets,
        artifact_targets=artifact_targets,
        operation_targets=operation_targets,
        tags=tags,
    )


def _command_list_to_execution(command: list[str]) -> ExecutionSpec:
    if not command:
        raise ValueError("memory_budget_lane requires a non-empty command")
    binary, *argv = command
    if binary == "polylogue":
        return polylogue_execution(*argv)
    if binary == "pytest":
        return pytest_execution(*argv)
    if binary == "devtools" and argv:
        return devtools_execution(argv[0], *argv[1:])
    return command_execution(binary, *argv)


def composite_lane(
    name: str,
    description: str,
    timeout_s: int,
    *sub_lanes: str,
    category: str,
    family: str | None = None,
    origin: str = "authored.validation-lane.composite",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneEntry:
    return LaneEntry(
        name=name,
        description=description,
        timeout_s=timeout_s,
        category=category,
        family=family,
        execution=composite_execution(*sub_lanes),
        origin=origin,
        path_targets=path_targets,
        artifact_targets=artifact_targets,
        operation_targets=operation_targets,
        tags=tags,
    )


__all__ = [
    "cli_lane",
    "composite_lane",
    "devtools_lane",
    "LaneEntry",
    "memory_budget_lane",
    "polylogue_lane",
    "pytest_lane",
]

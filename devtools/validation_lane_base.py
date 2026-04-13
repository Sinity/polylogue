"""Shared validation-lane config and command builders."""

from __future__ import annotations

from devtools.command_catalog import control_plane_argv
from devtools.execution_specs import command_execution, composite_execution, pytest_execution
from devtools.lane_models import LaneEntry


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
    category: str,
    origin: str = "authored.validation-lane",
    path_targets: tuple[str, ...] = (),
    artifact_targets: tuple[str, ...] = (),
    operation_targets: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
) -> LaneEntry:
    return cli_lane(
        name,
        description,
        timeout_s,
        "polylogue",
        "--plain",
        *args,
        category=category,
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
    return devtools_lane(
        name,
        description,
        timeout_s,
        "query-memory-budget",
        "--max-rss-mb",
        str(max_rss_mb),
        "--",
        *command,
        category=category,
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
    category: str,
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

"""Shared validation-lane config and command builders."""

from __future__ import annotations

from dataclasses import dataclass

from devtools.command_catalog import control_plane_argv
from polylogue.scenarios import ScenarioMetadata


@dataclass(frozen=True)
class LaneConfig(ScenarioMetadata):
    """Configuration for a validation lane."""

    name: str
    description: str
    timeout_s: int
    command: list[str] | None = None
    sub_lanes: tuple[str, ...] = ()

    @property
    def is_composite(self) -> bool:
        return bool(self.sub_lanes)


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
        command=[executable, *args],
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
        "pytest",
        *args,
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
        command=list(control_plane_argv(subcommand, *args)),
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
        sub_lanes=sub_lanes,
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

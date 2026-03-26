"""Shared validation-lane config and command builders."""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class LaneConfig:
    """Configuration for a validation lane."""

    name: str
    description: str
    timeout_s: int
    command: list[str] | None = None
    sub_lanes: tuple[str, ...] = ()

    @property
    def is_composite(self) -> bool:
        return bool(self.sub_lanes)


def pytest_lane(name: str, description: str, timeout_s: int, *args: str) -> LaneConfig:
    return LaneConfig(
        name=name,
        description=description,
        timeout_s=timeout_s,
        command=[sys.executable, "-m", "pytest", *args],
    )


def module_lane(name: str, description: str, timeout_s: int, module: str, *args: str) -> LaneConfig:
    return LaneConfig(
        name=name,
        description=description,
        timeout_s=timeout_s,
        command=[sys.executable, "-m", module, *args],
    )


def polylogue_lane(name: str, description: str, timeout_s: int, *args: str) -> LaneConfig:
    return module_lane(name, description, timeout_s, "polylogue", "--plain", *args)


def memory_budget_lane(
    name: str,
    description: str,
    timeout_s: int,
    *,
    max_rss_mb: int,
    command: list[str],
) -> LaneConfig:
    return module_lane(
        name,
        description,
        timeout_s,
        "devtools.query_memory_budget",
        "--max-rss-mb",
        str(max_rss_mb),
        "--",
        *command,
    )


def composite_lane(name: str, description: str, timeout_s: int, *sub_lanes: str) -> LaneConfig:
    return LaneConfig(
        name=name,
        description=description,
        timeout_s=timeout_s,
        sub_lanes=sub_lanes,
    )


__all__ = [
    "LaneConfig",
    "composite_lane",
    "memory_budget_lane",
    "module_lane",
    "polylogue_lane",
    "pytest_lane",
]

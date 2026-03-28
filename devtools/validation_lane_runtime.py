"""Validation lane registry and execution helpers."""

from __future__ import annotations

import subprocess

from devtools.validation_lane_base import LaneConfig
from devtools.validation_lane_catalog_composites import COMPOSITE_LANES
from devtools.validation_lane_catalog_contracts import CONTRACT_LANES
from devtools.validation_lane_catalog_live import LIVE_LANES

LANES: dict[str, LaneConfig] = {
    **CONTRACT_LANES,
    **LIVE_LANES,
    **COMPOSITE_LANES,
}
VALID_LANES = frozenset(LANES)


def parse_lane(lane_name: str) -> LaneConfig:
    if lane_name not in LANES:
        raise ValueError(
            f"Invalid lane: {lane_name!r}. Valid lanes: {', '.join(sorted(VALID_LANES))}"
        )
    return LANES[lane_name]


def build_lane_command(lane: LaneConfig) -> list[str]:
    if lane.command is None:
        raise ValueError(f"Lane {lane.name!r} is composite and has no direct command")
    return lane.command


def print_lane(lane: LaneConfig, *, indent: str = "") -> None:
    print(f"{indent}{lane.name}: {lane.description}")
    if lane.is_composite:
        for child_name in lane.sub_lanes:
            print_lane(parse_lane(child_name), indent=indent + "  ")
    else:
        print(f"{indent}  command: {' '.join(build_lane_command(lane))}")
        print(f"{indent}  timeout: {lane.timeout_s}s")


def run_lane(lane: LaneConfig) -> int:
    if lane.is_composite:
        print(f"Validation lane: {lane.name} — {lane.description}")
        for child_name in lane.sub_lanes:
            exit_code = run_lane(parse_lane(child_name))
            if exit_code != 0:
                return exit_code
        return 0

    cmd = build_lane_command(lane)
    print(f"Validation lane: {lane.name} — {lane.description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Timeout: {lane.timeout_s}s")
    print()

    try:
        result = subprocess.run(cmd, timeout=lane.timeout_s)
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"\nLane {lane.name!r} timed out after {lane.timeout_s}s")
        return 2


__all__ = [
    "LANES",
    "VALID_LANES",
    "build_lane_command",
    "parse_lane",
    "print_lane",
    "run_lane",
]

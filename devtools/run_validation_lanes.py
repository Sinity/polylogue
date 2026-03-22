"""Run named validation lanes for the remaining operator frontier.

Usage:
    python -m devtools.run_validation_lanes --list
    python -m devtools.run_validation_lanes --lane machine-contract
    python -m devtools.run_validation_lanes --lane frontier-local --dry-run
"""

from __future__ import annotations

import subprocess
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


LANES: dict[str, LaneConfig] = {
    "machine-contract": LaneConfig(
        name="machine-contract",
        description="Root CLI JSON success/failure envelopes and runtime-health machine surfaces",
        timeout_s=180,
        command=[sys.executable, "-m", "pytest", "-q", "-n", "0", "-m", "machine_contract"],
    ),
    "query-routing": LaneConfig(
        name="query-routing",
        description="Query-first CLI route planning, integration, and streamed read-surface proofs",
        timeout_s=240,
        command=[sys.executable, "-m", "pytest", "-q", "-n", "0", "-m", "query_routing"],
    ),
    "tui": LaneConfig(
        name="tui",
        description="Textual Mission Control screens and interaction-state coverage",
        timeout_s=240,
        command=[sys.executable, "-m", "pytest", "-q", "-n", "0", "-m", "tui"],
    ),
    "chaos": LaneConfig(
        name="chaos",
        description="Hostility, interruption, and chronology integration coverage",
        timeout_s=900,
        command=[sys.executable, "-m", "pytest", "-q", "-n", "0", "-m", "chaos"],
    ),
    "scale-fast": LaneConfig(
        name="scale-fast",
        description="Fast storage scale budgets",
        timeout_s=120,
        command=[sys.executable, "-m", "devtools.run_scale_lanes", "--lane", "fast"],
    ),
    "scale-slow": LaneConfig(
        name="scale-slow",
        description="Slow local storage scale budgets",
        timeout_s=360,
        command=[sys.executable, "-m", "devtools.run_scale_lanes", "--lane", "slow"],
    ),
    "long-haul-small": LaneConfig(
        name="long-haul-small",
        description="Small reproducible benchmark/long-haul campaign",
        timeout_s=1800,
        command=[sys.executable, "-m", "devtools.run_campaign", "--scale", "small"],
    ),
    "live-exercises": LaneConfig(
        name="live-exercises",
        description="Operator-run live archive showcase/QA exercise lane",
        timeout_s=1800,
        command=[
            sys.executable,
            "-m",
            "polylogue",
            "--plain",
            "qa",
            "--live",
            "--only",
            "exercises",
            "--tier",
            "0",
            "--json",
        ],
    ),
    "frontier-local": LaneConfig(
        name="frontier-local",
        description="Non-live local closure lane for machine/query/TUI/chaos validation",
        timeout_s=1500,
        sub_lanes=("machine-contract", "query-routing", "tui", "chaos"),
    ),
    "frontier-extended": LaneConfig(
        name="frontier-extended",
        description="Local closure lane plus fast scale and small long-haul campaign",
        timeout_s=3600,
        sub_lanes=("frontier-local", "scale-fast", "long-haul-small"),
    ),
}

VALID_LANES = frozenset(LANES)


def parse_lane(lane_name: str) -> LaneConfig:
    """Parse and validate a lane name."""
    if lane_name not in LANES:
        raise ValueError(
            f"Invalid lane: {lane_name!r}. Valid lanes: {', '.join(sorted(VALID_LANES))}"
        )
    return LANES[lane_name]


def build_lane_command(lane: LaneConfig) -> list[str]:
    """Build the concrete subprocess command for a non-composite lane."""
    if lane.command is None:
        raise ValueError(f"Lane {lane.name!r} is composite and has no direct command")
    return lane.command


def _print_lane(lane: LaneConfig, *, indent: str = "") -> None:
    print(f"{indent}{lane.name}: {lane.description}")
    if lane.is_composite:
        for child_name in lane.sub_lanes:
            _print_lane(parse_lane(child_name), indent=indent + "  ")
    else:
        print(f"{indent}  command: {' '.join(build_lane_command(lane))}")
        print(f"{indent}  timeout: {lane.timeout_s}s")


def run_lane(lane: LaneConfig) -> int:
    """Execute a validation lane."""
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


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lane",
        choices=sorted(VALID_LANES),
        help="Validation lane to run",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_lanes",
        help="List available lanes and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected lane command(s) without running them",
    )
    args = parser.parse_args(argv)

    if args.list_lanes:
        print("Available validation lanes:")
        for lane_name in sorted(VALID_LANES):
            lane = parse_lane(lane_name)
            print(f"  {lane.name}: {lane.description}")
        return 0

    if not args.lane:
        parser.error("--lane is required unless --list is used")

    lane = parse_lane(args.lane)
    if args.dry_run:
        _print_lane(lane)
        return 0

    return run_lane(lane)


if __name__ == "__main__":
    sys.exit(main())

"""Run named validation lanes for the remaining operator frontier.

Usage:
    devtools run-validation-lanes --list
    devtools run-validation-lanes --lane machine-contract
    devtools run-validation-lanes --lane frontier-local --dry-run
    devtools run-validation-lanes --lane archive-intelligence --dry-run
"""

from __future__ import annotations

import sys

from devtools.validation_catalog import ValidationLaneEntry as LaneConfig
from devtools.validation_lane_runtime import (
    LANES,
    VALID_LANES,
    build_lane_command,
    parse_lane,
    print_lane,
    run_lane,
)


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
        print_lane(lane)
        return 0

    return run_lane(lane)


__all__ = [
    "LANES",
    "VALID_LANES",
    "LaneConfig",
    "build_lane_command",
    "main",
    "parse_lane",
]


if __name__ == "__main__":
    sys.exit(main())

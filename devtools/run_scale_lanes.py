"""Route scale test lanes through the shared lane substrate.

Usage: devtools run-scale-lanes --lane fast|slow|stretch

Lanes:
  fast    - Unit-level scale tests (200 conversations, <10s)
  slow    - Storage-heavy tests marked @pytest.mark.slow (~30s)
  stretch - Both fast + slow combined with extended timeouts
"""

from __future__ import annotations

import subprocess
import sys

from devtools.validation_lane_base import LaneConfig, cli_lane

LANES: dict[str, LaneConfig] = {
    "fast": cli_lane(
        "fast",
        "Unit-level scale tests (200 conversations)",
        60,
        "pytest",
        "-v",
        "tests/unit/storage/test_scale.py",
        "-x",
        "--timeout=30",
        origin="authored.scale-lane",
        tags=("scale", "fast"),
    ),
    "slow": cli_lane(
        "slow",
        "Storage-heavy tests marked @pytest.mark.slow",
        300,
        "pytest",
        "-v",
        "-m",
        "slow",
        "tests/unit/storage/",
        "--timeout=120",
        origin="authored.scale-lane",
        tags=("scale", "slow"),
    ),
    "stretch": cli_lane(
        "stretch",
        "All scale and slow tests combined",
        600,
        "pytest",
        "-v",
        "tests/unit/storage/test_scale.py",
        "-m",
        "slow or not slow",
        "tests/unit/storage/",
        "--timeout=120",
        origin="authored.scale-lane",
        tags=("scale", "stretch"),
    ),
}

VALID_LANES = frozenset(LANES.keys())


def build_pytest_command(lane: LaneConfig) -> list[str]:
    """Build the full pytest command for a lane."""
    if lane.command is None:
        raise ValueError(f"Lane {lane.name!r} is composite and has no direct command")
    return lane.command


def parse_lane(lane_name: str) -> LaneConfig:
    """Parse and validate a lane name."""
    if lane_name not in LANES:
        raise ValueError(f"Invalid lane: {lane_name!r}. Valid lanes: {', '.join(sorted(VALID_LANES))}")
    return LANES[lane_name]


def run_lane(lane: LaneConfig) -> int:
    """Execute a scale test lane via subprocess."""
    cmd = build_pytest_command(lane)
    print(f"Scale lane: {lane.name} — {lane.description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Timeout: {lane.timeout_s}s")
    print()

    try:
        result = subprocess.run(
            cmd,
            timeout=lane.timeout_s,
        )
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
        required=True,
        choices=sorted(VALID_LANES),
        help="Scale test lane to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pytest command without running it",
    )
    args = parser.parse_args(argv)

    lane = parse_lane(args.lane)

    if args.dry_run:
        cmd = build_pytest_command(lane)
        print(" ".join(cmd))
        return 0

    return run_lane(lane)


if __name__ == "__main__":
    sys.exit(main())

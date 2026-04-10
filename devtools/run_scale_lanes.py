"""Route scale test lanes to appropriate pytest invocations.

Usage: devtools run-scale-lanes --lane fast|slow|stretch

Lanes:
  fast    - Unit-level scale tests (200 conversations, <10s)
  slow    - Storage-heavy tests marked @pytest.mark.slow (~30s)
  stretch - Both fast + slow combined with extended timeouts
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class LaneConfig:
    """Configuration for a scale test lane."""

    name: str
    description: str
    pytest_args: list[str]
    timeout_s: int


# Lane definitions
LANES: dict[str, LaneConfig] = {
    "fast": LaneConfig(
        name="fast",
        description="Unit-level scale tests (200 conversations)",
        pytest_args=[
            "tests/unit/storage/test_scale.py",
            "-x",
            "--timeout=30",
        ],
        timeout_s=60,
    ),
    "slow": LaneConfig(
        name="slow",
        description="Storage-heavy tests marked @pytest.mark.slow",
        pytest_args=[
            "-m",
            "slow",
            "tests/unit/storage/",
            "--timeout=120",
        ],
        timeout_s=300,
    ),
    "stretch": LaneConfig(
        name="stretch",
        description="All scale and slow tests combined",
        pytest_args=[
            "tests/unit/storage/test_scale.py",
            "-m",
            "slow or not slow",
            "tests/unit/storage/",
            "--timeout=120",
        ],
        timeout_s=600,
    ),
}

VALID_LANES = frozenset(LANES.keys())


def build_pytest_command(lane: LaneConfig) -> list[str]:
    """Build the full pytest command for a lane.

    Returns:
        List of command parts suitable for subprocess.run().
    """
    return [sys.executable, "-m", "pytest", "-v"] + lane.pytest_args


def parse_lane(lane_name: str) -> LaneConfig:
    """Parse and validate a lane name.

    Args:
        lane_name: One of 'fast', 'slow', 'stretch'.

    Returns:
        Corresponding LaneConfig.

    Raises:
        ValueError: If lane_name is not valid.
    """
    if lane_name not in LANES:
        raise ValueError(f"Invalid lane: {lane_name!r}. Valid lanes: {', '.join(sorted(VALID_LANES))}")
    return LANES[lane_name]


def run_lane(lane: LaneConfig) -> int:
    """Execute a scale test lane via subprocess.

    Args:
        lane: Lane configuration to execute.

    Returns:
        Process exit code.
    """
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
    """Main entry point.

    Args:
        argv: Command-line arguments (for testing).

    Returns:
        Exit code: 0 = all passed, 1 = failures, 2 = timeout.
    """
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

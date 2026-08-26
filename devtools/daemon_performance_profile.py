"""Managed entrypoint for the daemon/CLI production-route profile."""

from __future__ import annotations

from devtools.run_tests import main as managed_test_main


def main(argv: list[str] | None = None) -> int:
    """Run the named daemon profile through the repository test harness."""

    del argv
    return managed_test_main(
        [
            "tests/benchmarks/test_daemon_uds.py",
            "tests/benchmarks/test_daemon_operation_profile.py",
            "--benchmark-enable",
            "-p",
            "no:xdist",
            "-v",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Managed entrypoint for the complete installed CLI interaction profile."""

from __future__ import annotations

from devtools.run_tests import main as managed_test_main


def main(argv: list[str] | None = None) -> int:
    """Run cold installed CLI plus the direct daemon operation workloads."""

    del argv
    return managed_test_main(
        [
            "tests/benchmarks/test_cli_cold_start.py",
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

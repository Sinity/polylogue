"""Validate the provisioned native testmon graph before a lane is published."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from devtools.testmon_bootstrap import (
    TESTMON_DATA_RELPATH,
    NativeTestmonState,
    inspect_native_testmon_environment,
    testmon_environment_digest,
)


def check_provisioned_testmon(repo_root: Path) -> tuple[str, NativeTestmonState]:
    """Return the expected environment name and its provisioned graph state."""
    environment_name = testmon_environment_digest(repo_root)
    state = inspect_native_testmon_environment(
        repo_root / TESTMON_DATA_RELPATH,
        environment_name=environment_name,
    )
    return environment_name, state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a machine-readable result")
    args = parser.parse_args(argv)

    root = Path(os.getcwd()).resolve()
    try:
        environment_name, state = check_provisioned_testmon(root)
        payload = {
            "database": str(TESTMON_DATA_RELPATH),
            "environment": environment_name,
            "state": state.status,
            "reason": state.reason,
            "corpus_count": state.environment.corpus_count if state.environment else None,
        }
    except Exception as exc:  # Provisioning must report a typed command failure.
        payload = {
            "database": str(TESTMON_DATA_RELPATH),
            "state": "error",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    # A seed graph that does not name this environment is useless, not
    # dangerous: affected verification already refuses to represent it as
    # coverage. Discarding it leaves the workspace in the state of one that was
    # never seeded, which provisions.
    database = root / TESTMON_DATA_RELPATH
    payload["discarded"] = payload["state"] not in {"valid", "error"} and database.exists()
    if payload["discarded"]:
        database.unlink()

    if args.json:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"testmon provision: {payload['state']}: {payload['reason']}")
        if payload.get("environment"):
            print(f"expected environment: {payload['environment']}")
        if payload["discarded"]:
            print(f"discarded the unusable seed at {TESTMON_DATA_RELPATH}")

    return 1 if payload["state"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The checkout-local pytest-testmon graph: where it lives and whether it works.

One datafile per checkout, at ``.cache/testmon/testmondata``, under one fixed
environment name. Every managed pytest run traces into it and writes back, so
the graph is advanced rather than recomputed. A worktree is provisioned by
copying master's datafile: paths are repo-relative and fingerprints are by
content, so a copy is valid immediately.

An absent datafile is not a failure — the next run seeds it. A datafile that
cannot be opened, or that no compatible testmon wrote, is: selecting against
it would silently skip tests.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

TESTMON_DATA_RELPATH = Path(".cache/testmon/testmondata")

#: The single environment every managed run traces and selects under. Pinned so
#: a dependency bump, a Hypothesis profile or a Python patch release cannot
#: rename the environment and orphan the graph.
TESTMON_ENVIRONMENT = "polylogue"

#: Tables a datafile written by a compatible pytest-testmon carries.
_REQUIRED_TABLES = frozenset({"environment", "node"})

_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


class TestmonGraphStatus(StrEnum):
    ABSENT = "absent"
    USABLE = "usable"
    UNUSABLE = "unusable"


@dataclass(frozen=True, slots=True)
class TestmonGraphState:
    status: TestmonGraphStatus
    reason: str

    @property
    def usable(self) -> bool:
        return self.status is TestmonGraphStatus.USABLE


def testmon_datafile(root: Path) -> Path:
    return root / TESTMON_DATA_RELPATH


def inspect_testmon_graph(root: Path) -> TestmonGraphState:
    """Report whether the local datafile can back a selecting run."""
    data_path = testmon_datafile(root)
    if not data_path.is_file() or data_path.stat().st_size == 0:
        return TestmonGraphState(TestmonGraphStatus.ABSENT, "no testmon datafile")
    try:
        connection = sqlite3.connect(f"file:{data_path}?mode=ro", uri=True, timeout=10)
    except sqlite3.Error as exc:
        return TestmonGraphState(TestmonGraphStatus.UNUSABLE, f"the testmon datafile cannot be opened: {exc}")
    try:
        with contextlib.closing(connection):
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    except sqlite3.Error as exc:
        return TestmonGraphState(TestmonGraphStatus.UNUSABLE, f"the testmon datafile is corrupt: {exc}")
    missing = _REQUIRED_TABLES - tables
    if missing:
        return TestmonGraphState(
            TestmonGraphStatus.UNUSABLE,
            f"the testmon datafile was written by an incompatible testmon version (no {', '.join(sorted(missing))})",
        )
    return TestmonGraphState(TestmonGraphStatus.USABLE, "testmon datafile present")


def discard_testmon_graph(root: Path) -> None:
    """Remove the datafile and its SQLite sidecars.

    A sidecar without its database reads as damaged state, so they go together.
    """
    data_path = testmon_datafile(root)
    for path in (data_path, *(data_path.with_name(data_path.name + suffix) for suffix in _SIDECAR_SUFFIXES)):
        with contextlib.suppress(FileNotFoundError, OSError):
            path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report the provisioned testmon datafile, discarding a broken one.")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable result")
    args = parser.parse_args(argv)

    root = Path(os.getcwd()).resolve()
    state = inspect_testmon_graph(root)
    # A broken copy is worse than none: an absent datafile reseeds on the next
    # run, a broken one stops the tier.
    discarded = state.status is TestmonGraphStatus.UNUSABLE
    if discarded:
        discard_testmon_graph(root)
    payload = {
        "database": str(TESTMON_DATA_RELPATH),
        "environment": TESTMON_ENVIRONMENT,
        "status": str(state.status),
        "reason": state.reason,
        "discarded": discarded,
    }
    if args.json:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"testmon provision: {state.status}: {state.reason}")
        if discarded:
            print(f"discarded the unusable datafile at {TESTMON_DATA_RELPATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

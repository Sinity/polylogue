"""The checkout-local pytest-testmon graph: where it lives and whether it works.

One datafile per checkout, at ``.cache/testmon/testmondata``, under one fixed
environment name. Every managed pytest run traces into it and writes back, so
the graph is advanced rather than recomputed. A worktree is provisioned by
copying master's datafile: paths are repo-relative and fingerprints are by
content, so a copy is valid immediately.

An absent datafile is not a failure — the next run seeds it. A datafile that
cannot be opened, or that the installed testmon would not read, is: testmon
deletes a datafile whose data version differs from its own and starts over,
which is a silent full re-execution masquerading as selection.

testmon keys its graph on the installed package set: when that changes it
drops the environment row and every recorded test with it, and the run
re-executes everything. That is legitimate, and it is reported here so the
receipt says why a selected run ran the whole corpus.
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

from testmon.common import drop_patch_version, get_system_packages
from testmon.db import DATA_VERSION as TESTMON_DATA_VERSION

TESTMON_DATA_RELPATH = Path(".cache/testmon/testmondata")

#: The single environment every managed run traces and selects under. Pinned so
#: a Hypothesis profile or a settings module cannot rename the environment and
#: orphan the graph.
TESTMON_ENVIRONMENT = "polylogue"

#: testmon attributes covered lines to tests through coverage's dynamic
#: contexts, which the sys.monitoring core does not support: traced under it,
#: every test depends on nothing but its own file and selection skips every
#: test on every source change. Managed runs pin the C tracer.
TESTMON_COVERAGE_CORE = "ctrace"

#: Tables the installed pytest-testmon writes and reads.
_REQUIRED_TABLES = frozenset({"environment", "test_execution", "file_fp", "test_execution_file_fp"})

#: Files under this prefix are tests, not the code under test.
_TEST_PREFIX = "tests/"

_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


class TestmonGraphStatus(StrEnum):
    ABSENT = "absent"
    USABLE = "usable"
    UNUSABLE = "unusable"


@dataclass(frozen=True, slots=True)
class TestmonGraphState:
    status: TestmonGraphStatus
    reason: str
    #: Set when the graph is usable but the installed packages or interpreter
    #: differ from what it was written under: testmon will re-execute every
    #: test this run and record the new environment.
    full_rerun_cause: str | None = None

    @property
    def usable(self) -> bool:
        return self.status is TestmonGraphStatus.USABLE


def testmon_datafile(root: Path) -> Path:
    return root / TESTMON_DATA_RELPATH


def current_environment_key() -> tuple[str, str]:
    """(system_packages, python_version) exactly as testmon records them."""
    packages = drop_patch_version(get_system_packages())
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return packages, version


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
            data_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            environment = None
            recorded_tests = 0
            source_dependencies = 0
            if data_version == TESTMON_DATA_VERSION and tables >= _REQUIRED_TABLES:
                environment = connection.execute(
                    "SELECT system_packages, python_version FROM environment WHERE environment_name = ? ORDER BY id DESC",
                    (TESTMON_ENVIRONMENT,),
                ).fetchone()
                recorded_tests = int(connection.execute("SELECT count(*) FROM test_execution").fetchone()[0])
                source_dependencies = int(
                    connection.execute(
                        "SELECT count(*) FROM file_fp WHERE substr(filename, 1, ?) != ?",
                        (len(_TEST_PREFIX), _TEST_PREFIX),
                    ).fetchone()[0]
                )
    except sqlite3.Error as exc:
        return TestmonGraphState(TestmonGraphStatus.UNUSABLE, f"the testmon datafile is corrupt: {exc}")
    if data_version != TESTMON_DATA_VERSION:
        return TestmonGraphState(
            TestmonGraphStatus.UNUSABLE,
            f"the testmon datafile carries data version {data_version}; the installed pytest-testmon "
            f"reads version {TESTMON_DATA_VERSION} and would silently replace it",
        )
    missing = _REQUIRED_TABLES - tables
    if missing:
        return TestmonGraphState(
            TestmonGraphStatus.UNUSABLE,
            f"the testmon datafile was written by an incompatible testmon version (no {', '.join(sorted(missing))})",
        )
    if recorded_tests and not source_dependencies:
        return TestmonGraphState(
            TestmonGraphStatus.UNUSABLE,
            f"the testmon datafile records {recorded_tests} tests and no dependency on any source file: "
            "it was traced without dynamic contexts and cannot select",
        )
    cause = None
    if environment is not None:
        packages, version = current_environment_key()
        if environment[1] != version:
            cause = f"the interpreter changed ({environment[1]} -> {version})"
        elif environment[0] != packages:
            cause = "the installed packages changed"
    return TestmonGraphState(TestmonGraphStatus.USABLE, "testmon datafile present", cause)


def discard_testmon_graph(root: Path) -> None:
    """Remove the datafile and its SQLite sidecars.

    A sidecar without its database reads as damaged state, so they go together.
    """
    data_path = testmon_datafile(root)
    for path in (data_path, *(data_path.with_name(data_path.name + suffix) for suffix in _SIDECAR_SUFFIXES)):
        with contextlib.suppress(FileNotFoundError, OSError):
            path.unlink()


def snapshot_testmon_graph(source: Path, destination: Path) -> bool:
    """Copy a datafile another run may be writing, through SQLite's backup API.

    A byte copy of a live database is torn: it can capture a partial
    transaction, and it leaves the source's ``-wal`` behind, so the committed
    tail the copy depends on is missing. The backup API reads under SQLite's
    own locking and writes one self-contained file with no sidecars.

    An absent, unreadable, or non-SQLite source is not a failure: the next run
    seeds the graph from scratch. Returns whether a snapshot was written.
    """
    if not source.is_file() or source.stat().st_size == 0:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        source_connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=30)
    except sqlite3.Error:
        return False
    try:
        with (
            contextlib.closing(source_connection),
            contextlib.closing(sqlite3.connect(destination)) as destination_connection,
        ):
            source_connection.backup(destination_connection)
    except (sqlite3.Error, OSError):
        # A partial destination is the failure mode this function exists to
        # prevent; leave nothing behind for the provision check to accept.
        with contextlib.suppress(FileNotFoundError, OSError):
            destination.unlink()
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report the provisioned testmon datafile, discarding a broken one.")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable result")
    parser.add_argument(
        "--seed",
        metavar="DATAFILE",
        help="snapshot this datafile into the checkout before inspecting it",
    )
    args = parser.parse_args(argv)

    root = Path(os.getcwd()).resolve()
    seeded = False
    if args.seed:
        seeded = snapshot_testmon_graph(Path(args.seed), testmon_datafile(root))
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
        "full_rerun_cause": state.full_rerun_cause,
        "discarded": discarded,
        "seeded": seeded,
    }
    if args.json:
        json.dump(payload, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"testmon provision: {state.status}: {state.reason}")
        if state.full_rerun_cause:
            print(f"the next run re-executes every test: {state.full_rerun_cause}")
        if discarded:
            print(f"discarded the unusable datafile at {TESTMON_DATA_RELPATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The checkout-local pytest-testmon graph: where it lives and whether it works.

One datafile per checkout, at ``.cache/testmon/testmondata``, under environments
bound to declared pytest configuration, conftest policies and Hypothesis profile.
Managed affected and complete runs trace into it, advancing the graph. A worktree is provisioned by
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
import fnmatch
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from testmon.common import drop_patch_version, get_system_packages
from testmon.db import DATA_VERSION as TESTMON_DATA_VERSION

TESTMON_DATA_RELPATH = Path(".cache/testmon/testmondata")
PRIMARY_WORKTREE_ENV = "POLYLOGUE_PRIMARY_WORKTREE"
DEFAULT_PRIMARY_WORKTREE = Path("/realm/project/polylogue")

#: Namespace for compatible managed execution environments.
TESTMON_ENVIRONMENT = "polylogue"


def testmon_environment(root: Path, profile: str | None = None) -> str:
    """Bind graph edges to fixture policy, pytest configuration and execution budget.

    These inputs can introduce dependencies without executing any previously
    covered line. Native source fingerprints alone cannot prove selection.
    """
    profile = profile or os.environ.get("HYPOTHESIS_PROFILE", "default").strip() or "default"
    paths = {"pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg", "conftest.py"}
    paths.update(path.relative_to(root).as_posix() for path in (root / "tests").rglob("conftest.py"))
    digest = hashlib.sha256()
    digest.update(len(profile.encode()).to_bytes(8, "big"))
    digest.update(profile.encode())
    for name in sorted(paths):
        digest.update(len(name.encode()).to_bytes(8, "big"))
        digest.update(name.encode())
        try:
            file_digest = hashlib.sha256()
            with (root / name).open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    file_digest.update(chunk)
            digest.update(b"present" + file_digest.digest())
        except FileNotFoundError:
            digest.update(b"missing")
    return TESTMON_ENVIRONMENT + ":" + digest.hexdigest()


#: testmon attributes covered lines to tests through coverage's dynamic
#: contexts, which the sys.monitoring core does not support: traced under it,
#: every test depends on nothing but its own file and selection skips every
#: test on every source change. Managed runs pin the C tracer.
TESTMON_COVERAGE_CORE = "ctrace"

#: Tables the installed pytest-testmon writes and reads.
_REQUIRED_TABLES = frozenset(
    {"metadata", "environment", "test_execution", "file_fp", "test_execution_file_fp", "suite_execution_file_fsha"}
)

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
    #: Number of test executions currently represented by the graph.  These
    #: counts are evidence about the graph that was inspected, not a claim
    #: about what a later pytest invocation will select.
    recorded_tests: int | None = None
    #: Number of non-test files with recorded dependencies.  A graph with
    #: tests but no source dependencies is unusable for affected selection.
    source_dependencies: int | None = None

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


def inspect_testmon_graph(root: Path, *, datafile: Path | None = None, profile: str | None = None) -> TestmonGraphState:
    """Report whether the local datafile can back a selecting run."""
    data_path = testmon_datafile(root) if datafile is None else datafile
    if Path(str(data_path) + ".authority-unavailable").exists():
        return TestmonGraphState(
            TestmonGraphStatus.UNUSABLE, "execution source authority unavailable; explicit collection is required"
        )
    try:
        if not data_path.is_file() or data_path.stat().st_size == 0:
            return TestmonGraphState(TestmonGraphStatus.ABSENT, "no testmon datafile")
    except OSError as exc:
        return TestmonGraphState(TestmonGraphStatus.UNUSABLE, f"the testmon datafile cannot be inspected: {exc}")
    try:
        connection = sqlite3.connect(data_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    except sqlite3.Error as exc:
        return TestmonGraphState(TestmonGraphStatus.UNUSABLE, f"the testmon datafile cannot be opened: {exc}")
    try:
        with contextlib.closing(connection):
            data_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            environment = None
            recorded_tests: int | None = None
            source_dependencies: int | None = None
            if data_version == TESTMON_DATA_VERSION and tables >= _REQUIRED_TABLES:
                required_columns = {
                    "metadata": {"dataid", "data"},
                    "environment": {"id", "environment_name", "system_packages", "python_version"},
                    "test_execution": {"id", "environment_id", "test_name", "duration", "failed", "forced"},
                    "file_fp": {"id", "filename", "method_checksums", "mtime", "fsha"},
                    "test_execution_file_fp": {"test_execution_id", "fingerprint_id"},
                    "suite_execution_file_fsha": {"suite_execution_id", "filename", "fsha"},
                }
                for table, expected in required_columns.items():
                    actual = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
                    if not expected <= actual:
                        raise sqlite3.DatabaseError(f"incompatible {table} columns")
                recorded_tests = 0
                source_dependencies = 0
                environment = connection.execute(
                    "SELECT id, system_packages, python_version FROM environment WHERE environment_name = ? ORDER BY id DESC",
                    (testmon_environment(root, profile),),
                ).fetchone()
                if environment is None:
                    return TestmonGraphState(
                        TestmonGraphStatus.USABLE,
                        "testmon datafile present",
                        "pytest configuration, fixtures or Hypothesis profile changed; explicit collection is required",
                        recorded_tests=recorded_tests,
                        source_dependencies=source_dependencies,
                    )
                recorded_tests = int(
                    connection.execute(
                        "SELECT count(*) FROM test_execution WHERE environment_id = ?", (environment[0],)
                    ).fetchone()[0]
                )
                source_dependencies = int(
                    connection.execute(
                        "SELECT count(DISTINCT file_fp.id) FROM file_fp "
                        "JOIN test_execution_file_fp ON fingerprint_id = file_fp.id "
                        "JOIN test_execution ON test_execution.id = test_execution_id "
                        "WHERE environment_id = ? AND substr(filename, 1, ?) != ?",
                        (environment[0], len(_TEST_PREFIX), _TEST_PREFIX),
                    ).fetchone()[0]
                )
    except OSError as exc:
        return TestmonGraphState(TestmonGraphStatus.UNUSABLE, f"execution policy could not be read: {exc}")
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
            recorded_tests=recorded_tests,
            source_dependencies=source_dependencies,
        )
    cause = None
    if environment is not None:
        packages, version = current_environment_key()
        if environment[2] != version:
            cause = f"the interpreter changed ({environment[2]} -> {version})"
        elif environment[1] != packages:
            cause = "the installed packages changed"
    return TestmonGraphState(
        TestmonGraphStatus.USABLE,
        "testmon datafile present",
        cause,
        recorded_tests=recorded_tests,
        source_dependencies=source_dependencies,
    )


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
    if Path(str(source) + ".authority-unavailable").exists():
        return False
    if not source.is_file() or source.stat().st_size == 0:
        return False
    if source.absolute() == destination.absolute():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        source_connection = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    except sqlite3.Error:
        with contextlib.suppress(FileNotFoundError, OSError):
            temporary.unlink()
        return False
    try:
        with (
            contextlib.closing(source_connection),
            contextlib.closing(sqlite3.connect(temporary)) as destination_connection,
        ):
            source_connection.backup(destination_connection)
    except (sqlite3.Error, OSError):
        # A partial destination is the failure mode this function exists to
        # prevent; leave nothing behind for the provision check to accept.
        with contextlib.suppress(FileNotFoundError, OSError):
            temporary.unlink()
        return False
    if Path(str(source) + ".authority-unavailable").exists():
        temporary.unlink(missing_ok=True)
        return False
    os.replace(temporary, destination)
    for suffix in _SIDECAR_SUFFIXES:
        with contextlib.suppress(FileNotFoundError, OSError):
            destination.with_name(destination.name + suffix).unlink()
    return True


#: The declared corpus's test-file shapes and the root it is collected from,
#: mirroring ``devtools.pytest_invocation.CLOSED_WORLD_COLLECTION_ARGS``. The
#: file set has to be derived the same way the collection derives it, or the
#: comparison below reports files pytest never looks at.
_TEST_FILE_PATTERNS: Final = ("test_*.py", "*_test.py", "fuzz_*.py")
_TEST_ROOT: Final = "tests"
_COLLECTION_IGNORED: Final = ("tests/benchmarks/",)


def declared_test_files(root: Path) -> frozenset[str]:
    """Every repo-relative test file the declared closed-world collection reaches."""
    base = root / _TEST_ROOT
    if not base.is_dir():
        return frozenset()
    found = set()
    for path in base.rglob("*.py"):
        if not any(fnmatch.fnmatch(path.name, pattern) for pattern in _TEST_FILE_PATTERNS):
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith(_COLLECTION_IGNORED):
            continue
        found.add(relative)
    return frozenset(found)


def recorded_test_names(datafile: Path, *, environment: str) -> frozenset[str] | None:
    """The test node IDs a graph has an execution for.

    ``None`` when the datafile cannot be read, which is not an empty graph.
    """
    if not datafile.is_file():
        return None
    try:
        connection = sqlite3.connect(datafile.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
        with contextlib.closing(connection):
            return frozenset(
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT test_name FROM test_execution JOIN environment ON environment.id = test_execution.environment_id "
                    "WHERE environment_name = ?",
                    (environment,),
                )
            )
    except sqlite3.Error:
        return None


def primary_worktree() -> Path:
    return Path(os.environ.get(PRIMARY_WORKTREE_ENV, DEFAULT_PRIMARY_WORKTREE)).expanduser()


def should_seed(root: Path, seed: Path, *, profile: str | None = None) -> bool:
    """Whether the seed would back a better run than the checkout's own graph.

    A checkout that keeps its datafile between runs accumulates fingerprints
    the seed does not have, so a usable local graph is never replaced with a
    seed that would select the same or re-execute more. The seed wins only
    when the local graph is absent or unusable, or when the local graph would
    force a full re-execution and the seed would not.

    Age is not the comparison. The primary graph is rewritten by every run
    made there, so it is almost always the newer file; copying it over a
    checkout that runs repeatedly discards that checkout's own fingerprints
    and imports the primary's package set, which is how a checkout whose graph
    matched its own environment ends up re-executing the corpus. A graph the
    running checkout wrote can only over-select, never under-select: an
    unrecorded test is unknown and runs.

    That over-selection is also why an environment-current local graph still
    loses to a seed whose recorded test node IDs strictly contain its own: an
    interrupted corpus run leaves a usable but partial graph, and keeping it
    re-executes every test it never reached on each later selecting run. The
    comparison is over node IDs, not files: a seed that touches every file
    but misses tests the local graph recorded would turn those tests unknown.
    """
    environment = testmon_environment(root, profile)
    local = inspect_testmon_graph(root, profile=profile)
    if not local.usable:
        return True
    local_tests: frozenset[str] | None = None
    if local.full_rerun_cause is None:
        local_tests = recorded_test_names(testmon_datafile(root), environment=environment)
        if local_tests is None:
            return False
        # A read of the live seed rules out the common case without copying
        # it; the snapshot below is still what decides.
        live_seed_tests = recorded_test_names(seed, environment=environment)
        if live_seed_tests is None or not live_seed_tests > local_tests:
            return False
    with tempfile.TemporaryDirectory(prefix="testmon-seed-") as scratch:
        probe_root = Path(scratch)
        probe = testmon_datafile(probe_root)
        # Compare against the snapshot, not the live seed: the snapshot is
        # what would be installed, and the seed can change between reads.
        if not snapshot_testmon_graph(seed, probe):
            return False
        candidate = inspect_testmon_graph(root, datafile=probe, profile=profile)
        if not candidate.usable or candidate.full_rerun_cause is not None:
            return False
        if local_tests is None:
            return True
        seed_tests = recorded_test_names(probe, environment=environment)
    return seed_tests is not None and seed_tests > local_tests


def sync_testmon_graph(root: Path, *, source: Path | None = None, profile: str | None = None) -> bool:
    """Refresh a checkout from the primary graph when the primary would select better."""
    source = source or testmon_datafile(primary_worktree())
    destination = testmon_datafile(root)
    if source.absolute() == destination.absolute() or not source.is_file():
        return False
    if not should_seed(root, source, profile=profile):
        return False
    return snapshot_testmon_graph(source, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report the provisioned testmon datafile, discarding a broken one.")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable result")
    parser.add_argument(
        "--seed",
        metavar="DATAFILE",
        help="snapshot this datafile into the checkout when the checkout's own graph is absent, broken, or would re-execute everything while the seed would select",
    )
    args = parser.parse_args(argv)

    root = Path(os.getcwd()).resolve()
    seeded = False
    if args.seed and should_seed(root, Path(args.seed)):
        seeded = snapshot_testmon_graph(Path(args.seed), testmon_datafile(root))
    state = inspect_testmon_graph(root)
    # A broken copy is worse than none: an absent datafile reseeds on the next
    # run, a broken one stops the tier.
    discarded = (
        state.status is TestmonGraphStatus.UNUSABLE
        and not Path(str(testmon_datafile(root)) + ".authority-unavailable").exists()
    )
    if discarded:
        discard_testmon_graph(root)
    payload = {
        "database": str(TESTMON_DATA_RELPATH),
        "environment": testmon_environment(root),
        "status": str(state.status),
        "reason": state.reason,
        "full_rerun_cause": state.full_rerun_cause,
        "recorded_tests": state.recorded_tests,
        "source_dependencies": state.source_dependencies,
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

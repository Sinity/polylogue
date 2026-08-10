"""Typed safety contract for reusable pytest-testmon state.

The testmon database answers two different questions which must not share a
boolean marker:

* did collection and dependency capture cover every promised node?
* did that run establish a green release baseline?

A failed test can still have a complete dependency graph.  Such a graph is
usable for affected-test selection, but it is never evidence that the suite
is releasable.  This module is the single parser and SQLite validator used by
verification, worktree bootstrap, and the checkout guard.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any


class CollectionStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class GraphStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


class BaselineStatus(StrEnum):
    GREEN = "green"
    RED = "red"


class BindingMode(StrEnum):
    EXACT = "exact"
    RELATIVE_FILE_FINGERPRINTS = "relative-file-fingerprints"


class VerificationScope(StrEnum):
    AFFECTED = "affected"
    RELEASE_BASELINE = "release-baseline"
    NARROW_TERMINAL = "narrow-terminal"
    NON_TEST = "non-test"


class TerminalAuthorization(StrEnum):
    NARROW_TERMINAL = "narrow-terminal"


@dataclass(frozen=True, slots=True)
class TestmonIdentity:
    git_head: str | None
    worktree_fingerprint: str
    python: str
    skip_slow: bool
    lab: bool
    git_tree: str | None = None
    terminal_authorization: str | None = None
    dependency_environment: str = ""
    pytest_harness: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TestmonIdentity:
        git_head = value.get("git_head")
        if git_head is not None and (not isinstance(git_head, str) or not git_head):
            raise ValueError("identity.git_head must be a non-empty string or null")
        git_tree = value.get("git_tree")
        if git_tree is not None and (not isinstance(git_tree, str) or not git_tree):
            raise ValueError("identity.git_tree must be a non-empty string or null")
        worktree = value.get("worktree_fingerprint")
        python = value.get("python")
        if not isinstance(worktree, str) or not worktree:
            raise ValueError("identity.worktree_fingerprint must be a non-empty string")
        if not isinstance(python, str) or not python:
            raise ValueError("identity.python must be a non-empty string")
        dependency_environment = value.get("dependency_environment")
        pytest_harness = value.get("pytest_harness")
        if dependency_environment is None:
            dependency_environment = ""
        if pytest_harness is None:
            pytest_harness = ""
        if not isinstance(dependency_environment, str):
            raise ValueError("identity.dependency_environment must be a string")
        if not isinstance(pytest_harness, str):
            raise ValueError("identity.pytest_harness must be a string")
        if not isinstance(value.get("skip_slow"), bool) or not isinstance(value.get("lab"), bool):
            raise ValueError("identity selection flags must be booleans")
        terminal_authorization = value.get("terminal_authorization")
        if terminal_authorization is not None and terminal_authorization not in {
            authorization.value for authorization in TerminalAuthorization
        }:
            raise ValueError("identity.terminal_authorization is invalid")
        return cls(
            git_head,
            worktree,
            python,
            value["skip_slow"],
            value["lab"],
            git_tree,
            terminal_authorization,
            dependency_environment,
            pytest_harness,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "git_head": self.git_head,
            "worktree_fingerprint": self.worktree_fingerprint,
            "python": self.python,
            "skip_slow": self.skip_slow,
            "lab": self.lab,
            "git_tree": self.git_tree,
            "terminal_authorization": self.terminal_authorization,
            "dependency_environment": self.dependency_environment,
            "pytest_harness": self.pytest_harness,
        }


def _fingerprint_files(checkout_root: Path, relative_paths: Sequence[str]) -> str:
    """Hash named checkout inputs, preserving absent inputs as typed state."""
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        digest.update(relative_path.encode())
        digest.update(b"\0")
        try:
            contents = (checkout_root / relative_path).read_bytes()
        except OSError:
            digest.update(b"missing")
        else:
            digest.update(contents)
        digest.update(b"\0")
    return digest.hexdigest()


def _installed_distributions() -> tuple[tuple[str, str], ...] | None:
    """Return the active environment's normalized installed distributions."""
    try:
        distributions = []
        for distribution in importlib.metadata.distributions():
            name = distribution.metadata.get("Name")
            version = distribution.version
            if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
                return None
            distributions.append((name.casefold(), version))
    except (OSError, TypeError, ValueError, importlib.metadata.PackageNotFoundError):
        return None
    return tuple(sorted(distributions))


def testmon_runtime_identity(checkout_root: Path) -> tuple[str, str] | None:
    """Identify the lock, installed dependencies, and pytest execution harness.

    A testmon graph is reusable only under this exact dependency environment.
    The application lock catches declared changes; installed distributions and
    pytest-specific configuration catch a stale or differently provisioned
    virtual environment even when ``sys.version`` is unchanged.
    """
    distributions = _installed_distributions()
    if distributions is None:
        return None
    normalized_root = checkout_root.resolve()
    dependency_payload = {
        "lock_inputs": _fingerprint_files(normalized_root, ("uv.lock", "pyproject.toml")),
        "distributions": distributions,
    }
    harness_payload = {
        "configuration": _fingerprint_files(
            normalized_root,
            ("pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg", "tests/conftest.py"),
        ),
        "environment": {
            key: os.environ.get(key) for key in ("PYTEST_ADDOPTS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTEST_PLUGINS")
        },
        "pytest_distributions": tuple(
            item for item in distributions if item[0] in {"pytest", "pytest-testmon", "pytest-xdist", "pluggy"}
        ),
    }
    return (
        hashlib.sha256(json.dumps(dependency_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        hashlib.sha256(json.dumps(harness_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    )


def _identity_matches_runtime(identity: TestmonIdentity, *, checkout_root: Path, protocol_version: int) -> bool:
    """Keep pre-binding protocol receipts parseable but never reusable today."""
    if protocol_version < 5:
        return True
    runtime_identity = testmon_runtime_identity(checkout_root)
    return (
        runtime_identity is not None
        and (
            identity.dependency_environment,
            identity.pytest_harness,
        )
        == runtime_identity
    )


@dataclass(frozen=True, slots=True)
class TestmonBinding:
    mode: BindingMode
    checkout_root: str
    source_checkout_root: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TestmonBinding:
        raw_mode = value.get("mode")
        if not isinstance(raw_mode, str):
            raise ValueError("binding.mode is invalid")
        try:
            mode = BindingMode(raw_mode)
        except ValueError as exc:
            raise ValueError("binding.mode is invalid") from exc
        checkout_root = value.get("checkout_root")
        source = value.get("source_checkout_root")
        if not isinstance(checkout_root, str) or not checkout_root:
            raise ValueError("binding.checkout_root must be a non-empty string")
        if not Path(checkout_root).is_absolute():
            raise ValueError("binding.checkout_root must be absolute")
        if source is not None and (not isinstance(source, str) or not source):
            raise ValueError("binding.source_checkout_root must be a non-empty string or null")
        if source is not None and not Path(source).is_absolute():
            raise ValueError("binding.source_checkout_root must be absolute")
        if mode is BindingMode.EXACT and source is not None:
            raise ValueError("exact bindings cannot have a source checkout")
        if mode is BindingMode.RELATIVE_FILE_FINGERPRINTS:
            if source is None:
                raise ValueError("rebound bindings require a source checkout")
            if Path(source).resolve() == Path(checkout_root).resolve():
                raise ValueError("rebound binding source and destination must differ")
        return cls(mode, checkout_root, source)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "checkout_root": self.checkout_root,
            "source_checkout_root": self.source_checkout_root,
        }


@dataclass(frozen=True, slots=True)
class GraphInspection:
    status: GraphStatus
    recorded_count: int
    dependency_edge_count: int
    missing_nodeids: tuple[str, ...]
    orphan_execution_edges: int
    orphan_fingerprint_edges: int
    error: str | None
    failed_nodeids: tuple[str, ...]

    @property
    def usable_for_selection(self) -> bool:
        return self.status is GraphStatus.COMPLETE

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "recorded_count": self.recorded_count,
            "dependency_edge_count": self.dependency_edge_count,
            "missing_nodeids": list(self.missing_nodeids),
            "orphan_execution_edges": self.orphan_execution_edges,
            "orphan_fingerprint_edges": self.orphan_fingerprint_edges,
            "error": self.error,
            "failed_nodeids": list(self.failed_nodeids),
        }


@dataclass(frozen=True, slots=True)
class TestmonSeedStamp:
    protocol_version: int
    collection_status: CollectionStatus
    expected_nodeids: tuple[str, ...]
    selected_nodeids_omitted: int
    baseline_status: BaselineStatus
    release_baseline_allowed: bool
    baseline_exit_code: int
    graph: GraphInspection
    identity: TestmonIdentity
    binding: TestmonBinding
    testmon_data: str
    run_id: str
    artifact_dir: str

    @property
    def affected_selection_allowed(self) -> bool:
        return (
            self.collection_status is CollectionStatus.COMPLETE
            and self.selected_nodeids_omitted == 0
            and self.graph.usable_for_selection
        )

    @property
    def expected_digest(self) -> str:
        return hashlib.sha256("\n".join(sorted(self.expected_nodeids)).encode()).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "status": "usable",
            "collection": {
                "status": self.collection_status.value,
                "expected_count": len(self.expected_nodeids),
                "expected_digest": self.expected_digest,
                "selected_nodeids": list(self.expected_nodeids),
                "selected_nodeids_omitted": self.selected_nodeids_omitted,
            },
            "baseline": {
                "status": self.baseline_status.value,
                "exit_code": self.baseline_exit_code,
                "release_baseline_allowed": self.release_baseline_allowed,
            },
            "graph": self.graph.as_dict(),
            "identity": self.identity.as_dict(),
            "binding": self.binding.as_dict(),
            "testmon_data": self.testmon_data,
            "run_id": self.run_id,
            "artifact_dir": self.artifact_dir,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, protocol_version: int) -> TestmonSeedStamp:
        if value.get("protocol_version") != protocol_version or value.get("status") != "usable":
            raise ValueError("seed stamp is not a current usable testmon stamp")
        collection = value.get("collection")
        baseline = value.get("baseline")
        graph = value.get("graph")
        identity = value.get("identity")
        binding = value.get("binding")
        if not all(isinstance(item, Mapping) for item in (collection, baseline, graph, identity, binding)):
            raise ValueError("seed stamp has incomplete typed state")
        assert isinstance(collection, Mapping)
        assert isinstance(baseline, Mapping)
        assert isinstance(graph, Mapping)
        assert isinstance(identity, Mapping)
        assert isinstance(binding, Mapping)
        if collection.get("status") != CollectionStatus.COMPLETE.value:
            raise ValueError("seed stamp collection is not complete")
        nodeids = collection.get("selected_nodeids")
        if (
            not isinstance(nodeids, list)
            or not nodeids
            or any(not isinstance(item, str) or not item for item in nodeids)
        ):
            raise ValueError("seed stamp selected nodeids are missing or malformed")
        if len(set(nodeids)) != len(nodeids):
            raise ValueError("seed stamp selected nodeids are not unique")
        omitted = collection.get("selected_nodeids_omitted")
        if not isinstance(omitted, int) or isinstance(omitted, bool) or omitted != 0:
            raise ValueError("seed stamp has controlled collection omissions")
        if collection.get("expected_count") != len(nodeids):
            raise ValueError("seed stamp expected count does not match selected nodeids")
        expected_digest = hashlib.sha256("\n".join(sorted(nodeids)).encode()).hexdigest()
        if collection.get("expected_digest") != expected_digest:
            raise ValueError("seed stamp expected nodeid digest is stale")
        raw_baseline_status = baseline.get("status")
        if not isinstance(raw_baseline_status, str):
            raise ValueError("seed stamp baseline status is invalid")
        try:
            baseline_status = BaselineStatus(raw_baseline_status)
        except ValueError as exc:
            raise ValueError("seed stamp baseline status is invalid") from exc
        exit_code = baseline.get("exit_code")
        release_allowed = baseline.get("release_baseline_allowed")
        if not isinstance(exit_code, int) or isinstance(exit_code, bool) or not isinstance(release_allowed, bool):
            raise ValueError("seed stamp baseline fields are malformed")
        if release_allowed != (baseline_status is BaselineStatus.GREEN):
            raise ValueError("release permission does not match baseline status")
        if baseline_status is BaselineStatus.GREEN and exit_code != 0:
            raise ValueError("green seed stamp must have a zero exit code")
        graph_status = graph.get("status")
        if not isinstance(graph_status, str):
            raise ValueError("seed stamp graph status is invalid")
        try:
            status = GraphStatus(graph_status)
        except ValueError as exc:
            raise ValueError("seed stamp graph status is invalid") from exc
        if status is not GraphStatus.COMPLETE:
            raise ValueError("seed stamp graph is not complete")
        graph_expected = [
            "recorded_count",
            "dependency_edge_count",
            "orphan_execution_edges",
            "orphan_fingerprint_edges",
        ]
        graph_counts = {key: graph.get(key) for key in graph_expected}
        if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in graph_counts.values()):
            raise ValueError("seed stamp graph counts are malformed")
        dependency_edge_count = graph.get("dependency_edge_count")
        if not isinstance(dependency_edge_count, int) or isinstance(dependency_edge_count, bool):
            raise ValueError("seed stamp dependency edge count is malformed")
        if (
            graph.get("recorded_count") != len(nodeids)
            or dependency_edge_count < len(nodeids)
            or graph.get("orphan_execution_edges") != 0
            or graph.get("orphan_fingerprint_edges") != 0
        ):
            raise ValueError("seed stamp graph coverage is incomplete")
        missing_nodeids = graph.get("missing_nodeids")
        if (
            not isinstance(missing_nodeids, list)
            or any(not isinstance(item, str) or not item for item in missing_nodeids)
            or not set(missing_nodeids).issubset(nodeids)
        ):
            raise ValueError("seed stamp missing-node ledger is malformed")
        if graph.get("error") is not None or missing_nodeids:
            raise ValueError("seed stamp graph has missing or erroneous nodes")
        graph_nodeids = graph.get("failed_nodeids", [])
        if (
            not isinstance(graph_nodeids, list)
            or any(not isinstance(item, str) or not item for item in graph_nodeids)
            or not set(graph_nodeids).issubset(nodeids)
            or len(set(graph_nodeids)) != len(graph_nodeids)
        ):
            raise ValueError("seed stamp graph failure ledger is malformed")
        if baseline_status is BaselineStatus.GREEN and graph_nodeids:
            raise ValueError("green seed stamp cannot contain failed graph nodes")
        testmon_data = value.get("testmon_data")
        run_id = value.get("run_id")
        artifact_dir = value.get("artifact_dir")
        if not all(isinstance(item, str) and item for item in (testmon_data, run_id, artifact_dir)):
            raise ValueError("seed stamp provenance is incomplete")
        assert isinstance(testmon_data, str)
        assert isinstance(run_id, str)
        assert isinstance(artifact_dir, str)
        typed_binding = TestmonBinding.from_mapping(binding)
        if not _is_bound_run_artifact(
            artifact_dir,
            checkout_root=Path(typed_binding.checkout_root),
            run_id=run_id,
        ):
            raise ValueError("seed stamp artifact directory is not checkout-bound")
        return cls(
            protocol_version,
            CollectionStatus.COMPLETE,
            tuple(nodeids),
            0,
            baseline_status,
            release_allowed,
            exit_code,
            GraphInspection(
                status,
                graph["recorded_count"],
                dependency_edge_count,
                tuple(graph.get("missing_nodeids", [])),
                graph["orphan_execution_edges"],
                graph["orphan_fingerprint_edges"],
                graph.get("error"),
                tuple(graph_nodeids),
            ),
            TestmonIdentity.from_mapping(identity),
            typed_binding,
            testmon_data,
            run_id,
            artifact_dir,
        )

    def rebound(self, *, checkout_root: Path, inherited_from: Path) -> TestmonSeedStamp:
        return replace(
            self,
            binding=TestmonBinding(
                BindingMode.RELATIVE_FILE_FINGERPRINTS,
                str(checkout_root.resolve()),
                str(inherited_from.resolve()),
            ),
        )


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_bound_run_artifact(raw: object, *, checkout_root: Path, run_id: str) -> bool:
    if not isinstance(raw, str) or not raw or not run_id:
        return False
    path = Path(raw)
    if path.is_absolute() or path.parts[:3] != (".cache", "verify", "runs"):
        return False
    if path.parts[3:] != (run_id,):
        return False
    try:
        artifact_dir = (checkout_root / path).resolve()
        artifact_dir.relative_to((checkout_root / ".cache" / "verify" / "runs" / run_id).resolve())
        receipt = json.loads((artifact_dir / "run.json").read_text(encoding="utf-8"))
        if not isinstance(receipt, Mapping):
            return False
        return (
            receipt.get("run_id") == run_id
            and isinstance(receipt.get("checkout_root"), str)
            and Path(receipt["checkout_root"]).resolve() == checkout_root.resolve()
            and receipt.get("artifact_dir") == str(Path(".cache") / "verify" / "runs" / run_id)
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return False


def seed_marker_is_checkout_bound(
    marker_path: Path,
    *,
    checkout_root: Path,
    protocol_version: int,
) -> bool:
    """Validate only the typed ownership envelope of a seed marker.

    This intentionally does not open or fingerprint SQLite. The checkout guard
    uses this cheap predicate for every entrypoint; verify preflight performs
    the exhaustive graph validation before authorizing selection.
    """
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            return False
        stamp = TestmonSeedStamp.from_mapping(payload, protocol_version=protocol_version)
        return Path(stamp.binding.checkout_root).resolve() == checkout_root.resolve()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return False


def attempt_is_checkout_bound(
    attempt: Mapping[str, Any],
    *,
    checkout_root: Path,
    protocol_version: int,
    reusable_only: bool = True,
) -> bool:
    """Check a seed-attempt receipt without inspecting its SQLite graph."""
    allowed_statuses = {"reusable", "complete"} if reusable_only else {"running", "incomplete", "reusable", "complete"}
    if attempt.get("protocol_version") != protocol_version or attempt.get("status") not in allowed_statuses:
        return False
    identity = attempt.get("identity")
    expected = attempt.get("expected_nodeids")
    selection = attempt.get("selection")
    if not isinstance(identity, Mapping) or not isinstance(expected, list) or not isinstance(selection, Mapping):
        return False
    if not expected or any(not isinstance(nodeid, str) or not nodeid for nodeid in expected):
        return False
    if len(set(expected)) != len(expected):
        return False
    if (
        not isinstance(attempt.get("expected_count"), int)
        or isinstance(attempt.get("expected_count"), bool)
        or attempt.get("expected_count") != len(expected)
    ):
        return False
    expected_digest = hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest()
    if attempt.get("expected_digest") != expected_digest:
        return False
    try:
        typed_identity = TestmonIdentity.from_mapping(identity)
    except ValueError:
        return False
    if not _identity_matches_runtime(typed_identity, checkout_root=checkout_root, protocol_version=protocol_version):
        return False
    omitted = selection.get("selected_nodeids_omitted")
    selected_count = selection.get("selected_count")
    if (
        not isinstance(omitted, int)
        or isinstance(omitted, bool)
        or omitted != 0
        or not isinstance(selected_count, int)
        or isinstance(selected_count, bool)
        or selected_count != len(expected)
    ):
        return False
    recorded_data = attempt.get("testmon_data")
    run_id = attempt.get("run_id")
    artifact_dir = attempt.get("artifact_dir")
    if (
        not isinstance(recorded_data, str)
        or not recorded_data
        or not isinstance(run_id, str)
        or not run_id
        or not isinstance(artifact_dir, str)
        or not artifact_dir
        or not _is_bound_run_artifact(artifact_dir, checkout_root=checkout_root, run_id=run_id)
    ):
        return False
    raw_binding = attempt.get("binding")
    if raw_binding is None:
        binding = TestmonBinding(BindingMode.EXACT, str(checkout_root.resolve()))
    elif isinstance(raw_binding, Mapping):
        try:
            binding = TestmonBinding.from_mapping(raw_binding)
        except ValueError:
            return False
    else:
        return False
    if Path(binding.checkout_root).resolve() != checkout_root.resolve():
        return False
    raw_permission = attempt.get("release_baseline_allowed")
    if raw_permission is not None and not isinstance(raw_permission, bool):
        return False
    raw_scope = attempt.get("verification_scope")
    if raw_scope is not None and raw_scope not in {scope.value for scope in VerificationScope}:
        return False
    if reusable_only and raw_permission is not False:
        return False
    if reusable_only:
        outcomes = attempt.get("node_outcomes")
        if not isinstance(outcomes, list) or len(outcomes) != len(expected):
            return False
        nodeids = [item.get("nodeid") for item in outcomes if isinstance(item, Mapping)]
        if len(nodeids) != len(outcomes) or set(nodeids) != set(expected) or len(set(nodeids)) != len(nodeids):
            return False
        if any(item.get("outcome") not in {"passed", "failed", "error", "skipped"} for item in outcomes):
            return False
    return True


def inspect_testmon_database(path: Path, expected_nodeids: Sequence[str]) -> GraphInspection:
    """Validate the real testmon schema and every expected dependency edge."""
    expected = tuple(expected_nodeids)
    if not path.is_file() or not expected or len(set(expected)) != len(expected):
        return GraphInspection(
            GraphStatus.INCOMPLETE, 0, 0, expected, 0, 0, "missing or malformed expected nodeids", ()
        )
    try:
        with contextlib.closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                return GraphInspection(GraphStatus.INVALID, 0, 0, expected, 0, 0, "sqlite integrity check failed", ())
            required = {"test_execution", "test_execution_file_fp", "file_fp"}
            tables = {str(row[0]) for row in connection.execute("select name from sqlite_master where type='table'")}
            if not required <= tables:
                return GraphInspection(GraphStatus.INVALID, 0, 0, expected, 0, 0, "testmon schema is incomplete", ())
            required_columns = {
                "test_execution": {"id", "test_name", "failed"},
                "test_execution_file_fp": {"test_execution_id", "fingerprint_id"},
                "file_fp": {"id", "filename", "fsha"},
            }
            for table, columns in required_columns.items():
                actual = {str(row[1]) for row in connection.execute(f"pragma table_info({table})")}
                if not columns <= actual:
                    return GraphInspection(
                        GraphStatus.INVALID,
                        0,
                        0,
                        expected,
                        0,
                        0,
                        f"testmon schema is missing columns from {table}",
                        (),
                    )
            executions = connection.execute(
                "select id, test_name, failed from test_execution where test_name is not null"
            ).fetchall()
            latest: dict[str, tuple[int, bool]] = {}
            execution_ids: set[int] = set()
            for execution_id, test_name, failed in executions:
                if (
                    not isinstance(execution_id, int)
                    or isinstance(execution_id, bool)
                    or execution_id <= 0
                    or not isinstance(test_name, str)
                    or not test_name
                    or not isinstance(failed, int)
                    or isinstance(failed, bool)
                    or failed not in (0, 1)
                ):
                    return GraphInspection(
                        GraphStatus.INVALID, 0, 0, expected, 0, 0, "testmon execution row is malformed", ()
                    )
                if execution_id in execution_ids:
                    return GraphInspection(
                        GraphStatus.INVALID, 0, 0, expected, 0, 0, "testmon execution ids are not unique", ()
                    )
                execution_ids.add(execution_id)
                name = test_name
                prior = latest.get(name)
                if prior is None or execution_id > prior[0]:
                    latest[name] = (execution_id, failed == 1)
            missing = tuple(sorted(set(expected) - latest.keys()))
            expected_ids = {latest[nodeid][0] for nodeid in expected if nodeid in latest}
            edge_rows = connection.execute(
                "select test_execution_id, fingerprint_id from test_execution_file_fp"
            ).fetchall()
            fingerprints = connection.execute("select id, filename, fsha from file_fp").fetchall()
            fingerprint_ids: set[int] = set()
            for fingerprint_id, filename, fsha in fingerprints:
                if (
                    not isinstance(fingerprint_id, int)
                    or isinstance(fingerprint_id, bool)
                    or fingerprint_id <= 0
                    or not isinstance(filename, str)
                    or not filename
                    or Path(filename).is_absolute()
                    or ".." in Path(filename).parts
                    or not isinstance(fsha, str)
                    or not fsha
                ):
                    return GraphInspection(
                        GraphStatus.INVALID, 0, 0, expected, 0, 0, "testmon fingerprint row is malformed", ()
                    )
                if fingerprint_id in fingerprint_ids:
                    return GraphInspection(
                        GraphStatus.INVALID, 0, 0, expected, 0, 0, "testmon fingerprint ids are not unique", ()
                    )
                fingerprint_ids.add(fingerprint_id)
            for execution_id, fingerprint_id in edge_rows:
                if (
                    not isinstance(execution_id, int)
                    or isinstance(execution_id, bool)
                    or execution_id <= 0
                    or not isinstance(fingerprint_id, int)
                    or isinstance(fingerprint_id, bool)
                    or fingerprint_id <= 0
                ):
                    return GraphInspection(
                        GraphStatus.INVALID, 0, 0, expected, 0, 0, "testmon dependency edge is malformed", ()
                    )
            orphan_execution_edges = sum(1 for row in edge_rows if row[0] not in execution_ids)
            orphan_fingerprint_edges = sum(1 for row in edge_rows if row[1] not in fingerprint_ids)
            edge_counts: dict[int, int] = {}
            for execution_id, _fingerprint_id in edge_rows:
                edge_counts[execution_id] = edge_counts.get(execution_id, 0) + 1
            uncovered = tuple(
                sorted(nodeid for nodeid in expected if nodeid in latest and edge_counts.get(latest[nodeid][0], 0) == 0)
            )
            missing = tuple(sorted(set(missing) | set(uncovered)))
            failed = tuple(sorted(nodeid for nodeid in expected if nodeid in latest and latest[nodeid][1]))
            edge_count = sum(edge_counts.get(execution_id, 0) for execution_id in expected_ids)
            status = (
                GraphStatus.COMPLETE
                if not missing and not orphan_execution_edges and not orphan_fingerprint_edges
                else GraphStatus.INCOMPLETE
            )
            return GraphInspection(
                status,
                len(expected) - len(missing),
                edge_count,
                missing,
                orphan_execution_edges,
                orphan_fingerprint_edges,
                None,
                failed,
            )
    except (OSError, sqlite3.Error, UnicodeError, TypeError, ValueError, OverflowError) as exc:
        return GraphInspection(GraphStatus.INVALID, 0, 0, expected, 0, 0, str(exc), ())


def validate_stamp(
    stamp_path: Path,
    data_path: Path,
    *,
    checkout_root: Path,
    protocol_version: int,
) -> TestmonSeedStamp | None:
    """Parse and re-check a stamp against its current SQLite graph."""
    try:
        payload = json.loads(stamp_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            return None
        stamp = TestmonSeedStamp.from_mapping(payload, protocol_version=protocol_version)
        if not stamp.release_baseline_allowed:
            return None
        if (
            stamp.identity.skip_slow
            and stamp.identity.terminal_authorization != TerminalAuthorization.NARROW_TERMINAL.value
        ):
            return None
        if not _identity_matches_runtime(
            stamp.identity, checkout_root=checkout_root, protocol_version=protocol_version
        ):
            return None
        if Path(stamp.binding.checkout_root).resolve() != checkout_root.resolve():
            return None
        if file_fingerprint(data_path) != stamp.testmon_data:
            return None
        graph = inspect_testmon_database(data_path, stamp.expected_nodeids)
        if graph != stamp.graph:
            return None
        return stamp
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def refresh_stamp(stamp: TestmonSeedStamp, data_path: Path) -> TestmonSeedStamp | None:
    """Refresh mutable SQLite provenance after a successful affected run."""
    graph = inspect_testmon_database(data_path, stamp.expected_nodeids)
    if not graph.usable_for_selection:
        return None
    try:
        return replace(stamp, graph=graph, testmon_data=file_fingerprint(data_path))
    except OSError:
        return None


def stamp_from_attempt(
    attempt: Mapping[str, Any],
    data_path: Path,
    *,
    checkout_root: Path,
    protocol_version: int,
    published_marker: bool = True,
) -> TestmonSeedStamp | None:
    """Parse a complete attempt, withholding release authority until publication."""
    if attempt.get("protocol_version") != protocol_version or attempt.get("status") not in {"reusable", "complete"}:
        return None
    selection = attempt.get("selection")
    expected = attempt.get("expected_nodeids")
    identity = attempt.get("identity")
    if not isinstance(selection, Mapping) or not isinstance(expected, list) or not isinstance(identity, Mapping):
        return None
    assert isinstance(selection, Mapping)
    assert isinstance(identity, Mapping)
    omitted = selection.get("selected_nodeids_omitted")
    selected_count = selection.get("selected_count")
    if (
        not isinstance(omitted, int)
        or isinstance(omitted, bool)
        or omitted != 0
        or not isinstance(selected_count, int)
        or isinstance(selected_count, bool)
        or selected_count != len(expected)
        or not expected
        or any(not isinstance(nodeid, str) or not nodeid for nodeid in expected)
        or len(set(expected)) != len(expected)
    ):
        return None
    expected_count = attempt.get("expected_count")
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count != len(expected):
        return None
    expected_digest = attempt.get("expected_digest")
    if (
        not isinstance(expected_digest, str)
        or expected_digest != hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest()
    ):
        return None
    recorded_data = attempt.get("testmon_data")
    if not isinstance(recorded_data, str) or not recorded_data or not data_path.is_file():
        return None
    try:
        if file_fingerprint(data_path) != recorded_data:
            return None
    except OSError:
        return None
    run_id = attempt.get("run_id")
    artifact_dir = attempt.get("artifact_dir")
    if not isinstance(run_id, str) or not run_id or not isinstance(artifact_dir, str) or not artifact_dir:
        return None
    if not _is_bound_run_artifact(artifact_dir, checkout_root=checkout_root, run_id=run_id):
        return None
    outcomes = attempt.get("node_outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != len(expected):
        return None
    if any(not isinstance(item, Mapping) for item in outcomes):
        return None
    outcome_items = [item for item in outcomes if isinstance(item, Mapping)]
    if any(
        not isinstance(item.get("nodeid"), str) or not item.get("nodeid") or item.get("nodeid") not in expected
        for item in outcome_items
    ):
        return None
    outcome_by_node = {item["nodeid"]: item.get("outcome") for item in outcome_items}
    if set(outcome_by_node) != set(expected):
        return None
    if len(outcome_by_node) != len(outcomes) or any(
        not isinstance(nodeid, str) or not nodeid for nodeid in outcome_by_node
    ):
        return None
    if any(outcome not in {"passed", "failed", "error", "skipped"} for outcome in outcome_by_node.values()):
        return None
    exit_code = attempt.get("exit_code")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        return None
    graph = inspect_testmon_database(data_path, [str(nodeid) for nodeid in expected])
    if not graph.usable_for_selection:
        return None
    try:
        typed_identity = TestmonIdentity.from_mapping(identity)
    except ValueError:
        return None
    if not _identity_matches_runtime(typed_identity, checkout_root=checkout_root, protocol_version=protocol_version):
        return None
    baseline = (
        BaselineStatus.GREEN
        if attempt.get("status") == "complete"
        and exit_code == 0
        and all(outcome in {"passed", "skipped"} for outcome in outcome_by_node.values())
        and not graph.failed_nodeids
        else BaselineStatus.RED
    )
    raw_scope = attempt.get("verification_scope")
    if raw_scope is not None and raw_scope not in {scope.value for scope in VerificationScope}:
        return None
    terminal_authorized = (
        typed_identity.skip_slow is True
        and raw_scope == VerificationScope.NARROW_TERMINAL.value
        and typed_identity.terminal_authorization == TerminalAuthorization.NARROW_TERMINAL.value
    )
    if baseline is BaselineStatus.GREEN and typed_identity.skip_slow and not terminal_authorized:
        baseline = BaselineStatus.RED
    if not published_marker:
        baseline = BaselineStatus.RED
    raw_permission = attempt.get("release_baseline_allowed")
    if baseline is BaselineStatus.GREEN and (
        raw_scope != VerificationScope.NARROW_TERMINAL.value
        if typed_identity.skip_slow
        else raw_scope != VerificationScope.RELEASE_BASELINE.value
    ):
        baseline = BaselineStatus.RED
    if baseline is BaselineStatus.GREEN and raw_permission is not True:
        baseline = BaselineStatus.RED
    if raw_permission is not None and not isinstance(raw_permission, bool):
        return None
    if published_marker and raw_permission is not None and raw_permission != (baseline is BaselineStatus.GREEN):
        return None
    raw_binding = attempt.get("binding")
    if raw_binding is None:
        typed_binding = TestmonBinding(BindingMode.EXACT, str(checkout_root.resolve()))
    elif isinstance(raw_binding, Mapping):
        try:
            typed_binding = TestmonBinding.from_mapping(raw_binding)
        except ValueError:
            return None
        if Path(typed_binding.checkout_root).resolve() != checkout_root.resolve():
            return None
    else:
        return None
    return TestmonSeedStamp(
        protocol_version,
        CollectionStatus.COMPLETE,
        tuple(str(nodeid) for nodeid in expected),
        0,
        baseline,
        baseline is BaselineStatus.GREEN,
        exit_code,
        graph,
        typed_identity,
        typed_binding,
        recorded_data,
        run_id,
        artifact_dir,
    )


__all__ = [
    "BaselineStatus",
    "BindingMode",
    "CollectionStatus",
    "GraphInspection",
    "GraphStatus",
    "TestmonBinding",
    "TestmonIdentity",
    "TestmonSeedStamp",
    "TerminalAuthorization",
    "VerificationScope",
    "attempt_is_checkout_bound",
    "file_fingerprint",
    "inspect_testmon_database",
    "refresh_stamp",
    "seed_marker_is_checkout_bound",
    "stamp_from_attempt",
    "testmon_runtime_identity",
    "validate_stamp",
]

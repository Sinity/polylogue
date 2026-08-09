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

import hashlib
import json
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


@dataclass(frozen=True, slots=True)
class TestmonIdentity:
    git_head: str | None
    worktree_fingerprint: str
    python: str
    skip_slow: bool
    lab: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TestmonIdentity:
        git_head = value.get("git_head")
        if git_head is not None and (not isinstance(git_head, str) or not git_head):
            raise ValueError("identity.git_head must be a non-empty string or null")
        worktree = value.get("worktree_fingerprint")
        python = value.get("python")
        if not isinstance(worktree, str) or not worktree:
            raise ValueError("identity.worktree_fingerprint must be a non-empty string")
        if not isinstance(python, str) or not python:
            raise ValueError("identity.python must be a non-empty string")
        if not isinstance(value.get("skip_slow"), bool) or not isinstance(value.get("lab"), bool):
            raise ValueError("identity selection flags must be booleans")
        return cls(git_head, worktree, python, value["skip_slow"], value["lab"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "git_head": self.git_head,
            "worktree_fingerprint": self.worktree_fingerprint,
            "python": self.python,
            "skip_slow": self.skip_slow,
            "lab": self.lab,
        }


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
        testmon_data = value.get("testmon_data")
        run_id = value.get("run_id")
        artifact_dir = value.get("artifact_dir")
        if not all(isinstance(item, str) and item for item in (testmon_data, run_id, artifact_dir)):
            raise ValueError("seed stamp provenance is incomplete")
        assert isinstance(testmon_data, str)
        assert isinstance(run_id, str)
        assert isinstance(artifact_dir, str)
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
            TestmonBinding.from_mapping(binding),
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


def inspect_testmon_database(path: Path, expected_nodeids: Sequence[str]) -> GraphInspection:
    """Validate the real testmon schema and every expected dependency edge."""
    expected = tuple(expected_nodeids)
    if not path.is_file() or not expected or len(set(expected)) != len(expected):
        return GraphInspection(
            GraphStatus.INCOMPLETE, 0, 0, expected, 0, 0, "missing or malformed expected nodeids", ()
        )
    try:
        with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
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
) -> TestmonSeedStamp | None:
    """Promote only a complete attempt, including a red one, into a stamp."""
    if attempt.get("protocol_version") != protocol_version or attempt.get("status") not in {
        "incomplete",
        "reusable",
        "complete",
    }:
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
    baseline = (
        BaselineStatus.GREEN
        if attempt.get("status") == "complete"
        and exit_code == 0
        and all(outcome in {"passed", "skipped"} for outcome in outcome_by_node.values())
        and not graph.failed_nodeids
        else BaselineStatus.RED
    )
    try:
        typed_identity = TestmonIdentity.from_mapping(identity)
    except ValueError:
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
        TestmonBinding(BindingMode.EXACT, str(checkout_root.resolve())),
        file_fingerprint(data_path),
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
    "file_fingerprint",
    "inspect_testmon_database",
    "refresh_stamp",
    "stamp_from_attempt",
    "validate_stamp",
]

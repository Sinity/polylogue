"""Pinned product routes for the remaining CLI evidence views."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from polylogue.analysis.lineage_graph import CompactLineageGraph
from polylogue.analysis.topology import SessionTopology
from polylogue.config import Config
from polylogue.operations.daemon_protocol import validate_operation_result
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.storage_records import SessionBuilder


def _seed(root: Path) -> tuple[str, str]:
    root.mkdir()
    (
        SessionBuilder(root / "index.db", "view-probe-a")
        .provider("codex")
        .title("Build marker")
        .add_message("one", role="user", text="shared build marker")
        .add_message("two", role="assistant", text="shared build marker solved")
        .save()
    )
    (
        SessionBuilder(root / "index.db", "view-probe-b")
        .provider("codex")
        .title("Build marker followup")
        .add_message("one", role="user", text="shared build marker followup")
        .save()
    )
    with ArchiveStore.open_existing(root) as archive:
        connection = archive.index_connection
        assert connection is not None
        rows = connection.execute("SELECT session_id FROM sessions ORDER BY native_id").fetchall()
        return str(rows[0][0]), str(rows[1][0])


def test_effective_context_uses_the_declared_pinned_operation(tmp_path: Path) -> None:
    """Removing the product route or its row mapper loses the two authored messages."""

    root = tmp_path / "archive"
    first, _ = _seed(root)
    with ArchiveStore.open_existing(root) as archive:
        result = execute_read_operation(
            "read.effective_context",
            {"session_id": first, "at_position": None},
            archive=archive,
            serving_identity="test",
        )
    validate_operation_result("read.effective_context", result)
    body = result["payload"]
    assert isinstance(body, dict)
    assert body["session_id"] == first
    messages = body["messages"]
    assert isinstance(messages, list)
    assert [message["text"] for message in messages] == ["shared build marker", "shared build marker solved"]


def test_lineage_and_topology_use_the_pinned_graph_engines(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    first, _ = _seed(root)
    with ArchiveStore.open_existing(root) as archive:
        lineage = execute_read_operation(
            "read.lineage",
            {"session_id": first, "node_offset": 0, "node_limit": 1, "edge_offset": 0, "edge_limit": 1},
            archive=archive,
            serving_identity="test",
        )
        topology = execute_read_operation(
            "read.topology",
            {"session_id": first, "node_offset": 0, "node_limit": 1, "edge_limit": 1},
            archive=archive,
            serving_identity="test",
        )
    validate_operation_result("read.lineage", lineage)
    validate_operation_result("read.topology", topology)
    lineage_payload = lineage["payload"]
    topology_payload = topology["payload"]
    assert isinstance(lineage_payload, dict)
    assert isinstance(topology_payload, dict)
    assert lineage_payload["seed_id"] == first
    assert topology_payload["target_id"] == first
    assert topology_payload["nodes"]

    async def facade_results() -> tuple[CompactLineageGraph | None, SessionTopology | None]:
        from polylogue.api import Polylogue

        config = Config(archive_root=root, db_path=root / "index.db", render_root=tmp_path / "render", sources=[])
        async with Polylogue.open(config=config) as api:
            return (
                await api.compact_lineage(first, node_offset=0, node_limit=1, edge_offset=0, edge_limit=1),
                await api.get_session_topology(first, node_offset=0, node_limit=1, edge_limit=1),
            )

    facade_lineage, facade_topology = asyncio.run(facade_results())
    assert facade_lineage is not None
    assert facade_topology is not None
    from polylogue.operations.topology_envelope import topology_public_envelope

    assert lineage_payload == facade_lineage.model_dump(mode="json")
    assert topology_payload == topology_public_envelope(facade_topology, session_id=first)


def test_neighbors_keeps_ranked_evidence_on_the_operation_route(tmp_path: Path) -> None:
    """A stubbed route or a dropped seed would not return the second session."""

    root = tmp_path / "archive"
    first, second = _seed(root)
    with ArchiveStore.open_existing(root) as archive:
        result = execute_read_operation(
            "read.neighbors",
            {"session_id": first, "query": None, "origin": None, "limit": 10, "window_hours": 24},
            archive=archive,
            serving_identity="test",
        )
    validate_operation_result("read.neighbors", result)
    body = result["payload"]
    assert isinstance(body, dict)
    candidates = body["neighbors"]
    assert isinstance(candidates, list)
    assert candidates[0]["session"]["id"] == second
    assert candidates[0]["rank"] == 1
    assert candidates[0]["reasons"]


def test_correlation_retains_requested_repo_without_client_archive_read(tmp_path: Path) -> None:
    """The declared route carries the repo override into its product result."""

    root = tmp_path / "archive"
    first, _ = _seed(root)
    with ArchiveStore.open_existing(root) as archive:
        result = execute_read_operation(
            "read.correlation",
            {
                "session_id": first,
                "repo_path": str(root),
                "since_hours": 2,
                "confidence_threshold": 0.3,
            },
            archive=archive,
            serving_identity="test",
        )
    validate_operation_result("read.correlation", result)
    body = result["payload"]
    assert isinstance(body, dict)
    assert body["session_id"] == first
    assert body["repo"] == str(root)
    assert body["checkout_commits"] == []


def test_correlation_reads_typed_refs_from_the_pinned_index(tmp_path: Path) -> None:
    """Dropping the typed ref projection would lose authoritative issue evidence."""

    root = tmp_path / "archive"
    first, _ = _seed(root)
    with sqlite3.connect(root / "index.db") as connection:
        connection.execute(
            "INSERT INTO session_refs (session_id, position, kind, repo, ref_number, url) VALUES (?, ?, ?, ?, ?, ?)",
            (first, 0, "issue", "team/project", 42, "https://github.com/team/project/issues/42"),
        )
    with ArchiveStore.open_existing(root) as archive:
        result = execute_read_operation(
            "read.correlation",
            {"session_id": first, "repo_path": str(root), "since_hours": 2, "confidence_threshold": 0.3},
            archive=archive,
            serving_identity="test",
        )
    body = result["payload"]
    assert isinstance(body, dict)
    refs = body["issue_refs"]
    assert isinstance(refs, list)
    assert any(ref["owner"] == "team" and ref["repo"] == "project" and ref["number"] == 42 for ref in refs)

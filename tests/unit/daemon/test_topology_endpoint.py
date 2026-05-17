"""Reader lineage endpoint contracts (#1121).

Pins the shape of ``GET /api/conversations/{id}/topology`` against a
seeded archive that contains:

- a multi-level resolved tree (root → child → grandchild) so the
  endpoint must return all three nodes plus their resolved edges;
- a sibling branch attached to the root with ``branch_type='sidechain'``
  so the edge ``kind`` enum is exercised;
- a leaf conversation with no parent and no descendants — the
  ``readiness='empty'`` state the reader renders explicitly;
- a missing conversation id — the endpoint must return ``404`` with the
  shared error envelope rather than an empty topology.

The lane mirrors ``test_web_reader.py``'s in-process HTTP harness so the
reader UI exercises the same endpoint code path it ships against.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

pytestmark = pytest.mark.xdist_group("web-reader")


@contextmanager
def _running_server(workspace: dict[str, Path]) -> Iterator[str]:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

    _seed_lineage_db(workspace)
    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    server.auth_token = ""
    server.api_host = "127.0.0.1"
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, name="topology-test", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _archive_db_path(workspace: dict[str, Path]) -> Path:
    return workspace["data_root"] / "polylogue" / "polylogue.db"


def _seed_lineage_db(workspace: dict[str, Path]) -> None:
    from polylogue.storage.sqlite.schema_ddl_archive import (
        ARCHIVE_STORAGE_DDL,
        MESSAGE_FTS_DDL,
        RECALL_PACKS_DDL,
        SAVED_VIEWS_DDL,
        USER_ANNOTATIONS_DDL,
        USER_MARKS_DDL,
    )

    db = _archive_db_path(workspace)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.executescript(ARCHIVE_STORAGE_DDL)
    conn.executescript(MESSAGE_FTS_DDL)
    conn.executescript(USER_MARKS_DDL)
    conn.executescript(USER_ANNOTATIONS_DDL)
    conn.executescript(SAVED_VIEWS_DDL)
    conn.executescript(RECALL_PACKS_DDL)
    # rows: (id, parent, branch_type, title)
    rows = [
        ("root", None, None, "Root session"),
        ("child", "root", "continuation", "Resumed child"),
        ("grandchild", "child", "continuation", "Resumed grandchild"),
        ("sidebranch", "root", "sidechain", "Subagent sidechain"),
        ("orphan", None, None, "Isolated leaf"),
    ]
    for cid, parent, branch, title in rows:
        conn.execute(
            "INSERT INTO conversations("
            "conversation_id, provider_name, provider_conversation_id, title, "
            "content_hash, version, parent_conversation_id, branch_type) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (cid, "claude-code", f"p-{cid}", title, f"h-{cid}", 1, parent, branch),
        )
    conn.commit()
    conn.close()


def _get_json(base_url: str, path: str) -> object:
    req = Request(f"{base_url}{path}")
    with urlopen(req, timeout=10) as resp:
        assert resp.status == 200
        return json.loads(resp.read())


def _get_json_ex(base_url: str, path: str) -> tuple[int, dict[str, object]]:
    req = Request(f"{base_url}{path}")
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except HTTPError as exc:
        body = exc.read().decode()
        try:
            return exc.code, json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return exc.code, {}


class TestTopologyEndpoint:
    """``GET /api/conversations/{id}/topology`` — bounded SessionTopology envelope."""

    def test_root_returns_full_resolved_subtree(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as base_url:
            payload = _get_json(base_url, "/api/conversations/root/topology")
        topo = cast(dict[str, object], payload)
        assert topo["target_id"] == "root"
        assert topo["root_id"] == "root"
        assert topo["cycle_detected"] is False
        assert topo["readiness"] == "ok"
        nodes = cast(list[dict[str, object]], topo["nodes"])
        node_ids = {str(n["conversation_id"]) for n in nodes}
        assert node_ids == {"root", "child", "grandchild", "sidebranch"}
        edges = cast(list[dict[str, object]], topo["edges"])
        edge_pairs = {(str(e.get("parent_id")), str(e["child_id"]), str(e["kind"])) for e in edges}
        assert ("root", "child", "continuation") in edge_pairs
        assert ("child", "grandchild", "continuation") in edge_pairs
        assert ("root", "sidebranch", "sidechain") in edge_pairs

    def test_descendant_walks_up_to_root_and_includes_siblings(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as base_url:
            payload = _get_json(base_url, "/api/conversations/grandchild/topology")
        topo = cast(dict[str, object], payload)
        assert topo["target_id"] == "grandchild"
        assert topo["root_id"] == "root"
        nodes = cast(list[dict[str, object]], topo["nodes"])
        node_ids = {str(n["conversation_id"]) for n in nodes}
        # Walking up from grandchild to root means descendants of root —
        # including the sidebranch sibling of the parent chain — appear.
        assert node_ids == {"root", "child", "grandchild", "sidebranch"}

    def test_isolated_leaf_renders_empty_readiness(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as base_url:
            payload = _get_json(base_url, "/api/conversations/orphan/topology")
        topo = cast(dict[str, object], payload)
        assert topo["target_id"] == "orphan"
        assert topo["root_id"] == "orphan"
        assert cast(list[object], topo["nodes"]) == [
            {
                "conversation_id": "orphan",
                "provider_name": "claude-code",
                "title": "Isolated leaf",
                "depth": 0,
                "is_root": True,
            }
        ]
        assert topo["edges"] == []
        assert topo["readiness"] == "empty"
        assert topo["node_count"] == 1
        assert topo["unresolved_edge_count"] == 0

    def test_missing_conversation_returns_404_envelope(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as base_url:
            status, body = _get_json_ex(base_url, "/api/conversations/does-not-exist/topology")
        assert status == 404
        assert body["ok"] is False
        assert body["error"] == "not_found"

    def test_node_limit_is_bounded_and_signals_truncation(self, workspace_env: dict[str, Path]) -> None:
        """The reader must never receive an unbounded subtree (#1121 AC).

        Requesting ``limit=2`` against the 4-node root subtree must return
        exactly two BFS-ordered nodes plus a ``truncated_count`` of 2 and
        ``readiness='partial'`` so the UI can render the partial-state chip.
        """
        with _running_server(workspace_env) as base_url:
            payload = _get_json(base_url, "/api/conversations/root/topology?limit=2")
        topo = cast(dict[str, object], payload)
        nodes = cast(list[dict[str, object]], topo["nodes"])
        assert len(nodes) == 2
        assert topo["node_limit"] == 2
        assert topo["node_count"] == 4
        assert topo["truncated_count"] == 2
        assert topo["readiness"] == "partial"
        # Edges referencing dropped nodes are filtered so the UI never
        # plots a dangling endpoint.
        edges = cast(list[dict[str, object]], topo["edges"])
        kept_ids = {str(n["conversation_id"]) for n in nodes}
        for edge in edges:
            assert str(edge["child_id"]) in kept_ids
            parent = edge.get("parent_id")
            if parent is not None:
                assert str(parent) in kept_ids

    def test_node_limit_hard_cap_rejects_unbounded_request(self, workspace_env: dict[str, Path]) -> None:
        """Clients cannot escape the hard cap with ``limit=999999``."""
        with _running_server(workspace_env) as base_url:
            payload = _get_json(base_url, "/api/conversations/root/topology?limit=999999")
        topo = cast(dict[str, object], payload)
        assert topo["node_limit"] == 1000


class TestLineageReaderShell:
    """Web shell exposes the Lineage inspector tab and its load hook."""

    def test_lineage_tab_present_in_inspector(self, workspace_env: dict[str, Path]) -> None:
        with _running_server(workspace_env) as base_url:
            req = Request(f"{base_url}/")
            with urlopen(req, timeout=10) as resp:
                body = resp.read().decode()
        assert 'data-tab="lineage"' in body
        assert "function renderInspectorLineage(" in body
        assert "function loadLineage(" in body
        # The compare-with-parent affordance is reachable from the
        # lineage tab; #1121 AC requires the entry point to exist.
        assert "function openCompareWith(" in body
        # Readiness vocabulary surfaces explicit complete/partial/no-lineage
        # states rather than collapsing them into a single OK chip.
        assert "no lineage" in body
        assert "partial" in body

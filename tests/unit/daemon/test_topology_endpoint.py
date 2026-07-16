"""Lineage / topology endpoint contracts for the reader (#1121).

``GET /api/sessions/{id}/topology`` returns the bounded
:class:`polylogue.insights.topology.SessionTopology` envelope shaped by
:mod:`polylogue.daemon.topology_http`. Tests cover:

- the pure envelope projection (readiness chip vocabulary, edge filter,
  hard cap);
- end-to-end dispatch against a seeded archive containing a multi-level
  tree, isolated leaf, and missing id;
- the structural smoke contract for the reader shell: the HTML carries
  the new Lineage tab, JS entry points, and readiness vocabulary.

Visual smoke (multi-level tree / leaf / no-lineage screenshots) is
deferred to the Playwright-based lane in #865; the HTML/JSON contracts
asserted here pin the same three states at the bytes-on-the-wire level.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from polylogue.core.types import SessionId
from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer
from polylogue.daemon.topology_http import (
    DEFAULT_NODE_LIMIT,
    MAX_NODE_LIMIT,
    READINESS_EMPTY,
    READINESS_OK,
    READINESS_PARTIAL,
    build_topology_envelope,
    coerce_node_limit,
)
from polylogue.daemon.web_shell import WEB_SHELL_HTML
from polylogue.insights.topology import (
    SessionTopology,
    TopologyEdge,
    TopologyEdgeKind,
    TopologyNode,
)
from tests.infra.storage_records import SessionBuilder, db_setup

# ---------------------------------------------------------------------------
# In-process handler harness (mirrors test_cost_panel_endpoint.py)
# ---------------------------------------------------------------------------


class _MockServer:
    auth_token = ""
    api_host = "127.0.0.1"
    archive_query_executor = ThreadPoolExecutor(max_workers=1)
    archive_query_admission = threading.BoundedSemaphore(64)  # generous: not under test


class _MockHeaders:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self._headers = headers or {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._headers.get(key, default)


def _make_handler(method: str, path: str) -> DaemonAPIHandler:
    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.server = cast(DaemonAPIHTTPServer, _MockServer())
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.command = method
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.headers = cast(Message, _MockHeaders({"Content-Length": "0"}))
    handler.rfile = BytesIO(b"")
    handler.wfile = BytesIO()
    return handler


def _capture_responses(handler: DaemonAPIHandler) -> tuple[MagicMock, MagicMock]:
    send_error = MagicMock()
    send_json = MagicMock()
    handler._send_error = send_error  # type: ignore[method-assign]
    handler._send_json = send_json  # type: ignore[method-assign]
    return send_error, send_json


def _seed_lineage(db_path: Path) -> None:
    """Three-generation tree plus a sibling sidechain plus an isolated leaf.

    Shape:

        root ── continuation ── grandchild
             └─ sidechain
        isolated  (no parent, no children)
    """

    SessionBuilder(db_path, "root").provider("claude-code").title("Root").add_message(
        role="user", text="kickoff"
    ).save()
    SessionBuilder(db_path, "child").provider("claude-code").title("Resume").parent_session("ext-root").branch_type(
        "continuation"
    ).add_message(role="user", text="continue").save()
    SessionBuilder(db_path, "grandchild").provider("claude-code").title("Fork").parent_session("ext-child").branch_type(
        "fork"
    ).add_message(role="user", text="fork").save()
    SessionBuilder(db_path, "side").provider("claude-code").title("Side").parent_session("ext-root").branch_type(
        "sidechain"
    ).add_message(role="user", text="side").save()
    SessionBuilder(db_path, "isolated").provider("claude-code").title("Lone").add_message(
        role="user", text="alone"
    ).save()


# Archive session ids derive from the builder's provider_session_id
# (``ext-<conv_id>``) and the claude-code origin.
def _native(token: str) -> str:
    return f"claude-code-session:ext-{token}"


# ---------------------------------------------------------------------------
# Pure helpers — envelope shape, readiness vocabulary, bounds
# ---------------------------------------------------------------------------


def _topology(
    *,
    target: str = "child",
    root: str = "root",
    nodes: list[tuple[str, int, bool]] | None = None,
    edges: list[tuple[str, str | None, TopologyEdgeKind, bool, str | None]] | None = None,
    cycle: bool = False,
) -> SessionTopology:
    if nodes is None:
        nodes = [("root", 0, True), ("child", 1, False)]
    if edges is None:
        edges = [("child", "root", TopologyEdgeKind.CONTINUATION, True, None)]
    node_tuple = tuple(
        TopologyNode(
            session_id=SessionId(cid),
            origin="claude-code",
            title=cid.title(),
            depth=depth,
            is_root=is_root,
        )
        for cid, depth, is_root in nodes
    )
    edge_tuple = tuple(
        TopologyEdge(
            child_id=SessionId(child),
            parent_id=SessionId(parent) if parent else None,
            parent_native_id=native,
            kind=kind,
            resolved=resolved,
        )
        for child, parent, kind, resolved, native in edges
    )
    return SessionTopology(
        target_id=SessionId(target),
        root_id=SessionId(root),
        nodes=node_tuple,
        edges=edge_tuple,
        cycle_detected=cycle,
    )


class TestCoerceNodeLimit:
    """Bound-checking on the ``?limit=`` query parameter."""

    def test_missing_returns_default(self) -> None:
        assert coerce_node_limit(None) == DEFAULT_NODE_LIMIT

    def test_empty_returns_default(self) -> None:
        assert coerce_node_limit("") == DEFAULT_NODE_LIMIT

    def test_valid_in_range(self) -> None:
        assert coerce_node_limit("17") == 17

    def test_zero_rejected(self) -> None:
        assert coerce_node_limit("0") is None

    def test_negative_rejected(self) -> None:
        assert coerce_node_limit("-5") is None

    def test_above_hard_cap_rejected(self) -> None:
        assert coerce_node_limit(str(MAX_NODE_LIMIT + 1)) is None

    def test_non_numeric_rejected(self) -> None:
        assert coerce_node_limit("abc") is None


class TestEnvelopeProjection:
    """``build_topology_envelope`` projects ``SessionTopology`` into JSON."""

    def test_singleton_is_empty_readiness(self) -> None:
        topo = _topology(
            target="root",
            root="root",
            nodes=[("root", 0, True)],
            edges=[],
        )
        env = build_topology_envelope(topo)
        assert env["readiness"] == READINESS_EMPTY
        assert env["node_count"] == 1
        assert env["truncated_count"] == 0
        assert env["unresolved_edge_count"] == 0
        assert env["cycle_detected"] is False
        assert env["edges"] == []

    def test_multi_node_is_ok_readiness(self) -> None:
        topo = _topology()
        env = build_topology_envelope(topo)
        assert env["readiness"] == READINESS_OK
        assert env["node_count"] == 2
        node_ids = [cast(dict[str, object], n)["session_id"] for n in cast(list[object], env["nodes"])]
        assert node_ids == ["root", "child"]

    def test_unresolved_edge_marks_partial(self) -> None:
        topo = _topology(
            target="child",
            root="root",
            nodes=[("root", 0, True), ("child", 1, False)],
            edges=[
                ("child", "root", TopologyEdgeKind.CONTINUATION, True, None),
                ("child", None, TopologyEdgeKind.UNRESOLVED_NATIVE, False, "native-xyz"),
            ],
        )
        env = build_topology_envelope(topo)
        assert env["readiness"] == READINESS_PARTIAL
        assert env["unresolved_edge_count"] == 1

    def test_cycle_marks_partial(self) -> None:
        topo = _topology(cycle=True)
        env = build_topology_envelope(topo)
        assert env["readiness"] == READINESS_PARTIAL
        assert env["cycle_detected"] is True

    def test_node_limit_is_bounded_and_signals_truncation(self) -> None:
        """#1121 AC: lineage rendering is bounded — does not unbound expand."""
        topo = _topology(
            nodes=[("root", 0, True), ("child", 1, False), ("grand", 2, False)],
            edges=[
                ("child", "root", TopologyEdgeKind.CONTINUATION, True, None),
                ("grand", "child", TopologyEdgeKind.FORK, True, None),
            ],
        )
        env = build_topology_envelope(topo, node_limit=2)
        assert env["node_count"] == 2
        assert env["total_node_count"] == 3
        assert env["truncated_count"] == 1
        assert env["readiness"] == READINESS_PARTIAL
        # The edge pointing at the dropped grandchild must be filtered so
        # the UI does not plot a dangling endpoint.
        edge_children = [cast(dict[str, object], e)["child_id"] for e in cast(list[object], env["edges"])]
        assert "grand" not in edge_children

    def test_node_limit_hard_cap_enforced(self) -> None:
        topo = _topology()
        env = build_topology_envelope(topo, node_limit=MAX_NODE_LIMIT * 10)
        assert env["node_limit"] == MAX_NODE_LIMIT


# ---------------------------------------------------------------------------
# End-to-end endpoint dispatch
# ---------------------------------------------------------------------------


class TestTopologyEndpointDispatch:
    """``GET /api/sessions/{id}/topology`` routes to the topology handler."""

    def test_unknown_session_returns_404(self, workspace_env: dict[str, Path]) -> None:
        _seed_lineage(db_setup(workspace_env))
        handler = _make_handler("GET", "/api/sessions/does-not-exist/topology")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()
        send_json.assert_not_called()
        send_error.assert_called_once()
        status, code = send_error.call_args.args
        assert status == HTTPStatus.NOT_FOUND
        assert code == "not_found"

    def test_descendant_request_anchors_at_root(self, workspace_env: dict[str, Path]) -> None:
        """Requesting topology from a deep descendant must walk back to root."""
        _seed_lineage(db_setup(workspace_env))
        handler = _make_handler("GET", f"/api/sessions/{_native('grandchild')}/topology")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()
        send_error.assert_not_called()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["target_id"] == _native("grandchild")
        assert payload["root_id"] == _native("root")
        node_ids = {n["session_id"] for n in payload["nodes"]}
        assert node_ids == {_native("root"), _native("child"), _native("grandchild"), _native("side")}
        # BFS order: root first, depth-1 siblings before depth-2 grandchild.
        depths = [n["depth"] for n in payload["nodes"]]
        assert depths[0] == 0
        assert depths == sorted(depths)
        assert payload["readiness"] == READINESS_OK

    def test_isolated_session_renders_empty_readiness(self, workspace_env: dict[str, Path]) -> None:
        """#1121 AC: no-lineage state is rendered explicitly with a readiness chip."""
        _seed_lineage(db_setup(workspace_env))
        handler = _make_handler("GET", f"/api/sessions/{_native('isolated')}/topology")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()
        send_error.assert_not_called()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["readiness"] == READINESS_EMPTY
        assert payload["node_count"] == 1
        assert payload["edges"] == []

    def test_node_limit_hard_cap_rejects_unbounded_request(self, workspace_env: dict[str, Path]) -> None:
        """#1121 AC: bounded — operator cannot widen past the daemon hard cap."""
        _seed_lineage(db_setup(workspace_env))
        handler = _make_handler(
            "GET",
            f"/api/sessions/{_native('root')}/topology?limit={MAX_NODE_LIMIT + 1}",
        )
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()
        send_json.assert_not_called()
        send_error.assert_called_once()
        status, code = send_error.call_args.args
        assert status == HTTPStatus.BAD_REQUEST
        assert code == "invalid_limit"

    def test_node_limit_clamps_payload_size(self, workspace_env: dict[str, Path]) -> None:
        _seed_lineage(db_setup(workspace_env))
        handler = _make_handler("GET", f"/api/sessions/{_native('root')}/topology?limit=2")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()
        send_error.assert_not_called()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["node_count"] == 2
        assert payload["truncated_count"] >= 1
        # Edges referencing dropped nodes must be filtered out.
        kept_ids = {n["session_id"] for n in payload["nodes"]}
        for edge in payload["edges"]:
            if edge["resolved"]:
                assert edge["child_id"] in kept_ids
                assert edge["parent_id"] in kept_ids


# ---------------------------------------------------------------------------
# Reader-shell structural smoke (HTML contract for the Lineage tab)
# ---------------------------------------------------------------------------


class TestLineageReaderShell:
    """The shipped HTML must carry the Lineage tab, JS hooks, and chip vocab."""

    def test_lineage_tab_button_present(self) -> None:
        assert 'data-tab="lineage"' in WEB_SHELL_HTML

    def test_lineage_js_entry_points_present(self) -> None:
        for hook in (
            "loadLineage",
            "renderInspectorLineage",
            "openCompareWithParent",
            "lineageReadinessChip",
        ):
            assert hook in WEB_SHELL_HTML, f"missing JS hook: {hook}"

    def test_readiness_chip_vocabulary_present(self) -> None:
        # Reader uses MK3 q-* chip vocabulary shared with cost/provenance.
        for chip in ("q-canonical", "q-partial", "q-unavailable", "q-unresolved"):
            assert chip in WEB_SHELL_HTML, f"missing chip class: {chip}"

    def test_compare_with_parent_action_delegates_to_compare_workspace(self) -> None:
        # The "Compare with parent" affordance delegates to the existing
        # compare workspace route from #1124 — does not implement its own.
        assert "openCompareWithParent" in WEB_SHELL_HTML
        assert "loadWorkspaceRoute" in WEB_SHELL_HTML

    @pytest.mark.parametrize("placeholder", ["__LINEAGE_JS__", "__PROVENANCE_JS__"])
    def test_no_unresolved_template_placeholders(self, placeholder: str) -> None:
        # If interpolation wires up correctly the placeholders must not
        # appear in the final shipped HTML.
        assert placeholder not in WEB_SHELL_HTML

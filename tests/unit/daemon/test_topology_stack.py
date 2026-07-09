"""Parent-chain topology stack endpoint contracts (#1203).

The reader's #1203 work bridges the lineage topology projection (#1121)
into the stack workspace. This test module covers:

- the pure envelope projection (chain ordering, branch kind, sibling
  surfacing, isolated-leaf shape);
- end-to-end dispatch against a seeded archive containing a
  three-generation tree plus a sibling sidechain and an isolated leaf;
- the reader-shell structural contract: the HTML carries the branch
  chip helpers, the "open chain" affordances, and the parent-chain JS
  hooks so the topology -> workspace bridge is reachable from the UI.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer
from polylogue.daemon.topology_http import build_parent_chain_envelope
from polylogue.daemon.web_shell import WEB_SHELL_HTML
from polylogue.insights.topology import (
    SessionTopology,
    TopologyEdge,
    TopologyEdgeKind,
    TopologyNode,
)
from polylogue.types import SessionId
from tests.infra.storage_records import SessionBuilder, db_setup


class _MockServer:
    auth_token = ""
    api_host = "127.0.0.1"
    archive_query_executor = ThreadPoolExecutor(max_workers=1)


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

    Mirrors the seed shape from ``test_topology_endpoint.py`` so the
    stack envelope can be cross-checked against the unbounded topology
    walk.
    """

    SessionBuilder(db_path, "root").provider("claude-code").title("Root").add_message(
        role="user", text="kickoff"
    ).save()
    SessionBuilder(db_path, "child").provider("claude-code").title("Resume").parent_session("ext-root").branch_type(
        "subagent"
    ).add_message(role="user", text="continue").save()
    SessionBuilder(db_path, "grandchild").provider("claude-code").title("Fork").parent_session("ext-child").branch_type(
        "continuation"
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


def _make_topology(
    *,
    target: str,
    nodes: list[tuple[str, int, bool]],
    edges: list[tuple[str, str | None, TopologyEdgeKind, bool]],
    cycle: bool = False,
) -> SessionTopology:
    node_tuple = tuple(
        TopologyNode(
            session_id=SessionId(cid),
            source_name="claude-code",
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
            kind=kind,
            resolved=resolved,
        )
        for child, parent, kind, resolved in edges
    )
    root_id = nodes[0][0]
    return SessionTopology(
        target_id=SessionId(target),
        root_id=SessionId(root_id),
        nodes=node_tuple,
        edges=edge_tuple,
        cycle_detected=cycle,
    )


class TestParentChainProjection:
    """``build_parent_chain_envelope`` projects ``SessionTopology`` into stack."""

    def test_isolated_leaf_returns_single_element_chain(self) -> None:
        topo = _make_topology(
            target="root",
            nodes=[("root", 0, True)],
            edges=[],
        )
        env = build_parent_chain_envelope(topo)
        assert env["chain_ids"] == ["root"]
        assert env["ancestors"] == []
        assert env["descendants"] == []
        assert env["branch_kind"] is None
        assert env["parent_id"] is None
        assert env["focus_id"] == "root"

    def test_chain_is_ordered_root_to_target_to_descendants(self) -> None:
        # root -> child (subagent) -> grand (continuation), plus a sibling.
        topo = _make_topology(
            target="child",
            nodes=[
                ("root", 0, True),
                ("child", 1, False),
                ("grand", 2, False),
                ("sibling", 1, False),
            ],
            edges=[
                ("child", "root", TopologyEdgeKind.SUBAGENT, True),
                ("grand", "child", TopologyEdgeKind.CONTINUATION, True),
                ("sibling", "root", TopologyEdgeKind.FORK, True),
            ],
        )
        env = build_parent_chain_envelope(topo)
        assert env["chain_ids"] == ["root", "child", "grand"]
        assert env["ancestors"] == ["root"]
        assert env["descendants"] == ["grand"]
        assert env["branch_kind"] == "subagent"
        assert env["parent_id"] == "root"
        assert env["focus_id"] == "child"
        assert env["siblings"] == ["sibling"]

    def test_descendants_can_be_excluded(self) -> None:
        topo = _make_topology(
            target="child",
            nodes=[("root", 0, True), ("child", 1, False), ("grand", 2, False)],
            edges=[
                ("child", "root", TopologyEdgeKind.CONTINUATION, True),
                ("grand", "child", TopologyEdgeKind.CONTINUATION, True),
            ],
        )
        env = build_parent_chain_envelope(topo, include_descendants=False)
        assert env["chain_ids"] == ["root", "child"]
        assert env["descendants"] == []

    def test_unresolved_edge_does_not_surface_as_branch_kind(self) -> None:
        # A target whose only incoming edge is unresolved must not lie
        # about its branch kind — the chip falls back to root semantics.
        topo = _make_topology(
            target="child",
            nodes=[("root", 0, True), ("child", 0, True)],
            edges=[
                ("child", None, TopologyEdgeKind.UNRESOLVED_NATIVE, False),
            ],
        )
        env = build_parent_chain_envelope(topo)
        assert env["branch_kind"] is None
        assert env["parent_id"] is None
        assert env["chain_ids"] == ["child"]


class TestParentChainEndpointDispatch:
    """``GET /api/sessions/{id}/topology/parent-chain`` routing."""

    def test_unknown_session_returns_404(self, workspace_env: dict[str, Path]) -> None:
        _seed_lineage(db_setup(workspace_env))
        handler = _make_handler("GET", "/api/sessions/missing/topology/parent-chain")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()
        send_json.assert_not_called()
        send_error.assert_called_once()
        status, code = send_error.call_args.args
        assert status == HTTPStatus.NOT_FOUND
        assert code == "not_found"

    def test_grandchild_walks_back_to_root_and_orders_chain(self, workspace_env: dict[str, Path]) -> None:
        _seed_lineage(db_setup(workspace_env))
        handler = _make_handler("GET", f"/api/sessions/{_native('grandchild')}/topology/parent-chain")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()
        send_error.assert_not_called()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        # Oldest-to-newest ordering: root then the subagent child then the
        # grandchild itself. The sidechain sibling is NOT part of the chain.
        assert payload["chain_ids"][0] == _native("root")
        assert payload["chain_ids"][1] == _native("child")
        assert payload["chain_ids"][2] == _native("grandchild")
        assert payload["focus_id"] == _native("grandchild")
        assert payload["parent_id"] == _native("child")
        # Sibling is surfaced separately for the popover, not inline.
        assert _native("side") not in payload["chain_ids"]

    def test_isolated_session_returns_single_element_chain(self, workspace_env: dict[str, Path]) -> None:
        _seed_lineage(db_setup(workspace_env))
        handler = _make_handler("GET", f"/api/sessions/{_native('isolated')}/topology/parent-chain")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()
        send_error.assert_not_called()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["chain_ids"] == [_native("isolated")]
        assert payload["ancestors"] == []
        assert payload["branch_kind"] is None

    def test_descendants_query_param_excludes_descendants(self, workspace_env: dict[str, Path]) -> None:
        _seed_lineage(db_setup(workspace_env))
        handler = _make_handler("GET", f"/api/sessions/{_native('child')}/topology/parent-chain?descendants=0")
        send_error, send_json = _capture_responses(handler)
        handler.do_GET()
        send_error.assert_not_called()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["chain_ids"] == [_native("root"), _native("child")]
        assert payload["descendants"] == []


class TestReaderShellTopologyHooks:
    """The shipped HTML must carry the chip helpers and the chain action."""

    def test_branch_chip_helper_present(self) -> None:
        assert "renderTopologyBranchChip" in WEB_SHELL_HTML

    def test_open_chain_helper_present(self) -> None:
        assert "openParentChainAsStack" in WEB_SHELL_HTML
        assert "renderOpenParentChainButton" in WEB_SHELL_HTML

    def test_open_chain_action_is_in_lineage_inspector(self) -> None:
        # The lineage tab exposes the same action so it is reachable from
        # either the session header or the inspector.
        assert "Open chain as stack" in WEB_SHELL_HTML

    def test_branch_chip_uses_q_vocabulary(self) -> None:
        # The chip uses the MK3 q-* class vocabulary so the operator can
        # tell at a glance whether the branch kind is canonical / derived
        # / degraded / missing — matches the chip vocabulary in the rest
        # of the shell.
        for cls in ("q-canonical", "q-heuristic", "q-estimated", "q-unavailable", "q-unresolved"):
            assert cls in WEB_SHELL_HTML

    def test_branch_chip_clickable_into_lineage_tab(self) -> None:
        assert "openLineageInspector" in WEB_SHELL_HTML

    def test_parent_chain_endpoint_is_called(self) -> None:
        # JS routes through ``/api/sessions/{id}/topology/parent-chain``
        # so the daemon owns the lineage walk; the reader never re-derives
        # the chain client-side.
        assert "/topology/parent-chain" in WEB_SHELL_HTML

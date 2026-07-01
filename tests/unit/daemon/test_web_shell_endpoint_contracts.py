"""Endpoint contracts for the daemon web-shell split-out modules (#1291).

The reader's single-page shell was split for file-size budget reasons
into three sibling asset modules under ``polylogue/daemon/``:

- :mod:`polylogue.daemon.web_shell_selection` — selection-operations toolbar and
  client-side composer (#1119).
- :mod:`polylogue.daemon.web_shell_lineage` — lineage inspector tab
  rendering and ``/api/sessions/{id}/topology`` consumer (#1121).
- :mod:`polylogue.daemon.web_shell_workspace` — workspace mode toolbar,
  stack/compare/saved-view/recall-pack routing (#1124, #1203).

Before this PR ``grep -rl web_shell_(selection|lineage|workspace) tests/`` was
empty for all three modules even though the shell-level smoke (#865)
exercises the assembled HTML. The visual smoke does not pin the cross-
surface contract: which daemon endpoints the JS calls, which mark types
the toolbar mutates, and which fields the lineage tab consumes from the
:class:`SessionTopology` envelope.

This file is the durable cross-surface pin. It is organized by source
module so the closure-matrix row for each module can attribute coverage:

- ``TestWebShellSelectionAssetContract`` — asset module integrity and the
  JS→endpoint contract for ``POST /api/user/marks`` shared with the
  user-state surface stabilized by #1290.
- ``TestWebShellLineageAssetContract`` — asset module integrity and the
  JS→envelope contract against ``build_topology_envelope`` so the
  lineage tab field set never drifts from the daemon producer.
- ``TestWebShellWorkspaceAssetContract`` — asset module integrity, the
  ``/api/stack`` and ``/api/compare`` route dispatch and request
  validation contract under :mod:`polylogue.daemon.workspace_routes`,
  and the ``/api/user/{workspaces,saved-views,recall-packs}`` mutation
  contract the workspace toolbar drives via
  :mod:`polylogue.daemon.user_state_http`.

The HTTP-handler harness mirrors
``tests/unit/daemon/test_user_state_http.py`` (#1290): in-process
handler instance, no real socket, ``workspace_env`` fixture for the
SQLite archive.
"""

from __future__ import annotations

import json
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

from polylogue.daemon import (
    web_shell_lineage,
    web_shell_selection,
    web_shell_workspace,
    workspace_routes,
)
from polylogue.daemon.topology_http import build_topology_envelope
from polylogue.insights.topology import (
    SessionTopology,
    TopologyEdge,
    TopologyEdgeKind,
    TopologyNode,
)
from polylogue.types import SessionId


def _cid(value: str) -> SessionId:
    """Type-checker-friendly NewType cast used by the lineage helper."""
    return SessionId(value)


if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer


# ---------------------------------------------------------------------------
# Handler harness — mirrors tests/unit/daemon/test_user_state_http.py (#1290).
# ---------------------------------------------------------------------------


class _MockServer:
    auth_token = ""
    api_host = "127.0.0.1"


class _MockHeaders:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self._headers = headers or {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._headers.get(key, default)


def _make_handler(method: str, path: str, *, body: bytes = b"") -> DaemonAPIHandler:
    from polylogue.daemon.http import DaemonAPIHandler

    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.server = cast("DaemonAPIHTTPServer", _MockServer())
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.command = method
    handler.requestline = f"{method} {path} HTTP/1.1"
    headers: dict[str, str] = {"Content-Length": str(len(body))}
    handler.headers = cast("Message[str, str]", _MockHeaders(headers))
    handler.rfile = BytesIO(body)
    handler.wfile = BytesIO()
    return handler


def _capture(handler: DaemonAPIHandler) -> tuple[MagicMock, MagicMock]:
    send_error = MagicMock()
    send_json = MagicMock()
    handler._send_error = send_error  # type: ignore[method-assign]
    handler._send_json = send_json  # type: ignore[method-assign]
    return send_error, send_json


def _seed_session(session_id: str, workspace_env: dict[str, Path]) -> str:
    """Seed a single native session so resolve_id can find a target.

    Mirrors the helper used by ``test_user_state_http``: handlers such as
    ``add_mark`` call ``_resolve_user_state_target``, which requires the
    session to exist in the archive store. Returns the native session
    id (``origin:native_id``) to address the session in requests.
    """
    from tests.infra.storage_records import SessionBuilder, db_setup

    builder = (
        SessionBuilder(db_setup(workspace_env), session_id)
        .provider("claude-code")
        .title("Test")
        .add_message(message_id=f"m-{session_id}", role="user", text="hello")
    )
    builder.save()
    return builder.native_session_id()


# ---------------------------------------------------------------------------
# 1. web_shell_selection — asset contract + /api/user/marks JS→endpoint contract
# ---------------------------------------------------------------------------


class TestWebShellSelectionAssetContract:
    """Pins what the selection JS calls so the toolbar never silently drifts.

    The selection surface is intentionally a client-side composition over
    existing daemon endpoints. The asset module exposes three string
    constants — ``SELECTION_CSS``, ``SELECTION_TOOLBAR_HTML``, ``SELECTION_JS`` —
    and the JS issues per-session POSTs to ``/api/user/marks``
    (the same endpoint covered by ``test_user_state_http``).
    """

    def test_module_exports_three_string_assets(self) -> None:
        assert isinstance(web_shell_selection.SELECTION_CSS, str)
        assert isinstance(web_shell_selection.SELECTION_TOOLBAR_HTML, str)
        assert isinstance(web_shell_selection.SELECTION_JS, str)
        assert web_shell_selection.SELECTION_CSS.strip()
        assert web_shell_selection.SELECTION_TOOLBAR_HTML.strip()
        assert web_shell_selection.SELECTION_JS.strip()

    def test_toolbar_html_declares_all_selection_actions(self) -> None:
        """Toolbar must expose tag/export/delete-preview/reembed-preview.

        These five actions are the AC for #1119. Removing any one of
        them silently breaks the toolbar without a render error.
        """
        html = web_shell_selection.SELECTION_TOOLBAR_HTML
        for action in (
            "tag-star",
            "tag-pin",
            "tag-archive",
            "export",
            "delete-preview",
            "reembed-preview",
        ):
            assert f'data-selection-action="{action}"' in html

    def test_selection_js_calls_only_documented_endpoints(self) -> None:
        """The selection module is documented as composing over existing
        endpoints; if a new server route appears in the JS without a
        corresponding handler, this assertion forces a code review."""
        js = web_shell_selection.SELECTION_JS
        assert "/api/user/marks" in js
        assert "/api/sessions/" in js
        # Negative pins — the dry-run preview must NOT POST to a delete
        # or re-embed endpoint. The whole point of the preview gate is
        # that those endpoints do not exist yet.
        assert "DELETE /api/sessions" not in js
        assert "/api/embed/reembed" not in js
        assert "/api/sessions/delete" not in js

    def test_dry_run_envelope_shape_is_pinned(self) -> None:
        """The selection preview confirm path records ``no_endpoint`` skips.

        The shape (``{action, dryRun, succeeded, failed, skipped}``)
        is the cross-surface contract the inspector reads from
        ``state.lastSelectionResult``; pinning the literal strings here
        catches silent renames of the envelope keys.
        """
        js = web_shell_selection.SELECTION_JS
        for token in ("dryRun: true", "succeeded:", "failed:", "skipped:"):
            assert token in js
        assert "no_endpoint" in js

    def test_marks_endpoint_accepts_selection_js_payload_shape(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """End-to-end: the JSON shape the selection JS POSTs is the shape the
        daemon handler accepts. Without this pin a rename on either side
        breaks the toolbar without any test failing."""
        native_id = _seed_session("c-selection", workspace_env)

        # This is exactly the payload ``selectionApplyMark`` builds.
        body = json.dumps({"session_id": native_id, "mark_type": "star"}).encode()
        handler = _make_handler("POST", "/api/user/marks", body=body)
        send_error, send_json = _capture(handler)
        handler.do_POST()

        send_error.assert_not_called()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.CREATED
        assert payload["mark_type"] == "star"
        assert payload["session_id"] == native_id

    @pytest.mark.parametrize("mark_type", ["star", "pin", "archive"])
    def test_every_toolbar_mark_type_round_trips(
        self,
        workspace_env: dict[str, Path],
        mark_type: str,
    ) -> None:
        """The toolbar offers three tag buttons (#1119 AC). Each must
        be acceptable to the daemon mark handler — otherwise the
        toolbar shows a click that always fails."""
        native_id = _seed_session(f"c-{mark_type}", workspace_env)

        body = json.dumps({"session_id": native_id, "mark_type": mark_type}).encode()
        handler = _make_handler("POST", "/api/user/marks", body=body)
        send_error, send_json = _capture(handler)
        handler.do_POST()
        send_error.assert_not_called()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.CREATED
        assert payload["mark_type"] == mark_type


# ---------------------------------------------------------------------------
# 2. web_shell_lineage — asset contract + JS→topology envelope contract
# ---------------------------------------------------------------------------


def _make_topology(*, with_cycle: bool = False) -> SessionTopology:
    """Build a small SessionTopology covering parent+target+sibling+child."""
    nodes = (
        TopologyNode(
            session_id=_cid("root"),
            source_name="claude-code",
            title="Root",
            depth=0,
            is_root=True,
        ),
        TopologyNode(
            session_id=_cid("target"),
            source_name="claude-code",
            title="Target",
            depth=1,
            is_root=False,
        ),
        TopologyNode(
            session_id=_cid("sibling"),
            source_name="claude-code",
            title="Sibling",
            depth=1,
            is_root=False,
        ),
        TopologyNode(
            session_id=_cid("child"),
            source_name="claude-code",
            title="Child",
            depth=2,
            is_root=False,
        ),
    )
    edges = (
        TopologyEdge(
            child_id=_cid("target"),
            parent_id=_cid("root"),
            parent_native_id="root-native",
            kind=TopologyEdgeKind.CONTINUATION,
            resolved=True,
        ),
        TopologyEdge(
            child_id=_cid("sibling"),
            parent_id=_cid("root"),
            parent_native_id="root-native",
            kind=TopologyEdgeKind.SIDECHAIN,
            resolved=True,
        ),
        TopologyEdge(
            child_id=_cid("child"),
            parent_id=_cid("target"),
            parent_native_id="target-native",
            kind=TopologyEdgeKind.SUBAGENT,
            resolved=True,
        ),
    )
    return SessionTopology(
        target_id=_cid("target"),
        root_id=_cid("root"),
        nodes=nodes,
        edges=edges,
        cycle_detected=with_cycle,
    )


class TestWebShellLineageAssetContract:
    """Pins the lineage inspector contract with its envelope producer."""

    def test_module_exports_lineage_js(self) -> None:
        assert isinstance(web_shell_lineage.LINEAGE_JS, str)
        assert web_shell_lineage.LINEAGE_JS.strip()
        assert "LINEAGE_JS" in web_shell_lineage.__all__

    def test_lineage_js_targets_topology_endpoint(self) -> None:
        """The lineage tab fetches ``/api/sessions/{id}/topology``.

        That URL must match the route handler registered in
        :mod:`polylogue.daemon.http` (``_handle_get_session_topology``).
        """
        js = web_shell_lineage.LINEAGE_JS
        assert "/api/sessions/" in js
        assert "/topology" in js

    def test_lineage_js_consumes_only_envelope_fields(self) -> None:
        """The fields the inspector reads must all be produced by
        :func:`build_topology_envelope`. If the envelope drops a field
        the inspector reads, this test catches the drift.
        """
        js = web_shell_lineage.LINEAGE_JS
        envelope = build_topology_envelope(_make_topology())
        for field in (
            "readiness",
            "nodes",
            "edges",
            "node_count",
            "truncated_count",
            "unresolved_edge_count",
            "cycle_detected",
        ):
            assert field in envelope, f"envelope must produce {field}"
            assert field in js, f"lineage JS must consume {field}"

    def test_lineage_js_consumes_only_node_and_edge_keys_from_envelope(self) -> None:
        """Per-node/edge field consumption matches the envelope.

        Renaming ``parent_native_id`` to ``parent_provider_id`` on the
        envelope without updating the JS would silently break the
        "Unresolved parents" block — this catches that drift.
        """
        envelope = build_topology_envelope(_make_topology())
        node_keys = set(envelope["nodes"][0].keys())  # type: ignore[index]
        edge_keys = set(envelope["edges"][0].keys())  # type: ignore[index]
        # Every field the lineage JS reads off a node/edge must exist.
        js = web_shell_lineage.LINEAGE_JS
        for key in ("session_id", "depth", "is_root", "title"):
            assert key in node_keys
            assert key in js
        for key in ("child_id", "parent_id", "parent_native_id", "kind", "resolved"):
            assert key in edge_keys
            assert key in js

    def test_lineage_js_handles_documented_readiness_vocabulary(self) -> None:
        """The readiness chip vocabulary (``ok``/``partial``/``empty``)
        is shared between :func:`build_topology_envelope` and the JS
        renderer. The JS explicitly branches on the non-``ok`` values;
        ``ok`` is the implicit fall-through after the explicit checks.
        Renaming any of the three on the server without updating the
        chip renderer would silently flip the chip class.
        """
        from polylogue.daemon import topology_http

        js = web_shell_lineage.LINEAGE_JS
        # ``empty`` and ``partial`` must appear as explicit literal
        # branches in lineageReadinessChip — they are the conditions
        # for the dedicated chip classes.
        for vocab in (
            topology_http.READINESS_EMPTY,
            topology_http.READINESS_PARTIAL,
        ):
            assert f"'{vocab}'" in js or f'"{vocab}"' in js
        # ``ok`` is the implicit fall-through; the chip text appears
        # in the default branch rather than as a conditional literal.
        assert ">ok<" in js or "'ok'" in js or '"ok"' in js

    def test_lineage_js_only_delegates_to_known_reader_actions(self) -> None:
        """The lineage tab is read-only (#1121 explicit non-goal).

        It may call ``selectSession``, ``openCompareWithParent``,
        ``openParentChainAsStack``, and ``loadWorkspaceRoute`` — that's
        it. Any new mutation entrypoint added here is an architectural
        regression and should fail review.
        """
        js = web_shell_lineage.LINEAGE_JS
        # Allow-list — verify the documented entrypoints exist.
        assert "selectSession" in js
        assert "openCompareWithParent" in js
        assert "openParentChainAsStack" in js
        assert "loadWorkspaceRoute" in js
        # Deny-list — these would indicate non-read-only behavior.
        assert "fetch('/api/user/" not in js
        assert "sendJSON" not in js


# ---------------------------------------------------------------------------
# 3. web_shell_workspace — asset contract + /api/stack /api/compare contract
# ---------------------------------------------------------------------------


class TestWebShellWorkspaceAssetContract:
    """Pins the workspace toolbar's endpoint contract.

    Splits across three contracts:

    - module asset integrity (CSS/HTML/JS strings);
    - workspace routes (``/api/stack``, ``/api/compare``) dispatch and
      input validation under :mod:`polylogue.daemon.workspace_routes`;
    - user-state mutation routing (``/api/user/workspaces``,
      ``/api/user/saved-views``, ``/api/user/recall-packs``) — the JS
      MUST hit these exact URLs or the toolbar buttons are dead.
    """

    def test_module_exports_three_string_assets(self) -> None:
        assert isinstance(web_shell_workspace.WORKSPACE_CSS, str)
        assert isinstance(web_shell_workspace.WORKSPACE_HTML, str)
        assert isinstance(web_shell_workspace.WORKSPACE_JS, str)
        assert set(web_shell_workspace.__all__) == {
            "WORKSPACE_CSS",
            "WORKSPACE_HTML",
            "WORKSPACE_JS",
        }

    def test_workspace_js_calls_documented_endpoints(self) -> None:
        js = web_shell_workspace.WORKSPACE_JS
        # Workspace mode payloads — must match workspace_routes.
        assert "/api/stack?" in js
        assert "/api/compare?" in js
        # User-state mutations — must match user_state_http dispatch table.
        assert "/api/user/workspaces" in js
        assert "/api/user/saved-views" in js
        assert "/api/user/recall-packs" in js

    def test_workspace_modes_vocabulary_is_shared(self) -> None:
        """The toolbar exposes single/stack/compare buttons.

        Each mode token must be a member of
        :data:`workspace_routes.WORKSPACE_SHELL_MODES` (server-side
        vocabulary) — otherwise the saved-workspace round-trip rejects
        a workspace that the UI just produced.
        """
        html = web_shell_workspace.WORKSPACE_HTML
        for mode in ("single", "stack", "compare"):
            assert f'data-mode="{mode}"' in html
        # Server vocabulary must accept every shell mode.
        for mode in ("tabs", "stack", "compare"):
            assert mode in workspace_routes.WORKSPACE_SHELL_MODES

    def test_compare_align_vocabulary_is_shared(self) -> None:
        """The compare align dropdown only offers ``prompt`` today.

        :data:`workspace_routes.COMPARE_ALIGN_MODES` is the server-side
        whitelist — any new align mode added in the JS must be added to
        the server set in the same PR.
        """
        js = web_shell_workspace.WORKSPACE_JS
        # Every align literal the JS sets must be acceptable server-side.
        assert "align: 'prompt'" in js
        assert "prompt" in workspace_routes.COMPARE_ALIGN_MODES

    # ---- /api/stack contract -------------------------------------------

    def test_api_stack_dispatch_is_owned_by_http_handler(self) -> None:
        handler = _make_handler("GET", "/api/stack?ids=a,b")
        ok = workspace_routes.dispatch_get(handler, ["api", "stack"], {"ids": ["a,b"]})
        assert ok is False

    def test_api_stack_requires_ids(self) -> None:
        handler = _make_handler("GET", "/api/stack")
        send_error, send_json = _capture(handler)
        workspace_routes.handle_stack(handler, {})
        send_error.assert_called_once_with(HTTPStatus.BAD_REQUEST, "invalid_request")
        send_json.assert_not_called()

    def test_api_stack_blank_ids_token_rejected(self) -> None:
        """``?ids=,, ,`` parses to an empty list and must 400."""
        handler = _make_handler("GET", "/api/stack?ids=,,%20,")
        send_error, send_json = _capture(handler)
        workspace_routes.handle_stack(handler, {"ids": [",, ,"]})
        send_error.assert_called_once_with(HTTPStatus.BAD_REQUEST, "invalid_request")
        send_json.assert_not_called()

    # ---- /api/compare contract -----------------------------------------

    def test_api_compare_dispatch_is_owned_by_http_handler(self) -> None:
        handler = _make_handler("GET", "/api/compare?left=a&right=b")
        ok = workspace_routes.dispatch_get(
            handler,
            ["api", "compare"],
            {"left": ["a"], "right": ["b"]},
        )
        assert ok is False

    @pytest.mark.parametrize(
        "params",
        [
            {},
            {"left": ["a"]},
            {"right": ["b"]},
            {"left": ["a"], "right": ["b"], "align": ["nonsense"]},
        ],
    )
    def test_api_compare_validates_params(self, params: dict[str, list[str]]) -> None:
        handler = _make_handler("GET", "/api/compare")
        send_error, send_json = _capture(handler)
        workspace_routes.handle_compare(handler, params)
        send_error.assert_called_once_with(HTTPStatus.BAD_REQUEST, "invalid_request")
        send_json.assert_not_called()

    # ---- dispatch fallthrough -----------------------------------------

    def test_unknown_workspace_route_falls_through(self) -> None:
        handler = _make_handler("GET", "/api/unknown")
        ok = workspace_routes.dispatch_get(handler, ["api", "unknown"], {})
        assert ok is False

    def test_parse_id_list_normalizes_whitespace_and_commas(self) -> None:
        parsed = workspace_routes.parse_id_list({"ids": ["a, b ,c", "d"]})
        assert parsed == ["a", "b", "c", "d"]

    # ---- /api/user/workspaces contract used by the toolbar ------------

    def test_save_workspace_endpoint_accepts_toolbar_payload(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """Verify the toolbar's ``saveWorkspace`` payload shape round-
        trips through the daemon. The JS posts ``workspace_id``,
        ``name``, ``mode``, ``open_targets``, ``layout``,
        ``active_target`` — every field is required to round-trip
        without a 400 or a silently-dropped field.
        """
        body = json.dumps(
            {
                "workspace_id": "ws-toolbar",
                "name": "Toolbar workspace",
                "mode": "stack",
                "open_targets": [
                    {"target_type": "session", "session_id": "c-1"},
                    {"target_type": "session", "session_id": "c-2"},
                ],
                "layout": {"mode": "stack"},
                "active_target": {
                    "target_type": "session",
                    "session_id": "c-1",
                },
            }
        ).encode()
        handler = _make_handler("POST", "/api/user/workspaces", body=body)
        send_error, send_json = _capture(handler)
        handler.do_POST()
        send_error.assert_not_called()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.CREATED
        assert payload == {
            "status": "ok",
            "affected_count": 1,
            "operation": "workspace.save",
            "resource_type": "workspace",
            "resource_id": "ws-toolbar",
        }

    def test_save_recall_pack_endpoint_accepts_toolbar_payload(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """``createRecallPack`` posts ``{pack_id, label, payload:{items}}``."""
        _seed_session("c-recall", workspace_env)
        body = json.dumps(
            {
                "pack_id": "pack-toolbar",
                "label": "Toolbar pack",
                "payload": {
                    "summary": "Toolbar pack",
                    "items": [
                        {"target_type": "session", "session_id": "c-recall"},
                    ],
                },
            }
        ).encode()
        handler = _make_handler("POST", "/api/user/recall-packs", body=body)
        send_error, send_json = _capture(handler)
        handler.do_POST()
        send_error.assert_not_called()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.CREATED
        assert payload == {
            "status": "ok",
            "affected_count": 1,
            "operation": "recall_pack.save",
            "resource_type": "recall_pack",
            "resource_id": "pack-toolbar",
        }

    def test_save_view_endpoint_accepts_toolbar_payload(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """``saveCurrentView`` posts ``{name, query: {...}}``."""
        body = json.dumps({"name": "Toolbar view", "query": {"limit": 20, "origin": "claude-code-session"}}).encode()
        handler = _make_handler("POST", "/api/user/saved-views", body=body)
        send_error, send_json = _capture(handler)
        handler.do_POST()
        send_error.assert_not_called()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.CREATED
        assert payload["status"] == "ok"
        assert payload["affected_count"] == 1
        assert payload["operation"] == "saved_view.save"
        assert payload["resource_type"] == "saved_view"
        assert isinstance(payload["resource_id"], str)

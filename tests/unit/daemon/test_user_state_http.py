"""Dedicated coverage for ``polylogue/daemon/user_state_http.py`` (#1290).

The user-state surface — marks, annotations, saved views, recall packs,
reader workspaces — is the daemon HTTP equivalent of the MCP mutation
tools stabilized by #819 / #1224. Before this module existed,
``grep -rl user_state_http tests/`` returned zero dedicated test files
and the closure matrix ``daemon.convergence`` row offered no specific
coverage.

Tests pin three layers:

1. **Dispatch routing.** Each ``/api/user/...`` path resolves to the
   right handler under ``dispatch_get`` / ``dispatch_post`` /
   ``dispatch_delete``, and unknown paths fall through (so the parent
   ``_dispatch_*`` returns 404 rather than 500).
2. **Envelope contract.** List handlers return ``{"items": [...],
   "total": N}`` matching the MCP tool contracts. Mutation handlers
   return the saved row plus a ``created`` flag and the right
   ``HTTPStatus.CREATED`` / ``HTTPStatus.OK`` distinction.
3. **Error paths.** Malformed JSON, missing required fields, and
   unknown ids produce ``HTTPStatus.BAD_REQUEST`` / ``NOT_FOUND``
   envelopes via ``_send_error`` — never a raw exception or 500.

The harness mirrors ``test_daemon_http_contracts.py``: in-process
handler instances, no real socket, real ``Polylogue`` facade against
the ``workspace_env`` SQLite DB so dispatched handlers exercise the
actual storage path.
"""

from __future__ import annotations

import json
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer


# ---------------------------------------------------------------------------
# Handler harness — mirrors tests/unit/daemon/test_daemon_http_contracts.py
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

    Several handlers (``add_mark``, ``save_annotation``) call
    ``_resolve_user_state_target`` which requires the session to exist
    in the archive store. Returns the archive session id
    (``origin:native_id``) to address the session in requests.
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
# 1. Dispatch routing — each path routes to the documented handler
# ---------------------------------------------------------------------------


class TestDispatchRouting:
    """``dispatch_get`` / ``dispatch_post`` / ``dispatch_delete`` cover all routes."""

    @pytest.mark.parametrize(
        ("path_segments", "handler_attr"),
        [
            (["marks"], "handle_list_marks"),
            (["annotations"], "handle_list_annotations"),
            (["annotations", "a-1"], "handle_get_annotation"),
            (["saved-views"], "handle_list_saved_views"),
            (["saved-views", "v-1"], "handle_get_saved_view"),
            (["recall-packs"], "handle_list_recall_packs"),
            (["recall-packs", "p-1"], "handle_get_recall_pack"),
            (["workspaces"], "handle_list_workspaces"),
            (["workspaces", "w-1"], "handle_get_workspace"),
        ],
    )
    def test_get_routes_dispatch_to_named_handler(
        self,
        path_segments: list[str],
        handler_attr: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.daemon import user_state_http

        called = MagicMock()
        monkeypatch.setattr(user_state_http, handler_attr, called)

        handler = _make_handler("GET", "/")
        handler._handle_user_state = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda h, *args: h(handler, *args)
        )
        ok = user_state_http.dispatch_get(handler, path_segments, {})
        assert ok is True
        called.assert_called_once()

    @pytest.mark.parametrize(
        ("path_segments", "handler_attr"),
        [
            (["marks"], "handle_create_mark"),
            (["annotations"], "handle_save_annotation"),
            (["saved-views"], "handle_save_view"),
            (["recall-packs"], "handle_save_recall_pack"),
            (["workspaces"], "handle_save_workspace"),
        ],
    )
    def test_post_routes_dispatch_to_named_handler(
        self,
        path_segments: list[str],
        handler_attr: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.daemon import user_state_http

        called = MagicMock()
        monkeypatch.setattr(user_state_http, handler_attr, called)

        handler = _make_handler("POST", "/")
        handler._handle_user_state = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda h: h(handler)
        )
        ok = user_state_http.dispatch_post(handler, path_segments)
        assert ok is True
        called.assert_called_once()

    @pytest.mark.parametrize(
        ("path_segments", "handler_attr"),
        [
            (["marks"], "handle_delete_mark"),
            (["annotations", "a-1"], "handle_delete_annotation"),
            (["saved-views", "v-1"], "handle_delete_saved_view"),
            (["recall-packs", "p-1"], "handle_delete_recall_pack"),
            (["workspaces", "w-1"], "handle_delete_workspace"),
        ],
    )
    def test_delete_routes_dispatch_to_named_handler(
        self,
        path_segments: list[str],
        handler_attr: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.daemon import user_state_http

        called = MagicMock()
        monkeypatch.setattr(user_state_http, handler_attr, called)

        handler = _make_handler("DELETE", "/")
        handler._handle_user_state = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda h, *args: h(handler, *args)
        )
        ok = user_state_http.dispatch_delete(handler, path_segments, {})
        assert ok is True
        called.assert_called_once()

    def test_unknown_get_path_falls_through(self) -> None:
        from polylogue.daemon import user_state_http

        handler = _make_handler("GET", "/")
        ok = user_state_http.dispatch_get(handler, ["unknown"], {})
        assert ok is False

    def test_unknown_post_path_falls_through(self) -> None:
        from polylogue.daemon import user_state_http

        handler = _make_handler("POST", "/")
        ok = user_state_http.dispatch_post(handler, ["marks", "extra"])
        assert ok is False

    def test_unknown_delete_path_falls_through(self) -> None:
        from polylogue.daemon import user_state_http

        handler = _make_handler("DELETE", "/")
        ok = user_state_http.dispatch_delete(handler, ["unknown"], {})
        assert ok is False

    def test_empty_id_in_get_path_does_not_route(self) -> None:
        """``/annotations/`` (empty id) must not match ``handle_get_annotation``.

        The dispatch table guards on ``path[1]`` truthiness — without
        that guard, an empty id would silently call the get handler
        with ``''`` and hit the storage layer.
        """
        from polylogue.daemon import user_state_http

        handler = _make_handler("GET", "/")
        ok = user_state_http.dispatch_get(handler, ["annotations", ""], {})
        assert ok is False


# ---------------------------------------------------------------------------
# 2. List handlers — envelope contract
# ---------------------------------------------------------------------------


class TestListEnvelopeContract:
    """List handlers return ``{"items": [...], "total": N}``."""

    @pytest.mark.parametrize(
        "path",
        [
            "/api/user/marks",
            "/api/user/annotations",
            "/api/user/saved-views",
            "/api/user/recall-packs",
            "/api/user/workspaces",
        ],
    )
    def test_empty_list_envelope_shape(
        self,
        workspace_env: dict[str, Path],
        path: str,
    ) -> None:
        handler = _make_handler("GET", path)
        send_error, send_json = _capture(handler)
        handler.do_GET()

        send_error.assert_not_called()
        send_json.assert_called_once()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert isinstance(payload, dict)
        assert set(payload.keys()) == {"items", "total"}
        assert isinstance(payload["items"], list)
        assert payload["items"] == []
        assert payload["total"] == 0


# ---------------------------------------------------------------------------
# 3. Marks — happy path, validation, idempotency
# ---------------------------------------------------------------------------


class TestMarksContract:
    """``handle_create_mark`` + ``handle_list_marks`` + ``handle_delete_mark``."""

    def test_create_then_list_then_delete_roundtrip(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        conv_id = _seed_session("c-mark", workspace_env)

        # CREATE
        body = json.dumps({"session_id": conv_id, "mark_type": "star"}).encode()
        handler = _make_handler("POST", "/api/user/marks", body=body)
        send_error, send_json = _capture(handler)
        handler.do_POST()
        send_error.assert_not_called()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.CREATED
        assert payload["status"] == "ok"
        assert payload["operation"] == "mark.add"
        assert payload["affected_count"] == 1
        assert payload["mark_type"] == "star"
        assert payload["session_id"] == conv_id

        # CREATE again — idempotent, unchanged and OK (not CREATED)
        handler = _make_handler("POST", "/api/user/marks", body=body)
        _, send_json = _capture(handler)
        handler.do_POST()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["status"] == "unchanged"
        assert payload["detail"] == "already_present"
        assert payload["affected_count"] == 0

        # LIST — envelope shape, contains the mark
        handler = _make_handler("GET", "/api/user/marks")
        _, send_json = _capture(handler)
        handler.do_GET()
        _, payload = send_json.call_args.args
        assert payload["total"] >= 1
        assert any(m.get("mark_type") == "star" for m in payload["items"])

        # DELETE
        handler = _make_handler(
            "DELETE",
            f"/api/user/marks?session_id={conv_id}&mark_type=star",
        )
        _, send_json = _capture(handler)
        handler.do_DELETE()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["status"] == "deleted"
        assert payload["operation"] == "mark.delete"
        assert payload["affected_count"] == 1
        assert payload["mark_type"] == "star"

    @pytest.mark.parametrize(
        "body",
        [
            {},  # no session_id
            {"session_id": "c-x"},  # no mark_type
            {"session_id": "c-x", "mark_type": "invalid"},  # bad mark_type
            {"session_id": "c-x", "mark_type": "star", "target_type": "nonsense"},
        ],
    )
    def test_create_validates_body(
        self,
        workspace_env: dict[str, Path],
        body: dict[str, object],
    ) -> None:
        encoded = json.dumps(body).encode()
        handler = _make_handler("POST", "/api/user/marks", body=encoded)
        send_error, send_json = _capture(handler)
        handler.do_POST()
        send_error.assert_called_once()
        status, code = send_error.call_args.args
        assert status == HTTPStatus.BAD_REQUEST
        assert code in {"invalid_request", "invalid_target_type"}
        send_json.assert_not_called()

    def test_create_rejects_malformed_json(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        handler = _make_handler("POST", "/api/user/marks", body=b"{not-json")
        send_error, send_json = _capture(handler)
        handler.do_POST()
        send_error.assert_called_once_with(HTTPStatus.BAD_REQUEST, "invalid_request")
        send_json.assert_not_called()

    def test_create_rejects_non_object_json(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """A bare JSON array must be rejected — ``_read_json_body`` requires a dict."""
        handler = _make_handler("POST", "/api/user/marks", body=b"[1,2,3]")
        send_error, send_json = _capture(handler)
        handler.do_POST()
        send_error.assert_called_once_with(HTTPStatus.BAD_REQUEST, "invalid_request")
        send_json.assert_not_called()

    def test_delete_requires_session_id_and_mark_type(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        handler = _make_handler("DELETE", "/api/user/marks")
        send_error, send_json = _capture(handler)
        handler.do_DELETE()
        send_error.assert_called_once_with(HTTPStatus.BAD_REQUEST, "invalid_request")
        send_json.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Annotations — happy path, validation, 404
# ---------------------------------------------------------------------------


class TestAnnotationsContract:
    """``handle_save_annotation`` + ``handle_get_annotation`` + delete + list."""

    def test_save_then_get_then_list_then_delete(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        conv_id = _seed_session("c-ann", workspace_env)

        body = json.dumps(
            {
                "annotation_id": "ann-1",
                "session_id": conv_id,
                "note_text": "first note",
            }
        ).encode()
        handler = _make_handler("POST", "/api/user/annotations", body=body)
        _, send_json = _capture(handler)
        handler.do_POST()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.CREATED
        assert payload["status"] == "ok"
        assert payload["operation"] == "annotation.save"
        assert payload["affected_count"] == 1
        assert payload["resource_type"] == "annotation"
        assert payload["resource_id"] == "ann-1"

        # GET
        handler = _make_handler("GET", "/api/user/annotations/ann-1")
        send_error, send_json = _capture(handler)
        handler.do_GET()
        send_error.assert_not_called()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["annotation_id"] == "ann-1"

        # LIST
        handler = _make_handler("GET", "/api/user/annotations")
        _, send_json = _capture(handler)
        handler.do_GET()
        _, payload = send_json.call_args.args
        assert payload["total"] >= 1

        # Re-save updates the existing resource and remains an affected mutation.
        handler = _make_handler("POST", "/api/user/annotations", body=body)
        _, send_json = _capture(handler)
        handler.do_POST()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["status"] == "ok"
        assert payload["detail"] == "updated"
        assert payload["affected_count"] == 1

        # DELETE
        handler = _make_handler("DELETE", "/api/user/annotations/ann-1")
        _, send_json = _capture(handler)
        handler.do_DELETE()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["status"] == "deleted"
        assert payload["operation"] == "annotation.delete"
        assert payload["affected_count"] == 1
        assert payload["resource_type"] == "annotation"
        assert payload["resource_id"] == "ann-1"

    def test_get_missing_annotation_returns_404(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        handler = _make_handler("GET", "/api/user/annotations/does-not-exist")
        send_error, send_json = _capture(handler)
        handler.do_GET()
        send_error.assert_called_once_with(HTTPStatus.NOT_FOUND, "not_found")
        send_json.assert_not_called()

    @pytest.mark.parametrize(
        "body",
        [
            {},  # no session_id, no note
            {"session_id": "c-x"},  # missing note
            {"session_id": "c-x", "note_text": "  "},  # blank note
            {"session_id": "", "note_text": "x"},  # blank conv id
            {"session_id": "c-x", "note_text": "x", "target_type": "nonsense"},
        ],
    )
    def test_save_validates_body(
        self,
        workspace_env: dict[str, Path],
        body: dict[str, object],
    ) -> None:
        handler = _make_handler("POST", "/api/user/annotations", body=json.dumps(body).encode())
        send_error, send_json = _capture(handler)
        handler.do_POST()
        send_error.assert_called_once()
        status, code = send_error.call_args.args
        assert status == HTTPStatus.BAD_REQUEST
        assert code in {"invalid_request", "invalid_target_type"}
        send_json.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Saved views — happy path, validation, 404
# ---------------------------------------------------------------------------


class TestSavedViewsContract:
    """``handle_save_view`` + ``handle_get_saved_view`` + list + delete."""

    def test_save_then_get_then_list_then_delete(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        body = json.dumps(
            {
                "view_id": "v-1",
                "name": "Recent stars",
                "query": {"limit": 10},
            }
        ).encode()
        handler = _make_handler("POST", "/api/user/saved-views", body=body)
        _, send_json = _capture(handler)
        handler.do_POST()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.CREATED
        assert payload == {
            "status": "ok",
            "affected_count": 1,
            "operation": "saved_view.save",
            "resource_type": "saved_view",
            "resource_id": "v-1",
        }

        # GET
        handler = _make_handler("GET", "/api/user/saved-views/v-1")
        _, send_json = _capture(handler)
        handler.do_GET()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["view_id"] == "v-1"
        assert payload["name"] == "Recent stars"
        assert payload["query"] == {"limit": 10}
        assert isinstance(payload["query_json"], str)

        # LIST envelope
        handler = _make_handler("GET", "/api/user/saved-views")
        _, send_json = _capture(handler)
        handler.do_GET()
        _, payload = send_json.call_args.args
        assert payload["total"] >= 1
        assert all({"view_id", "name", "query", "query_json"} <= set(item.keys()) for item in payload["items"])

        # Re-save updates the existing resource and remains an affected mutation.
        updated_body = json.dumps(
            {
                "view_id": "v-1",
                "name": "Recent stars",
                "query": {"limit": 25},
            }
        ).encode()
        handler = _make_handler("POST", "/api/user/saved-views", body=updated_body)
        _, send_json = _capture(handler)
        handler.do_POST()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["status"] == "ok"
        assert payload["detail"] == "updated"
        assert payload["affected_count"] == 1
        assert payload["resource_type"] == "saved_view"
        assert payload["resource_id"] == "v-1"

        # DELETE
        handler = _make_handler("DELETE", "/api/user/saved-views/v-1")
        _, send_json = _capture(handler)
        handler.do_DELETE()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["status"] == "deleted"
        assert payload["operation"] == "saved_view.delete"
        assert payload["affected_count"] == 1
        assert payload["resource_type"] == "saved_view"
        assert payload["resource_id"] == "v-1"

    def test_get_missing_view_returns_404(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        handler = _make_handler("GET", "/api/user/saved-views/missing")
        send_error, send_json = _capture(handler)
        handler.do_GET()
        send_error.assert_called_once_with(HTTPStatus.NOT_FOUND, "not_found")
        send_json.assert_not_called()

    @pytest.mark.parametrize(
        "body",
        [
            {},  # no name, no query
            {"name": "x"},  # no query
            {"name": "", "query": {}},  # blank name
            {"name": "x", "query": "not-a-dict"},  # query not dict
            {"name": "x", "query": []},  # query not dict
        ],
    )
    def test_save_validates_body(
        self,
        workspace_env: dict[str, Path],
        body: dict[str, object],
    ) -> None:
        handler = _make_handler("POST", "/api/user/saved-views", body=json.dumps(body).encode())
        send_error, send_json = _capture(handler)
        handler.do_POST()
        send_error.assert_called_once_with(HTTPStatus.BAD_REQUEST, "invalid_request")
        send_json.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Recall packs — happy path, validation, 404
# ---------------------------------------------------------------------------


class TestRecallPacksContract:
    """``handle_save_recall_pack`` + ``handle_get_recall_pack`` + list + delete."""

    def test_save_then_get_then_list_then_delete(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        conv_id = _seed_session("c-pack", workspace_env)

        # Recall packs allow item resolution against the seeded session.
        body = json.dumps(
            {
                "pack_id": "rp-1",
                "label": "First pack",
                "payload": {
                    "items": [
                        {"type": "session", "session_id": conv_id},
                    ]
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
            "resource_id": "rp-1",
        }

        # GET
        handler = _make_handler("GET", "/api/user/recall-packs/rp-1")
        _, send_json = _capture(handler)
        handler.do_GET()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["pack_id"] == "rp-1"
        assert isinstance(payload["payload"], dict)
        assert isinstance(payload["session_ids"], list)

        # LIST
        handler = _make_handler("GET", "/api/user/recall-packs")
        _, send_json = _capture(handler)
        handler.do_GET()
        _, payload = send_json.call_args.args
        assert payload["total"] >= 1

        # DELETE
        handler = _make_handler("DELETE", "/api/user/recall-packs/rp-1")
        _, send_json = _capture(handler)
        handler.do_DELETE()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["status"] == "deleted"
        assert payload["operation"] == "recall_pack.delete"
        assert payload["affected_count"] == 1
        assert payload["resource_type"] == "recall_pack"
        assert payload["resource_id"] == "rp-1"

    def test_get_missing_pack_returns_404(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        handler = _make_handler("GET", "/api/user/recall-packs/missing")
        send_error, send_json = _capture(handler)
        handler.do_GET()
        send_error.assert_called_once_with(HTTPStatus.NOT_FOUND, "not_found")
        send_json.assert_not_called()

    @pytest.mark.parametrize(
        "body",
        [
            {},  # no pack_id, no label
            {"pack_id": "p", "label": ""},  # blank label
            {"pack_id": "", "label": "l"},  # blank pack_id
            {"pack_id": "p", "label": "l", "payload": "not-a-dict"},
            {"pack_id": "p", "label": "l", "payload": {}},  # missing items list
            {"pack_id": "p", "label": "l", "payload": {"items": "not-a-list"}},
            {"pack_id": "p", "label": "l", "payload": {"items": ["not-a-dict"]}},
        ],
    )
    def test_save_validates_body(
        self,
        workspace_env: dict[str, Path],
        body: dict[str, object],
    ) -> None:
        handler = _make_handler("POST", "/api/user/recall-packs", body=json.dumps(body).encode())
        send_error, send_json = _capture(handler)
        handler.do_POST()
        send_error.assert_called_once_with(HTTPStatus.BAD_REQUEST, "invalid_request")
        send_json.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Workspaces — happy path, validation, 404
# ---------------------------------------------------------------------------


class TestWorkspacesContract:
    """``handle_save_workspace`` + ``handle_get_workspace`` + list + delete."""

    def test_save_then_get_then_list_then_delete(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        body = json.dumps(
            {
                "workspace_id": "ws-1",
                "name": "Daily work",
                "mode": "tabs",
                "open_targets": [{"target_type": "session", "target_id": "c-1"}],
                "layout": {"split": "horizontal"},
                "active_target": {"target_type": "session", "target_id": "c-1"},
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
            "resource_id": "ws-1",
        }

        # Resave (no changes) — unchanged
        handler = _make_handler("POST", "/api/user/workspaces", body=body)
        _, send_json = _capture(handler)
        handler.do_POST()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["status"] == "ok"
        assert payload["detail"] == "updated"
        assert payload["affected_count"] == 1

        # GET
        handler = _make_handler("GET", "/api/user/workspaces/ws-1")
        _, send_json = _capture(handler)
        handler.do_GET()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["workspace_id"] == "ws-1"
        assert payload["mode"] == "tabs"
        assert isinstance(payload["open_targets"], list)
        assert isinstance(payload["layout"], dict)
        assert isinstance(payload["active_target"], dict)

        # LIST
        handler = _make_handler("GET", "/api/user/workspaces")
        _, send_json = _capture(handler)
        handler.do_GET()
        _, payload = send_json.call_args.args
        assert payload["total"] >= 1

        # DELETE
        handler = _make_handler("DELETE", "/api/user/workspaces/ws-1")
        _, send_json = _capture(handler)
        handler.do_DELETE()
        status, payload = send_json.call_args.args
        assert status == HTTPStatus.OK
        assert payload["status"] == "deleted"
        assert payload["operation"] == "workspace.delete"
        assert payload["affected_count"] == 1
        assert payload["resource_type"] == "workspace"
        assert payload["resource_id"] == "ws-1"

    def test_get_missing_workspace_returns_404(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        handler = _make_handler("GET", "/api/user/workspaces/missing")
        send_error, send_json = _capture(handler)
        handler.do_GET()
        send_error.assert_called_once_with(HTTPStatus.NOT_FOUND, "not_found")
        send_json.assert_not_called()

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"workspace_id": "w", "name": ""},  # blank name
            {"workspace_id": "", "name": "n"},  # blank id
            {"workspace_id": "w", "name": "n", "mode": "weird"},  # bad mode
            {"workspace_id": "w", "name": "n", "mode": "tabs", "open_targets": "not-a-list"},
            {"workspace_id": "w", "name": "n", "mode": "tabs", "open_targets": ["str"]},
            {"workspace_id": "w", "name": "n", "mode": "tabs", "layout": "no"},
            {"workspace_id": "w", "name": "n", "mode": "tabs", "active_target": []},
        ],
    )
    def test_save_validates_body(
        self,
        workspace_env: dict[str, Path],
        body: dict[str, object],
    ) -> None:
        handler = _make_handler("POST", "/api/user/workspaces", body=json.dumps(body).encode())
        send_error, send_json = _capture(handler)
        handler.do_POST()
        send_error.assert_called_once_with(HTTPStatus.BAD_REQUEST, "invalid_request")
        send_json.assert_not_called()


# ---------------------------------------------------------------------------
# 8. Payload normalization helpers — JSON decode failures fall back safely
# ---------------------------------------------------------------------------


class TestPayloadNormalization:
    """Internal helpers tolerate corrupt JSON in DB rows by defaulting."""

    def test_saved_view_payload_handles_corrupt_query_json(self) -> None:
        from polylogue.daemon import user_state_http

        row = {
            "view_id": "v",
            "name": "n",
            "query_json": "{not-json",
            "created_at": "2026-01-01T00:00:00Z",
        }
        payload = cast(dict[str, Any], user_state_http._saved_view_payload(row))
        assert payload["view_id"] == "v"
        assert payload["query"] is None
        assert payload["query_json"] == "{not-json"

    def test_recall_pack_payload_handles_corrupt_json(self) -> None:
        from polylogue.daemon import user_state_http

        row = {
            "pack_id": "p",
            "label": "l",
            "session_ids_json": "broken",
            "payload_json": "broken",
            "created_at": "2026-01-01T00:00:00Z",
        }
        payload = cast(dict[str, Any], user_state_http._recall_pack_payload(row))
        assert payload["session_ids"] == []
        assert payload["payload"] == {}

    def test_workspace_payload_handles_corrupt_json(self) -> None:
        from polylogue.daemon import user_state_http

        row = {
            "workspace_id": "w",
            "name": "n",
            "mode": "tabs",
            "open_targets_json": "broken",
            "layout_json": "broken",
            "active_target_json": "broken",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
        payload = cast(dict[str, Any], user_state_http._workspace_payload(row))
        assert payload["open_targets"] == []
        assert payload["layout"] == {}
        assert payload["active_target"] == {}

    def test_workspace_payload_coerces_wrong_types_to_defaults(self) -> None:
        """Even with valid JSON, wrong shapes (str instead of list/dict)
        must be coerced — the daemon must never echo a misshapen value
        to the reader."""
        from polylogue.daemon import user_state_http

        row = {
            "workspace_id": "w",
            "name": "n",
            "mode": "tabs",
            "open_targets_json": '"not-a-list"',
            "layout_json": "[1,2,3]",
            "active_target_json": "[]",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
        payload = cast(dict[str, Any], user_state_http._workspace_payload(row))
        assert payload["open_targets"] == []
        assert payload["layout"] == {}
        assert payload["active_target"] == {}

    def test_default_id_helpers_are_deterministic(self) -> None:
        """``_default_saved_view_id`` / ``_default_annotation_id`` are
        deterministic hashes — same inputs always produce the same id.
        This is what makes idempotent ``POST`` without an explicit id
        possible.
        """
        from polylogue.daemon import user_state_http

        archive = user_state_http._default_saved_view_id("name", '{"a":1}')
        v2 = user_state_http._default_saved_view_id("name", '{"a":1}')
        assert archive == v2
        assert archive.startswith("view-")

        a1 = user_state_http._default_annotation_id("session", "c", "note")
        a2 = user_state_http._default_annotation_id("session", "c", "note")
        assert a1 == a2
        assert a1.startswith("annotation-")

        # Different inputs → different ids.
        assert archive != user_state_http._default_saved_view_id("other", '{"a":1}')
        assert a1 != user_state_http._default_annotation_id("session", "c", "different")

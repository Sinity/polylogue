"""Production HTTP endpoint behavior for the declared user-overlay routes."""

from __future__ import annotations

import json
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from polylogue.daemon.http import DaemonAPIHandler
from polylogue.daemon.route_families.user_overlay import ROUTES


def _handler(method: str, path: str, *, body: dict[str, object] | bytes | None = None) -> DaemonAPIHandler:
    raw = body if isinstance(body, bytes) else json.dumps(body or {}).encode()
    handler = object.__new__(DaemonAPIHandler)
    handler.server = type("OverlayServer", (), {"archive_root": Path("/synthetic/archive")})()
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.command = method
    handler.requestline = f"{method} {path} HTTP/1.1"
    headers = Message()
    headers["Content-Length"] = str(len(raw))
    headers["Host"] = "127.0.0.1:8766"
    handler.headers = headers
    handler.rfile = BytesIO(raw)
    handler.wfile = BytesIO()
    handler._check_host_admission = lambda **_kwargs: True  # type: ignore[method-assign]
    handler._reject_credential_query = lambda: False  # type: ignore[method-assign]
    handler._check_auth = lambda *_args, **_kwargs: True  # type: ignore[method-assign]
    handler._check_cross_origin = lambda **_kwargs: True  # type: ignore[method-assign]
    handler._send_json = MagicMock()  # type: ignore[method-assign]
    handler._send_error = MagicMock()  # type: ignore[method-assign]
    return handler


def _sample_body(path: str) -> dict[str, object]:
    if path.endswith("/marks"):
        return {"session_id": "codex:one", "mark_type": "star"}
    if path.endswith("/annotations"):
        return {"session_id": "codex:one", "note_text": "note", "annotation_id": "a1"}
    if path.endswith("/saved-views"):
        return {"view_id": "v1", "name": "Recent", "query": {"limit": 10}}
    if path.endswith("/recall-packs"):
        return {"pack_id": "p1", "label": "Pack", "payload": {"items": []}}
    if path.endswith("/workspaces"):
        return {"workspace_id": "w1", "name": "Work", "mode": "tabs"}
    return {}


@pytest.mark.parametrize("route", ROUTES, ids=lambda route: f"{route.method} {route.path}")
def test_every_overlay_endpoint_invokes_its_declared_operation(route: Any) -> None:
    path = route.path.replace(":id", "item-1")
    if route.method == "DELETE" and path.endswith("/marks"):
        path += "?session_id=codex:one&mark_type=star"
    handler = _handler(route.method, path, body=_sample_body(route.path) if route.method == "POST" else None)
    calls: list[tuple[str, dict[str, object]]] = []

    def execute(request: Any) -> dict[str, object]:
        calls.append((request.operation, request.payload))
        if route.method == "POST":
            result: dict[str, object] = {
                "outcome": "completed",
                "result": {
                    "status": "ok",
                    "affected_count": 1,
                    "operation": route.domain_operation.removeprefix("user."),
                },
            }
        elif route.method == "DELETE":
            result = {
                "outcome": "completed",
                "result": {
                    "status": "deleted",
                    "affected_count": 1,
                    "operation": route.domain_operation.removeprefix("user."),
                },
            }
        elif route.path.endswith("/:id"):
            result = {"found": True, "item": {"id": "item-1"}}
        elif route.path == "/api/assertions":
            result = {"items": [], "total": 0, "limit": 20, "statuses": ["active", "candidate"]}
        else:
            result = {"items": [], "total": 0}
        if route.method != "GET":
            return {
                "outcome": "completed",
                "result": {"outcome": "completed", "effect": "committed", "result": result["result"]},
            }
        return {"outcome": "completed", "result": result}

    handler._execute_daemon_operation = execute  # type: ignore[method-assign]
    getattr(handler, f"do_{route.method}")()
    cast(MagicMock, handler._send_error).assert_not_called()
    cast(MagicMock, handler._send_json).assert_called_once()
    assert calls[0][0] == route.domain_operation
    status, payload = cast(MagicMock, handler._send_json).call_args.args
    assert status == (HTTPStatus.CREATED if route.method == "POST" else HTTPStatus.OK)
    assert isinstance(payload, dict)
    if route.method == "POST":
        assert payload["status"] == "ok"
    elif route.method == "DELETE":
        assert payload["status"] == "deleted"


@pytest.mark.parametrize(
    "path",
    [
        "/api/user/annotations/missing",
        "/api/user/saved-views/missing",
        "/api/user/recall-packs/missing",
        "/api/user/workspaces/missing",
    ],
)
def test_absent_overlay_resource_returns_404(path: str) -> None:
    handler = _handler("GET", path)
    handler.__dict__["_execute_daemon_operation"] = lambda _request: {
        "outcome": "completed",
        "result": {"found": False, "item": None},
    }
    handler.do_GET()
    cast(MagicMock, handler._send_error).assert_called_once_with(HTTPStatus.NOT_FOUND, "not_found")
    cast(MagicMock, handler._send_json).assert_not_called()


@pytest.mark.parametrize(
    "path,body,code",
    [
        ("/api/user/marks", {}, "invalid_request"),
        ("/api/user/marks", {"session_id": "s", "mark_type": "unknown"}, "invalid_request"),
        ("/api/user/annotations", {"session_id": "s", "note_text": " "}, "invalid_request"),
        ("/api/user/saved-views", {"name": "", "query": {}}, "invalid_request"),
        ("/api/user/recall-packs", {"pack_id": "p", "label": "P", "payload": {}}, "invalid_request"),
        ("/api/user/workspaces", {"workspace_id": "w", "name": "W", "mode": "invalid"}, "invalid_request"),
    ],
)
def test_invalid_overlay_body_is_refused_before_operation(path: str, body: dict[str, object], code: str) -> None:
    handler = _handler("POST", path, body=body)
    operation = MagicMock()
    handler._execute_daemon_operation = operation  # type: ignore[method-assign]
    handler.do_POST()
    cast(MagicMock, handler._send_error).assert_called_once_with(HTTPStatus.BAD_REQUEST, code)
    operation.assert_not_called()

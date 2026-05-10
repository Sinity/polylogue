"""Verify maintenance HTTP API endpoints exist and are reachable.

Uses the DaemonAPIHandler class itself (unit-style), not a live server.
"""

from __future__ import annotations

import json
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer


class MockServer:
    auth_token = None
    api_host = "127.0.0.1"


class MockHeaders:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self._headers = headers or {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._headers.get(key, default)


def _make_handler(
    path: str,
    body: dict[str, object] | None = None,
    content_length: int | None = None,
) -> DaemonAPIHandler:
    """Build a DaemonAPIHandler with mocked request attributes."""
    from polylogue.daemon.http import DaemonAPIHandler

    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    # Stand-ins: BaseHTTPRequestHandler typing wants the real server/Message,
    # but the routes under test never touch fields we don't simulate here.
    handler.server = cast("DaemonAPIHTTPServer", MockServer())
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.command = "POST"
    handler.requestline = f"POST {path} HTTP/1.1"
    handler.headers = cast("Message[str, str]", MockHeaders())

    if body is not None:
        raw = json.dumps(body).encode("utf-8")
        cl = content_length if content_length is not None else len(raw)
        cast(MockHeaders, handler.headers)._headers["Content-Length"] = str(cl)
        handler.rfile = BytesIO(raw)
    else:
        cast(MockHeaders, handler.headers)._headers["Content-Length"] = "0"
        handler.rfile = BytesIO(b"")

    return handler


class TestMaintenanceAPIRoutes:
    """Route dispatch: POST /api/maintenance/plan and /api/maintenance/run."""

    def test_plan_route_dispatched(self) -> None:
        """POST /api/maintenance/plan calls _handle_maintenance_plan."""
        handler = _make_handler("/api/maintenance/plan", body={"targets": []})
        with patch.object(handler, "_handle_maintenance_plan") as mock:
            handler.do_POST()
            mock.assert_called_once()

    def test_run_route_dispatched(self) -> None:
        """POST /api/maintenance/run calls _handle_maintenance_run."""
        handler = _make_handler("/api/maintenance/run", body={"targets": [], "dry_run": True})
        with patch.object(handler, "_handle_maintenance_run") as mock:
            handler.do_POST()
            mock.assert_called_once()

    def test_unknown_maintenance_route_404(self) -> None:
        """POST /api/maintenance/status returns 404."""
        handler = _make_handler("/api/maintenance/status/x")
        with patch.object(handler, "_send_error") as mock:
            handler.do_POST()
            mock.assert_called_once_with(HTTPStatus.NOT_FOUND, "not_found")

    def test_plan_returns_backfill_operation_dict(self) -> None:
        """_handle_maintenance_plan returns a BackfillOperation as JSON."""
        handler = _make_handler("/api/maintenance/plan", body={"targets": ["session_insights"]})
        with patch.object(handler, "_send_json") as mock:
            with patch("polylogue.maintenance.planner.preview_backfill") as mock_preview:
                from polylogue.maintenance.planner import BackfillKind, BackfillOperation, BackfillStatus

                fake_op = BackfillOperation(
                    operation_id="test-001",
                    kind=BackfillKind.BACKFILL,
                    targets=("session_insights",),
                    status=BackfillStatus.PENDING,
                    affected_rows=10,
                    estimated_time_s=0.2,
                )
                mock_preview.return_value = fake_op
                handler._handle_maintenance_plan()
                mock.assert_called_once()
                call_args = mock.call_args[0]
                assert call_args[0] == HTTPStatus.OK
                assert call_args[1]["operation_id"] == "test-001"
                assert call_args[1]["targets"] == ["session_insights"]

    def test_run_returns_backfill_operation_dict(self) -> None:
        """_handle_maintenance_run returns a BackfillOperation as JSON."""
        handler = _make_handler("/api/maintenance/run", body={"targets": ["fts_repair"], "dry_run": False})
        with patch.object(handler, "_send_json") as mock:
            with patch("polylogue.maintenance.planner.execute_backfill") as mock_exec:
                from polylogue.maintenance.planner import BackfillKind, BackfillOperation, BackfillStatus

                fake_op = BackfillOperation(
                    operation_id="test-002",
                    kind=BackfillKind.BACKFILL,
                    targets=("fts_repair",),
                    status=BackfillStatus.COMPLETED,
                    affected_rows=42,
                    started_at="2026-01-01T00:00:00+00:00",
                    completed_at="2026-01-01T00:00:01+00:00",
                )
                mock_exec.return_value = fake_op
                handler._handle_maintenance_run()
                mock.assert_called_once()
                call_args = mock.call_args[0]
                assert call_args[0] == HTTPStatus.OK
                assert call_args[1]["operation_id"] == "test-002"
                assert call_args[1]["status"] == "completed"

    def test_plan_invalid_json_400(self) -> None:
        """_handle_maintenance_plan returns 400 on invalid JSON body."""
        handler = _make_handler("/api/maintenance/plan")
        handler.rfile = BytesIO(b"not-json")
        cast(MockHeaders, handler.headers)._headers["Content-Length"] = str(len(b"not-json"))
        with patch.object(handler, "_send_error") as mock:
            with patch.object(handler, "_send_json"):
                handler._handle_maintenance_plan()
                mock.assert_called_once_with(HTTPStatus.BAD_REQUEST, "invalid_request")

    def test_run_invalid_json_400(self) -> None:
        """_handle_maintenance_run returns 400 on invalid JSON body."""
        handler = _make_handler("/api/maintenance/run")
        handler.rfile = BytesIO(b"{broken")
        cast(MockHeaders, handler.headers)._headers["Content-Length"] = str(len(b"{broken"))
        with patch.object(handler, "_send_error") as mock:
            with patch.object(handler, "_send_json"):
                handler._handle_maintenance_run()
                mock.assert_called_once_with(HTTPStatus.BAD_REQUEST, "invalid_request")

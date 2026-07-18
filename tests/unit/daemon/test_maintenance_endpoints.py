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
    auth_token: str | None = None
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

    def test_rebuild_index_route_dispatched(self) -> None:
        handler = _make_handler("/api/maintenance/rebuild-index", body={})
        with patch.object(handler, "_handle_rebuild_index") as mock:
            handler.do_POST()
            mock.assert_called_once()

    def test_unknown_maintenance_post_route_404(self) -> None:
        """POST /api/maintenance/status returns 404 — status is GET-only."""
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
                    kind=BackfillKind.DERIVED_REBUILD,
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
                    kind=BackfillKind.INDEX_REPAIR,
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

    def test_rebuild_index_runs_the_typed_service_inside_the_route_executor(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from polylogue.maintenance.rebuild_index import RebuildIndexReceipt

        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
        handler = _make_handler("/api/maintenance/rebuild-index", body={"promote": False, "raw_ids": ["raw-1"]})
        receipt = RebuildIndexReceipt(
            archive_root=str(tmp_path),
            raw_session_count=1,
            selected_raw_count=1,
            skipped_by_blob_limit_count=0,
            status="replayed",
            materialized=True,
            materialization={},
            generation={"generation_id": "candidate-1", "active": False},
            readiness={"checked": True, "blocked_surface_count": 0},
            replay={"classified_full_count": 1, "replayed_logical_source_count": 1, "quarantined_raw_count": 0},
        )
        handler.server.write_bridge = type(
            "Bridge",
            (),
            {"run_sync": lambda _self, _actor, function, *args: function(*args)},
        )()
        with patch(
            "polylogue.maintenance.rebuild_index.rebuild_index_from_source_sync", return_value=receipt
        ) as rebuild:
            with patch.object(handler, "_send_json") as send:
                handler._handle_rebuild_index()
        request = rebuild.call_args.args[0]
        assert request.raw_ids == ("raw-1",)
        assert request.promote is False
        assert send.call_args.args == (HTTPStatus.OK, receipt.to_dict())


class TestMaintenanceRegistryEndpoints:
    """GET /api/maintenance/status/<op_id> and /api/maintenance/operations (#1197)."""

    def test_status_route_dispatched(self) -> None:
        """GET /api/maintenance/status/<op_id> routes to _handle_maintenance_status."""
        handler = _make_handler("/api/maintenance/status/op-1")
        with patch.object(handler, "_handle_maintenance_status") as mock:
            handler._dispatch_get(["api", "maintenance", "status", "op-1"], {})
            mock.assert_called_once_with("op-1")

    def test_operations_route_dispatched(self) -> None:
        """GET /api/maintenance/operations routes to _handle_maintenance_operations."""
        handler = _make_handler("/api/maintenance/operations")
        with patch.object(handler, "_handle_maintenance_operations") as mock:
            handler._dispatch_get(["api", "maintenance", "operations"], {})
            mock.assert_called_once_with()

    def test_status_not_found_returns_404(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """A missing op-id returns 404."""
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
        handler = _make_handler("/api/maintenance/status/missing")
        with patch.object(handler, "_send_error") as mock_err:
            handler._handle_maintenance_status("missing")
            mock_err.assert_called_once_with(HTTPStatus.NOT_FOUND, "not_found")

    def test_status_returns_envelope_with_metadata(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """A persisted op returns the shared envelope plus updated_at / state_path."""
        from polylogue.config import Config
        from polylogue.core.json import dumps as json_dumps
        from polylogue.maintenance.planner import (
            BackfillKind,
            BackfillOperation,
            BackfillStatus,
            MaintenanceScope,
        )
        from polylogue.maintenance.replay import state_path_for

        archive_root_path = tmp_path / "archive"
        archive_root_path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root_path))
        config = Config(archive_root=archive_root_path, render_root=tmp_path / "render", sources=[])
        op = BackfillOperation(
            operation_id="op-h1",
            kind=BackfillKind.DERIVED_REBUILD,
            targets=("session_insights",),
            status=BackfillStatus.RUNNING,
            scope=MaintenanceScope(targets=("session_insights",)),
        )
        path = state_path_for(config, "op-h1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json_dumps(
                {
                    "operation_id": "op-h1",
                    "targets": ["session_insights"],
                    "cursor": "target:0",
                    "started_at": "2026-05-17T00:00:00+00:00",
                    "updated_at": "2026-05-17T00:00:01+00:00",
                    "dry_run": False,
                    "repaired_count": 0,
                    "failure_count": 0,
                    "results": [],
                    "operation": op.to_dict(),
                }
            )
        )

        handler = _make_handler("/api/maintenance/status/op-h1")
        with patch.object(handler, "_send_json") as mock_json:
            handler._handle_maintenance_status("op-h1")
            mock_json.assert_called_once()
            body = mock_json.call_args[0][1]
            assert body["envelope"]["operation_id"] == "op-h1", f"unexpected body: {body}"
            assert body["envelope"]["status"] == "running"
            assert body["updated_at"] == "2026-05-17T00:00:01+00:00"
            assert body["state_path"].endswith("op-h1.json")

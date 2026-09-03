"""Production UDS operation route contracts."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from http import HTTPStatus
from pathlib import Path

import pytest

from polylogue.cli.daemon_client import DaemonClient
from polylogue.daemon.http import DaemonAPIHandler
from polylogue.daemon.uds import DaemonAPIUnixHTTPServer
from polylogue.operations.daemon_protocol import DAEMON_OPERATION_PROTOCOL, DaemonOperationRequest


def test_one_uds_operation_request_returns_typed_result_without_health_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AF_UNIX paths are capped at 108 bytes; the managed TMPDIR is far longer.
    runtime = Path(tempfile.mkdtemp(prefix="plg-operation-test-", dir="/tmp"))
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "index.db").write_bytes(b"index")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))

    class Handler(DaemonAPIHandler):
        def _handle_health(self) -> None:
            raise AssertionError("operation path must not issue a health probe")

        def _handle_cli_query(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send_json(HTTPStatus.OK, {"items": [], "total": body["params"].get("limit", 0)})

        def _handle_query_units(self, params: dict[str, list[str]]) -> None:
            self._send_json(HTTPStatus.OK, {"items": [], "expression": params.get("expression", [""])[0]})

    socket_path = runtime / "daemon.sock"
    server = DaemonAPIUnixHTTPServer(socket_path, Handler)
    server.auth_token = ""
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        envelope = DaemonClient(socket_path, timeout_s=2).operation(
            "cli.query",
            {"params": {"limit": 7}},
            archive_root=str(archive),
        )
        units = DaemonClient(socket_path, timeout_s=2).operation(
            "query.units",
            {"params": {"expression": "origin:codex", "limit": 3}},
            archive_root=str(archive),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        shutil.rmtree(runtime, ignore_errors=True)

    assert envelope is not None
    assert envelope["protocol"] == "polylogue.daemon-operation/v1"
    assert envelope["readiness"]["ready"] is True
    assert envelope["authority"]["writes"] == "daemon-owned"
    assert envelope["progress"]["state"] == "complete"
    assert envelope["result"] == {"items": [], "total": 7}
    assert units is not None
    assert units["result"] == {"items": [], "expression": "origin:codex"}


def test_operation_route_bounds_the_serialized_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = Path(tempfile.mkdtemp(prefix="plg-operation-limit-"))
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "index.db").write_bytes(b"index")
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setattr("polylogue.operations.daemon_protocol.MAX_OPERATION_RESULT_BYTES", 256)

    class Handler(DaemonAPIHandler):
        def _handle_cli_query(self) -> None:
            self._send_json(HTTPStatus.OK, {"items": ["x" * 512]})

    socket_path = runtime / "daemon.sock"
    server = DaemonAPIUnixHTTPServer(socket_path, Handler)
    server.auth_token = ""
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request = DaemonOperationRequest(
        operation="cli.query",
        payload={},
        request_id="oversized-response",
    )
    try:
        response = DaemonClient(socket_path, timeout_s=2).request_json(
            "POST",
            "/api/operation",
            request.to_dict(),
            accepted_statuses=frozenset({413}),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        shutil.rmtree(runtime, ignore_errors=True)

    assert response is not None
    assert response["protocol"] == DAEMON_OPERATION_PROTOCOL
    assert response["error"] == {
        "code": "result_too_large",
        "detail": "operation result exceeds the bounded result size",
    }

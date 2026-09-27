"""``POST /api/demo/augment`` submits the declared demo-augment operation.

The route used to apply the demo writes itself through the write bridge while
``maintenance.demo.augment`` held a second copy of the same effect for the
CLI. It now only validates the body and submits the declared operation, whose
handler is the one executor (polylogue-t46).
"""

from __future__ import annotations

import json
from http import HTTPStatus
from pathlib import Path
from typing import Any

import pytest

from polylogue.operations.daemon_protocol import DaemonOperationRequest
from tests.infra.daemon_http_harness import MockDaemonServer, capture_responses, make_daemon_handler


class _Server(MockDaemonServer):
    def __init__(self, archive_root: Path) -> None:
        super().__init__()
        self.archive_root = archive_root


def _handler(tmp_path: Path, body: bytes) -> Any:
    return make_daemon_handler("POST", "/api/demo/augment", body=body, server=_Server(tmp_path))


@pytest.mark.parametrize("with_overlays", [False, True])
def test_demo_augment_route_submits_the_declared_operation(tmp_path: Path, with_overlays: bool) -> None:
    """Both overlay values reach ``maintenance.demo.augment`` unchanged.

    Anti-vacuity: restore the route's direct ``apply_demo_post_ingest_augmentation``
    call and no operation is submitted, so ``submitted`` stays empty.
    """
    handler = _handler(tmp_path, json.dumps({"with_overlays": with_overlays}).encode())
    submitted: list[DaemonOperationRequest] = []

    def execute(request: DaemonOperationRequest) -> dict[str, object]:
        submitted.append(request)
        return {"operation": request.operation, "outcome": "completed", "result": {"augmented": True}}

    handler._execute_daemon_operation = execute
    send_error, send_json = capture_responses(handler)

    handler.do_POST()

    send_error.assert_not_called()
    assert [(request.operation, request.payload) for request in submitted] == [
        ("maintenance.demo.augment", {"with_overlays": with_overlays})
    ]
    assert submitted[0].archive_root == str(tmp_path)
    send_json.assert_called_once_with(HTTPStatus.OK, {"ok": True, "augmented": True, "overlays": with_overlays})


def test_demo_augment_route_reports_a_refused_operation_as_its_envelope(tmp_path: Path) -> None:
    """A refused operation is not reported as a successful augmentation."""
    handler = _handler(tmp_path, b"{}")
    refused = {"operation": "maintenance.demo.augment", "outcome": "rejected", "error": {"code": "busy"}}
    handler._execute_daemon_operation = lambda request: dict(refused)
    sent: list[dict[str, object]] = []
    handler._send_daemon_operation = sent.append
    send_error, send_json = capture_responses(handler)

    handler.do_POST()

    send_error.assert_not_called()
    send_json.assert_not_called()
    assert sent == [refused]


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"with_overlays": "yes"}'])
def test_demo_augment_route_refuses_malformed_bodies_before_submitting(tmp_path: Path, body: bytes) -> None:
    handler = _handler(tmp_path, body)
    submitted: list[object] = []
    handler._execute_daemon_operation = submitted.append
    send_error, _ = capture_responses(handler)

    handler.do_POST()

    send_error.assert_called_once_with(HTTPStatus.BAD_REQUEST, "invalid_request")
    assert submitted == []

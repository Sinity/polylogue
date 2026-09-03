"""Typed daemon operation protocol contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.operations.daemon_protocol import (
    DAEMON_OPERATION_PROTOCOL,
    DaemonOperationRequest,
    archive_identity,
)


def test_operation_request_rejects_untyped_payloads() -> None:
    with pytest.raises(ValueError, match="payload must be an object"):
        DaemonOperationRequest.from_dict({"operation": "status", "payload": []})


def test_operation_request_requires_the_negotiated_protocol() -> None:
    with pytest.raises(ValueError, match="unsupported daemon operation protocol"):
        DaemonOperationRequest.from_dict({"operation": "status", "payload": {}, "protocol": "v0"})

    with pytest.raises(ValueError, match="unsupported daemon operation protocol"):
        DaemonOperationRequest.from_dict({"operation": "status", "payload": {}})


def test_operation_request_requires_a_declared_operation_and_exchange_identity() -> None:
    request = {
        "protocol": DAEMON_OPERATION_PROTOCOL,
        "operation": "status",
        "payload": {},
        "request_id": "request-1",
    }

    parsed = DaemonOperationRequest.from_dict(request)

    assert parsed.operation == "status"
    assert parsed.request_id == "request-1"
    with pytest.raises(ValueError, match="operation is not declared"):
        DaemonOperationRequest.from_dict({**request, "operation": "unknown"})
    with pytest.raises(ValueError, match="request_id must be a non-empty string"):
        DaemonOperationRequest.from_dict({key: value for key, value in request.items() if key != "request_id"})
    with pytest.raises(ValueError, match="index_schema_version must be an integer"):
        DaemonOperationRequest.from_dict({**request, "index_schema_version": True})
    with pytest.raises(ValueError, match="deadline_ms must be a positive integer"):
        DaemonOperationRequest.from_dict({**request, "deadline_ms": True})


def test_archive_identity_contains_readiness_and_generation(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "index.db").write_bytes(b"fixture")
    identity, generation, readiness = archive_identity(archive, schema_version=24, daemon_version="test")

    assert identity["root"] == str(archive)
    assert generation["index_schema_version"] == 24
    assert generation["index_size_bytes"] == 7
    assert readiness == {"state": "ready", "ready": True, "reason": None}
    assert DAEMON_OPERATION_PROTOCOL == "polylogue.daemon-operation/v1"

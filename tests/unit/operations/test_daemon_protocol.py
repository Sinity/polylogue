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

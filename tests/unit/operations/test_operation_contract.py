"""Unit tests for the generic Operation contract and the import instance.

Covers serialization, validation, and the immutability/extra-forbid
guarantees of the new typed scheduling contract introduced in #1247.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import ValidationError

from polylogue.operations import (
    ImportAck,
    ImportRequest,
    OperationAck,
    OperationFollowUp,
    OperationKind,
    OperationRequest,
    OperationStatus,
)
from polylogue.operations.operation_contract import _require_operation_kind


class TestOperationFollowUp:
    """Follow-up hints carry the polling triple for accepted operations."""

    def test_minimal_construction(self) -> None:
        follow_up = OperationFollowUp(status_endpoint="daemon.import_status")
        assert follow_up.status_endpoint == "daemon.import_status"
        assert follow_up.status_token is None
        assert follow_up.poll_after_ms == 0

    def test_full_construction(self) -> None:
        follow_up = OperationFollowUp(
            status_endpoint="/api/operations/op-123",
            status_token="opaque-cursor",
            poll_after_ms=500,
        )
        assert follow_up.status_token == "opaque-cursor"
        assert follow_up.poll_after_ms == 500

    def test_to_dict_is_json_shaped(self) -> None:
        follow_up = OperationFollowUp(
            status_endpoint="daemon.import_status",
            poll_after_ms=250,
        )
        payload = follow_up.to_dict()
        assert payload == {
            "status_endpoint": "daemon.import_status",
            "status_token": None,
            "poll_after_ms": 250,
        }

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            OperationFollowUp(
                status_endpoint="x",
                extra_unknown_field=1,  # type: ignore[call-arg]
            )

    def test_is_frozen(self) -> None:
        follow_up = OperationFollowUp(status_endpoint="x")
        with pytest.raises(ValidationError):
            follow_up.poll_after_ms = 9999


class TestImportRequestValidation:
    """ImportRequest pins kind to IMPORT and enforces explicit fields."""

    def test_minimal_construction(self) -> None:
        req = ImportRequest(
            source_path="/inbox/sessions.jsonl",
            source_name="claude-code-export",
        )
        assert req.source_path == "/inbox/sessions.jsonl"
        assert req.source_name == "claude-code-export"
        assert req.staged_path is None
        assert req.idempotency_key is None
        assert req.operation_kind is OperationKind.IMPORT

    def test_full_construction(self) -> None:
        req = ImportRequest(
            source_path="https://example.com/export.zip",
            source_name="manual-upload",
            staged_path="/var/polylogue/inbox/abc.zip",
            idempotency_key="upload-2026-05-18-abc",
        )
        assert req.staged_path == "/var/polylogue/inbox/abc.zip"
        assert req.idempotency_key == "upload-2026-05-18-abc"

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            ImportRequest(source_path="/x")  # type: ignore[call-arg]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ImportRequest(
                source_path="/x",
                source_name="n",
                provider="claude-code",  # type: ignore[call-arg]
            )

    def test_is_frozen(self) -> None:
        req = ImportRequest(source_path="/x", source_name="n")
        with pytest.raises(ValidationError):
            req.source_path = "/y"

    def test_to_dict_roundtrip(self) -> None:
        req = ImportRequest(
            source_path="/inbox/a.json",
            source_name="export-1",
            staged_path="/staged/a.json",
            idempotency_key="k1",
        )
        payload = req.to_dict()
        assert payload == {
            "idempotency_key": "k1",
            "source_path": "/inbox/a.json",
            "source_name": "export-1",
            "staged_path": "/staged/a.json",
        }
        restored = ImportRequest.model_validate(payload)
        assert restored == req


class TestImportAckFactories:
    """ImportAck factory helpers pin kind=IMPORT and produce the right status."""

    def test_accept(self) -> None:
        follow_up = OperationFollowUp(status_endpoint="daemon.import_status")
        ack = ImportAck.accept_import(
            operation_id="op-123",
            follow_up=follow_up,
            message="queued",
        )
        assert ack.operation_id == "op-123"
        assert ack.kind is OperationKind.IMPORT
        assert ack.status is OperationStatus.ACCEPTED
        assert ack.follow_up == follow_up
        assert ack.error is None
        assert ack.is_accepted() is True

    def test_reject(self) -> None:
        ack = ImportAck.reject_import(
            operation_id="op-bad",
            error="unsupported source type",
        )
        assert ack.status is OperationStatus.REJECTED
        assert ack.error == "unsupported source type"
        assert ack.follow_up is None
        assert ack.is_accepted() is False

    def test_pending(self) -> None:
        follow_up = OperationFollowUp(
            status_endpoint="daemon.import_status",
            poll_after_ms=250,
        )
        ack = ImportAck.pending_import(operation_id="op-1", follow_up=follow_up)
        assert ack.status is OperationStatus.PENDING
        assert ack.is_accepted() is True


class TestOperationAckSerialization:
    """OperationAck reduces to byte-stable JSON for surface parity."""

    def test_to_dict_accepted(self) -> None:
        follow_up = OperationFollowUp(
            status_endpoint="daemon.import_status",
            poll_after_ms=100,
        )
        ack = OperationAck.accepted(
            operation_id="op-1",
            kind=OperationKind.IMPORT,
            follow_up=follow_up,
            message="ok",
        )
        payload = ack.to_dict()
        assert payload == {
            "operation_id": "op-1",
            "kind": "import",
            "status": "accepted",
            "message": "ok",
            "error": None,
            "follow_up": {
                "status_endpoint": "daemon.import_status",
                "status_token": None,
                "poll_after_ms": 100,
            },
        }

    def test_to_dict_rejected_omits_follow_up_payload(self) -> None:
        ack = OperationAck.rejected(
            operation_id="op-2",
            kind=OperationKind.IMPORT,
            error="invalid path",
        )
        payload = ack.to_dict()
        assert payload["status"] == "rejected"
        assert payload["error"] == "invalid path"
        assert payload["follow_up"] is None

    def test_model_validate_roundtrip(self) -> None:
        follow_up = OperationFollowUp(status_endpoint="x")
        ack = OperationAck.accepted(
            operation_id="op-3",
            kind=OperationKind.MAINTENANCE,
            follow_up=follow_up,
        )
        restored = OperationAck.model_validate(ack.to_dict())
        assert restored == ack

    def test_kind_serialized_as_string(self) -> None:
        ack = OperationAck.accepted(
            operation_id="op-4",
            kind=OperationKind.MAINTENANCE,
            follow_up=OperationFollowUp(status_endpoint="x"),
        )
        assert ack.to_dict()["kind"] == "maintenance"

    def test_status_serialized_as_string(self) -> None:
        ack = OperationAck.rejected(
            operation_id="op-5",
            kind=OperationKind.IMPORT,
            error="nope",
        )
        assert ack.to_dict()["status"] == "rejected"

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            OperationAck(
                operation_id="op",
                kind=OperationKind.IMPORT,
                status=OperationStatus.ACCEPTED,
                extra="forbidden",  # type: ignore[call-arg]
            )

    def test_is_frozen(self) -> None:
        ack = OperationAck.accepted(
            operation_id="op-6",
            kind=OperationKind.IMPORT,
            follow_up=OperationFollowUp(status_endpoint="x"),
        )
        with pytest.raises(ValidationError):
            ack.status = OperationStatus.FAILED


class TestRequireOperationKind:
    """Subclasses must declare their operation_kind class variable."""

    def test_import_request_declares_kind(self) -> None:
        assert _require_operation_kind(ImportRequest) is OperationKind.IMPORT

    def test_missing_kind_raises(self) -> None:
        class BadRequest(OperationRequest):
            # Intentionally omits operation_kind.
            value: str = ""

        with pytest.raises(TypeError):
            _require_operation_kind(BadRequest)

    def test_custom_subclass_with_kind(self) -> None:
        class CustomRequest(OperationRequest):
            operation_kind: ClassVar[OperationKind] = OperationKind.MAINTENANCE

            scope: str

        assert _require_operation_kind(CustomRequest) is OperationKind.MAINTENANCE
        req = CustomRequest(scope="all")
        assert req.scope == "all"
        assert req.operation_kind is OperationKind.MAINTENANCE


class TestOperationStatusEnum:
    """OperationStatus is a closed enum — adding values is an explicit change."""

    def test_known_values(self) -> None:
        assert {s.value for s in OperationStatus} == {
            "accepted",
            "rejected",
            "pending",
            "running",
            "completed",
            "failed",
        }

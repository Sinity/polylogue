"""Canonical operation acceptance and validated browser upload metadata."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from polylogue.core.enums import OPERATION_LIFECYCLE_STATUSES, require_operation_lifecycle_status
from polylogue.core.enums import OperationStatus as CoreOperationStatus
from polylogue.operations import ImportRequest, OperationKind, OperationStatus


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
        # operation_kind is a ClassVar (not a pydantic field), so it is
        # deliberately absent from the wire payload — matching the shape
        # from before the polylogue-a7xr.14 generic-base collapse.
        assert payload == {
            "idempotency_key": "k1",
            "source_path": "/inbox/a.json",
            "source_name": "export-1",
            "staged_path": "/staged/a.json",
        }
        restored = ImportRequest.model_validate(payload)
        assert restored == req


class TestOperationStatusEnum:
    """OperationStatus is a closed enum — adding values is an explicit change."""

    def test_known_values(self) -> None:
        assert OperationStatus is CoreOperationStatus
        assert {s.value for s in OperationStatus} == {
            "accepted",
            "rejected",
            "pending",
            "running",
            "completed",
            "failed",
            "interrupted",
        }

    def test_lifecycle_subset_excludes_admission_states(self) -> None:
        assert OPERATION_LIFECYCLE_STATUSES == (
            OperationStatus.RUNNING,
            OperationStatus.COMPLETED,
            OperationStatus.FAILED,
            OperationStatus.INTERRUPTED,
        )
        assert require_operation_lifecycle_status(OperationStatus.INTERRUPTED) is OperationStatus.INTERRUPTED
        with pytest.raises(ValueError, match="not a run lifecycle status"):
            require_operation_lifecycle_status(OperationStatus.ACCEPTED)

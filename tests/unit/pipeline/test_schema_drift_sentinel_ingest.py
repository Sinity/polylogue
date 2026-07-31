"""End-to-end drift classification through ``ingest_record`` (polylogue-da1).

Pins the AC that a drift-shaped record is classified AND still ingests
normally -- the sentinel augments ingest, it never gates it. Mirrors the
schema-resolution-reuse fixture pattern in
``test_ingest_worker_reuses_schema_resolution_and_walks_drift``
(tests/unit/pipeline/test_resilience.py) but drives the classification
result through to ``IngestRecordResult.schema_drift`` instead of only
checking the ``include_drift`` flag passed to ``validate()``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.raw_payload.decode import JSONValue
from polylogue.core.enums import Provider
from polylogue.schemas import ValidationResult
from polylogue.schemas.drift_sentinel import FIELD_CHANGED, NEW_FIELD, UNSEEN_SHAPE
from polylogue.schemas.packages import SchemaResolution
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.runtime import RawSessionRecord

pytestmark = pytest.mark.uses_real_clock(
    "polylogue-da1 format-drift sentinel ingest tests build RawSessionRecord fixtures via the same helper as test_resilience.py; acquired_at/file_mtime are opaque now() metadata with no production timing comparison."
)


def _make_raw_record(raw_id: str, provider: str, content: bytes, path: str = "/exports/test.json") -> RawSessionRecord:
    from polylogue.storage.blob_store import get_blob_store

    blob_store = get_blob_store()
    actual_raw_id, blob_size = blob_store.write_from_bytes(content)
    now = datetime.now(timezone.utc).isoformat()
    return RawSessionRecord(
        raw_id=actual_raw_id,
        source_name=provider,
        source_path=path,
        source_index=None,
        blob_size=blob_size,
        acquired_at=now,
        file_mtime=now,
    )


def _rig_ingest(
    monkeypatch: pytest.MonkeyPatch,
    *,
    resolution_reason: str,
    is_valid: bool,
    drift_warnings: list[str],
) -> None:
    """Wire ingest_record's schema resolution + validation to fixed values.

    Same technique as the existing schema-resolution-reuse test: a fake
    registry (so ``_resolve_plan_schema`` returns a controlled
    ``SchemaResolution``), a fake ``SchemaValidator.for_payload`` (so
    ``validate()`` returns a controlled ``ValidationResult``), and a fake
    ``parse_payload`` (so materialization always succeeds with one session,
    regardless of what the "provider" payload actually contains).
    """
    resolution = SchemaResolution(
        provider="chatgpt",
        package_version="v1",
        element_kind="session_document",
        exact_structure_id="shape-1" if resolution_reason == "exact_structure" else None,
        bundle_scope=None,
        reason=resolution_reason,  # type: ignore[arg-type]
    )

    class _FakeRegistry:
        def resolve_payload(
            self,
            provider: str | Provider,
            payload: JSONValue,
            *,
            source_path: str | None = None,
        ) -> SchemaResolution:
            return resolution

    monkeypatch.setattr("polylogue.pipeline.services.ingest_worker._SCHEMA_REGISTRY", _FakeRegistry())

    class _FakeValidator:
        provider = "chatgpt"

        def validation_samples(self, payload: JSONValue, max_samples: int | None = None) -> list[JSONValue]:
            return [payload]

        def validate(self, _sample: object, *, include_drift: bool | None = None) -> ValidationResult:
            return ValidationResult(
                is_valid=is_valid,
                errors=[] if is_valid else ["root: 'id' is a required property"],
                drift_warnings=list(drift_warnings),
            )

    def _fake_for_payload(
        provider: str | Provider,
        payload: JSONValue,
        *,
        source_path: str | None = None,
        schema_resolution: SchemaResolution | None = None,
        strict: bool = True,
    ) -> _FakeValidator:
        return _FakeValidator()

    def _fake_parse_payload(
        provider: str | Provider,
        payload: JSONValue,
        fallback_id: str,
        _depth: int = 0,
        *,
        schema_resolution: SchemaResolution | None = None,
        source_path: str | None = None,
    ) -> Sequence[ParsedSession]:
        return [
            ParsedSession(
                source_name=Provider.CHATGPT,
                provider_session_id=fallback_id,
                title="Test Session",
                created_at="2023-11-14T22:13:20Z",
                updated_at="2023-11-14T22:13:21Z",
                messages=[ParsedMessage(provider_message_id="msg-1", role=Role.USER, text="hello")],
                attachments=[],
            )
        ]

    monkeypatch.setattr("polylogue.schemas.validator.SchemaValidator.for_payload", _fake_for_payload)
    monkeypatch.setattr("polylogue.sources.dispatch.parse_payload", _fake_parse_payload)


def _basic_chatgpt_payload() -> bytes:
    return json.dumps(
        {
            "id": "conv-1",
            "title": "Test Session",
            "mapping": {},
            "create_time": 1700000000,
            "update_time": 1700000001,
        }
    ).encode()


def test_unseen_shape_record_still_ingests_and_classifies_as_unseen_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A payload resolving to package_default (no real candidate) still
    materializes a session (raw payload/content never blocked) and is
    classified unseen_shape."""
    _rig_ingest(monkeypatch, resolution_reason="package_default", is_valid=True, drift_warnings=[])
    from polylogue.pipeline.services.ingest_worker import ingest_record

    record = _make_raw_record("unseen", "chatgpt", _basic_chatgpt_payload())
    result = ingest_record(record, str(tmp_path / "archive"), "advisory")

    assert result.error is None
    assert result.sessions, "a drift-shaped record must still ingest its session"
    assert result.schema_drift is not None
    assert result.schema_drift.classification == UNSEEN_SHAPE
    assert result.schema_drift.origin == "chatgpt-export"
    assert result.schema_drift.raw_id == record.raw_id


def test_new_optional_field_record_still_ingests_and_classifies_as_benign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A payload with an extra permitted field ingests normally and is
    classified new_field (benign)."""
    _rig_ingest(
        monkeypatch,
        resolution_reason="exact_structure",
        is_valid=True,
        drift_warnings=["Unexpected field: metadata.newThing"],
    )
    from polylogue.pipeline.services.ingest_worker import ingest_record

    record = _make_raw_record("new-field", "chatgpt", _basic_chatgpt_payload())
    result = ingest_record(record, str(tmp_path / "archive"), "advisory")

    assert result.error is None
    assert result.sessions, "a benign drift record must still ingest its session"
    assert result.schema_drift is not None
    assert result.schema_drift.classification == NEW_FIELD
    assert result.schema_drift.unseen_key_signature == "metadata.newThing"


def test_missing_field_record_still_ingests_in_advisory_mode_and_classifies_as_risky(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC: 'ingest never fails or skips on drift'. A record whose validation
    fails (known field disappeared/type changed) is the risky case, but in
    advisory mode (the default live-ingest mode) it still ingests and its
    raw payload is stored -- only STRICT mode fails on validation errors."""
    _rig_ingest(monkeypatch, resolution_reason="exact_structure", is_valid=False, drift_warnings=[])
    from polylogue.pipeline.services.ingest_worker import ingest_record

    record = _make_raw_record("field-changed", "chatgpt", _basic_chatgpt_payload())
    result = ingest_record(record, str(tmp_path / "archive"), "advisory")

    assert result.error is None
    assert result.sessions, "a risky drift record must still ingest under advisory validation"
    assert result.schema_drift is not None
    assert result.schema_drift.classification == FIELD_CHANGED


def test_exact_match_with_no_drift_records_no_schema_drift_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean, exact-structure match with no unknown fields carries no
    drift signal at all -- nothing to record."""
    _rig_ingest(monkeypatch, resolution_reason="exact_structure", is_valid=True, drift_warnings=[])
    from polylogue.pipeline.services.ingest_worker import ingest_record

    record = _make_raw_record("clean", "chatgpt", _basic_chatgpt_payload())
    result = ingest_record(record, str(tmp_path / "archive"), "advisory")

    assert result.error is None
    assert result.sessions
    assert result.schema_drift is None

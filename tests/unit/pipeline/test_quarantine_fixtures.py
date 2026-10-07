"""Quarantine behavior owned by the retained validation service."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.core.enums import Provider, ValidationMode, ValidationStatus
from polylogue.pipeline.services.validation import ValidationService
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.storage_records import admit_raw_record


@pytest.mark.asyncio
async def test_validation_api_persists_malformed_jsonl_quarantine_without_payload_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed content is quarantined by the live validation API without leaking it."""
    private_text = "SYNTHETIC_PRIVATE_PAYLOAD_41A9"
    payload = (
        '{"type":"session_meta","payload":{"id":"synthetic"}}\n'
        '{"type":"message","role":"user","content":"' + private_text + '"}\n'
        '{"type":"message" "role":"assistant"}\n'
    ).encode()
    blob_root = tmp_path / "blob"
    raw_id, blob_size = BlobStore(blob_root).write_from_bytes(payload)
    acquired_at = "2026-01-01T00:00:00+00:00"
    record = RawSessionRecord(
        raw_id=raw_id,
        source_name="codex",
        source_path="synthetic/session.jsonl",
        canonical_source_path="synthetic/session.jsonl",
        payload_provider=Provider.CODEX,
        source_index=None,
        blob_size=blob_size,
        acquired_at=acquired_at,
        file_mtime=acquired_at,
    )
    backend = SQLiteBackend(db_path=tmp_path / "validation.db")
    service = ValidationService(backend)
    monkeypatch.setattr(service, "_schema_validation_mode", lambda: ValidationMode.STRICT)
    monkeypatch.setattr("polylogue.pipeline.services.validation_flow.blob_store_root", lambda: blob_root)
    try:
        await admit_raw_record(backend, record)
        result = await service.validate_raw_ids(raw_ids=[raw_id])
        stored = await backend.get_raw_session(raw_id)

        assert len(result.records) == 1
        validation = result.records[0]
        assert validation.validation_status is ValidationStatus.FAILED
        assert validation.parse_error is not None
        assert "Malformed JSONL lines" in validation.parse_error
        assert private_text not in validation.parse_error
        assert stored is not None
        assert stored.validation_status is ValidationStatus.FAILED
        assert stored.parsed_at is None
        assert stored.parse_error == validation.parse_error
    finally:
        await backend.close()

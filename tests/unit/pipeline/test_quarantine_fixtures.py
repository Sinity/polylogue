"""Quarantine behavior owned by the retained validation service."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider, ValidationMode, ValidationStatus
from polylogue.pipeline.services.validation import ValidationService
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.retained_jsonl import prepared_source_fixture, retained_raw_fixture
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


@pytest.mark.parametrize("mode", ["off", "advisory", "strict"])
@pytest.mark.parametrize(
    ("provider", "suffix"),
    [
        (Provider.CODEX, "jsonl"),
        (Provider.CLAUDE_CODE, "jsonl"),
        (Provider.CHATGPT, "json"),
        (Provider.GEMINI, "json"),
    ],
)
def test_zero_length_retained_raw_is_terminal_for_every_provider_and_validation_mode(
    tmp_path: Path, mode: str, provider: Provider, suffix: str
) -> None:
    """Empty acquired bytes are terminal decoder evidence, regardless of schema mode or provider."""
    from polylogue.core.raw_failure_evidence import RAW_FAILURE_VALIDATION_FAILURE_KINDS
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    blob_hash, _size = BlobStore(root / "blob").write_from_bytes(b"")
    with retained_raw_fixture(
        root=root,
        provider=provider,
        blob_hash=blob_hash,
        source_path=str(tmp_path / "exports" / f"empty.{suffix}"),
    ) as (_source_read, raw_id):
        pass

    async def replay() -> tuple[object, list[str]]:
        refused: list[str] = []
        async with prepared_live_convergence_owner(root, validation_mode=ValidationMode.from_string(mode)) as owner:
            receipts = (
                await owner.replay_retained_raw_ids(
                    (raw_id,), on_terminal_refusal=lambda _keys, refusal: refused.append(refusal.raw_id)
                )
            ).require_complete()
            return receipts, refused

    receipts, refused = asyncio.run(replay())

    assert receipts == (), (provider, mode, receipts)
    assert refused == [raw_id], (provider, mode, refused)
    with prepared_source_fixture(root) as source_read:
        assert source_read.raw_parser_census_is_current(raw_id)
        refusal = source_read.raw_terminal_decode_refusal(raw_id)
        assert refusal is not None
        assert refusal.kind.value in RAW_FAILURE_VALIDATION_FAILURE_KINDS
        assert str(refusal).strip()
    with sqlite3.connect(root / "source.db") as source:
        state = source.execute(
            "SELECT parsed_at_ms, parse_error FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone()
        artifact_kinds = {
            str(row[0]) for row in source.execute("SELECT artifact_kind FROM raw_artifacts WHERE raw_id=?", (raw_id,))
        }
    assert state is not None
    assert state[0] is None
    assert state[1] == str(refusal)
    assert artifact_kinds & RAW_FAILURE_VALIDATION_FAILURE_KINDS
    with sqlite3.connect(root / "index.db") as index:
        assert index.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)


@pytest.mark.parametrize("mode", ["off", "advisory", "strict"])
def test_complete_malformed_jsonl_is_terminal_without_publishing_a_session_in_every_mode(
    tmp_path: Path, mode: str
) -> None:
    """Schema mode changes validation only; it never makes complete malformed JSONL publishable."""
    from polylogue.core.raw_failure_evidence import RAW_FAILURE_VALIDATION_FAILURE_KINDS
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    payload = (
        b'{"parentUuid":null,"type":"user","message":{"role":"user","content":"safe"},'
        b'"uuid":"m1","timestamp":"2025-01-01T00:00:00Z"}\n'
        b'{"parentUuid":null "type":"user","message":{"role":"user","content":"PRIVATE_BAD_LINE"},'
        b'"uuid":"m2","timestamp":"2025-01-01T00:00:01Z"}\n'
    )
    blob_hash, _size = BlobStore(root / "blob").write_from_bytes(payload)
    with retained_raw_fixture(
        root=root,
        provider=Provider.CLAUDE_CODE,
        blob_hash=blob_hash,
        source_path=str(tmp_path / "projects" / "proj" / "malformed.jsonl"),
    ) as (_source_read, raw_id):
        pass

    async def replay() -> tuple[object, list[str]]:
        refused: list[str] = []
        async with prepared_live_convergence_owner(root, validation_mode=ValidationMode.from_string(mode)) as owner:
            receipts = (
                await owner.replay_retained_raw_ids(
                    (raw_id,), on_terminal_refusal=lambda _keys, refusal: refused.append(refusal.raw_id)
                )
            ).require_complete()
            return receipts, refused

    receipts, refused = asyncio.run(replay())

    assert receipts == ()
    assert refused == [raw_id]
    with prepared_source_fixture(root) as source_read:
        assert source_read.raw_parser_census_is_current(raw_id)
        refusal = source_read.raw_terminal_decode_refusal(raw_id)
        assert refusal is not None
        assert refusal.kind.value in RAW_FAILURE_VALIDATION_FAILURE_KINDS
        assert "PRIVATE_BAD_LINE" not in str(refusal)
        assert "line" in str(refusal).lower() or "malformed" in str(refusal).lower()
    with sqlite3.connect(root / "index.db") as index:
        assert index.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)

"""Regression tests: malformed JSONL counts are accurate and surfaced (#1745).

Two surfaces are covered:

1. The stream parse plan now scans the *whole* stream for malformed-line
   accounting even in advisory mode (``scan_full`` was previously tied to
   STRICT, undercounting past the 64-record sample). The accurate, whole-file
   count lands on the durable artifact observation for the blob.
2. ``_validate_parse_plan`` surfaces malformed lines on a schema-eligible plan:
   STRICT fails the record; ADVISORY does not fail but emits a WARNING (rather
   than silently demoting the loss).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.archive.artifact_taxonomy import ArtifactClassification, ArtifactKind
from polylogue.core.enums import Provider, ValidationMode, ValidationStatus
from polylogue.pipeline.services.ingest_worker import (
    _IngestContext,
    _ParsePlan,
    _validate_parse_plan,
    ingest_record,
)
from polylogue.storage.artifacts.inspection import inspect_raw_artifact
from polylogue.storage.blob_store import BlobStore, reset_blob_store
from polylogue.storage.runtime import RawSessionRecord


@pytest.fixture
def blob_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[BlobStore]:
    root = tmp_path / "blobs"
    store = BlobStore(root)
    monkeypatch.setattr("polylogue.paths.blob_store_root", lambda: root)
    reset_blob_store()
    yield store
    reset_blob_store()


def _valid_line(i: int) -> bytes:
    return (
        b'{"type":"user","uuid":"u%d","sessionId":"s1","parentUuid":null,'
        b'"cwd":"/tmp","message":{"role":"user","content":"hi %d"}}\n' % (i, i)
    )


def _make_record(store: BlobStore, content: bytes, *, source_path: str) -> RawSessionRecord:
    raw_id, blob_size = store.write_from_bytes(content)
    return RawSessionRecord(
        raw_id=raw_id,
        source_name="claude-code",
        source_path=source_path,
        payload_provider=Provider.CLAUDE_CODE,
        source_index=None,
        blob_size=blob_size,
        acquired_at="2026-01-01T00:00:00+00:00",
        file_mtime=None,
    )


def test_advisory_stream_counts_malformed_past_sample_boundary(blob_store: BlobStore, tmp_path: Path) -> None:
    """Malformed lines past the 64-sample boundary are counted; record not failed.

    The accurate whole-file count is surfaced on the durable artifact
    observation for the same blob (not the 64-sample undercount).
    """
    lines = [_valid_line(i) for i in range(120)]
    lines.insert(80, b"{ not valid json line A\n")
    lines.insert(101, b"{ not valid json line B\n")
    content = b"".join(lines)

    record = _make_record(blob_store, content, source_path="agent-z/session.jsonl")

    result = ingest_record(record, str(tmp_path / "archive"), "advisory", blob_root_str=str(blob_store.root))
    # Advisory mode does not fail the record.
    assert result.validation_status != ValidationStatus.FAILED.value

    observation = inspect_raw_artifact(record)
    assert observation.malformed_jsonl_lines >= 2


def _context(record: RawSessionRecord, *, mode: ValidationMode, tmp_path: Path) -> _IngestContext:
    return _IngestContext(
        raw_record=record,
        raw_source=tmp_path / "unused",
        archive_root=tmp_path / "archive",
        validation_mode=mode,
        measure_serialized_size=False,
        source_name=record.source_name or "",
        fallback_timestamp=None,
    )


def _schema_eligible_plan(malformed: int) -> _ParsePlan:
    artifact = ArtifactClassification(
        provider=Provider.CLAUDE_CODE,
        kind=ArtifactKind.SESSION_DOCUMENT,
        parse_as_session=True,
        schema_eligible=True,
        default_priority=100,
        reason="test schema-eligible artifact",
    )
    return _ParsePlan(
        provider=Provider.CLAUDE_CODE,
        payload_provider=str(Provider.CLAUDE_CODE),
        artifact=artifact,
        mode="payload",
        schema_payload={"k": "v"},
        payload={"k": "v"},
        malformed_jsonl_lines=malformed,
        malformed_jsonl_detail="line 7: broken",
    )


def test_strict_fails_on_malformed_schema_eligible_plan(blob_store: BlobStore, tmp_path: Path) -> None:
    """STRICT mode fails a schema-eligible plan carrying malformed lines."""
    record = _make_record(blob_store, _valid_line(0), source_path="conv.json")
    plan = _schema_eligible_plan(malformed=3)

    validation = _validate_parse_plan(_context(record, mode=ValidationMode.STRICT, tmp_path=tmp_path), plan)

    assert validation.status is ValidationStatus.FAILED
    assert validation.parse_error is not None
    assert "Malformed JSONL" in (validation.parse_error or "")


def test_advisory_warns_but_does_not_fail_schema_eligible_plan(
    blob_store: BlobStore, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """ADVISORY mode surfaces malformed lines at WARNING without failing (#1745)."""
    record = _make_record(blob_store, _valid_line(0), source_path="conv.json")
    plan = _schema_eligible_plan(malformed=3)

    with caplog.at_level(logging.WARNING):
        validation = _validate_parse_plan(_context(record, mode=ValidationMode.ADVISORY, tmp_path=tmp_path), plan)

    assert validation.status is not ValidationStatus.FAILED
    # The loss is surfaced, not silently demoted.
    assert any("Malformed JSONL lines counted in advisory mode" in rec.message for rec in caplog.records)

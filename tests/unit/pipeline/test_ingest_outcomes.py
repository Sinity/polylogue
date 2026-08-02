"""Real-production-pipeline fixtures for typed ingest-attempt dispositions (polylogue-cnu3).

Every test here drives the actual ``ingest_record()`` subprocess-worker
entrypoint (or the actual archive-write exception classifier / a real
``ingest_attempts`` row round trip) -- never a hand-rolled stand-in for the
pipeline -- and asserts the resulting ``outcome_code``/``retryable`` are the
exact expected :class:`~polylogue.core.enums.IngestOutcome`. Removing the
structured mapping in ``polylogue.pipeline.ingest_outcomes`` (or reverting a
classification call site in ``ingest_worker.py``/``batch.py``) makes these
fail.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from polylogue.core.enums import IngestOutcome
from polylogue.pipeline.ingest_outcomes import (
    classify_archive_write_exception,
    classify_decode_exception,
    classify_parse_exception,
    success_disposition,
)
from polylogue.pipeline.services.ingest_worker import ingest_record
from polylogue.storage.runtime import RawSessionRecord

pytestmark = pytest.mark.uses_real_clock(
    "Mirrors tests/unit/pipeline/test_resilience.py's _make_raw_record: acquired_at/file_mtime are"
    " opaque metadata for a subprocess-worker fixture, not a production timing comparison."
)


def _make_raw_record(
    raw_id: str,
    provider: str,
    content: bytes,
    path: str = "/exports/test.json",
) -> RawSessionRecord:
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


def test_valid_chatgpt_payload_classifies_success(tmp_path: Path) -> None:
    """A real, well-formed ChatGPT export session classifies as SUCCESS."""
    payload = json.dumps(
        {
            "id": "conv-1",
            "title": "Real Session",
            "create_time": 1700000000,
            "update_time": 1700000001,
            "mapping": {
                "root": {
                    "id": "root",
                    "message": None,
                    "parent": None,
                    "children": ["m1"],
                },
                "m1": {
                    "id": "m1",
                    "message": {
                        "id": "m1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["hello world"]},
                        "create_time": 1700000000,
                    },
                    "parent": "root",
                    "children": [],
                },
            },
        }
    ).encode()
    record = _make_raw_record("real-session", "chatgpt", payload)

    result = ingest_record(record, str(tmp_path / "archive"), "off")

    assert result.error is None
    assert result.sessions, "fixture must actually materialize a session for this to be a meaningful SUCCESS proof"
    assert result.outcome_code == IngestOutcome.SUCCESS.value
    assert result.retryable is False


def test_zero_length_blob_classifies_corrupt_input(tmp_path: Path) -> None:
    """An empty acquired blob is CORRUPT_INPUT, not a parser defect."""
    record = _make_raw_record("empty-blob", "chatgpt", b"")

    result = ingest_record(record, str(tmp_path / "archive"), "off")

    assert result.error is not None
    assert result.outcome_code == IngestOutcome.CORRUPT_INPUT.value
    assert result.retryable is False


def test_undecodable_bytes_classify_corrupt_input_via_real_decode_failure() -> None:
    """A genuine decode-stage exception (invalid UTF-8) classifies as CORRUPT_INPUT.

    Exercises ``classify_decode_exception`` against a real ``UnicodeDecodeError``
    raised by the standard library, proving the classification is type-based
    (never text-matched against the resulting error string).
    """
    try:
        b"\xff\xfe\x00\x01".decode("utf-8")
    except UnicodeDecodeError as exc:
        disposition = classify_decode_exception(exc)
    else:
        pytest.fail("expected a real UnicodeDecodeError")

    assert disposition.outcome is IngestOutcome.CORRUPT_INPUT
    assert disposition.retryable is False
    assert disposition.evidence_ref == "decode:UnicodeDecodeError"


def test_chatgpt_payload_with_no_messages_classifies_unsupported_shape(tmp_path: Path) -> None:
    """A recognized-but-empty ChatGPT payload (no mapping content) is UNSUPPORTED_SHAPE.

    This is the real production path for artifacts a detector claims but that
    yield zero materializable sessions -- distinct from a hard validation
    rejection or a parser bug.
    """
    payload = json.dumps(
        {
            "title": "No Messages",
            "mapping": {},
            "create_time": 1700000000,
            "update_time": 1700000001,
        }
    ).encode()
    record = _make_raw_record("no-messages", "chatgpt", payload)

    result = ingest_record(record, str(tmp_path / "archive"), "off")

    assert not result.sessions
    assert result.outcome_code == IngestOutcome.UNSUPPORTED_SHAPE.value
    assert result.retryable is False


def test_strict_schema_validation_rejection_classifies_validation_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STRICT-mode schema validation failure is VALIDATION_REJECTED.

    This is the flagship question the bead names: "how often did strict
    validation reject real input" must be answerable without text-matching
    ``error_message``. The schema-package resolution is faked (matching the
    existing convention in ``test_ingest_worker_reuses_schema_resolution_and_
    walks_drift``) so this test isolates the real ``_validate_parse_plan``
    STRICT-rejection branch in ``ingest_worker.py`` from a specific packaged
    schema's content.
    """
    from polylogue.schemas import ValidationResult

    payload = json.dumps({"id": "conv-1", "title": "T", "mapping": {}}).encode()
    record = _make_raw_record("strict-reject", "chatgpt", payload)

    class _RejectingValidator:
        provider = "chatgpt"

        def validation_samples(self, payload: object, max_samples: int | None = None) -> list[object]:
            return [payload]

        def validate(self, _sample: object, *, include_drift: bool | None = None) -> ValidationResult:
            return ValidationResult(is_valid=False, errors=["missing required field 'foo'"])

    monkeypatch.setattr(
        "polylogue.schemas.validator.SchemaValidator.for_payload",
        lambda *args, **kwargs: _RejectingValidator(),
    )

    result = ingest_record(record, str(tmp_path / "archive"), "strict")

    assert result.validation_status == "failed"
    assert result.outcome_code == IngestOutcome.VALIDATION_REJECTED.value
    assert result.retryable is False


def test_unexpected_parse_exception_classifies_parser_defect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuinely unexpected parser exception is PARSER_DEFECT, not a guess."""
    payload = json.dumps({"id": "conv-1", "title": "T", "mapping": {"root": {}}}).encode()
    record = _make_raw_record("parser-bug", "chatgpt", payload)

    def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("real unexpected parser bug")

    monkeypatch.setattr("polylogue.sources.dispatch.parse_payload", _boom)

    result = ingest_record(record, str(tmp_path / "archive"), "off")

    assert result.outcome_code == IngestOutcome.PARSER_DEFECT.value
    assert result.retryable is False
    assert result.evidence_ref == "parse:RuntimeError"


def test_pydantic_validation_error_classifies_validation_rejected_by_type() -> None:
    """A real ``pydantic.ValidationError`` classifies as VALIDATION_REJECTED by type."""
    from pydantic import BaseModel, ValidationError

    class _Strict(BaseModel):
        value: int

    try:
        _Strict.model_validate({"value": "not-an-int-and-not-coercible-xyz"})
    except ValidationError as exc:
        disposition = classify_parse_exception(exc)
    else:
        pytest.fail("expected a real pydantic.ValidationError")

    assert disposition.outcome is IngestOutcome.VALIDATION_REJECTED
    assert disposition.retryable is False


def test_real_sqlite_lock_classifies_transient_error(tmp_path: Path) -> None:
    """A genuine ``sqlite3.OperationalError: database is locked`` is TRANSIENT_ERROR.

    Reproduces real lock contention with two live connections rather than
    constructing a fake exception, proving ``classify_archive_write_exception``
    keys off ``is_transient_sqlite_lock``'s structural check.
    """
    db_path = tmp_path / "contended.db"
    holder = sqlite3.connect(str(db_path), timeout=0)
    holder.execute("CREATE TABLE t (x INTEGER)")
    holder.execute("BEGIN EXCLUSIVE")
    holder.execute("INSERT INTO t VALUES (1)")

    contender = sqlite3.connect(str(db_path), timeout=0)
    try:
        with pytest.raises(sqlite3.OperationalError) as excinfo:
            contender.execute("INSERT INTO t VALUES (2)")
    finally:
        holder.rollback()
        holder.close()
        contender.close()

    disposition = classify_archive_write_exception(excinfo.value)
    assert disposition.outcome is IngestOutcome.TRANSIENT_ERROR
    assert disposition.retryable is True
    assert disposition.evidence_ref == "archive_write:OperationalError"


def test_non_transient_database_error_classifies_parser_defect_not_swallowed() -> None:
    """A non-lock ``sqlite3.OperationalError`` (e.g. schema mismatch) is NOT retried silently."""
    exc = sqlite3.OperationalError("no such table: sessions")
    disposition = classify_archive_write_exception(exc)
    assert disposition.outcome is IngestOutcome.PARSER_DEFECT
    assert disposition.retryable is False


def test_ingest_attempt_round_trip_persists_typed_disposition(tmp_path: Path) -> None:
    """Every field AC1 names round-trips through a real ``ingest_attempts`` row."""
    from polylogue.sources.live.cursor import CursorStore

    cursor_store = CursorStore(tmp_path / "index.sqlite", ops_db_path=tmp_path / "ops.db")
    attempt_id = cursor_store.begin_ingest_attempt(paths=[Path("/x/a.jsonl")], input_bytes=10, queued_file_count=1)
    ok = cursor_store.finish_ingest_attempt(
        attempt_id, status="completed", phase="completed", disposition=success_disposition(evidence_ref="ok")
    )
    assert ok

    conn = sqlite3.connect(tmp_path / "ops.db")
    try:
        row = conn.execute(
            "SELECT outcome_code, retryable, evidence_ref FROM ingest_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row == (IngestOutcome.SUCCESS.value, 0, "ok")


def test_legacy_ingest_attempt_row_defaults_to_legacy_unknown(tmp_path: Path) -> None:
    """A row written without an explicit disposition (a legacy caller) stays legacy_unknown (AC4)."""
    from polylogue.sources.live.cursor import CursorStore

    cursor_store = CursorStore(tmp_path / "index.sqlite", ops_db_path=tmp_path / "ops.db")
    attempt_id = cursor_store.begin_ingest_attempt(paths=[Path("/x/legacy.jsonl")], input_bytes=10, queued_file_count=1)

    conn = sqlite3.connect(tmp_path / "ops.db")
    try:
        row = conn.execute(
            "SELECT outcome_code, retryable FROM ingest_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row == (IngestOutcome.LEGACY_UNKNOWN.value, None)

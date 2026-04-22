"""Synthetic regression fixtures for known quarantine shapes.

The live archive has recurring quarantine cases that the operator brief
documents but that were not previously captured as automated tests.
Two patterns dominate:

1. **Zero-length source file** — the export file exists but has 0 bytes.
   Typical for interrupted Codex session writes. The decoder surfaces
   ``Input is a zero-length, empty document``. The raw id for a real
   zero-length blob is the SHA-256 of the empty string
   (``e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855``).

2. **Malformed JSONL mid-stream** — the JSONL export is mostly well-formed
   but a single record has invalid JSON (missing comma, unbalanced brace,
   etc.). In STRICT validation mode this surfaces as
   ``Malformed JSONL lines: N (first bad line X: <detail>)``.

These tests pin the quarantine contract: ``ingest_record`` must produce
``error``/``parse_error`` in both cases, and when the result is
persisted via ``mark_raw_parsed`` the raw conversation lands in a
quarantined state (``parsed_at is None AND parse_error is not None``).

If the quarantine policy ever shifts — say a future decoder starts
tolerating empty blobs silently, or JSONL validation relaxes out of
STRICT by default — these tests fail loudly so the regression is
visible at PR time, not after a live archive run silently drops data.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest

from polylogue.pipeline.services.ingest_worker import ingest_record
from polylogue.pipeline.services.validation_flow import validate_raw_ids
from polylogue.storage.blob_store import get_blob_store
from polylogue.storage.raw_ingest_artifacts import RawIngestArtifactState
from polylogue.storage.raw_state_models import RawConversationState
from polylogue.storage.store import RawConversationRecord
from polylogue.types import ValidationMode, ValidationStatus

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _make_raw_record(content: bytes, provider: str, path: str) -> RawConversationRecord:
    """Store ``content`` in the blob store and wrap a RawConversationRecord."""
    raw_id, size = get_blob_store().write_from_bytes(content)
    now = datetime.now(timezone.utc).isoformat()
    return RawConversationRecord(
        raw_id=raw_id,
        provider_name=provider,
        source_name="quarantine-fixture",
        source_path=path,
        source_index=None,
        blob_size=size,
        acquired_at=now,
        file_mtime=now,
    )


# ---------------------------------------------------------------------------
# Synthetic fixtures — the shape, not the bytes, of real quarantine cases.
# ---------------------------------------------------------------------------


def zero_length_bytes() -> bytes:
    """Zero-length export body. SHA-256 matches EMPTY_SHA256."""
    return b""


def claude_code_malformed_jsonl_bytes() -> bytes:
    """Valid claude-code JSONL with one record that has a missing comma.

    Structure: two well-formed records surrounding one bad record. The
    bad record drops the comma between ``parentUuid`` and ``type``,
    yielding ``unexpected character`` at the point the decoder expects
    either a comma or the object's closing brace.
    """
    good_a = (
        b'{"parentUuid":null,"type":"user",'
        b'"message":{"role":"user","content":"hello"},'
        b'"uuid":"m1","timestamp":"2025-01-01T00:00:00Z"}'
    )
    bad = (
        b'{"parentUuid":null '  # <- missing comma here
        b'"type":"user",'
        b'"message":{"role":"user","content":"bad"},'
        b'"uuid":"m2","timestamp":"2025-01-01T00:00:01Z"}'
    )
    good_b = (
        b'{"parentUuid":"m1","type":"assistant",'
        b'"message":{"role":"assistant","content":[{"type":"text","text":"hi"}]},'
        b'"uuid":"m3","timestamp":"2025-01-01T00:00:02Z"}'
    )
    return good_a + b"\n" + bad + b"\n" + good_b + b"\n"


def codex_malformed_jsonl_bytes() -> bytes:
    """Valid codex JSONL with one record that is not valid JSON.

    Codex sessions are JSONL with a ``session_meta`` envelope followed
    by ``message`` records. A missing closing brace in a ``message``
    record is a realistic corruption shape.
    """
    meta = b'{"type":"session_meta","payload":{"id":"session-x","timestamp":"2025-01-01T00:00:00Z"}}'
    good = (
        b'{"type":"message","id":"msg-1","role":"user",'
        b'"timestamp":"2025-01-01T00:00:01Z",'
        b'"content":[{"type":"input_text","text":"ping"}]}'
    )
    bad = (
        b'{"type":"message","id":"msg-2","role":"assistant",'
        b'"timestamp":"2025-01-01T00:00:02Z"'  # <- missing closing brace + comma
        b'"content":[{"type":"output_text","text":"pong"}]'
    )
    return meta + b"\n" + good + b"\n" + bad + b"\n"


# ---------------------------------------------------------------------------
# Zero-length quarantine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider,source_path",
    [
        ("codex", "/exports/codex-session.jsonl"),
        ("claude-code", "/exports/claude-code-session.jsonl"),
        ("chatgpt", "/exports/chatgpt.json"),
        ("gemini", "/exports/gemini.json"),
    ],
)
def test_zero_length_blob_quarantines_across_providers(tmp_path: Path, provider: str, source_path: str) -> None:
    """Every provider produces a decoder-level quarantine on zero-length input.

    The error originates in ``build_raw_payload_envelope`` and surfaces
    identically regardless of provider — it's a pre-parser failure that
    all provider paths share. Contract: ``error`` and ``parse_error``
    set, ``validation_status == FAILED``, no conversations produced.
    """
    record = _make_raw_record(zero_length_bytes(), provider, source_path)
    assert record.raw_id == EMPTY_SHA256

    result = ingest_record(record, str(tmp_path / "archive"), "strict")

    assert result.error is not None
    assert result.parse_error is not None
    assert "zero-length" in result.error
    assert result.parse_error == result.error
    assert result.validation_status == ValidationStatus.FAILED.value
    assert result.conversations == []


def test_zero_length_blob_quarantine_survives_non_strict_mode(tmp_path: Path) -> None:
    """Zero-length input fails even with validation OFF — it's a decoder error.

    The STRICT/OFF distinction governs schema validation, not blob
    decoding. An empty document has no bytes to parse, so the error
    fires before any validation step runs.
    """
    record = _make_raw_record(zero_length_bytes(), "codex", "/exports/codex.jsonl")
    result = ingest_record(record, str(tmp_path / "archive"), "off")

    assert result.error is not None
    assert "zero-length" in result.error
    assert result.parse_error is not None


# ---------------------------------------------------------------------------
# Malformed JSONL quarantine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider,fixture",
    [
        ("claude-code", claude_code_malformed_jsonl_bytes),
        ("codex", codex_malformed_jsonl_bytes),
    ],
)
def test_malformed_jsonl_mid_stream_quarantines_in_strict_mode(
    tmp_path: Path,
    provider: str,
    fixture: Callable[[], bytes],
) -> None:
    """Stream-record providers quarantine a single malformed JSONL record.

    Under STRICT validation, even a single bad line triggers the
    ``Malformed JSONL lines: N (first bad line X: ...)`` quarantine.
    The envelope records ``malformed_jsonl_lines`` during sampling and
    surfaces it as a parse error rather than silently dropping data.
    """
    record = _make_raw_record(fixture(), provider, f"/exports/{provider}.jsonl")
    result = ingest_record(record, str(tmp_path / "archive"), "strict")

    assert result.error is not None, "malformed JSONL must surface an error in STRICT mode"
    assert result.error.startswith("Malformed JSONL lines:")
    assert result.parse_error == result.error
    assert result.validation_status == ValidationStatus.FAILED.value
    assert result.conversations == []


def test_malformed_jsonl_tolerated_in_validation_off_mode(tmp_path: Path) -> None:
    """With validation disabled, well-formed records around a bad line still parse.

    STRICT is the only mode that promotes malformed-line detection to a
    fatal parse error. With validation OFF the bad line is silently
    skipped and the surrounding records produce one conversation. This
    test pins that asymmetry so a future "always strict" change would
    fail here rather than silently break the OFF contract.
    """
    record = _make_raw_record(
        claude_code_malformed_jsonl_bytes(),
        "claude-code",
        "/exports/claude-code.jsonl",
    )
    result = ingest_record(record, str(tmp_path / "archive"), "off")

    assert result.error is None
    assert result.conversations, "valid surrounding records should still parse"


# ---------------------------------------------------------------------------
# Persistence lifecycle — ingest_record → mark_raw_parsed → quarantined
# ---------------------------------------------------------------------------


async def test_quarantine_state_round_trip_through_mark_raw_parsed(tmp_path: Path) -> None:
    """Persisting an ingest error leaves the raw row in a quarantined state.

    Quarantine = ``parsed_at is None AND parse_error is not None`` (see
    ``RawIngestArtifactState.quarantined``). Verifies the full lifecycle:
    ``ingest_record`` surfaces the error, ``mark_raw_parsed`` persists it,
    ``RawConversationState`` round-trips, and the derived artifact state
    classifies the row as quarantined.
    """
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

    record = _make_raw_record(zero_length_bytes(), "codex", "/exports/codex.jsonl")

    result = ingest_record(record, str(tmp_path / "archive"), "strict")
    assert result.error is not None

    backend = SQLiteBackend(db_path=tmp_path / "archive.db")
    try:
        await backend.save_raw_conversation(record)
        await backend.mark_raw_parsed(record.raw_id, error=result.error)

        stored = await backend.get_raw_conversation(record.raw_id)
        assert stored is not None
        assert stored.parse_error is not None
        assert stored.parsed_at is None

        state = RawIngestArtifactState.from_state(
            RawConversationState(
                raw_id=stored.raw_id,
                parsed_at=stored.parsed_at,
                parse_error=stored.parse_error,
            )
        )
        assert state.quarantined is True
        assert state.parsed is False
    finally:
        await backend.close()


async def test_validation_flow_persists_decode_quarantine_state(tmp_path: Path) -> None:
    """Validation persistence records both decode-failure shapes as quarantines."""
    from polylogue.storage.backends.async_sqlite import SQLiteBackend

    cases = [
        (
            _make_raw_record(zero_length_bytes(), "codex", "/exports/empty-codex.jsonl"),
            "zero-length",
        ),
        (
            _make_raw_record(codex_malformed_jsonl_bytes(), "codex", "/exports/malformed-codex.jsonl"),
            "Malformed JSONL lines:",
        ),
    ]

    backend = SQLiteBackend(db_path=tmp_path / "archive.db")
    try:
        for record, _expected_error in cases:
            await backend.save_raw_conversation(record)

        result = await validate_raw_ids(
            repository=backend,
            raw_ids=[record.raw_id for record, _expected_error in cases],
            persist=True,
            validation_mode=ValidationMode.STRICT,
            raw_batch_size=10,
        )

        assert len(result.records) == 2
        assert result.parseable_raw_ids == []
        assert set(result.invalid_raw_ids) == {record.raw_id for record, _expected_error in cases}

        for record, expected_error in cases:
            validation_record = next(item for item in result.records if item.raw_id == record.raw_id)
            assert validation_record.validation_status == ValidationStatus.FAILED
            assert validation_record.parse_error is not None
            assert expected_error in validation_record.parse_error

            stored = await backend.get_raw_conversation(record.raw_id)
            assert stored is not None
            assert stored.validation_status == ValidationStatus.FAILED
            assert stored.validation_error is not None
            assert expected_error in stored.validation_error
            assert stored.parse_error is not None
            assert expected_error in stored.parse_error
            assert stored.parsed_at is None

            state = RawIngestArtifactState.from_state(
                RawConversationState(
                    raw_id=stored.raw_id,
                    parsed_at=stored.parsed_at,
                    parse_error=stored.parse_error,
                    validation_status=stored.validation_status,
                )
            )
            assert state.quarantined is True
            assert state.needs_parse_backlog() is False
    finally:
        await backend.close()

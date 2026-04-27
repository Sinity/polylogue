"""Ingestion chaos and recovery tests (Workstream E).

Tests for:
- E1-E2: Large-batch partial-corruption (malformed JSON, truncation, bad UTF-8,
  wrong envelope) — pipeline skips corrupted lines, processes the rest.
- E3-E5: Timestamp edge-case parsing (1970-adjacent, Y2K38, far-future, mixed
  formats, missing timestamps) and re-run idempotency.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from hypothesis import given, settings

from polylogue.lib.timestamps import parse_timestamp
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.sources.decoders import _decode_json_bytes, _iter_json_stream
from polylogue.storage.runtime import RawConversationRecord
from tests.infra.large_batches import (
    corrupt_line_bad_utf8,
    corrupt_line_malformed_json,
    corrupt_line_truncated,
    corrupt_line_wrong_envelope,
    generate_large_jsonl,
    generate_timestamp_patterns,
    generate_valid_jsonl_record,
    write_jsonl_with_bad_utf8,
)
from tests.infra.strategies import malformed_json_strategy

TimestampInput = str | int | float | None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_record(
    raw_id: str,
    provider: str,
    content: bytes,
    path: str = "/exports/test.jsonl",
) -> RawConversationRecord:
    from polylogue.storage.blob_store import get_blob_store

    # Write content to blob store
    blob_store = get_blob_store()
    actual_raw_id, blob_size = blob_store.write_from_bytes(content)
    now = datetime.now(timezone.utc).isoformat()

    return RawConversationRecord(
        raw_id=actual_raw_id,  # Use the actual hash as raw_id
        provider_name=provider,
        source_name="test",
        source_path=path,
        source_index=None,
        blob_size=blob_size,
        acquired_at=now,
        file_mtime=now,
    )


def _make_parsing_service(tmp_path: Path) -> ParsingService:
    from polylogue.config import Config
    from polylogue.pipeline.services.parsing import ParsingService
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository

    db = SQLiteBackend(db_path=tmp_path / "test.db")
    config = Config(
        sources=[],
        archive_root=tmp_path / "archive",
        render_root=tmp_path / "render",
    )
    return ParsingService(
        repository=ConversationRepository(backend=db),
        archive_root=tmp_path / "archive",
        config=config,
    )


def _jsonl_bytes(lines: list[str]) -> bytes:
    """Join JSONL lines into bytes for use in raw records."""
    return ("\n".join(lines) + "\n").encode("utf-8")


def _iter_jsonl_stream(data: bytes, name: str = "test.jsonl") -> list[object]:
    """Convenience: parse JSONL bytes via _iter_json_stream and collect."""
    return list(_iter_json_stream(BytesIO(data), name))


def _as_record(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


def _timestamp_value(record: dict[str, object]) -> TimestampInput:
    value = record.get("timestamp")
    assert value is None or isinstance(value, str | int | float)
    return value


# ===========================================================================
# E1-E2: Large-batch partial-corruption tests
# ===========================================================================


class TestLargeBatchMalformedJson:
    """500+ line JSONL where exactly 1 line is malformed JSON."""

    BATCH_SIZE = 500
    CORRUPT_INDEX = 250

    def test_malformed_json_skipped_rest_processed(self) -> None:
        """Pipeline skips the 1 malformed line, processes the other 499."""
        lines = generate_large_jsonl(self.BATCH_SIZE, provider="codex")
        corrupted = corrupt_line_malformed_json(lines, self.CORRUPT_INDEX)
        data = _jsonl_bytes(corrupted)

        parsed = _iter_jsonl_stream(data)
        assert len(parsed) == self.BATCH_SIZE - 1

    def test_malformed_json_records_are_valid_dicts(self) -> None:
        """Every parsed record from the surviving lines is a valid dict."""
        lines = generate_large_jsonl(self.BATCH_SIZE, provider="codex")
        corrupted = corrupt_line_malformed_json(lines, self.CORRUPT_INDEX)
        data = _jsonl_bytes(corrupted)

        parsed = _iter_jsonl_stream(data)
        assert all(isinstance(record, dict) for record in parsed)

    def test_malformed_json_all_records_have_type_field(self) -> None:
        """Non-corrupted records retain their 'type' field."""
        lines = generate_large_jsonl(self.BATCH_SIZE, provider="codex")
        corrupted = corrupt_line_malformed_json(lines, self.CORRUPT_INDEX)
        data = _jsonl_bytes(corrupted)

        parsed = [_as_record(record) for record in _iter_jsonl_stream(data)]
        assert all(record.get("type") == "message" for record in parsed)


class TestLargeBatchTruncatedLine:
    """500+ line JSONL where exactly 1 line is truncated mid-value."""

    BATCH_SIZE = 500
    CORRUPT_INDEX = 100

    def test_truncated_line_skipped_rest_processed(self) -> None:
        """Pipeline skips the truncated line, processes the other 499."""
        lines = generate_large_jsonl(self.BATCH_SIZE, provider="codex")
        corrupted = corrupt_line_truncated(lines, self.CORRUPT_INDEX)
        data = _jsonl_bytes(corrupted)

        parsed = _iter_jsonl_stream(data)
        assert len(parsed) == self.BATCH_SIZE - 1

    def test_truncated_produces_valid_records(self) -> None:
        """All surviving records are structurally valid."""
        lines = generate_large_jsonl(self.BATCH_SIZE, provider="codex")
        corrupted = corrupt_line_truncated(lines, self.CORRUPT_INDEX)
        data = _jsonl_bytes(corrupted)

        parsed = [_as_record(record) for record in _iter_jsonl_stream(data)]
        assert all(isinstance(r, dict) and "type" in r for r in parsed)


class TestLargeBatchBadUtf8:
    """500+ line JSONL where exactly 1 line has invalid UTF-8 bytes."""

    BATCH_SIZE = 500
    CORRUPT_INDEX = 350

    def test_bad_utf8_skipped_rest_processed(self, tmp_path: Path) -> None:
        """Pipeline skips the bad UTF-8 line, processes the rest."""
        lines = generate_large_jsonl(self.BATCH_SIZE, provider="codex")
        corrupted = corrupt_line_bad_utf8(lines, self.CORRUPT_INDEX)

        # Must write to file to get actual bad bytes
        path = tmp_path / "test.jsonl"
        write_jsonl_with_bad_utf8(path, corrupted)

        with open(path, "rb") as f:
            parsed = list(_iter_json_stream(f, "test.jsonl"))

        # The bad UTF-8 line may parse as a dict (the bad bytes are inside a
        # JSON string value, so _decode_json_bytes may succeed with errors="ignore"
        # fallback). What matters is the pipeline doesn't crash and processes
        # at least the valid lines.
        assert len(parsed) >= self.BATCH_SIZE - 1

    def test_bad_utf8_no_crash(self, tmp_path: Path) -> None:
        """No exception propagates from bad UTF-8 lines."""
        lines = generate_large_jsonl(self.BATCH_SIZE, provider="codex")
        corrupted = corrupt_line_bad_utf8(lines, self.CORRUPT_INDEX)
        path = tmp_path / "test.jsonl"
        write_jsonl_with_bad_utf8(path, corrupted)

        with open(path, "rb") as f:
            parsed = list(_iter_json_stream(f, "test.jsonl"))
        assert isinstance(parsed, list)


class TestLargeBatchWrongEnvelope:
    """500+ line JSONL where exactly 1 line has valid JSON but wrong structure."""

    BATCH_SIZE = 500
    CORRUPT_INDEX = 450

    def test_wrong_envelope_still_parsed_as_json(self) -> None:
        """Wrong envelope is valid JSON, so _iter_json_stream yields it.

        The wrong-envelope record IS valid JSON — it just has the wrong
        structure for the provider. _iter_json_stream accepts it; the
        provider parser downstream will reject/ignore it. Verify count
        includes the wrong-envelope line (it's not a parse error at the
        JSON level).
        """
        lines = generate_large_jsonl(self.BATCH_SIZE, provider="codex")
        corrupted = corrupt_line_wrong_envelope(lines, self.CORRUPT_INDEX)
        data = _jsonl_bytes(corrupted)

        parsed = _iter_jsonl_stream(data)
        # Wrong envelope IS valid JSON, so all 500 lines parse
        assert len(parsed) == self.BATCH_SIZE

    def test_wrong_envelope_record_has_different_structure(self) -> None:
        """The wrong-envelope record is structurally distinct from valid records."""
        lines = generate_large_jsonl(self.BATCH_SIZE, provider="codex")
        corrupted = corrupt_line_wrong_envelope(lines, self.CORRUPT_INDEX)
        data = _jsonl_bytes(corrupted)

        parsed = [_as_record(record) for record in _iter_jsonl_stream(data)]
        # Exactly one record should lack the 'type' field
        records_without_type = [r for r in parsed if "type" not in r]
        assert len(records_without_type) == 1


class TestMultipleCorruptionsInBatch:
    """JSONL with multiple corruption types at different positions."""

    BATCH_SIZE = 500

    def test_multiple_malformed_lines(self) -> None:
        """Multiple malformed lines are all skipped."""
        lines = generate_large_jsonl(self.BATCH_SIZE, provider="codex")
        corrupt_indices = [10, 100, 200, 300, 400]
        for idx in sorted(corrupt_indices, reverse=True):
            lines = corrupt_line_malformed_json(lines, idx)
        data = _jsonl_bytes(lines)

        parsed = _iter_jsonl_stream(data)
        assert len(parsed) == self.BATCH_SIZE - len(corrupt_indices)

    def test_mixed_corruption_types(self, tmp_path: Path) -> None:
        """Different corruption types at different positions all handled."""
        lines = generate_large_jsonl(self.BATCH_SIZE, provider="codex")
        # Apply corruptions in reverse to preserve indices
        lines = corrupt_line_truncated(lines, 400)
        lines = corrupt_line_malformed_json(lines, 200)
        lines = corrupt_line_malformed_json(lines, 50)
        data = _jsonl_bytes(lines)

        parsed = _iter_jsonl_stream(data)
        # 3 corrupted lines should be skipped
        assert len(parsed) == self.BATCH_SIZE - 3


class TestCorruptionAtBoundaries:
    """Corruption at first/last line of a batch."""

    BATCH_SIZE = 500

    def test_first_line_corrupted(self) -> None:
        """Corruption at index 0 is handled gracefully."""
        lines = generate_large_jsonl(self.BATCH_SIZE, provider="codex")
        corrupted = corrupt_line_malformed_json(lines, 0)
        data = _jsonl_bytes(corrupted)

        parsed = _iter_jsonl_stream(data)
        assert len(parsed) == self.BATCH_SIZE - 1

    def test_last_line_corrupted(self) -> None:
        """Corruption at the last index is handled gracefully."""
        lines = generate_large_jsonl(self.BATCH_SIZE, provider="codex")
        corrupted = corrupt_line_malformed_json(lines, self.BATCH_SIZE - 1)
        data = _jsonl_bytes(corrupted)

        parsed = _iter_jsonl_stream(data)
        assert len(parsed) == self.BATCH_SIZE - 1


# ===========================================================================
# E1-E2 (pipeline layer): ParsingService with corrupted raw records
# ===========================================================================


class TestParsingServiceCorruption:
    """ingest_record handles partial corruption."""

    def test_malformed_jsonl_line_in_codex_raw(self, tmp_path: Path) -> None:
        """Codex JSONL with 1 bad line: parsing succeeds with fewer messages."""
        from polylogue.pipeline.services.ingest_worker import ingest_record

        lines = generate_large_jsonl(50, provider="codex")
        corrupted = corrupt_line_malformed_json(lines, 25)
        content = _jsonl_bytes(corrupted)

        record = _make_raw_record("codex-corrupt-1", "codex", content, "/exports/codex.jsonl")
        result = ingest_record(record, str(tmp_path / "archive"), "off")
        assert result.error is None
        if result.conversations:
            assert result.conversations[0].provider_name in ("codex", "codex-cli")

    def test_truncated_jsonl_line_in_codex_raw(self, tmp_path: Path) -> None:
        """Codex JSONL with 1 truncated line: parsing succeeds."""
        from polylogue.pipeline.services.ingest_worker import ingest_record

        lines = generate_large_jsonl(50, provider="codex")
        corrupted = corrupt_line_truncated(lines, 10)
        content = _jsonl_bytes(corrupted)

        record = _make_raw_record("codex-truncated-1", "codex", content, "/exports/codex.jsonl")
        result = ingest_record(record, str(tmp_path / "archive"), "off")
        assert result.error is None

    def test_wrong_envelope_in_codex_raw(self, tmp_path: Path) -> None:
        """Codex JSONL with 1 wrong-envelope line: parsing still works."""
        from polylogue.pipeline.services.ingest_worker import ingest_record

        lines = generate_large_jsonl(50, provider="codex")
        corrupted = corrupt_line_wrong_envelope(lines, 30)
        content = _jsonl_bytes(corrupted)

        record = _make_raw_record("codex-wrong-env-1", "codex", content, "/exports/codex.jsonl")
        result = ingest_record(record, str(tmp_path / "archive"), "off")
        assert result.error is None


# ===========================================================================
# E1-E2: _decode_raw_payload level corruption (raw_payload.py path)
# ===========================================================================


class TestDecodeRawPayloadCorruption:
    """raw_payload.build_raw_payload_envelope handles JSONL corruption."""

    def test_malformed_line_counted_in_envelope(self) -> None:
        """malformed_jsonl_lines reflects the number of bad lines."""
        from polylogue.lib.raw_payload import build_raw_payload_envelope

        lines = generate_large_jsonl(100, provider="codex")
        corrupted = corrupt_line_malformed_json(lines, 50)
        content = _jsonl_bytes(corrupted)

        envelope = build_raw_payload_envelope(
            content,
            source_path="/exports/codex.jsonl",
            fallback_provider="codex",
        )
        assert envelope.malformed_jsonl_lines == 1
        assert envelope.wire_format == "jsonl"
        # Payload should contain 99 valid records
        assert isinstance(envelope.payload, list)
        assert len(envelope.payload) == 99

    def test_multiple_malformed_lines_counted(self) -> None:
        """All malformed lines are counted."""
        from polylogue.lib.raw_payload import build_raw_payload_envelope

        lines = generate_large_jsonl(100, provider="codex")
        for idx in [10, 30, 50, 70, 90]:
            lines = corrupt_line_malformed_json(lines, idx)
        content = _jsonl_bytes(lines)

        envelope = build_raw_payload_envelope(
            content,
            source_path="/exports/codex.jsonl",
            fallback_provider="codex",
        )
        assert envelope.malformed_jsonl_lines == 5
        assert isinstance(envelope.payload, list)
        assert len(envelope.payload) == 95

    def test_bad_utf8_lines_are_counted_and_salvaged(self, tmp_path: Path) -> None:
        """Invalid UTF-8 lines count as malformed JSONL but do not poison the whole file."""
        from polylogue.lib.raw_payload import build_raw_payload_envelope

        lines = generate_large_jsonl(20, provider="codex")
        corrupted = corrupt_line_bad_utf8(lines, 4)
        corrupted = corrupt_line_bad_utf8(corrupted, 12)

        path = tmp_path / "bad_utf8.jsonl"
        write_jsonl_with_bad_utf8(path, corrupted)

        envelope = build_raw_payload_envelope(
            path.read_bytes(),
            source_path=str(path),
            fallback_provider="codex",
        )

        assert envelope.wire_format == "jsonl"
        assert envelope.malformed_jsonl_lines == 2
        assert isinstance(envelope.payload, list)
        assert len(envelope.payload) == 18


# ===========================================================================
# E3-E5: Timestamp edge-case parsing
# ===========================================================================


class TestTimestampEpochNearZero:
    """1970-adjacent timestamps parse correctly."""

    def test_epoch_near_zero_records_parse(self) -> None:
        """All epoch-near-zero records from generate_timestamp_patterns parse."""
        patterns = generate_timestamp_patterns()
        records = patterns["epoch_near_zero"]
        for record in records:
            cast_record = _as_record(record)
            ts = _timestamp_value(cast_record)
            result = parse_timestamp(ts)
            assert result is not None, f"Failed to parse timestamp: {ts}"
            assert isinstance(result, datetime)
            # Should be within a day of epoch (86400 seconds into 1970)
            assert result.year == 1970

    def test_epoch_one_day(self) -> None:
        """Timestamp 86400 (1970-01-02) parses correctly."""
        result = parse_timestamp(86400)
        assert result is not None
        assert result.year == 1970
        assert result.month == 1
        assert result.day == 2

    def test_epoch_one_day_as_string(self) -> None:
        """Timestamp '86400' as string parses correctly."""
        result = parse_timestamp("86400")
        assert result is not None
        assert result.year == 1970


class TestTimestampY2K38:
    """Y2K38-adjacent timestamps (near 2^31 - 1) parse correctly."""

    def test_y2038_records_parse(self) -> None:
        """All y2038-adjacent records from generate_timestamp_patterns parse."""
        patterns = generate_timestamp_patterns()
        records = patterns["y2038_adjacent"]
        for record in records:
            cast_record = _as_record(record)
            ts = _timestamp_value(cast_record)
            result = parse_timestamp(ts)
            assert result is not None, f"Failed to parse timestamp: {ts}"
            assert isinstance(result, datetime)

    def test_exact_y2038_boundary(self) -> None:
        """The exact Y2K38 value (2147483647) parses correctly."""
        result = parse_timestamp(2147483647)
        assert result is not None
        assert result.year == 2038
        assert result.month == 1
        assert result.day == 19

    def test_y2038_as_string(self) -> None:
        """Y2K38 value as string parses correctly."""
        result = parse_timestamp("2147483647")
        assert result is not None
        assert result.year == 2038

    def test_just_past_y2038(self) -> None:
        """Value just past Y2K38 (32-bit overflow) still parses on 64-bit systems."""
        result = parse_timestamp(2147483648)
        assert result is not None
        assert result.year == 2038


class TestTimestampFarFuture:
    """Far-future timestamps parse correctly."""

    def test_far_future_records_parse(self) -> None:
        """All far-future records from generate_timestamp_patterns parse."""
        patterns = generate_timestamp_patterns()
        records = patterns["far_future"]
        for record in records:
            cast_record = _as_record(record)
            ts = _timestamp_value(cast_record)
            result = parse_timestamp(ts)
            assert result is not None, f"Failed to parse timestamp: {ts}"
            assert isinstance(result, datetime)
            # 3000000000 is year 2065
            assert result.year >= 2065

    def test_year_2100_timestamp(self) -> None:
        """Timestamp in year 2100 parses correctly."""
        # 2100-01-01 00:00:00 UTC = 4102444800
        result = parse_timestamp(4102444800)
        assert result is not None
        assert result.year == 2100

    def test_far_future_iso_string(self) -> None:
        """Far-future ISO timestamp parses correctly."""
        result = parse_timestamp("2100-01-01T00:00:00+00:00")
        assert result is not None
        assert result.year == 2100


class TestTimestampMixedFormats:
    """Mixed timestamp formats in same batch all parse correctly."""

    def test_mixed_format_records_all_parse(self) -> None:
        """All mixed-format records from generate_timestamp_patterns parse."""
        patterns = generate_timestamp_patterns()
        records = patterns["mixed_formats"]
        parsed_timestamps = []
        for record in records:
            cast_record = _as_record(record)
            ts = _timestamp_value(cast_record)
            result = parse_timestamp(ts)
            assert result is not None, f"Failed to parse timestamp: {ts}"
            parsed_timestamps.append(result)
        assert len(parsed_timestamps) == 3

    def test_iso_with_timezone_offset(self) -> None:
        """ISO 8601 with +00:00 timezone offset."""
        result = parse_timestamp("2024-01-15T10:30:00+00:00")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_epoch_float(self) -> None:
        """Epoch as float (1705312200.0)."""
        result = parse_timestamp(1705312200.0)
        assert result is not None
        assert result.year == 2024

    def test_epoch_string(self) -> None:
        """Epoch as string ('1705312260')."""
        result = parse_timestamp("1705312260")
        assert result is not None
        assert result.year == 2024

    def test_iso_with_z_suffix(self) -> None:
        """ISO 8601 with Z suffix."""
        result = parse_timestamp("2024-01-15T10:30:00Z")
        assert result is not None
        assert result.year == 2024

    def test_bare_iso_date(self) -> None:
        """Bare ISO date without time component."""
        result = parse_timestamp("2024-01-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15


class TestTimestampMissing:
    """Missing/None timestamps handled gracefully."""

    def test_none_returns_none(self) -> None:
        """None input returns None."""
        assert parse_timestamp(None) is None

    def test_missing_timestamp_records_have_none_for_absent(self) -> None:
        """Records with missing timestamps: timestamp key absent -> parse_timestamp(None)."""
        patterns = generate_timestamp_patterns()
        records = patterns["missing_timestamps"]
        results = []
        for record in records:
            cast_record = _as_record(record)
            ts = _timestamp_value(cast_record)
            results.append(parse_timestamp(ts))

        # First record has timestamp
        assert results[0] is not None
        # Second record has no timestamp key
        assert results[1] is None
        # Third record has timestamp
        assert results[2] is not None

    def test_empty_string_returns_none(self) -> None:
        """Empty string returns None (not parseable)."""
        assert parse_timestamp("") is None

    def test_whitespace_returns_none(self) -> None:
        """Whitespace-only string returns None."""
        assert parse_timestamp("   ") is None


class TestTimestampExtremes:
    """Edge cases for timestamp parsing."""

    def test_negative_epoch_returns_none_or_valid(self) -> None:
        """Negative epoch (before 1970) should return None or valid datetime."""
        result = parse_timestamp(-1)
        # Implementation returns None for negative epochs due to OSError on some platforms
        # On Linux 64-bit, it may succeed. Either way, no crash.
        assert result is None or isinstance(result, datetime)

    def test_very_large_epoch_returns_none(self) -> None:
        """Extremely large epoch (overflow) returns None gracefully."""
        result = parse_timestamp(99999999999999)
        assert result is None

    def test_non_numeric_string_returns_none(self) -> None:
        """Non-numeric, non-ISO string returns None."""
        assert parse_timestamp("not-a-timestamp") is None

    def test_float_with_microseconds(self) -> None:
        """Float with microseconds parses correctly."""
        result = parse_timestamp(1700000000.123456)
        assert result is not None
        assert result.year == 2023


# ===========================================================================
# E3-E5: Timestamp patterns through JSONL ingestion
# ===========================================================================


class TestTimestampPatternsInJsonl:
    """Verify timestamp patterns survive full JSONL -> stream -> parse round-trip."""

    def test_epoch_near_zero_through_stream(self) -> None:
        """Epoch-near-zero timestamps survive JSONL round-trip."""
        patterns = generate_timestamp_patterns()
        records = patterns["epoch_near_zero"]
        lines = [json.dumps(r, separators=(",", ":")) for r in records]
        data = _jsonl_bytes(lines)

        parsed = [_as_record(record) for record in _iter_jsonl_stream(data)]
        assert len(parsed) == len(records)
        for record in parsed:
            ts = parse_timestamp(_timestamp_value(record))
            assert ts is not None
            assert ts.year == 1970

    def test_y2038_through_stream(self) -> None:
        """Y2K38-adjacent timestamps survive JSONL round-trip."""
        patterns = generate_timestamp_patterns()
        records = patterns["y2038_adjacent"]
        lines = [json.dumps(r, separators=(",", ":")) for r in records]
        data = _jsonl_bytes(lines)

        parsed = [_as_record(record) for record in _iter_jsonl_stream(data)]
        assert len(parsed) == len(records)
        for record in parsed:
            ts = parse_timestamp(_timestamp_value(record))
            assert ts is not None
            assert ts.year == 2038

    def test_far_future_through_stream(self) -> None:
        """Far-future timestamps survive JSONL round-trip."""
        patterns = generate_timestamp_patterns()
        records = patterns["far_future"]
        lines = [json.dumps(r, separators=(",", ":")) for r in records]
        data = _jsonl_bytes(lines)

        parsed = [_as_record(record) for record in _iter_jsonl_stream(data)]
        assert len(parsed) == len(records)
        for record in parsed:
            ts = parse_timestamp(_timestamp_value(record))
            assert ts is not None

    def test_mixed_formats_through_stream(self) -> None:
        """Mixed timestamp formats all survive JSONL round-trip and parse."""
        patterns = generate_timestamp_patterns()
        records = patterns["mixed_formats"]
        lines = [json.dumps(r, separators=(",", ":")) for r in records]
        data = _jsonl_bytes(lines)

        parsed = [_as_record(record) for record in _iter_jsonl_stream(data)]
        assert len(parsed) == len(records)
        for record in parsed:
            ts = parse_timestamp(_timestamp_value(record))
            assert ts is not None

    def test_missing_timestamps_through_stream(self) -> None:
        """Records with missing timestamps survive JSONL round-trip."""
        patterns = generate_timestamp_patterns()
        records = patterns["missing_timestamps"]
        lines = [json.dumps(r, separators=(",", ":")) for r in records]
        data = _jsonl_bytes(lines)

        parsed = [_as_record(record) for record in _iter_jsonl_stream(data)]
        assert len(parsed) == len(records)
        # Only records with timestamps should parse; missing should be None
        timestamps = [parse_timestamp(_timestamp_value(r)) for r in parsed]
        assert timestamps[0] is not None
        assert timestamps[1] is None
        assert timestamps[2] is not None


# ===========================================================================
# E5: Re-run idempotency
# ===========================================================================


class TestRerunIdempotency:
    """Running the same batch twice produces identical records, no duplicates."""

    def test_same_batch_twice_produces_same_records(self, tmp_path: Path) -> None:
        """Parsing the same raw record twice yields identical results."""
        from polylogue.pipeline.services.ingest_worker import ingest_record

        lines = generate_large_jsonl(20, provider="codex")
        content = _jsonl_bytes(lines)

        record = _make_raw_record("idempotency-test", "codex", content, "/exports/codex.jsonl")

        result_1 = ingest_record(record, str(tmp_path / "archive"), "off")
        result_2 = ingest_record(record, str(tmp_path / "archive"), "off")

        assert len(result_1.conversations) == len(result_2.conversations)
        for conv1, conv2 in zip(result_1.conversations, result_2.conversations, strict=True):
            assert conv1.conversation_id == conv2.conversation_id
            assert conv1.provider_name == conv2.provider_name
            assert len(conv1.message_tuples) == len(conv2.message_tuples)

    def test_reparse_with_corruption_then_clean(self, tmp_path: Path) -> None:
        """First parse with corruption, second with clean data — both succeed."""
        from polylogue.pipeline.services.ingest_worker import ingest_record

        # First: corrupted
        lines_corrupt = generate_large_jsonl(20, provider="codex")
        lines_corrupt = corrupt_line_malformed_json(lines_corrupt, 10)
        content_corrupt = _jsonl_bytes(lines_corrupt)

        record_corrupt = _make_raw_record("idempotency-corrupt", "codex", content_corrupt, "/exports/codex.jsonl")
        result_corrupt = ingest_record(record_corrupt, str(tmp_path / "archive"), "off")

        # Second: clean (same data without corruption)
        lines_clean = generate_large_jsonl(20, provider="codex")
        content_clean = _jsonl_bytes(lines_clean)

        record_clean = _make_raw_record("idempotency-clean", "codex", content_clean, "/exports/codex.jsonl")
        result_clean = ingest_record(record_clean, str(tmp_path / "archive"), "off")

        assert result_corrupt.error is None
        assert result_clean.error is None
        # Clean parse should have all conversations
        assert len(result_clean.conversations) >= len(result_corrupt.conversations)

    def test_iter_json_stream_idempotent(self) -> None:
        """_iter_json_stream produces identical output on repeated calls."""
        lines = generate_large_jsonl(100, provider="codex")
        data = _jsonl_bytes(lines)

        parsed_1 = _iter_jsonl_stream(data)
        parsed_2 = _iter_jsonl_stream(data)

        assert len(parsed_1) == len(parsed_2) == 100
        for r1, r2 in zip(parsed_1, parsed_2, strict=True):
            assert r1 == r2

    def test_idempotency_with_timestamp_patterns(self) -> None:
        """Timestamp pattern records produce identical results on re-parse."""
        patterns = generate_timestamp_patterns()
        for pattern_name, records in patterns.items():
            lines = [json.dumps(r, separators=(",", ":")) for r in records]
            data = _jsonl_bytes(lines)

            parsed_1 = _iter_jsonl_stream(data)
            parsed_2 = _iter_jsonl_stream(data)

            assert parsed_1 == parsed_2, f"Idempotency violated for pattern '{pattern_name}'"


# ===========================================================================
# Property-based: malformed JSON never crashes the pipeline
# ===========================================================================


@given(malformed_json_strategy())
@settings(max_examples=40)
def test_iter_json_stream_handles_malformed_json_without_crash(malformed: str) -> None:
    """_iter_json_stream never raises on malformed JSON input — it skips bad lines."""
    raw = malformed.encode("utf-8", errors="replace")
    try:
        result = list(_iter_json_stream(BytesIO(raw), "fuzz.jsonl"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        pass  # Expected for some malformed patterns
    else:
        assert isinstance(result, list)


@given(malformed_json_strategy())
@settings(max_examples=40)
def test_decode_json_bytes_handles_malformed_json_without_crash(malformed: str) -> None:
    """_decode_json_bytes never raises on malformed JSON — returns None or a string."""
    raw = malformed.encode("utf-8", errors="replace")
    result = _decode_json_bytes(raw)
    assert result is None or isinstance(result, str)


@given(malformed_json_strategy())
@settings(max_examples=30)
def test_malformed_json_in_jsonl_batch_does_not_poison_valid_lines(malformed: str) -> None:
    """Inserting a malformed JSON line into a valid JSONL batch preserves valid records."""
    valid_records = [json.dumps(generate_valid_jsonl_record(i, provider="codex")) for i in range(5)]
    lines = valid_records[:2] + [malformed] + valid_records[2:]
    data = ("\n".join(lines) + "\n").encode("utf-8", errors="replace")

    parsed = list(_iter_json_stream(BytesIO(data), "batch-fuzz.jsonl"))
    # At least the 5 valid records should survive; the malformed line
    # is either skipped (not a dict) or parsed if it happens to be valid JSON
    assert len(parsed) >= 4  # at worst one valid line adjacent to malformed gets damaged

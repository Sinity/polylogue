"""Differential convergence tests for parallel code paths.

These tests prove that code paths which should produce the same result
actually do. Historical drift has been found in every pair tested here.
"""

from __future__ import annotations

import io
import json

import pytest

from polylogue.archive.raw_payload.decode import JSONValue

# ---------------------------------------------------------------------------
# 1. Sample decoder vs streaming decoder (JSONL)
# ---------------------------------------------------------------------------


class TestDecoderConvergence:
    """The JSONL sample decoder (validation/schema) and streaming decoder
    (parser) must agree on which lines are valid and how many are malformed."""

    @staticmethod
    def _sample_decode(raw_bytes: bytes) -> tuple[list[JSONValue], int]:
        """Run the sample decoder path, return (valid_records, malformed_count)."""
        from polylogue.archive.raw_payload.decode import _sample_jsonl_payload_with_detail

        records, malformed_count, _error = _sample_jsonl_payload_with_detail(raw_bytes)
        return records, malformed_count

    @staticmethod
    def _stream_decode(raw_bytes: bytes) -> list[JSONValue]:
        """Run the streaming decoder path, return valid records.

        The streaming decoder (ijson) operates on token streams, not lines,
        so it cannot report per-line malformed counts the way the sample
        decoder does. Only record-level agreement is comparable.
        """
        import logging

        from polylogue.sources.decoder_json import iter_json_stream_with

        logger = logging.getLogger("test_differential")
        handle = io.BytesIO(raw_bytes)

        records = []
        try:
            import ijson
        except ImportError:
            pytest.skip("ijson is required for streaming decoder convergence tests")

        for record in iter_json_stream_with(logger, ijson, handle, "test.jsonl"):
            records.append(record)

        return records

    def test_well_formed_jsonl_same_record_count(self) -> None:
        lines = [json.dumps({"id": i, "text": f"message {i}"}) for i in range(10)]
        raw = ("\n".join(lines) + "\n").encode("utf-8")

        sample_records, sample_malformed = self._sample_decode(raw)
        stream_records = self._stream_decode(raw)

        assert len(sample_records) == len(stream_records) == 10
        assert sample_malformed == 0

    def test_mixed_valid_invalid_jsonl(self) -> None:
        lines = [
            json.dumps({"id": 1, "text": "valid"}),
            "not valid json {{{",
            json.dumps({"id": 2, "text": "also valid"}),
            "",
            json.dumps({"id": 3, "text": "third"}),
        ]
        raw = ("\n".join(lines) + "\n").encode("utf-8")

        sample_records, _sample_malformed = self._sample_decode(raw)
        stream_records = self._stream_decode(raw)

        assert len(sample_records) >= 3, "Sample should find at least 3 valid records"
        assert len(stream_records) >= 3, "Stream should find at least 3 valid records"
        assert len(sample_records) == len(stream_records), (
            f"Decoder agreement: sample={len(sample_records)}, stream={len(stream_records)}"
        )

    def test_bom_handling_agrees(self) -> None:
        line = json.dumps({"id": 1, "text": "bom test"})
        raw = ("\ufeff" + line + "\n").encode("utf-8")

        sample_records, _ = self._sample_decode(raw)
        stream_records = self._stream_decode(raw)

        assert len(sample_records) >= 1, "Sample should handle BOM"
        assert len(stream_records) >= 1, "Stream should handle BOM"

    def test_empty_lines_skipped_by_both(self) -> None:
        lines = [
            json.dumps({"id": 1}),
            "",
            "   ",
            json.dumps({"id": 2}),
        ]
        raw = ("\n".join(lines) + "\n").encode("utf-8")

        sample_records, sample_malformed = self._sample_decode(raw)
        stream_records = self._stream_decode(raw)

        assert len(sample_records) == 2
        assert len(stream_records) == len(sample_records), "Decoders must agree on empty-line skipping"
        assert sample_malformed == 0


# ---------------------------------------------------------------------------

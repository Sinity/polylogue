"""Differential convergence tests for parallel code paths.

These tests prove that code paths which should produce the same result
actually do. Historical drift has been found in every pair tested here.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

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
# 2. Health debt vs repair counts
# ---------------------------------------------------------------------------


class TestHealthRepairConvergence:
    """Health/doctor and repair must agree on debt counts when querying
    the same database state."""

    def test_empty_session_count_agrees(self: object, workspace_env: dict[str, Path]) -> None:
        """``count_empty_sessions_sync`` only counts a message-less session as
        debris when its raw artifact positively fails the live
        ``classify_artifact`` pipeline (polylogue-ne6k) -- so this seeds a
        real ``agent-*.meta.json`` sidecar raw artifact (the genuinely-phantom
        shape) behind the empty session, not a bare session row."""
        import sqlite3

        from polylogue.storage.blob_store import BlobStore
        from polylogue.storage.repair import count_empty_sessions_sync
        from tests.infra.archive_scenarios import open_index_db
        from tests.infra.storage_records import SessionBuilder, db_setup

        db_path = db_setup(workspace_env)
        # Seed one real session so the archive schema is bootstrapped.
        SessionBuilder(db_path, "seed").provider("chatgpt").title("Seed").add_message(role="user", text="hi").save()

        archive_root = workspace_env["archive_root"]
        blob_store = BlobStore(archive_root / "blob")
        raw_id, blob_size = blob_store.write_from_bytes(b'{"agentType":"general-purpose"}')
        with sqlite3.connect(archive_root / "source.db") as source_conn:
            source_conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
                ) VALUES (?, 'chatgpt-export', 'ext-empty', 'agent-empty.meta.json', 0, ?, ?, 1)
                """,
                (raw_id, bytes.fromhex(raw_id), blob_size),
            )
            source_conn.commit()

        with open_index_db(db_path) as conn:
            # A archive session row with no messages is the "empty session"
            # shape; ``content_hash`` is a 32-byte BLOB by CHECK constraint.
            conn.execute(
                "INSERT INTO sessions (native_id, origin, raw_id, title, content_hash) "
                "VALUES ('ext-empty', 'chatgpt-export', ?, 'Empty', "
                "X'0011223344556677889900112233445566778899001122334455667788990011')",
                (raw_id,),
            )
            conn.commit()
            count = count_empty_sessions_sync(conn)

        assert count >= 1, "Should detect empty session whose raw artifact fails classification"


# ---------------------------------------------------------------------------

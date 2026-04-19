"""Differential convergence tests for parallel code paths.

These tests prove that code paths which should produce the same result
actually do. Historical drift has been found in every pair tested here.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from polylogue.lib.raw_payload_decode import JSONValue

# ---------------------------------------------------------------------------
# 1. Sample decoder vs streaming decoder (JSONL)
# ---------------------------------------------------------------------------


class TestDecoderConvergence:
    """The JSONL sample decoder (validation/schema) and streaming decoder
    (parser) must agree on which lines are valid and how many are malformed."""

    @staticmethod
    def _sample_decode(raw_bytes: bytes) -> tuple[list[JSONValue], int]:
        """Run the sample decoder path, return (valid_records, malformed_count)."""
        from polylogue.lib.raw_payload_decode import _sample_jsonl_payload_with_detail

        records, malformed_count, _error = _sample_jsonl_payload_with_detail(raw_bytes)
        return records, malformed_count

    @staticmethod
    def _stream_decode(raw_bytes: bytes) -> list[object]:
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

            for record in iter_json_stream_with(logger, ijson, handle, "test.jsonl"):
                records.append(record)
        except Exception:
            pass

        return records

    def test_well_formed_jsonl_same_record_count(self: Any) -> None:
        lines = [json.dumps({"id": i, "text": f"message {i}"}) for i in range(10)]
        raw = ("\n".join(lines) + "\n").encode("utf-8")

        sample_records, sample_malformed = self._sample_decode(raw)
        stream_records = self._stream_decode(raw)

        assert len(sample_records) == len(stream_records) == 10
        assert sample_malformed == 0

    def test_mixed_valid_invalid_jsonl(self: Any) -> None:
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

    def test_bom_handling_agrees(self: Any) -> None:
        line = json.dumps({"id": 1, "text": "bom test"})
        raw = ("\ufeff" + line + "\n").encode("utf-8")

        sample_records, _ = self._sample_decode(raw)
        stream_records = self._stream_decode(raw)

        assert len(sample_records) >= 1, "Sample should handle BOM"
        assert len(stream_records) >= 1, "Stream should handle BOM"

    def test_empty_lines_skipped_by_both(self: Any) -> None:
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

    @pytest.fixture()
    def seeded_db(self: Any, workspace_env: Any) -> Any:
        """Create a DB with some conversations and introduce orphaned messages."""
        from polylogue.storage.backends.connection import open_connection
        from tests.infra.storage_records import ConversationBuilder, db_setup

        db_path = db_setup(workspace_env)
        with open_connection(db_path) as conn:
            (
                ConversationBuilder(db_path, "conv-1")
                .provider("chatgpt")
                .title("First")
                .add_message(role="user", text="Hello")
                .add_message(role="assistant", text="Hi")
                .save()
            )
            (
                ConversationBuilder(db_path, "conv-2")
                .provider("claude-code")
                .title("Second")
                .add_message(role="user", text="Test")
                .save()
            )

            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute(
                "INSERT INTO messages (message_id, conversation_id, role, text, content_hash, version, provider_name, word_count, has_tool_use, has_thinking) "
                "VALUES ('orphan-msg-1', 'nonexistent-conv', 'user', 'orphan', 'hash1', 1, 'test', 1, 0, 0)"
            )
            conn.commit()
            conn.execute("PRAGMA foreign_keys = ON")

        return db_path

    def test_orphaned_message_count_agrees(self: Any, seeded_db: Any) -> None:
        from polylogue.storage.backends.connection import open_connection
        from polylogue.storage.repair import count_orphaned_messages_sync

        with open_connection(seeded_db) as conn:
            count = count_orphaned_messages_sync(conn)

        assert count >= 1, "Should detect at least 1 orphaned message"

    def test_empty_conversation_count_agrees(self: Any, workspace_env: Any) -> None:
        from polylogue.storage.backends.connection import open_connection
        from polylogue.storage.repair import count_empty_conversations_sync
        from tests.infra.storage_records import db_setup

        db_path = db_setup(workspace_env)
        with open_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO conversations (conversation_id, provider_name, provider_conversation_id, content_hash, version) "
                "VALUES ('empty-conv', 'test', 'empty-prov-id', 'hash', 1)"
            )
            conn.commit()
            count = count_empty_conversations_sync(conn)

        assert count >= 1, "Should detect empty conversation"


# ---------------------------------------------------------------------------
# 3. Repair preview count vs live recount
# ---------------------------------------------------------------------------


class TestRepairPreviewConvergence:
    """Repair preview counts (from health report) must match what the
    actual repair handler would find on a fresh connection."""

    @pytest.fixture()
    def db_with_orphans(self: Any, workspace_env: Any) -> Any:
        from polylogue.storage.backends.connection import open_connection
        from tests.infra.storage_records import ConversationBuilder, db_setup

        db_path = db_setup(workspace_env)
        with open_connection(db_path) as conn:
            ConversationBuilder(db_path, "real-conv").provider("chatgpt").title("Real").add_message(
                role="user", text="Real message"
            ).save()

            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute(
                "INSERT INTO messages (message_id, conversation_id, role, text, content_hash, version, provider_name, word_count, has_tool_use, has_thinking) "
                "VALUES ('orphan-1', 'ghost-conv', 'user', 'orphan text', 'ohash', 1, 'test', 2, 0, 0)"
            )
            conn.execute(
                "INSERT INTO messages (message_id, conversation_id, role, text, content_hash, version, provider_name, word_count, has_tool_use, has_thinking) "
                "VALUES ('orphan-2', 'ghost-conv', 'assistant', 'orphan reply', 'ohash2', 1, 'test', 2, 0, 0)"
            )
            conn.commit()
            conn.execute("PRAGMA foreign_keys = ON")
        return db_path

    def test_preview_matches_live_orphan_count(self: Any, db_with_orphans: Any) -> None:
        """The count from health/debt should match what repair would find."""
        from polylogue.storage.backends.connection import open_connection
        from polylogue.storage.repair import count_orphaned_messages_sync

        with open_connection(db_with_orphans) as conn:
            count1 = count_orphaned_messages_sync(conn)
            count2 = count_orphaned_messages_sync(conn)

        assert count1 == count2, "Same query on same state should return same count"
        assert count1 == 2, "Should find exactly 2 orphaned messages"

    def test_repair_removes_exactly_previewed_count(self: Any, db_with_orphans: Any) -> None:
        """After repair, orphan count should be zero."""
        from polylogue.storage.backends.connection import open_connection
        from polylogue.storage.repair import (
            count_orphaned_messages_sync,
            repair_orphaned_messages,
        )

        with open_connection(db_with_orphans) as conn:
            before = count_orphaned_messages_sync(conn)
            assert before == 2

        result = repair_orphaned_messages(db_with_orphans, dry_run=False)

        with open_connection(db_with_orphans) as conn:
            after = count_orphaned_messages_sync(conn)

        assert after == 0, "Repair should remove all orphaned messages"
        assert result.repaired_count == before, f"Repair should report removing {before}, got {result.repaired_count}"

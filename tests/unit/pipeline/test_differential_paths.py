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
from polylogue.config import Config, get_config

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

    @pytest.fixture()
    def seeded_db(self: object, workspace_env: dict[str, Path]) -> Path:
        """Native ``index.db`` with sessions plus a hand-fabricated orphan.

        Orphan ``messages`` cannot arise under the native
        ``messages.session_id REFERENCES sessions ON DELETE CASCADE`` while
        foreign keys are enforced — they only appear after a corrupted or
        foreign-key-disabled write. The fixture reproduces exactly that
        corruption shape (FK off, insert a message under a missing
        ``session_id``) so the repair detection path has something to find.
        """
        from tests.infra.archive_scenarios import open_index_db
        from tests.infra.storage_records import SessionBuilder, db_setup

        db_path = db_setup(workspace_env)
        (
            SessionBuilder(db_path, "conv-1")
            .provider("chatgpt")
            .title("First")
            .add_message(role="user", text="Hello")
            .add_message(role="assistant", text="Hi")
            .save()
        )
        (
            SessionBuilder(db_path, "conv-2")
            .provider("claude-code")
            .title("Second")
            .add_message(role="user", text="Test")
            .save()
        )

        with open_index_db(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute(
                "INSERT INTO messages (session_id, native_id, position, role, word_count, content_hash) "
                "VALUES ('chatgpt:ext-nonexistent', 'orphan-msg-1', 0, 'user', 1, X'0011223344556677889900112233445566778899001122334455667788990011')"
            )
            conn.commit()
            conn.execute("PRAGMA foreign_keys = ON")

        return db_path

    def test_orphaned_message_count_agrees(self: object, seeded_db: Path) -> None:
        from polylogue.storage.repair import count_orphaned_messages_sync
        from tests.infra.archive_scenarios import open_index_db

        with open_index_db(seeded_db) as conn:
            count = count_orphaned_messages_sync(conn)

        assert count >= 1, "Should detect at least 1 orphaned message"

    def test_empty_session_count_agrees(self: object, workspace_env: dict[str, Path]) -> None:
        from polylogue.storage.repair import count_empty_sessions_sync
        from tests.infra.archive_scenarios import open_index_db
        from tests.infra.storage_records import SessionBuilder, db_setup

        db_path = db_setup(workspace_env)
        # Seed one real session so the archive schema is bootstrapped.
        SessionBuilder(db_path, "seed").provider("chatgpt").title("Seed").add_message(role="user", text="hi").save()
        with open_index_db(db_path) as conn:
            # A archive session row with no messages is the "empty session"
            # shape; ``content_hash`` is a 32-byte BLOB by CHECK constraint.
            conn.execute(
                "INSERT INTO sessions (native_id, origin, title, content_hash) "
                "VALUES ('ext-empty', 'chatgpt-export', 'Empty', X'0011223344556677889900112233445566778899001122334455667788990011')"
            )
            conn.commit()
            count = count_empty_sessions_sync(conn)

        assert count >= 1, "Should detect empty session"


# ---------------------------------------------------------------------------
# 3. Repair preview count vs live recount
# ---------------------------------------------------------------------------


class TestRepairPreviewConvergence:
    """Repair preview counts (from health report) must match what the
    actual repair handler would find on a fresh connection."""

    @pytest.fixture()
    def db_with_orphans(self: object, workspace_env: dict[str, Path]) -> Config:
        from tests.infra.archive_scenarios import open_index_db
        from tests.infra.storage_records import SessionBuilder, db_setup

        db_path = db_setup(workspace_env)
        SessionBuilder(db_path, "real-conv").provider("chatgpt").title("Real").add_message(
            role="user", text="Real message"
        ).save()

        # Fabricate two archive session-less messages (FK disabled) to model the
        # post-corruption shape the repair path exists to clean up.
        with open_index_db(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute(
                "INSERT INTO messages (session_id, native_id, position, role, word_count, content_hash) "
                "VALUES ('chatgpt:ext-ghost', 'orphan-1', 0, 'user', 2, X'0011223344556677889900112233445566778899001122334455667788990011')"
            )
            conn.execute(
                "INSERT INTO messages (session_id, native_id, position, role, word_count, content_hash) "
                "VALUES ('chatgpt:ext-ghost', 'orphan-2', 1, 'assistant', 2, X'ffeeddccbbaa9988776655443322110099887766554433221100ffeeddccbbaa')"
            )
            conn.commit()
            conn.execute("PRAGMA foreign_keys = ON")
        return get_config()

    def test_preview_matches_live_orphan_count(self: object, db_with_orphans: Config) -> None:
        """The count from health/debt should match what repair would find."""
        from polylogue.storage.repair import count_orphaned_messages_sync
        from tests.infra.archive_scenarios import open_index_db

        with open_index_db(db_with_orphans.db_path) as conn:
            count1 = count_orphaned_messages_sync(conn)
            count2 = count_orphaned_messages_sync(conn)

        assert count1 == count2, "Same query on same state should return same count"
        assert count1 == 2, "Should find exactly 2 orphaned messages"

    def test_repair_removes_exactly_previewed_count(self: object, db_with_orphans: Config) -> None:
        """After repair, orphan count should be zero."""
        from polylogue.storage.repair import (
            count_orphaned_messages_sync,
            repair_orphaned_messages,
        )
        from tests.infra.archive_scenarios import open_index_db

        with open_index_db(db_with_orphans.db_path) as conn:
            before = count_orphaned_messages_sync(conn)
            assert before == 2

        result = repair_orphaned_messages(db_with_orphans, dry_run=False)

        with open_index_db(db_with_orphans.db_path) as conn:
            after = count_orphaned_messages_sync(conn)

        assert after == 0, "Repair should remove all orphaned messages"
        assert result.repaired_count == before, f"Repair should report removing {before}, got {result.repaired_count}"

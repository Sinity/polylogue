"""Direct tests for polylogue.schemas.sampling module.

Covers ProviderConfig, PROVIDERS dict, load_samples_from_db with missing DB,
and load_samples_from_sessions with JSONL fixtures.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from polylogue.schemas.observation import PROVIDERS, ProviderConfig
from polylogue.schemas.sampling import load_samples_from_db, load_samples_from_sessions


class TestProviderConfig:
    def test_init_minimal(self) -> None:
        config = ProviderConfig(
            name="test",
            description="Test provider",
        )
        assert config.name == "test"
        assert config.description == "Test provider"
        assert config.db_provider_name is None
        assert config.session_dir is None

    def test_init_with_db_provider(self) -> None:
        config = ProviderConfig(
            name="test",
            description="Test",
            db_provider_name="test_db",
        )
        assert config.db_provider_name == "test_db"

    def test_init_with_session_dir(self) -> None:
        path = Path.home() / ".test"
        config = ProviderConfig(
            name="test",
            description="Test",
            session_dir=path,
        )
        assert config.session_dir == path

    def test_defaults(self) -> None:
        config = ProviderConfig(name="test", description="Test")
        assert config.sample_granularity == "document"
        assert config.max_sessions is None
        assert config.record_type_key is None


class TestProvidersConfig:
    def test_all_five_providers_present(self) -> None:
        expected = {"chatgpt", "claude-code", "claude-ai", "gemini", "codex"}
        assert set(PROVIDERS.keys()) == expected

    def test_each_has_name_and_description(self) -> None:
        for name, config in PROVIDERS.items():
            assert config.name == name
            assert isinstance(config.description, str)
            assert len(config.description) > 0

    def test_chatgpt_has_db_provider(self) -> None:
        assert PROVIDERS["chatgpt"].db_provider_name == "chatgpt"

    def test_claude_ai_db_name_is_canonical(self) -> None:
        # Claude AI rows are stored under the canonical provider token.
        assert PROVIDERS["claude-ai"].db_provider_name == "claude-ai"

    def test_claude_code_has_db_provider(self) -> None:
        assert PROVIDERS["claude-code"].db_provider_name == "claude-code"

    def test_gemini_has_db_provider(self) -> None:
        assert PROVIDERS["gemini"].db_provider_name == "gemini"

    def test_codex_has_session_dir(self) -> None:
        assert PROVIDERS["codex"].session_dir is not None

    def test_codex_record_type_key(self) -> None:
        assert PROVIDERS["codex"].record_type_key == "type"

    def test_claude_code_record_type_key(self) -> None:
        assert PROVIDERS["claude-code"].record_type_key == "type"

    def test_chatgpt_no_record_type_key(self) -> None:
        assert PROVIDERS["chatgpt"].record_type_key is None

    def test_document_granularity_for_chatgpt(self) -> None:
        assert PROVIDERS["chatgpt"].sample_granularity == "document"

    def test_record_granularity_for_claude_code(self) -> None:
        assert PROVIDERS["claude-code"].sample_granularity == "record"


class TestLoadSamplesFromDb:
    def test_nonexistent_db_returns_empty(self, tmp_path: Path) -> None:
        result = load_samples_from_db("chatgpt", db_path=tmp_path / "nope.db")
        assert result == []

    def test_empty_db_returns_empty(self, tmp_path: Path) -> None:
        # Create a DB with schema but no data
        from polylogue.storage.backends.connection import open_connection
        db = tmp_path / "empty.db"
        with open_connection(db):
            pass
        result = load_samples_from_db("chatgpt", db_path=db)
        assert result == []

    def test_nonexistent_db_with_default_path(self) -> None:
        # When db_path=None and default doesn't exist, should return []
        from unittest.mock import patch

        fake_path = Path("/nonexistent/fake/db.db")
        with patch("polylogue.schemas.sampling.default_db_path", return_value=fake_path):
            result = load_samples_from_db("chatgpt")
            assert result == []

    def test_claude_ai_reads_db_rows_stored_under_claude(self, tmp_path: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        db = tmp_path / "claude.db"
        raw_content = json.dumps([
            {
                "uuid": "conv-1",
                "name": "Conversation",
                "summary": "Summary",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:05:00Z",
                "account": {"uuid": "acct-1"},
                "chat_messages": [],
            }
        ]).encode("utf-8")

        with open_connection(db) as conn:
            conn.execute(
                """
                INSERT INTO raw_conversations (
                    raw_id, provider_name, payload_provider, source_name, source_path,
                    source_index, raw_content, acquired_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "raw-claude-1",
                    "claude-ai",
                    "claude-ai",
                    "claude-ai",
                    "/tmp/conversations.json",
                    0,
                    raw_content,
                    datetime.now(tz=timezone.utc).isoformat(),
                ),
            )
            conn.commit()

        result = load_samples_from_db("claude-ai", db_path=db)
        assert len(result) == 1
        assert result[0]["uuid"] == "conv-1"

    def test_record_provider_sampling_streams_without_full_envelope(self, tmp_path: Path, monkeypatch) -> None:
        from polylogue.storage.backends.connection import open_connection

        db = tmp_path / "codex.db"
        raw_content = "\n".join(
            json.dumps(record)
            for record in [
                {"type": "session_meta", "id": "sess-1"},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x" * 2048}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
            ]
        ).encode("utf-8")

        with open_connection(db) as conn:
            conn.execute(
                """
                INSERT INTO raw_conversations (
                    raw_id, provider_name, payload_provider, source_name, source_path,
                    source_index, raw_content, acquired_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "raw-codex-1",
                    "codex",
                    "codex",
                    "codex",
                    "/tmp/session.jsonl",
                    0,
                    raw_content,
                    datetime.now(tz=timezone.utc).isoformat(),
                ),
            )
            conn.commit()

        monkeypatch.setattr(
            "polylogue.schemas.sampling.build_raw_payload_envelope",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not decode full payload")),
        )

        result = load_samples_from_db("codex", db_path=db, max_samples=2)
        assert len(result) == 2
        assert {sample["type"] for sample in result} == {"session_meta", "message"}
        message_sample = next(sample for sample in result if sample["type"] == "message")
        assert len(message_sample["content"][0]["text"]) == 1024


class TestLoadSamplesFromSessions:
    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        result = load_samples_from_sessions(tmp_path / "nope")
        assert result == []

    def test_reads_jsonl_files(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        jsonl_file = session_dir / "session-001.jsonl"
        records = [
            {"type": "session_meta", "id": "s1"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ]
        jsonl_file.write_text("\n".join(json.dumps(r) for r in records))

        result = load_samples_from_sessions(session_dir)
        assert len(result) == 2
        assert result[0]["type"] == "session_meta"
        assert result[1]["type"] == "message"

    def test_multiple_jsonl_files(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()

        for i in range(3):
            jsonl_file = session_dir / f"session-{i:03d}.jsonl"
            records = [{"type": "message", "n": j} for j in range(2)]
            jsonl_file.write_text("\n".join(json.dumps(r) for r in records))

        result = load_samples_from_sessions(session_dir)
        assert len(result) == 6  # 3 files * 2 records each

    def test_max_samples_limits_output(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        jsonl_file = session_dir / "session-001.jsonl"
        records = [{"type": "message", "n": i} for i in range(100)]
        jsonl_file.write_text("\n".join(json.dumps(r) for r in records))

        result = load_samples_from_sessions(session_dir, max_samples=5)
        assert len(result) <= 5

    def test_malformed_jsonl_skipped(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        jsonl_file = session_dir / "session-001.jsonl"
        # Valid, invalid, valid on three lines
        jsonl_file.write_text('{"valid": true}\n{invalid json\n{"also_valid": true}\n')

        result = load_samples_from_sessions(session_dir)
        assert len(result) == 2  # Only valid lines loaded

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        jsonl_file = session_dir / "session-001.jsonl"
        lines = [
            '{"a": 1}',
            '',  # empty
            '{"b": 2}',
            '   ',  # whitespace
            '{"c": 3}',
        ]
        jsonl_file.write_text("\n".join(lines))

        result = load_samples_from_sessions(session_dir)
        assert len(result) == 3

    def test_non_dict_json_skipped(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        jsonl_file = session_dir / "session-001.jsonl"
        lines = [
            '{"valid": true}',
            '[1, 2, 3]',  # array, not dict
            '"string"',  # string
            '{"also_valid": true}',
        ]
        jsonl_file.write_text("\n".join(lines))

        result = load_samples_from_sessions(session_dir)
        assert len(result) == 2

    def test_max_sessions_limits_files(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()

        # Create 5 session files
        for i in range(5):
            jsonl_file = session_dir / f"session-{i:03d}.jsonl"
            jsonl_file.write_text('{"type": "message"}\n')

        # Request only 2 sessions
        result = load_samples_from_sessions(session_dir, max_sessions=2)
        # Should have records from only 2 files
        assert len(result) <= 2

    def test_nested_session_dirs(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        subdir = session_dir / "subdir"
        subdir.mkdir()

        # File in root
        (session_dir / "session-001.jsonl").write_text('{"a": 1}\n')
        # File in subdir
        (subdir / "session-002.jsonl").write_text('{"b": 2}\n')

        result = load_samples_from_sessions(session_dir)
        # rglob should find both
        assert len(result) == 2

    def test_unreadable_file_skipped(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()

        # Valid file
        (session_dir / "session-001.jsonl").write_text('{"valid": true}\n')
        # Unreadable file (by chmod)
        bad_file = session_dir / "session-002.jsonl"
        bad_file.write_text('{"also_valid": true}\n')
        bad_file.chmod(0o000)

        try:
            result = load_samples_from_sessions(session_dir)
            # Should have at least the readable one
            assert len(result) >= 1
        finally:
            # Restore permissions for cleanup
            bad_file.chmod(0o644)

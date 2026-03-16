"""Direct tests for polylogue.schemas.sampling module.

Covers ProviderConfig, PROVIDERS dict, load_samples_from_db with missing DB,
and load_samples_from_sessions with JSONL fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.schemas.sampling import (
    PROVIDERS,
    ProviderConfig,
    load_samples_from_db,
    load_samples_from_sessions,
)


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

    def test_claude_ai_db_name_is_claude(self) -> None:
        # DB uses "claude" not "claude-ai"
        assert PROVIDERS["claude-ai"].db_provider_name == "claude"

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
        from polylogue.paths import db_path as default_db_path
        from unittest.mock import patch

        fake_path = Path("/nonexistent/fake/db.db")
        with patch("polylogue.schemas.sampling.default_db_path", return_value=fake_path):
            result = load_samples_from_db("chatgpt")
            assert result == []


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

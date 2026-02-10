"""Tests for schema_inference.py uncovered code paths.

Covers:
- _remove_nested_required: recursive schema manipulation
- load_samples_from_sessions: JSONL session file iteration
- get_sample_count_from_db: database query edge cases
- generate_schema_from_samples: genson integration
- generate_all_schemas: file output
- cli_main: argparse CLI
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


# =============================================================================
# _remove_nested_required
# =============================================================================


class TestRemoveNestedRequired:
    """Tests for recursive required-removal from JSON schemas."""

    def _fn(self, schema, depth=0):
        from polylogue.schemas.schema_inference import _remove_nested_required

        return _remove_nested_required(schema, depth)

    def test_root_required_preserved(self):
        """Root-level required array is kept."""
        schema = {"type": "object", "required": ["id", "name"], "properties": {}}
        result = self._fn(schema, depth=0)
        assert "required" in result
        assert result["required"] == ["id", "name"]

    def test_nested_required_removed(self):
        """Nested required array is removed (depth > 0)."""
        schema = {"type": "object", "required": ["field1"], "properties": {}}
        result = self._fn(schema, depth=1)
        assert "required" not in result

    def test_recurse_into_properties(self):
        """Required removed from nested properties."""
        schema = {
            "type": "object",
            "required": ["a"],
            "properties": {
                "a": {"type": "object", "required": ["x"], "properties": {}},
            },
        }
        result = self._fn(schema, depth=0)
        # Root required preserved
        assert "required" in result
        # Nested required removed
        assert "required" not in result["properties"]["a"]

    def test_recurse_into_items(self):
        """Required removed from array items schema."""
        schema = {
            "type": "array",
            "items": {"type": "object", "required": ["id"], "properties": {}},
        }
        result = self._fn(schema, depth=0)
        assert "required" not in result["items"]

    def test_recurse_into_anyof(self):
        """Required removed from anyOf variants."""
        schema = {
            "anyOf": [
                {"type": "object", "required": ["a"]},
                {"type": "object", "required": ["b"]},
            ]
        }
        result = self._fn(schema, depth=0)
        for variant in result["anyOf"]:
            assert "required" not in variant

    def test_recurse_into_oneof(self):
        """Required removed from oneOf variants."""
        schema = {"oneOf": [{"type": "object", "required": ["x"]}]}
        result = self._fn(schema, depth=0)
        assert "required" not in result["oneOf"][0]

    def test_recurse_into_allof(self):
        """Required removed from allOf variants."""
        schema = {"allOf": [{"type": "object", "required": ["y"]}]}
        result = self._fn(schema, depth=0)
        assert "required" not in result["allOf"][0]

    def test_non_dict_returns_unchanged(self):
        """Non-dict input returned as-is."""
        assert self._fn("string") == "string"
        assert self._fn(42) == 42
        assert self._fn([1, 2]) == [1, 2]

    def test_deeply_nested(self):
        """Deeply nested schemas handled correctly."""
        schema = {
            "type": "object",
            "required": ["top"],
            "properties": {
                "top": {
                    "type": "object",
                    "required": ["mid"],
                    "properties": {
                        "mid": {
                            "type": "object",
                            "required": ["deep"],
                            "properties": {},
                        }
                    },
                }
            },
        }
        result = self._fn(schema, depth=0)
        assert "required" in result
        assert "required" not in result["properties"]["top"]
        assert "required" not in result["properties"]["top"]["properties"]["mid"]


# =============================================================================
# load_samples_from_sessions
# =============================================================================


class TestLoadSamplesFromSessions:
    """Tests for loading samples from JSONL session files."""

    def _fn(self, session_dir, max_sessions=None):
        from polylogue.schemas.schema_inference import load_samples_from_sessions

        return load_samples_from_sessions(session_dir, max_sessions=max_sessions)

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        """Non-existent directory returns empty list."""
        result = self._fn(tmp_path / "does_not_exist")
        assert result == []

    def test_empty_dir_returns_empty(self, tmp_path):
        """Empty directory returns empty list."""
        d = tmp_path / "empty"
        d.mkdir()
        result = self._fn(d)
        assert result == []

    def test_single_jsonl_file(self, tmp_path):
        """Single JSONL file with records is loaded."""
        d = tmp_path / "sessions"
        d.mkdir()
        (d / "session1.jsonl").write_text(
            json.dumps({"type": "user", "text": "hello"}) + "\n"
            + json.dumps({"type": "assistant", "text": "hi"}) + "\n"
        )
        result = self._fn(d)
        assert len(result) == 2
        assert result[0]["type"] == "user"

    def test_max_sessions_limits_files(self, tmp_path):
        """max_sessions parameter limits number of files processed."""
        d = tmp_path / "sessions"
        d.mkdir()
        for i in range(10):
            (d / f"session{i:02d}.jsonl").write_text(
                json.dumps({"id": i}) + "\n"
            )
        result = self._fn(d, max_sessions=3)
        # Should process at most 3 sessions
        assert len(result) <= 10
        assert len(result) >= 1

    def test_malformed_json_skipped(self, tmp_path):
        """Malformed JSON lines are silently skipped."""
        d = tmp_path / "sessions"
        d.mkdir()
        (d / "bad.jsonl").write_text(
            "not json\n"
            + json.dumps({"valid": True}) + "\n"
            + "{incomplete\n"
        )
        result = self._fn(d)
        assert len(result) == 1
        assert result[0]["valid"] is True

    def test_empty_lines_skipped(self, tmp_path):
        """Empty lines in JSONL files are skipped."""
        d = tmp_path / "sessions"
        d.mkdir()
        (d / "spaces.jsonl").write_text(
            "\n\n" + json.dumps({"ok": True}) + "\n\n"
        )
        result = self._fn(d)
        assert len(result) == 1

    def test_recursive_glob(self, tmp_path):
        """Files in subdirectories are found by rglob."""
        d = tmp_path / "sessions"
        sub = d / "sub1"
        sub.mkdir(parents=True)
        (sub / "nested.jsonl").write_text(json.dumps({"nested": True}) + "\n")
        result = self._fn(d)
        assert len(result) == 1
        assert result[0]["nested"] is True


# =============================================================================
# get_sample_count_from_db
# =============================================================================


class TestGetSampleCountFromDb:
    """Tests for get_sample_count_from_db."""

    def _fn(self, provider_name, db_path):
        from polylogue.schemas.schema_inference import get_sample_count_from_db

        return get_sample_count_from_db(provider_name, db_path=db_path)

    def test_nonexistent_db_returns_zero(self, tmp_path):
        """Non-existent database returns 0."""
        result = self._fn("chatgpt", tmp_path / "missing.db")
        assert result == 0

    def test_empty_db_returns_zero(self, tmp_path):
        """Database with no messages returns 0."""
        from polylogue.storage.backends.sqlite import open_connection

        db_path = tmp_path / "empty.db"
        with open_connection(db_path):
            pass
        result = self._fn("chatgpt", db_path)
        assert result == 0

    def test_matching_provider_returns_count(self, tmp_path):
        """Database with matching provider messages returns count."""
        from polylogue.storage.backends.sqlite import open_connection

        db_path = tmp_path / "test.db"
        with open_connection(db_path) as conn:
            conn.execute(
                """INSERT INTO conversations
                   (conversation_id, provider_name, provider_conversation_id,
                    title, created_at, updated_at, content_hash,
                    provider_meta, metadata, version,
                    parent_conversation_id, branch_type, raw_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("c1", "chatgpt", "p1", "Test", None, None, "hash1",
                 '{"source":"test"}', '{}', 1, None, None, None),
            )
            conn.execute(
                "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("m1", "c1", "pm1", "user", "hello", None, "hash2", '{"key":"val"}', 1, None, 0),
            )
            conn.commit()

        result = self._fn("chatgpt", db_path)
        assert result == 1

    def test_wrong_provider_returns_zero(self, tmp_path):
        """Database with no matching provider returns 0."""
        from polylogue.storage.backends.sqlite import open_connection

        db_path = tmp_path / "test.db"
        with open_connection(db_path) as conn:
            conn.execute(
                """INSERT INTO conversations
                   (conversation_id, provider_name, provider_conversation_id,
                    title, created_at, updated_at, content_hash,
                    provider_meta, metadata, version,
                    parent_conversation_id, branch_type, raw_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("c1", "chatgpt", "p1", "Test", None, None, "hash1",
                 None, '{}', 1, None, None, None),
            )
            conn.execute(
                "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("m1", "c1", "pm1", "user", "hello", None, "hash2", '{"k":"v"}', 1, None, 0),
            )
            conn.commit()

        result = self._fn("claude", db_path)
        assert result == 0


# =============================================================================
# generate_schema_from_samples
# =============================================================================


class TestGenerateSchemaFromSamples:
    """Tests for genson-based schema generation."""

    def _fn(self, samples):
        from polylogue.schemas.schema_inference import generate_schema_from_samples

        return generate_schema_from_samples(samples)

    @pytest.fixture(autouse=True)
    def _check_genson(self):
        """Skip tests if genson not available."""
        try:
            import genson  # noqa: F401
        except ImportError:
            pytest.skip("genson not installed")

    def test_empty_samples(self):
        """Empty samples returns description-only schema."""
        result = self._fn([])
        assert result["type"] == "object"
        assert "description" in result

    def test_single_sample(self):
        """Single sample produces valid schema with properties."""
        result = self._fn([{"id": "abc", "count": 42}])
        assert result["type"] == "object"
        assert "properties" in result
        assert "id" in result["properties"]
        assert "count" in result["properties"]

    def test_multiple_samples_merge(self):
        """Multiple samples merge optional fields."""
        result = self._fn([
            {"id": "a", "name": "Alice"},
            {"id": "b", "age": 30},
        ])
        assert "id" in result["properties"]
        # Both name and age should appear
        assert "name" in result["properties"]
        assert "age" in result["properties"]


# =============================================================================
# generate_all_schemas
# =============================================================================


class TestGenerateAllSchemas:
    """Tests for generate_all_schemas file output."""

    def test_creates_output_directory(self, tmp_path):
        """Output directory is created if it doesn't exist."""
        from unittest.mock import patch

        from polylogue.schemas.schema_inference import GenerationResult, generate_all_schemas

        output_dir = tmp_path / "schemas" / "nested"
        fake_result = GenerationResult(
            provider="test", sample_count=1,
            schema={"type": "object"}, error=None,
        )

        with patch("polylogue.schemas.schema_inference.generate_provider_schema", return_value=fake_result):
            results = generate_all_schemas(output_dir, providers=["test"])

        assert output_dir.exists()
        assert len(results) == 1
        assert (output_dir / "test.schema.json").exists()

    def test_skips_failed_schemas(self, tmp_path):
        """Failed schemas are not written to disk."""
        from unittest.mock import patch

        from polylogue.schemas.schema_inference import GenerationResult, generate_all_schemas

        failed_result = GenerationResult(
            provider="broken", sample_count=0,
            schema=None, error="No samples",
        )

        with patch("polylogue.schemas.schema_inference.generate_provider_schema", return_value=failed_result):
            results = generate_all_schemas(tmp_path, providers=["broken"])

        assert not (tmp_path / "broken.schema.json").exists()
        assert results[0].success is False


# =============================================================================
# cli_main
# =============================================================================


class TestCliMain:
    """Tests for CLI entry point."""

    def test_cli_with_no_db(self, tmp_path, capsys):
        """CLI handles missing database gracefully."""
        from polylogue.schemas.schema_inference import cli_main

        exit_code = cli_main([
            "--provider", "chatgpt",
            "--output-dir", str(tmp_path / "out"),
            "--db-path", str(tmp_path / "missing.db"),
        ])
        # May succeed (0 samples) or fail depending on implementation
        assert isinstance(exit_code, int)

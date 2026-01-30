"""Tests for schema generation module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.schemas.schema_inference import (
    PROVIDERS,
    GenerationResult,
    collapse_dynamic_keys,
    generate_provider_schema,
    generate_schema_from_samples,
    is_dynamic_key,
    load_samples_from_db,
)


class TestDynamicKeyDetection:
    """Test dynamic key detection for schema collapsing."""

    def test_uuid_detected(self):
        assert is_dynamic_key("550e8400-e29b-41d4-a716-446655440000")
        assert is_dynamic_key("550E8400-E29B-41D4-A716-446655440000")

    def test_long_hex_detected(self):
        assert is_dynamic_key("a" * 24)
        assert is_dynamic_key("abcdef0123456789abcdef01")

    def test_message_ids_detected(self):
        assert is_dynamic_key("msg-abc123")
        assert is_dynamic_key("node-550e8400")
        assert is_dynamic_key("conv-123abc")

    def test_normal_keys_not_detected(self):
        assert not is_dynamic_key("content")
        assert not is_dynamic_key("role")
        assert not is_dynamic_key("text")
        assert not is_dynamic_key("id")


class TestSchemaCollapsing:
    """Test dynamic key collapsing in schemas."""

    def test_collapse_uuid_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "550e8400-e29b-41d4-a716-446655440000": {"type": "object"},
                "660f9511-f3ac-52e5-b827-557766551111": {"type": "object"},
            },
        }

        collapsed = collapse_dynamic_keys(schema)

        assert "title" in collapsed["properties"]
        assert "550e8400-e29b-41d4-a716-446655440000" not in collapsed["properties"]
        assert "additionalProperties" in collapsed

    def test_preserve_static_keys(self):
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "content": {"type": "object"},
            },
        }

        collapsed = collapse_dynamic_keys(schema)

        assert "id" in collapsed["properties"]
        assert "content" in collapsed["properties"]
        assert "additionalProperties" not in collapsed


class TestSchemaGeneration:
    """Test schema generation from samples."""

    def test_generate_from_simple_samples(self):
        samples = [
            {"role": "user", "text": "hello"},
            {"role": "assistant", "text": "hi there"},
        ]

        schema = generate_schema_from_samples(samples)

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "role" in schema["properties"]
        assert "text" in schema["properties"]

    def test_generate_from_complex_samples(self):
        samples = [
            {"id": "1", "content": {"parts": ["text"]}},
            {"id": "2", "content": {"parts": ["more text"], "type": "text"}},
        ]

        schema = generate_schema_from_samples(samples)

        assert "content" in schema["properties"]
        assert "parts" in schema["properties"]["content"]["properties"]

    def test_empty_samples_returns_placeholder(self):
        schema = generate_schema_from_samples([])
        assert "No samples available" in schema.get("description", "")


class TestProviderSchemaGeneration:
    """Test full provider schema generation."""

    @pytest.fixture
    def db_path(self):
        path = Path.home() / ".local/state/polylogue/polylogue.db"
        if not path.exists():
            pytest.skip("Polylogue database not found")
        return path

    def test_known_providers(self):
        """All configured providers should be known."""
        expected = {"chatgpt", "claude-code", "claude-ai", "gemini", "codex"}
        assert set(PROVIDERS.keys()) == expected

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "claude-ai", "gemini"])
    def test_generate_schema_from_db(self, db_path, provider):
        """Generate schema from database for each provider."""
        result = generate_provider_schema(provider, db_path=db_path, max_samples=100)

        # Should succeed (unless no data)
        if result.sample_count > 0:
            assert result.success, f"Failed: {result.error}"
            assert result.schema is not None
            assert result.schema.get("type") == "object"
            assert "properties" in result.schema

    def test_unknown_provider_returns_error(self):
        result = generate_provider_schema("unknown-provider")
        assert not result.success
        assert "Unknown provider" in (result.error or "")

    def test_result_dataclass(self):
        """GenerationResult should have correct properties."""
        success = GenerationResult(provider="test", schema={"type": "object"}, sample_count=10)
        assert success.success
        assert success.provider == "test"
        assert success.sample_count == 10

        failure = GenerationResult(provider="test", schema=None, sample_count=0, error="no data")
        assert not failure.success


class TestLoadSamples:
    """Test sample loading functions."""

    @pytest.fixture
    def db_path(self):
        path = Path.home() / ".local/state/polylogue/polylogue.db"
        if not path.exists():
            pytest.skip("Polylogue database not found")
        return path

    def test_load_limited_samples(self, db_path):
        """Should respect max_samples limit."""
        samples = load_samples_from_db("chatgpt", db_path=db_path, max_samples=10)

        # Should have at most 10 samples (may have fewer if DB has less)
        assert len(samples) <= 10

    def test_load_nonexistent_provider(self, db_path):
        """Unknown provider returns empty list."""
        samples = load_samples_from_db("nonexistent-provider", db_path=db_path)
        assert samples == []

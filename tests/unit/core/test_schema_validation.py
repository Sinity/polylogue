"""Focused validator and validation-contract tests for schema handling."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from polylogue.schemas import ValidationResult
from polylogue.schemas.validator import SchemaValidator, validate_provider_export


def test_schema_validator_loads_provider(mock_schema_dir):
    """SchemaValidator.for_provider loads the requested schema."""
    with patch("polylogue.schemas.registry.SCHEMA_DIR", mock_schema_dir):
        SchemaValidator._cache.clear()
        validator = SchemaValidator.for_provider("test-provider")
        assert validator.schema["required"] == ["id"]

        with pytest.raises(FileNotFoundError):
            SchemaValidator.for_provider("nonexistent")


def test_validate_valid_data(mock_schema_dir):
    """Valid payloads should pass with no errors or drift."""
    with patch("polylogue.schemas.registry.SCHEMA_DIR", mock_schema_dir):
        validator = SchemaValidator.for_provider("test-provider")
        result = validator.validate({"id": "123", "count": 10, "meta": {"source": "test"}})

    assert result.is_valid
    assert not result.errors
    assert not result.drift_warnings


def test_validate_detects_errors(mock_schema_dir):
    """Validation should report required-field and type errors."""
    with patch("polylogue.schemas.registry.SCHEMA_DIR", mock_schema_dir):
        validator = SchemaValidator.for_provider("test-provider")

        missing = validator.validate({"count": 10})
        wrong_type = validator.validate({"id": 123})

    assert not missing.is_valid
    assert any("id" in error for error in missing.errors)
    assert not wrong_type.is_valid
    assert any("123" in error for error in wrong_type.errors)


def test_validate_detects_drift(mock_schema_dir):
    """Strict mode should surface unexpected fields as drift."""
    with patch("polylogue.schemas.registry.SCHEMA_DIR", mock_schema_dir):
        validator = SchemaValidator.for_provider("open-provider", strict=True)
        result = validator.validate({"id": "123", "extra": "drift"})

    assert result.is_valid
    assert result.has_drift
    assert "extra" in result.drift_warnings[0]


def test_provider_alias_maps_claude_to_claude_ai(mock_schema_dir):
    """Runtime provider aliases should resolve to canonical schema providers."""
    alias_schema = {
        "type": "object",
        "properties": {"uuid": {"type": "string"}},
        "required": ["uuid"],
        "additionalProperties": False,
    }
    (mock_schema_dir / "claude-ai.schema.json").write_text(json.dumps(alias_schema), encoding="utf-8")

    with patch("polylogue.schemas.registry.SCHEMA_DIR", mock_schema_dir):
        SchemaValidator._cache.clear()
        validator = SchemaValidator.for_provider("claude")

    assert validator.provider == "claude-ai"
    assert "uuid" in validator.schema["properties"]


def test_validation_samples_record_mode_skips_non_record_documents():
    """Record-mode validation should ignore top-level document payloads."""
    validator = SchemaValidator(
        {
            "type": "object",
            "x-polylogue-sample-granularity": "record",
            "properties": {"type": {"type": "string"}},
        },
        provider="claude-code",
    )

    assert validator.validation_samples({"version": 1, "entries": []}, max_samples=16) == []


def test_schema_validator_prefers_registry_latest(monkeypatch):
    """Latest registry schemas should override packaged fallback files."""
    fake_schema = {
        "type": "object",
        "properties": {"from_registry": {"type": "string"}},
        "additionalProperties": False,
    }

    class _FakeRegistry:
        def get_schema(self, provider: str, version: str = "latest"):
            if provider == "chatgpt" and version == "latest":
                return fake_schema
            return None

    monkeypatch.setattr("polylogue.schemas.validator.SchemaRegistry", _FakeRegistry)
    SchemaValidator._cache.clear()

    validator = SchemaValidator.for_provider("chatgpt")
    assert "from_registry" in validator.schema["properties"]


def test_dynamic_key_maps_do_not_emit_drift_warnings():
    """Explicit dynamic-key containers should suppress key-level drift noise."""
    validator = SchemaValidator(
        {
            "type": "object",
            "properties": {
                "mapping": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": {"type": "object"},
                    "x-polylogue-dynamic-keys": True,
                }
            },
            "additionalProperties": False,
        },
        strict=True,
    )

    result = validator.validate(
        {
            "mapping": {
                "550e8400-e29b-41d4-a716-446655440000": {"id": "node-1"},
                "660f9511-f3ac-52e5-b827-557766551111": {"id": "node-2"},
            }
        }
    )
    assert result.is_valid
    assert not result.has_drift


def test_dynamic_identifier_keys_are_suppressed_in_additional_properties_maps():
    """Identifier-like additional-property keys should not count as drift."""
    validator = SchemaValidator(
        {
            "type": "object",
            "properties": {
                "mapping": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"role": {"type": "string"}},
                        "additionalProperties": False,
                    },
                }
            },
            "additionalProperties": False,
        },
        strict=True,
    )

    result = validator.validate(
        {"mapping": {"msg-550e8400-e29b-41d4-a716-446655440000": {"role": "assistant"}}}
    )
    assert result.is_valid
    assert not result.has_drift


def test_additional_properties_schema_still_detects_nested_drift():
    """Nested additionalProperties schemas must still recurse for drift."""
    validator = SchemaValidator(
        {
            "type": "object",
            "properties": {
                "mapping": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "additionalProperties": {"type": "string"},
                    },
                }
            },
            "additionalProperties": False,
        },
        strict=True,
    )

    result = validator.validate({"mapping": {"static-key": {"name": "node", "extra": "drift"}}})
    assert result.is_valid
    assert result.has_drift
    assert any("mapping.static-key.extra" in warning for warning in result.drift_warnings)


def test_validate_non_strict_mode_skips_drift_detection():
    """Non-strict validation should ignore unexpected-field drift."""
    validator = SchemaValidator(
        {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "additionalProperties": {"type": "string"},
        },
        strict=False,
    )

    result = validator.validate({"id": "ok", "extra": "allowed"})
    assert result.is_valid
    assert not result.has_drift


def test_format_error_includes_nested_array_path():
    """Error formatting should preserve nested array/object paths."""
    validator = SchemaValidator(
        {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"],
                    },
                }
            },
        },
        strict=False,
    )

    result = validator.validate({"items": [{}]})
    assert not result.is_valid
    assert any(error.startswith("items.0:") for error in result.errors)


def test_looks_dynamic_key_matches_expected_identifier_patterns():
    """Dynamic-key heuristic should match supported identifier families only."""
    validator = SchemaValidator({"type": "object"})

    assert validator._looks_dynamic_key("550e8400-e29b-41d4-a716-446655440000")
    assert validator._looks_dynamic_key("abcdef0123456789abcdef01")
    assert validator._looks_dynamic_key("msg-550e8400-e29b-41d4-a716-446655440000")
    assert not validator._looks_dynamic_key("title")
    assert not validator._looks_dynamic_key("message_text")


def test_available_providers(mock_schema_dir):
    """available_providers should reflect packaged schema files."""
    with patch("polylogue.schemas.registry.SCHEMA_DIR", mock_schema_dir):
        providers = SchemaValidator.available_providers()

    assert "test-provider" in providers
    assert "open-provider" in providers
    assert "nonexistent" not in providers


def test_validate_helper(mock_schema_dir):
    """validate_provider_export should delegate to SchemaValidator cleanly."""
    with patch("polylogue.schemas.registry.SCHEMA_DIR", mock_schema_dir):
        result = validate_provider_export({"id": "123"}, "test-provider")

    assert result.is_valid


def test_missing_provider_raises():
    """Unknown providers should fail with FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No schema found"):
        SchemaValidator.for_provider("nonexistent-provider")


class TestSyntheticRoundTrip:
    """Synthetic data should remain parser-compatible for supported providers."""

    @pytest.mark.parametrize("provider", ["chatgpt", "claude-code", "claude-ai", "codex", "gemini"])
    def test_synthetic_parses_successfully(self, provider: str, synthetic_source) -> None:
        from polylogue.sources import iter_source_conversations

        source = synthetic_source(provider, count=3, seed=42)
        conversations = list(iter_source_conversations(source))

        assert conversations, f"No conversations parsed for {provider}"
        for conversation in conversations:
            assert conversation.messages, f"Empty conversation for {provider}"
            assert any(message.text for message in conversation.messages), f"No message text for {provider}"


def test_validation_result_properties():
    """ValidationResult convenience behavior should stay stable."""
    valid = ValidationResult(is_valid=True)
    assert valid.is_valid
    assert not valid.has_drift
    valid.raise_if_invalid()

    invalid = ValidationResult(is_valid=False, errors=["missing field"])
    assert not invalid.is_valid
    with pytest.raises(ValueError, match="missing field"):
        invalid.raise_if_invalid()

    with_drift = ValidationResult(is_valid=True, drift_warnings=["new field: foo"])
    assert with_drift.is_valid
    assert with_drift.has_drift

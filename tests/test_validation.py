"""Tests for polylogue.validation functionality."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from polylogue.schemas.validator import SchemaValidator, validate_provider_export


@pytest.fixture
def mock_schema_dir(tmp_path):
    """Create a mock schema directory with test schemas."""
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()

    # Create a simple test schema
    test_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "count": {"type": "integer"},
            "meta": {"type": "object", "properties": {"source": {"type": "string"}}, "additionalProperties": False},
        },
        "required": ["id"],
        "additionalProperties": False,
    }

    (schema_dir / "test-provider.schema.json").write_text(json.dumps(test_schema), encoding="utf-8")

    # Create a schema permitting additional properties
    open_schema = {"type": "object", "properties": {"id": {"type": "string"}}, "additionalProperties": {}}
    (schema_dir / "open-provider.schema.json").write_text(json.dumps(open_schema), encoding="utf-8")

    return schema_dir


@patch("polylogue.schemas.validator.SCHEMA_DIR")
def test_schema_validator_loads_provider(mock_path_attr, mock_schema_dir):
    """SchemaValidator.for_provider loads correct schema."""
    # We need to patch the MODULE LEVEL attributes, assuming they are imported or accessed
    # The code uses SCHEMA_DIR global.
    # We patch it directly.
    pass  # Wait, patch decorator above patches it in the TEST CONTEXT, but we need to patch specifically where it's used.
    # Since we imported SchemaValidator, we should patch `polylogue.schemas.validator.SCHEMA_DIR`

    with patch("polylogue.schemas.validator.SCHEMA_DIR", mock_schema_dir):
        validator = SchemaValidator.for_provider("test-provider")
        assert validator.schema["required"] == ["id"]

        with pytest.raises(FileNotFoundError):
            SchemaValidator.for_provider("nonexistent")


@patch("polylogue.schemas.validator.SCHEMA_DIR")
def test_validate_valid_data(mock_path_attr, mock_schema_dir):
    """Validate returns is_valid=True for valid data."""
    with patch("polylogue.schemas.validator.SCHEMA_DIR", mock_schema_dir):
        validator = SchemaValidator.for_provider("test-provider")

        data = {"id": "123", "count": 10, "meta": {"source": "test"}}
        result = validator.validate(data)

        assert result.is_valid
        assert not result.errors
        assert not result.drift_warnings


@patch("polylogue.schemas.validator.SCHEMA_DIR")
def test_validate_detects_errors(mock_path_attr, mock_schema_dir):
    """Validate detects schema errors."""
    with patch("polylogue.schemas.validator.SCHEMA_DIR", mock_schema_dir):
        validator = SchemaValidator.for_provider("test-provider")

        # Missing required field
        result = validator.validate({"count": 10})
        assert not result.is_valid
        assert any("id" in e for e in result.errors)

        # Type mismatch
        result = validator.validate({"id": 123})  # int instead of string
        assert not result.is_valid
        assert any("123" in e for e in result.errors)


@patch("polylogue.schemas.validator.SCHEMA_DIR")
def test_validate_detects_drift(mock_path_attr, mock_schema_dir):
    """Validate detects drift (unexpected fields) in strict mode."""
    with patch("polylogue.schemas.validator.SCHEMA_DIR", mock_schema_dir):
        validator = SchemaValidator.for_provider("test-provider", strict=True)

        data = {"id": "123", "new_field": "something", "meta": {"source": "test", "extra": "value"}}

        # Schema has additionalProperties=False, so these are ERRORS in validation
        # Wait, if schema forbids additional properties, they are ERRORS, not drift.
        # Drift detection is for when schema ALLOWS them but we want to track them?
        # Or when validator catches them separately?

        # Let's check SchemaValidator logic:
        # SchemaValidator uses `additionalProperties` in schema.
        # Drift logic: "Check for unexpected fields... if key not in schema_props..."
        # If schema has additionalProperties=False, jsonschema validator flags it as ERROR.
        # So drift logic is for when schema is permissive (default) or allows it.

        # Let's use the OPEN schema for drift test.
        validator = SchemaValidator.for_provider("open-provider", strict=True)

        data = {"id": "123", "extra": "drift"}
        result = validator.validate(data)

        # Since additionalProperties=True, it IS VALID
        assert result.is_valid
        # But strict mode should detect drift
        assert result.has_drift
        assert "extra" in result.drift_warnings[0]


def test_available_providers(mock_schema_dir):
    """available_providers lists schema files."""
    with patch("polylogue.schemas.validator.SCHEMA_DIR", mock_schema_dir):
        providers = SchemaValidator.available_providers()
        assert "test-provider" in providers
        assert "open-provider" in providers
        assert "nonexistent" not in providers


def test_validate_helper(mock_schema_dir):
    """validate_provider_export helper works."""
    with patch("polylogue.schemas.validator.SCHEMA_DIR", mock_schema_dir):
        result = validate_provider_export({"id": "123"}, "test-provider")
        assert result.is_valid
